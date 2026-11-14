#!/usr/bin/env python3
"""
e5_cnn_baseline.py – Deep Convolutional Neural Network Baseline for Pothole Detection.
Fine-tunes a pretrained ResNet-18 on the RDD2022 dataset using zero-leakage sequence splits.

Key features:
  - Exact match with group-stratified splits from data/interim/features/feature_manifest.csv
  - Positive class loss weighting (w_pos = N_neg / N_pos = ~7.27) to address 12% imbalance
  - CosineAnnealingLR scheduling and AdamW optimization
  - Real-time tracking of Accuracy, Precision, Recall, F1 Pothole, ROC-AUC, and PR-AUC
  - Checkpoints best model on Validation PR-AUC
  - Full evaluation on held-out test split (2,323 images)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class RoadDamageDataset(Dataset):
    """PyTorch Dataset loading images and binary pothole targets from manifest."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_paths = self.df["image_path"].tolist()
        self.labels = self.df["has_pothole"].values.astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_p = self.image_paths[idx]
        try:
            image = Image.open(img_p).convert("RGB")
        except Exception as e:
            log.warning("Corrupted image %s (%s). Generating fallback blank tensor.", img_p, e)
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


def get_transforms(img_size: int = 224):
    """Data augmentations for training and deterministic normalization for eval."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


def build_resnet18_model(pretrained: bool = True) -> nn.Module:
    """Instantiates ResNet-18 and adapts classification head for binary prediction."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    # Custom binary head with dropout for regularization
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, 1),
    )
    return model


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluates model over dataloader and returns comprehensive classification metrics."""
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_logits = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            all_targets.append(targets.cpu().numpy())
            all_logits.append(outputs.cpu().numpy())

    total_samples = len(loader.dataset)
    epoch_loss = running_loss / max(1, total_samples)

    y_true = np.vstack(all_targets).flatten()
    logits = np.vstack(all_logits).flatten()
    y_prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -25, 25)))
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1_pot = f1_score(y_true, y_pred, zero_division=0)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

    metrics = {
        "loss": round(float(epoch_loss), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_pothole": round(float(f1_pot), 4),
        "f1_macro": round(float(f1_mac), 4),
        "roc_auc": round(float(roc), 4),
        "pr_auc": round(float(pr_auc), 4),
    }
    return metrics, y_true, y_prob


def plot_training_curves(metrics_df: pd.DataFrame, out_path: Path):
    """Plots training and validation loss, PR-AUC, and F1 progression."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    epochs = metrics_df["epoch"]

    # Panel 1: Losses
    ax1.plot(epochs, metrics_df["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, metrics_df["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Weighted BCE Loss", fontsize=11)
    ax1.set_title("ResNet-18: Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Panel 2: PR-AUC and F1 Score
    ax2.plot(epochs, metrics_df["val_pr_auc"], label="Val PR-AUC", color="#2ca02c", lw=2)
    ax2.plot(epochs, metrics_df["val_f1_pothole"], label="Val F1 (Pothole)", color="#d62728", lw=2, linestyle="--")
    best_pr_idx = metrics_df["val_pr_auc"].idxmax()
    best_pr = metrics_df.loc[best_pr_idx, "val_pr_auc"]
    best_ep = metrics_df.loc[best_pr_idx, "epoch"]
    ax2.scatter([best_ep], [best_pr], color="purple", s=70, zorder=5, label=f"Best PR-AUC ({best_pr:.3f} @ ep {best_ep})")

    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Metric Score", fontsize=11)
    ax2.set_title("ResNet-18: Validation PR-AUC and F1 Score", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    log.info("Saved CNN training curves to %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 CNN Baseline for Road Pothole Detection.")
    parser.add_argument("--manifest", type=Path, default=Path("data/interim/features/feature_manifest.csv"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--models-dir", type=Path, default=Path("results/models"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading dataset manifest from %s...", args.manifest)
    df = pd.read_csv(args.manifest)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    log.info("Partitions: Train=%d, Val=%d, Test=%d", len(train_df), len(val_df), len(test_df))
    n_neg = (train_df["has_pothole"] == 0).sum()
    n_pos = (train_df["has_pothole"] == 1).sum()
    pos_weight_val = float(n_neg / max(1, n_pos))
    log.info("Train class balance: Neg=%d, Pos=%d -> pos_weight=%.2f", n_neg, n_pos, pos_weight_val)

    train_transform, eval_transform = get_transforms(img_size=224)

    train_dataset = RoadDamageDataset(train_df, transform=train_transform)
    val_dataset = RoadDamageDataset(val_df, transform=eval_transform)
    test_dataset = RoadDamageDataset(test_df, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device)
    log.info("Using device: %s", device)

    # Model, weighted loss, optimizer, scheduler
    model = build_resnet18_model(pretrained=True).to(device)
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_pr = -1.0
    best_model_path = args.models_dir / "e5_resnet18_best.pt"
    history = []

    log.info("=" * 65)
    log.info("STARTING RESNET-18 FINE-TUNING (%d EPOCHS)", args.epochs)
    log.info("=" * 65)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

        scheduler.step()
        train_loss = running_train_loss / len(train_dataset)

        # Validation evaluation
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        val_pr = val_metrics["pr_auc"]
        val_f1 = val_metrics["f1_pothole"]
        val_rec = val_metrics["recall"]
        val_prec = val_metrics["precision"]

        log.info(
            "Epoch [%02d/%02d] Train Loss: %.4f | Val Loss: %.4f | Val PR-AUC: %.4f | F1: %.4f | Prec: %.4f | Rec: %.4f",
            epoch, args.epochs, train_loss, val_metrics["loss"], val_pr, val_f1, val_prec, val_rec
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1_pothole": val_metrics["f1_pothole"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "lr": scheduler.get_last_lr()[0],
        })

        if val_pr > best_val_pr:
            best_val_pr = val_pr
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_pr_auc": val_pr,
                "val_f1": val_f1,
            }, best_model_path)
            log.info(">>> New best model saved to %s (Val PR-AUC: %.4f)", best_model_path, val_pr)

    total_training_time = time.time() - start_time
    log.info("Fine-tuning completed in %.1f seconds (%.2f min).", total_training_time, total_training_time / 60)

    # Save training history table & plot curves
    history_df = pd.DataFrame(history)
    history_csv = args.tables_dir / "e5_cnn_training_metrics.csv"
    history_df.to_csv(history_csv, index=False)
    log.info("Saved training history to %s", history_csv)
    plot_training_curves(history_df, args.figures_dir / "e5_cnn_training_curves.png")

    # Load best checkpoint and evaluate on held-out test split
    log.info("=" * 65)
    log.info("TEST SET EVALUATION WITH BEST CHECKPOINT (Val PR-AUC: %.4f)", best_val_pr)
    log.info("=" * 65)
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, y_test_true, y_test_prob = evaluate(model, test_loader, criterion, device)
    test_metrics["model"] = "ResNet-18 (Deep Baseline)"
    test_metrics["best_val_epoch"] = checkpoint["epoch"]
    test_metrics["best_val_pr_auc"] = checkpoint["val_pr_auc"]

    # Save test predictions for ROC/PR curve plotting
    np.savez_compressed(
        args.models_dir / "e5_resnet18_test_preds.npz",
        y_true=y_test_true,
        y_prob=y_test_prob,
    )

    test_df_summary = pd.DataFrame([test_metrics])
    test_csv = args.tables_dir / "e5_cnn_test_metrics.csv"
    test_df_summary.to_csv(test_csv, index=False)
    log.info("Saved test metrics to %s", test_csv)

    log.info("-" * 75)
    log.info("RESNET-18 TEST SET RESULTS:")
    log.info("Accuracy:    %.4f", test_metrics["accuracy"])
    log.info("Precision:   %.4f", test_metrics["precision"])
    log.info("Recall:      %.4f", test_metrics["recall"])
    log.info("F1 Pothole:  %.4f", test_metrics["f1_pothole"])
    log.info("F1 Macro:    %.4f", test_metrics["f1_macro"])
    log.info("ROC-AUC:     %.4f", test_metrics["roc_auc"])
    log.info("PR-AUC:      %.4f", test_metrics["pr_auc"])
    log.info("-" * 75)


if __name__ == "__main__":
    main()
