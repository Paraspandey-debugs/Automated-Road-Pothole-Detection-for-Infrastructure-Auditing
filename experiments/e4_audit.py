#!/usr/bin/env python3
"""
e4_audit.py – YOLOv8 Fine-tuning & Road Infrastructure Severity Audit.
Steps:
  1. Fine-tunes YOLOv8n on D40 pothole detection using configs/data.yaml.
  2. Runs batch inference across test images to predict pothole bounding boxes.
  3. Computes image ground-truth severity: GT Area Ratio = sum(gt_box_w * gt_box_h).
  4. Computes image predicted severity: Pred Area Ratio = sum(pred_box_w * pred_box_h).
  5. Aggregates frame-level severities into street segments (group_id).
  6. Computes Spearman rank correlation between Ground Truth vs Predicted segment severity rankings.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def train_yolo(
    epochs: int = 10,
    imgsz: int = 512,
    batch: int = 8,
    workers: int = 2,
    data_yaml: str = "configs/data.yaml",
    device: str = "0",
):
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics package not installed. Run `pip install ultralytics`.")
        return None

    log.info(
        "Initializing YOLOv8n for pothole detection fine-tuning (%d epochs, batch=%d, workers=%d, imgsz=%d, device=%s)...",
        epochs,
        batch,
        workers,
        imgsz,
        device,
    )
    model = YOLO("yolov8n.pt")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        name="pothole_yolov8n",
        save=True,
        plots=True,
        device=device,
        verbose=True,
        exist_ok=True,
    )
    return model


def get_ground_truth_box_area(yolo_dir: Path, country: str, image_id: str) -> float:
    """Reads normalized YOLO label file and returns sum of bounding box areas (w * h)."""
    lbl_path = yolo_dir / country / "labels" / f"{image_id}.txt"
    if not lbl_path.exists():
        return 0.0
    total_area = 0.0
    try:
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    w, h = float(parts[3]), float(parts[4])
                    total_area += w * h
    except Exception:
        return 0.0
    return float(total_area)


def compute_severity_audit(
    labels_csv: Path,
    model_weights: Path,
    yolo_dir: Path,
    out_table_dir: Path,
    out_fig_dir: Path,
    device: str = "0",
) -> pd.DataFrame:
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics package not installed. Run `pip install ultralytics`.")
        return pd.DataFrame()

    df = pd.read_csv(labels_csv)
    test_df = df[df["split"] == "test"].copy() if "split" in df.columns else df.tail(500).copy()

    log.info("Loading trained weights from: %s", model_weights)
    model = YOLO(str(model_weights))

    pred_severities = []
    gt_severities = []

    log.info("Running detection inference and computing area ratios on %d test images...", len(test_df))

    # Pre-extract all image paths and compute ground-truth box areas
    image_paths = []
    for _, row in test_df.iterrows():
        image_paths.append(str(row["image_path"]))
        cname = row["country"]
        stem = row["image_id"]
        gt_area = get_ground_truth_box_area(yolo_dir, cname, stem)
        gt_severities.append(gt_area)

    # Batch prediction for maximum throughput and memory efficiency
    batch_size = 32
    for i in range(0, len(image_paths), batch_size):
        chunk = image_paths[i : i + batch_size]
        try:
            results = model.predict(chunk, verbose=False, conf=0.25, device=device, batch=len(chunk))
            for res in results:
                boxes = res.boxes
                if len(boxes) > 0:
                    xywhn = boxes.xywhn.cpu().numpy()
                    pred_area = float(np.sum(xywhn[:, 2] * xywhn[:, 3]))
                else:
                    pred_area = 0.0
                pred_severities.append(pred_area)
        except Exception as e:
            log.warning("Batch prediction fallback due to: %s", e)
            for img_p in chunk:
                pred_severities.append(0.0)

    test_df["gt_area_severity"] = gt_severities
    test_df["pred_area_severity"] = pred_severities

    # Aggregate by street segment / sequence
    segment_audit = test_df.groupby("group_id").agg(
        total_frames=("image_id", "count"),
        gt_segment_severity=("gt_area_severity", "mean"),
        pred_segment_severity=("pred_area_severity", "mean"),
        total_gt_potholes=("num_d40", "sum"),
    ).reset_index()

    rho, pval = spearmanr(segment_audit["gt_segment_severity"], segment_audit["pred_segment_severity"])
    log.info("=" * 65)
    log.info("INFRASTRUCTURE SEVERITY AUDIT RANKING CORRELATION:")
    log.info("Spearman Rank Correlation (rho): %.4f (p-value: %.4e)", rho, pval)
    log.info("=" * 65)

    out_table_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    segment_audit.to_csv(out_table_dir / "e4_segment_audit_ranking.csv", index=False)
    log.info("Saved severity audit rankings to %s", out_table_dir / "e4_segment_audit_ranking.csv")

    # Plot Severity Ranking Scatter and Correlation
    plt.figure(figsize=(7, 6))
    plt.scatter(
        segment_audit["gt_segment_severity"],
        segment_audit["pred_segment_severity"],
        color="#1f77b4",
        alpha=0.7,
        edgecolors="k",
        s=40,
    )
    plt.xlabel("Ground Truth Road Segment Severity (Mean Pothole Area Ratio)", fontsize=11)
    plt.ylabel("Predicted Road Segment Severity (Mean Pothole Area Ratio)", fontsize=11)
    plt.title(f"Infrastructure Audit Severity Correlation\nSpearman's $\\rho = {rho:.3f}$ ($p = {pval:.2e}$)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Trend line
    if len(segment_audit) > 1 and np.std(segment_audit["gt_segment_severity"]) > 1e-6:
        m, b = np.polyfit(segment_audit["gt_segment_severity"], segment_audit["pred_segment_severity"], 1)
        x_vals = np.linspace(segment_audit["gt_segment_severity"].min(), segment_audit["gt_segment_severity"].max(), 50)
        plt.plot(x_vals, m * x_vals + b, color="red", linestyle="--", lw=1.5, label=f"Trend Line (slope={m:.2f})")
        plt.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(out_fig_dir / "e4_severity_ranking_correlation.png", dpi=300)
    plt.close()
    log.info("Saved correlation plot to %s", out_fig_dir / "e4_severity_ranking_correlation.png")

    return segment_audit


def plot_yolo_training_curves(results_csv: Path, out_fig_dir: Path, out_table_dir: Path):
    """Parses ultralytics results.csv and plots training/validation loss curves."""
    if not results_csv.exists():
        log.warning("results.csv not found at %s", results_csv)
        return

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    out_fig_dir.mkdir(parents=True, exist_ok=True)
    out_table_dir.mkdir(parents=True, exist_ok=True)

    # Copy / save clean metrics table
    clean_cols = [c for c in df.columns if any(k in c for k in ["epoch", "loss", "mAP", "precision", "recall"])]
    df[clean_cols].to_csv(out_table_dir / "e4_yolo_training_metrics.csv", index=False)
    log.info("Saved YOLO training metrics to %s", out_table_dir / "e4_yolo_training_metrics.csv")

    epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss Curves (Train vs Val)
    ax1 = axes[0]
    if "train/box_loss" in df.columns and "val/box_loss" in df.columns:
        ax1.plot(epochs, df["train/box_loss"], label="Train Box Loss", color="#1f77b4", lw=2)
        ax1.plot(epochs, df["val/box_loss"], label="Val Box Loss", color="#1f77b4", linestyle="--", lw=2)
    if "train/cls_loss" in df.columns and "val/cls_loss" in df.columns:
        ax1.plot(epochs, df["train/cls_loss"], label="Train Cls Loss", color="#ff7f0e", lw=2)
        ax1.plot(epochs, df["val/cls_loss"], label="Val Cls Loss", color="#ff7f0e", linestyle="--", lw=2)
    if "train/dfl_loss" in df.columns and "val/dfl_loss" in df.columns:
        ax1.plot(epochs, df["train/dfl_loss"], label="Train DFL Loss", color="#2ca02c", lw=2)
        ax1.plot(epochs, df["val/dfl_loss"], label="Val DFL Loss", color="#2ca02c", linestyle="--", lw=2)

    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("YOLOv8n Training & Validation Loss Curves", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # mAP & Recall Curves
    ax2 = axes[1]
    if "metrics/mAP50(B)" in df.columns:
        ax2.plot(epochs, df["metrics/mAP50(B)"], label="mAP@0.5", color="#d62728", lw=2.5)
    if "metrics/mAP50-95(B)" in df.columns:
        ax2.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", color="#9467bd", lw=2)
    if "metrics/recall(B)" in df.columns:
        ax2.plot(epochs, df["metrics/recall(B)"], label="Recall", color="#8c564b", linestyle=":", lw=2)

    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Metric Score", fontsize=11)
    ax2.set_title("YOLOv8n Validation Detection Metrics", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_fig_dir / "e4_yolo_loss_curves.png", dpi=300)
    plt.close()
    log.info("Saved YOLO loss curves to %s", out_fig_dir / "e4_yolo_loss_curves.png")


def main():
    p = argparse.ArgumentParser(description="Run E4 detection and severity audit.")
    p.add_argument("--train", action="store_true", help="Fine-tune YOLOv8n model.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--weights", type=Path, default=Path("runs/detect/pothole_yolov8n/weights/best.pt"))
    p.add_argument("--labels-csv", type=Path, default=Path("labels.csv"))
    p.add_argument("--yolo-dir", type=Path, default=Path("data/yolo"))
    p.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    p.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    args = p.parse_args()

    if args.train:
        train_yolo(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
        )

    results_csv = Path("runs/detect/pothole_yolov8n/results.csv")
    if results_csv.exists():
        plot_yolo_training_curves(results_csv, args.figures_dir, args.tables_dir)

    if args.weights.exists():
        compute_severity_audit(args.labels_csv, args.weights, args.yolo_dir, args.tables_dir, args.figures_dir, device=args.device)
    else:
        log.warning("Weights %s not found. Run with --train first to fine-tune.", args.weights)


if __name__ == "__main__":
    main()
