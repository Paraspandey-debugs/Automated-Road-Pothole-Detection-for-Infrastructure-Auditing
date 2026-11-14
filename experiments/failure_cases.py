#!/usr/bin/env python3
"""
failure_cases.py – Generates publication-grade error analysis grid (False Positives and False Negatives).
Visualizes why classical and deep detectors fail on challenging real-world road artifacts:
  - False Positives: Manholes, dark tree shadows, expansion joints, asphalt patches, water puddles.
  - False Negatives: Low-contrast shallow potholes, overexposed glare, small distant cavities.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_yolo_boxes(yolo_dir: Path, country: str, image_id: str, img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
    """Loads normalized boxes and converts to pixel coordinates (xmin, ymin, xmax, ymax)."""
    lbl_path = yolo_dir / country / "labels" / f"{image_id}.txt"
    boxes = []
    if not lbl_path.exists():
        return boxes
    try:
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    xmin = int((xc - w / 2) * img_w)
                    ymin = int((yc - h / 2) * img_h)
                    xmax = int((xc + w / 2) * img_w)
                    ymax = int((yc + h / 2) * img_h)
                    boxes.append((max(0, xmin), max(0, ymin), min(img_w, xmax), min(img_h, ymax)))
    except Exception:
        pass
    return boxes


def make_failure_case_grid(
    manifest_csv: Path,
    X_test_path: Path,
    y_test_path: Path,
    model_path: Path,
    yolo_dir: Path,
    out_path: Path,
    n_examples: int = 3,
) -> None:
    df_manifest = pd.read_csv(manifest_csv)
    test_mask = (df_manifest["split"] == "test").values
    test_df = df_manifest[test_mask].reset_index(drop=True)

    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    log.info("Loading model from %s...", model_path)
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

    # Identify FP: True = 0, Pred = 1 (sorted by confidence descending)
    fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
    fp_sorted = fp_indices[np.argsort(-y_prob[fp_indices])]

    # Identify FN: True = 1, Pred = 0 (sorted by confidence ascending, i.e. most confident mistake)
    fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
    fn_sorted = fn_indices[np.argsort(y_prob[fn_indices])]

    log.info("Found %d False Positives and %d False Negatives.", len(fp_indices), len(fn_indices))

    selected_fps = fp_sorted[:n_examples]
    selected_fns = fn_sorted[:n_examples]

    fig, axes = plt.subplots(2, n_examples, figsize=(4.5 * n_examples, 8))

    # Row 1: False Positives
    for col_idx, idx in enumerate(selected_fps):
        ax = axes[0, col_idx]
        row = test_df.iloc[idx]
        img_p = row["image_path"]
        conf = y_prob[idx]
        cname = row["country"]

        img = cv2.imread(img_p)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        ax.set_title(f"FP: {cname} (Conf: {conf:.2f})\nPred: Pothole | GT: Normal/Crack", fontsize=10, color="darkred", fontweight="bold")
        ax.axis("off")

    # Row 2: False Negatives
    for col_idx, idx in enumerate(selected_fns):
        ax = axes[1, col_idx]
        row = test_df.iloc[idx]
        img_p = row["image_path"]
        conf = y_prob[idx]
        cname = row["country"]
        stem = row["image_id"]

        img = cv2.imread(img_p)
        if img is not None:
            h, w = img.shape[:2]
            boxes = load_yolo_boxes(yolo_dir, cname, stem, w, h)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for (xmin, ymin, xmax, ymax) in boxes:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
            ax.imshow(img)
        ax.set_title(f"FN: {cname} (Conf: {conf:.2f})\nPred: Normal | GT: Pothole [Green Box]", fontsize=10, color="navy", fontweight="bold")
        ax.axis("off")

    plt.suptitle("Qualitative Error Analysis: False Positives (Top) vs. False Negatives (Bottom)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Saved failure cases figure to %s", out_path)


def main():
    p = argparse.ArgumentParser(description="Generate failure case analysis grid.")
    p.add_argument("--manifest", type=Path, default=Path("data/interim/features/feature_manifest.csv"))
    p.add_argument("--features-dir", type=Path, default=Path("data/interim/features"))
    p.add_argument("--model", type=Path, default=Path("results/models/e1_random_forest.joblib"))
    p.add_argument("--yolo-dir", type=Path, default=Path("data/yolo"))
    p.add_argument("--out-fig", type=Path, default=Path("results/figures/failure_cases.png"))
    p.add_argument("--n-examples", type=int, default=3)
    args = p.parse_args()

    # Fallback model if RF not yet trained
    if not args.model.exists():
        fallback_models = list(Path("results/models").glob("*.joblib"))
        if fallback_models:
            args.model = fallback_models[0]
        else:
            log.warning("No trained models found in results/models. Run e1_models.py first.")
            return

    make_failure_case_grid(
        manifest_csv=args.manifest,
        X_test_path=args.features_dir / "X_test.npy",
        y_test_path=args.features_dir / "y_test.npy",
        model_path=args.model,
        yolo_dir=args.yolo_dir,
        out_path=args.out_fig,
        n_examples=args.n_examples,
    )


if __name__ == "__main__":
    main()
