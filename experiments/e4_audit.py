#!/usr/bin/env python3
"""
e4_audit.py – YOLOv8 Fine-tuning & Road Infrastructure Severity Audit.
Steps:
  1. Fine-tunes YOLOv8n on D40 pothole detection using configs/data.yaml.
  2. Runs batch inference across test images to predict pothole bounding boxes.
  3. Computes image severity index: Severity = sum(box_area) / image_area.
  4. Aggregates per-image severity into street segments (group_id).
  5. Computes Spearman rank correlation between Ground Truth vs Predicted segment severity rankings.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def train_yolo(epochs: int = 50, imgsz: int = 640, data_yaml: str = "configs/data.yaml"):
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics package not installed. Run `pip install ultralytics`.")
        return None

    log.info("Initializing YOLOv8n for pothole detection fine-tuning...")
    model = YOLO("yolov8n.pt")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        name="pothole_yolov8n",
        save=True,
        plots=True,
    )
    return model


def compute_severity_audit(labels_csv: Path, model_weights: Path, out_dir: Path) -> pd.DataFrame:
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics package not installed. Run `pip install ultralytics`.")
        return pd.DataFrame()

    df = pd.read_csv(labels_csv)
    test_df = df[df["split"] == "test"].copy() if "split" in df.columns else df.tail(500).copy()

    log.info("Loading trained weights: %s", model_weights)
    model = YOLO(str(model_weights))

    pred_severities = []
    log.info("Running detection inference on %d test images...", len(test_df))

    for _, row in test_df.iterrows():
        img_p = row["image_path"]
        w = row["width"]
        h = row["height"]
        img_area = float(w * h) if w > 0 and h > 0 else 1.0

        results = model.predict(img_p, verbose=False, conf=0.25)
        boxes = results[0].boxes
        if len(boxes) > 0:
            xywh = boxes.xywh.cpu().numpy()
            total_box_area = np.sum(xywh[:, 2] * xywh[:, 3])
            severity = total_box_area / img_area
        else:
            severity = 0.0

        pred_severities.append(float(severity))

    test_df["pred_severity"] = pred_severities
    test_df["gt_severity"] = test_df["num_d40"]

    segment_audit = test_df.groupby("group_id").agg(
        total_frames=("image_id", "count"),
        gt_segment_severity=("gt_severity", "mean"),
        pred_segment_severity=("pred_severity", "mean"),
    ).reset_index()

    rho, pval = spearmanr(segment_audit["gt_segment_severity"], segment_audit["pred_segment_severity"])
    log.info("=" * 60)
    log.info("INFRASTRUCTURE SEVERITY AUDIT RANKING CORRELATION:")
    log.info("Spearman Rank Correlation (rho): %.4f (p-value: %.4e)", rho, pval)
    log.info("=" * 60)

    out_dir.mkdir(parents=True, exist_ok=True)
    segment_audit.to_csv(out_dir / "e4_segment_audit_ranking.csv", index=False)
    log.info("Saved severity audit rankings to %s", out_dir / "e4_segment_audit_ranking.csv")
    return segment_audit


def main():
    p = argparse.ArgumentParser(description="Run E4 detection and severity audit.")
    p.add_argument("--train", action="store_true", help="Fine-tune YOLOv8n model.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--weights", type=Path, default=Path("runs/detect/pothole_yolov8n/weights/best.pt"))
    p.add_argument("--labels-csv", type=Path, default=Path("labels.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    args = p.parse_args()

    if args.train:
        train_yolo(epochs=args.epochs)

    if args.weights.exists():
        compute_severity_audit(args.labels_csv, args.weights, args.out_dir)
    else:
        log.warning("Weights %s not found. Run with --train first to fine-tune.", args.weights)


if __name__ == "__main__":
    main()
