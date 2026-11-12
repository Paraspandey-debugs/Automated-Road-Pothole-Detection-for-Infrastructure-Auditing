#!/usr/bin/env python3
"""
build.py – Batch-extracts, standardizes, and caches feature matrices for E1–E3.

Workflow:
  1. Loads labels_split.csv or labels.csv.
  2. Concurrently extracts Color + HOG + LBP + GLCM vectors for all images.
  3. Fits StandardScaler strictly on 'train' split.
  4. Transforms train, val, and test splits.
  5. Caches feature matrices and ground truth targets to .npy files in data/interim/features/.

Usage:
    python src/feats/build.py
"""

import argparse
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.feats.color import extract_color_features
from src.feats.glcm import extract_glcm_features
from src.feats.hog import extract_hog_features
from src.feats.lbp import extract_lbp_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def extract_features_single(
    img_path: str,
    img_size: Tuple[int, int] = (224, 224),
    denoise: bool = True,
) -> Tuple[str, Optional[np.ndarray]]:
    """Loads image, applies preprocessing, and concatenates all feature descriptors."""
    img = cv2.imread(img_path)
    if img is None:
        return img_path, None

    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
    if denoise:
        img = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)

    try:
        f_color = extract_color_features(img)
        f_hog = extract_hog_features(img)
        f_lbp = extract_lbp_features(img)
        f_glcm = extract_glcm_features(img)
        f_vector = np.concatenate([f_color, f_hog, f_lbp, f_glcm])
        return img_path, f_vector
    except Exception as e:
        log.warning("Feature extraction failed for %s: %s", img_path, e)
        return img_path, None


def main() -> None:
    p = argparse.ArgumentParser(description="Extract and cache classical features.")
    p.add_argument("--labels-csv", type=Path, default=Path("data/interim/labels_split.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/interim/features"))
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = p.parse_args()

    csv_path = args.labels_csv if args.labels_csv.exists() else Path("labels.csv")
    if not csv_path.exists():
        log.error("Labels CSV %s not found. Run labels.py and splits.py first.", csv_path)
        return

    df = pd.read_csv(csv_path)
    log.info("Loaded %d image entries from %s", len(df), csv_path)

    valid_mask = df["image_path"].apply(lambda p: Path(p).exists())
    df = df[valid_mask].copy()
    log.info("Found %d existing local image files to extract.", len(df))

    if df.empty:
        log.warning("No existing image files found on disk to extract.")
        return

    tasks = df["image_path"].tolist()
    results: Dict[str, np.ndarray] = {}

    log.info("Extracting features using %d CPU workers...", args.workers)
    with mp.Pool(processes=args.workers) as pool:
        for path_str, feat_vec in tqdm(
            pool.imap(extract_features_single, tasks, chunksize=50),
            total=len(tasks),
            desc="Extracting features",
        ):
            if feat_vec is not None:
                results[path_str] = feat_vec

    valid_indices = [idx for idx, path in enumerate(df["image_path"]) if path in results]
    df_aligned = df.iloc[valid_indices].reset_index(drop=True)

    X_all = np.array([results[p] for p in df_aligned["image_path"]], dtype=np.float32)
    y_all = df_aligned["has_pothole"].values.astype(np.int64)

    log.info("Extraction complete. Feature matrix shape: %s", X_all.shape)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if "split" in df_aligned.columns and "train" in df_aligned["split"].values:
        train_mask = (df_aligned["split"] == "train").values
        val_mask = (df_aligned["split"] == "val").values
        test_mask = (df_aligned["split"] == "test").values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_all[train_mask])
        X_val = scaler.transform(X_all[val_mask]) if np.any(val_mask) else np.empty((0, X_all.shape[1]))
        X_test = scaler.transform(X_all[test_mask]) if np.any(test_mask) else np.empty((0, X_all.shape[1]))

        np.save(args.out_dir / "X_train.npy", X_train)
        np.save(args.out_dir / "y_train.npy", y_all[train_mask])
        np.save(args.out_dir / "X_val.npy", X_val)
        np.save(args.out_dir / "y_val.npy", y_all[val_mask])
        np.save(args.out_dir / "X_test.npy", X_test)
        np.save(args.out_dir / "y_test.npy", y_all[test_mask])
        joblib.dump(scaler, args.out_dir / "scaler.pkl")
        log.info("Fitted and saved StandardScaler on train split.")

    np.save(args.out_dir / "X_raw.npy", X_all)
    np.save(args.out_dir / "y_raw.npy", y_all)
    df_aligned.to_csv(args.out_dir / "feature_manifest.csv", index=False)
    log.info("Cached all feature matrices in %s", args.out_dir)


if __name__ == "__main__":
    main()
