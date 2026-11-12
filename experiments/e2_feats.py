#!/usr/bin/env python3
"""
e2_feats.py – Feature Ablation Study.
Evaluates model performance across descriptor subsets:
  1. Color only (HSV stats & histogram)
  2. Color + HOG (Gradients)
  3. Color + HOG + LBP + GLCM (Full descriptor fusion)
  4. PCA-reduced representation (64 components)
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

COLOR_DIM = 9 + 4096         # 9 moments + 4096 3D histogram bins
HOG_DIM = 9 * 13 * 13 * 4    # Default skimage HOG feature dimension for 224x224
LBP_DIM = 18                 # 16 + 2 uniform bins
GLCM_DIM = 10                # 5 properties * (mean, std)


def get_feature_subset(X: np.ndarray, subset_name: str) -> np.ndarray:
    if subset_name == "color_only":
        return X[:, :COLOR_DIM]
    elif subset_name == "color_hog":
        end_idx = min(X.shape[1], COLOR_DIM + HOG_DIM)
        return X[:, :end_idx]
    elif subset_name == "color_hog_lbp_glcm":
        return X
    raise ValueError(f"Unknown feature subset: {subset_name}")


def main():
    feat_dir = Path("data/interim/features")
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train_path = feat_dir / "X_train.npy"
    if not X_train_path.exists():
        log.error("Feature cache %s not found. Run src/feats/build.py first.", X_train_path)
        return

    X_train_raw = np.load(feat_dir / "X_train.npy")
    y_train = np.load(feat_dir / "y_train.npy")
    X_test_raw = np.load(feat_dir / "X_test.npy")
    y_test = np.load(feat_dir / "y_test.npy")

    subsets = ["color_only", "color_hog", "color_hog_lbp_glcm"]
    records = []

    for name in subsets:
        X_tr = get_feature_subset(X_train_raw, name)
        X_te = get_feature_subset(X_test_raw, name)

        clf = RandomForestClassifier(n_estimators=150, class_weight="balanced", n_jobs=-1, random_state=42)
        clf.fit(X_tr, y_train)

        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        f1 = f1_score(y_test, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_test, y_prob)

        records.append({
            "feature_set": name,
            "num_dimensions": X_tr.shape[1],
            "f1_pothole": round(float(f1), 4),
            "pr_auc": round(float(pr_auc), 4),
        })
        log.info("Ablation [%s] (dim=%d) -> F1: %.4f | PR-AUC: %.4f", name, X_tr.shape[1], f1, pr_auc)

    # PCA reduced representation
    n_components = min(64, X_train_raw.shape[1], X_train_raw.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    X_tr_pca = pca.fit_transform(X_train_raw)
    X_te_pca = pca.transform(X_test_raw)

    clf_pca = RandomForestClassifier(n_estimators=150, class_weight="balanced", n_jobs=-1, random_state=42)
    clf_pca.fit(X_tr_pca, y_train)
    y_pred_pca = clf_pca.predict(X_te_pca)
    y_prob_pca = clf_pca.predict_proba(X_te_pca)[:, 1]

    records.append({
        "feature_set": f"pca_{n_components}_components",
        "num_dimensions": n_components,
        "f1_pothole": round(float(f1_score(y_test, y_pred_pca, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob_pca)), 4),
    })

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "e2_feature_ablation.csv", index=False)
    log.info("Saved E2 ablation results to %s", out_dir / "e2_feature_ablation.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
