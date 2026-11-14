#!/usr/bin/env python3
"""
e2_feats.py – Feature Ablation Study across Classical Descriptors & Models.
Evaluates model performance across descriptor representations:
  1. Color only (HSV moments + 3D color histogram, 4,105 dims)
  2. Color + HOG (Gradients & structural edges, 10,189 dims)
  3. Color + HOG + LBP + GLCM (Full multi-modal fusion, 10,217 dims)
  4. PCA-reduced representation (config-driven n_components, e.g. 64)

Models evaluated:
  - Random Forest
  - XGBoost
  - Logistic Regression
  - SVM

Outputs:
  - results/tables/e2_feature_ablation.csv
  - results/figures/e2_feature_ablation.png
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
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

COLOR_DIM = 9 + 4096         # 4,105
HOG_DIM = 9 * 13 * 13 * 4    # 6,084
LBP_DIM = 18                 # 18
GLCM_DIM = 10                # 10


def get_feature_subset(X: np.ndarray, subset_name: str) -> np.ndarray:
    if subset_name == "color_only":
        return X[:, :COLOR_DIM]
    elif subset_name == "color_hog":
        end_idx = min(X.shape[1], COLOR_DIM + HOG_DIM)
        return X[:, :end_idx]
    elif subset_name == "color_hog_lbp_glcm":
        return X
    raise ValueError(f"Unknown feature subset: {subset_name}")


def plot_ablation_chart(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 5.5))

    models = df["model"].unique()
    subsets = df["feature_set"].unique()

    x = np.arange(len(subsets))
    width = 0.8 / len(models)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, m in enumerate(models):
        m_df = df[df["model"] == m]
        # Align subsets
        f1_vals = [m_df[m_df["feature_set"] == s]["f1_pothole"].values[0] if s in m_df["feature_set"].values else 0 for s in subsets]
        plt.bar(x + i * width, f1_vals, width, label=m, color=colors[i % len(colors)], alpha=0.9)

    clean_labels = [s.replace("_", " ").title() for s in subsets]
    plt.xticks(x + width * (len(models) - 1) / 2, clean_labels, fontsize=10)
    plt.ylabel("F1 Score (Pothole Class)", fontsize=11)
    plt.title("Feature Ablation Study: Performance across Descriptors and Models", fontsize=12, fontweight="bold")
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    log.info("Saved ablation plot to %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 2: Feature Ablation Study.")
    parser.add_argument("--features-config", type=Path, default=Path("configs/features.yaml"))
    parser.add_argument("--experiments-config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--features-dir", type=Path, default=Path("data/interim/features"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # Read n_components from config
    pca_components = 64
    if args.experiments_config.exists():
        with open(args.experiments_config, "r", encoding="utf-8") as f:
            ecfg = yaml.safe_load(f)
            e2_sets = ecfg.get("e2", {}).get("sets", [])
            for s in e2_sets:
                if s.get("name") == "pca_reduced":
                    pca_components = s.get("n_components", 64)

    log.info("Loading cached feature representations...")
    X_train_raw = np.load(args.features_dir / "X_train.npy")
    y_train = np.load(args.features_dir / "y_train.npy")
    X_test_raw = np.load(args.features_dir / "X_test.npy")
    y_test = np.load(args.features_dir / "y_test.npy")

    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    pos_weight = float(neg_count / max(1, pos_count))

    feature_subsets = [
        ("color_only", "Color Only (HSV Stats & Hist)"),
        ("color_hog", "Color + HOG (Gradients)"),
        ("color_hog_lbp_glcm", "Color + HOG + LBP + GLCM (Full)"),
    ]

    records = []

    # Prepare PCA reduction
    n_components = min(pca_components, X_train_raw.shape[1], X_train_raw.shape[0])
    log.info("Computing PCA dimensionality reduction with %d components...", n_components)
    pca = PCA(n_components=n_components, random_state=args.seed)
    X_train_pca = pca.fit_transform(X_train_raw)
    X_test_pca = pca.transform(X_test_raw)

    all_sets = feature_subsets + [(f"pca_{n_components}", f"PCA Reduced ({n_components} dims)")]

    models_to_run = [
        ("Random_Forest", lambda: RandomForestClassifier(n_estimators=150, class_weight="balanced", n_jobs=-1, random_state=args.seed)),
        ("XGBoost", lambda: XGBClassifier(n_estimators=150, learning_rate=0.05, scale_pos_weight=pos_weight, random_state=args.seed, n_jobs=-1)),
        ("Logistic_Regression", lambda: LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=args.seed)),
    ]

    for model_name, model_fn in models_to_run:
        log.info("=" * 60)
        log.info("Running ablation for Model: %s", model_name)

        for set_key, set_desc in all_sets:
            if set_key.startswith("pca"):
                X_tr, X_te = X_train_pca, X_test_pca
            else:
                X_tr = get_feature_subset(X_train_raw, set_key)
                X_te = get_feature_subset(X_test_raw, set_key)

            clf = model_fn()
            clf.fit(X_tr, y_train)

            y_pred = clf.predict(X_te)
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_te)[:, 1]
            elif hasattr(clf, "decision_function"):
                df_val = clf.decision_function(X_te)
                y_prob = 1.0 / (1.0 + np.exp(-df_val))
            else:
                y_prob = y_pred.astype(float)

            f1 = f1_score(y_test, y_pred, zero_division=0)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            pr_auc = average_precision_score(y_test, y_prob)

            records.append({
                "model": model_name,
                "feature_set": set_key,
                "description": set_desc,
                "num_dimensions": X_tr.shape[1],
                "f1_pothole": round(float(f1), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "pr_auc": round(float(pr_auc), 4),
            })
            log.info("[%s | %s] Dims=%d -> F1: %.4f | PR-AUC: %.4f", model_name, set_key, X_tr.shape[1], f1, pr_auc)

    res_df = pd.DataFrame(records)
    out_table = args.tables_dir / "e2_feature_ablation.csv"
    res_df.to_csv(out_table, index=False)
    log.info("Saved ablation table to %s", out_table)

    plot_ablation_chart(res_df, args.figures_dir / "e2_feature_ablation.png")

    log.info("=" * 75)
    log.info("EXPERIMENT 2: FEATURE ABLATION SUMMARY")
    log.info("=" * 75)
    print(res_df[["model", "feature_set", "num_dimensions", "f1_pothole", "precision", "recall", "pr_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
