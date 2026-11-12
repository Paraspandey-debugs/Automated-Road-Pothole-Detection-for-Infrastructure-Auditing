#!/usr/bin/env python3
"""
e1_models.py – Benchmark Logistic Regression, SVM (RBF), Random Forest, and XGBoost.
Computes Accuracy, Precision, Recall, F1 (pothole & macro), PR-AUC, ROC-AUC,
and generates confusion matrices and PR curves in results/figures/.
"""
import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def evaluate_model(name: str, model, X_test, y_test, fig_dir: Path) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_pothole": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4) if len(np.unique(y_test)) > 1 else None,
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4) if len(np.unique(y_test)) > 1 else None,
    }

    if metrics["pr_auc"] is not None:
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(rec, prec, label=f"{name} (PR-AUC = {metrics['pr_auc']:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve – {name}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(fig_dir / f"e1_pr_curve_{name}.png", dpi=300)
        plt.close()

    return metrics


def main():
    feat_dir = Path("data/interim/features")
    out_dir = Path("results/tables")
    fig_dir = Path("results/figures")
    model_dir = Path("results/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    X_train_path = feat_dir / "X_train.npy"
    if not X_train_path.exists():
        log.error("Feature cache %s not found. Run src/feats/build.py first.", X_train_path)
        return

    X_train = np.load(feat_dir / "X_train.npy")
    y_train = np.load(feat_dir / "y_train.npy")
    X_test = np.load(feat_dir / "X_test.npy")
    y_test = np.load(feat_dir / "y_test.npy")

    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    pos_weight = float(neg_count / max(1, pos_count))

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "SVM_RBF": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.05, scale_pos_weight=pos_weight, random_state=42),
    }

    results = []
    for name, clf in models.items():
        log.info("Training %s...", name)
        clf.fit(X_train, y_train)
        metrics = evaluate_model(name, clf, X_test, y_test, fig_dir)
        results.append(metrics)
        joblib.dump(clf, model_dir / f"e1_{name.lower()}.joblib")
        log.info("%s -> F1 Pothole: %.4f | PR-AUC: %s", name, metrics["f1_pothole"], metrics["pr_auc"])

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "e1_model_comparison.csv", index=False)
    log.info("Saved E1 benchmark results to %s", out_dir / "e1_model_comparison.csv")
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
