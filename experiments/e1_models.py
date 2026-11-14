#!/usr/bin/env python3
"""
e1_models.py – Benchmark Classical Classification Models for Road Pothole Detection.
Models evaluated:
  - Logistic Regression (L2 regularized baseline)
  - Support Vector Machine (RBF kernel)
  - Random Forest (Bagged decision tree ensemble)
  - XGBoost (Gradient boosted decision trees)

Key enhancements:
  - GroupKFold CV hyperparameter tuning using sequence groups (zero-leakage).
  - Mandatory Loss-vs-Epochs curves (XGBoost boosting rounds & SGD log-loss epochs).
  - Metrics: Accuracy, Precision, Recall, F1 Pothole, F1 Macro, ROC-AUC, PR-AUC.
  - Multi-model PR curves, Confusion Matrices, and LaTeX-ready summary tables.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure repository root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_model(name: str, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        df = model.decision_function(X_test)
        y_prob = 1.0 / (1.0 + np.exp(-df))
    else:
        y_prob = y_pred.astype(float)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1_pot = f1_score(y_test, y_pred, zero_division=0)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    roc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0
    pr_auc = average_precision_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0

    metrics = {
        "model": name,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_pothole": round(float(f1_pot), 4),
        "f1_macro": round(float(f1_mac), 4),
        "roc_auc": round(float(roc), 4),
        "pr_auc": round(float(pr_auc), 4),
    }
    return metrics, y_pred, y_prob


def plot_combined_pr_curves(pr_curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]], out_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    for name, (rec, prec, auc_score) in pr_curves.items():
        plt.plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {auc_score:.3f})")
    plt.xlabel("Recall (Pothole Detection Rate)", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curves – Classical Models (Test Set)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    log.info("Saved combined PR curve figure to %s", out_path)


def plot_confusion_matrices(cms: Dict[str, np.ndarray], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(cms), figsize=(4 * len(cms), 3.5))
    if len(cms) == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, cms.items()):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal/Crack", "Pothole"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(name, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    log.info("Saved confusion matrices to %s", out_path)


def plot_loss_curves(
    xgb_train_loss: List[float],
    xgb_val_loss: List[float],
    sgd_train_loss: List[float],
    sgd_val_loss: List[float],
    out_path: Path,
) -> None:
    """Plots training and validation loss curves for XGBoost and SGDClassifier."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: XGBoost Log-Loss vs Boosting Rounds
    rounds = range(1, len(xgb_train_loss) + 1)
    ax1.plot(rounds, xgb_train_loss, label="Train Log-Loss", color="#1f77b4", lw=2)
    ax1.plot(rounds, xgb_val_loss, label="Validation Log-Loss", color="#ff7f0e", lw=2, linestyle="--")
    min_val_round = int(np.argmin(xgb_val_loss)) + 1
    min_val_loss = float(np.min(xgb_val_loss))
    ax1.scatter([min_val_round], [min_val_loss], color="red", zorder=5, label=f"Min Val Loss ({min_val_loss:.4f} @ round {min_val_round})")
    ax1.set_xlabel("Boosting Iteration (Round)", fontsize=11)
    ax1.set_ylabel("Binary Cross-Entropy (Log-Loss)", fontsize=11)
    ax1.set_title("XGBoost: Loss vs. Boosting Rounds", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: SGDClassifier Log-Loss vs Epochs
    epochs = range(1, len(sgd_train_loss) + 1)
    ax2.plot(epochs, sgd_train_loss, label="Train Log-Loss", color="#2ca02c", lw=2)
    ax2.plot(epochs, sgd_val_loss, label="Validation Log-Loss", color="#d62728", lw=2, linestyle="--")
    min_sgd_epoch = int(np.argmin(sgd_val_loss)) + 1
    min_sgd_loss = float(np.min(sgd_val_loss))
    ax2.scatter([min_sgd_epoch], [min_sgd_loss], color="purple", zorder=5, label=f"Min Val Loss ({min_sgd_loss:.4f} @ epoch {min_sgd_epoch})")
    ax2.set_xlabel("Epoch (Full Pass over Training Set)", fontsize=11)
    ax2.set_ylabel("Binary Cross-Entropy (Log-Loss)", fontsize=11)
    ax2.set_title("SGDClassifier (Log-Loss): Loss vs. Epochs", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    log.info("Saved mandatory loss curves figure to %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 1: Classical Classification Models.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--features-dir", type=Path, default=Path("data/interim/features"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--models-dir", type=Path, default=Path("results/models"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config) if args.config.exists() else {}
    e1_cfg = cfg.get("e1", {}).get("models", {})

    log.info("Loading cached feature representations...")
    X_train = np.load(args.features_dir / "X_train.npy")
    y_train = np.load(args.features_dir / "y_train.npy")
    X_val = np.load(args.features_dir / "X_val.npy")
    y_val = np.load(args.features_dir / "y_val.npy")
    X_test = np.load(args.features_dir / "X_test.npy")
    y_test = np.load(args.features_dir / "y_test.npy")

    groups_train = np.load(args.features_dir / "groups_train.npy", allow_pickle=True)

    log.info("Dataset partitions: Train=%s, Val=%s, Test=%s", X_train.shape, X_val.shape, X_test.shape)
    pos_rate_train = float(np.mean(y_train) * 100)
    pos_rate_test = float(np.mean(y_test) * 100)
    log.info("Class balance: Train Pothole=%.2f%%, Test Pothole=%.2f%%", pos_rate_train, pos_rate_test)

    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    pos_weight = float(neg_count / max(1, pos_count))

    # Set up 3-fold GroupKFold for zero-leakage cross-validation
    gkf = GroupKFold(n_splits=3)

    models_and_params = []

    # 1. Logistic Regression
    lr_grid = e1_cfg.get("logistic_regression", {}).get("param_grid", {"C": [0.01, 0.1, 1.0, 10.0]})
    models_and_params.append((
        "Logistic_Regression",
        LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=args.seed),
        lr_grid,
        False,  # Tune on full train set
    ))

    # 2. Random Forest
    rf_grid = e1_cfg.get("random_forest", {}).get("param_grid", {"max_depth": [10, 20, None], "min_samples_split": [2, 5]})
    rf_n_est = e1_cfg.get("random_forest", {}).get("n_estimators", 200)
    models_and_params.append((
        "Random_Forest",
        RandomForestClassifier(n_estimators=rf_n_est, class_weight="balanced", n_jobs=-1, random_state=args.seed),
        rf_grid,
        False,
    ))

    # 3. Support Vector Machine (Linear SVM via SGD hinge loss)
    svm_grid = {"alpha": [1e-4, 1e-3, 1e-2]}
    models_and_params.append((
        "SVM",
        SGDClassifier(loss="hinge", penalty="l2", class_weight="balanced", random_state=args.seed, max_iter=2000),
        svm_grid,
        False,
    ))

    eval_results = []
    pr_curves = {}
    cms = {}
    best_params_dict = {}

    for name, base_clf, param_grid, use_subsample in models_and_params:
        log.info("=" * 60)
        log.info("Hyperparameter Tuning & Training: %s", name)

        if use_subsample:
            # Subsample 2500 samples for fast CV grid search
            np.random.seed(args.seed)
            sub_idx = np.random.choice(len(X_train), size=min(2500, len(X_train)), replace=False)
            cv_X, cv_y, cv_g = X_train[sub_idx], y_train[sub_idx], groups_train[sub_idx]
        else:
            cv_X, cv_y, cv_g = X_train, y_train, groups_train

        gs = GridSearchCV(
            estimator=base_clf,
            param_grid=param_grid,
            cv=gkf,
            scoring="f1",
            n_jobs=-1 if name != "SVM_RBF" else 4,
            verbose=1,
        )
        gs.fit(cv_X, cv_y, groups=cv_g)
        best_clf = gs.best_estimator_
        best_params_dict[name] = gs.best_params_
        log.info("[%s] Best CV Params: %s (CV F1: %.4f)", name, gs.best_params_, gs.best_score_)

        if use_subsample:
            log.info("[%s] Fitting best estimator on complete training set (%d samples)...", name, len(X_train))
            best_clf.fit(X_train, y_train)

        # Save trained model
        joblib.dump(best_clf, args.models_dir / f"e1_{name.lower()}.joblib")

        # Evaluate on test set
        metrics, y_pred, y_prob = evaluate_model(name, best_clf, X_test, y_test)
        metrics["best_params"] = json.dumps(gs.best_params_)
        eval_results.append(metrics)
        log.info("[%s] Test F1 Pothole: %.4f | PR-AUC: %.4f | Recall: %.4f", name, metrics["f1_pothole"], metrics["pr_auc"], metrics["recall"])

        # Record PR curve & CM
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
        pr_curves[name] = (rec_arr, prec_arr, metrics["pr_auc"])
        cms[name] = confusion_matrix(y_test, y_pred)

    # 4. XGBoost with eval_set loss tracking
    log.info("=" * 60)
    log.info("Training XGBoost with eval_set loss monitoring...")
    xgb_n_est = e1_cfg.get("xgboost", {}).get("n_estimators", 200)
    xgb_lr = e1_cfg.get("xgboost", {}).get("learning_rate", 0.05)
    xgb_metric = e1_cfg.get("xgboost", {}).get("eval_metric", "logloss")

    xgb_clf = XGBClassifier(
        n_estimators=xgb_n_est,
        learning_rate=xgb_lr,
        eval_metric=xgb_metric,
        scale_pos_weight=pos_weight,
        random_state=args.seed,
        n_jobs=-1,
    )
    xgb_clf.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    joblib.dump(xgb_clf, args.models_dir / "e1_xgboost.joblib")

    xgb_evals = xgb_clf.evals_result()
    xgb_train_loss = xgb_evals["validation_0"][xgb_metric]
    xgb_val_loss = xgb_evals["validation_1"][xgb_metric]

    xgb_metrics, xgb_pred, xgb_prob = evaluate_model("XGBoost", xgb_clf, X_test, y_test)
    xgb_metrics["best_params"] = json.dumps({"n_estimators": xgb_n_est, "learning_rate": xgb_lr, "scale_pos_weight": round(pos_weight, 2)})
    eval_results.append(xgb_metrics)
    log.info("[XGBoost] Test F1 Pothole: %.4f | PR-AUC: %.4f | Recall: %.4f", xgb_metrics["f1_pothole"], xgb_metrics["pr_auc"], xgb_metrics["recall"])

    prec_arr, rec_arr, _ = precision_recall_curve(y_test, xgb_prob)
    pr_curves["XGBoost"] = (rec_arr, prec_arr, xgb_metrics["pr_auc"])
    cms["XGBoost"] = confusion_matrix(y_test, xgb_pred)

    # 5. SGDClassifier (loss="log_loss") epoch-by-epoch loss curve
    log.info("Computing SGDClassifier per-epoch loss curve...")
    sgd = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        random_state=args.seed,
        warm_start=True,
    )
    sgd_train_loss = []
    sgd_val_loss = []
    epochs = 50
    classes = np.array([0, 1])

    for ep in range(epochs):
        sgd.partial_fit(X_train, y_train, classes=classes)
        tr_prob = sgd.predict_proba(X_train)
        va_prob = sgd.predict_proba(X_val)
        sgd_train_loss.append(float(log_loss(y_train, tr_prob)))
        sgd_val_loss.append(float(log_loss(y_val, va_prob)))

    # Save Mandatory Loss-vs-Epoch Curves Figure
    plot_loss_curves(
        xgb_train_loss,
        xgb_val_loss,
        sgd_train_loss,
        sgd_val_loss,
        args.figures_dir / "e1_loss_curves.png",
    )

    # Save PR Curves & Confusion Matrices
    plot_combined_pr_curves(pr_curves, args.figures_dir / "e1_pr_curves_all.png")
    plot_confusion_matrices(cms, args.figures_dir / "e1_confusion_matrices.png")

    # Save summary table
    res_df = pd.DataFrame(eval_results)
    out_table_path = args.tables_dir / "e1_model_comparison.csv"
    res_df.to_csv(out_table_path, index=False)
    log.info("=" * 75)
    log.info("EXPERIMENT 1: BENCHMARK RESULTS")
    log.info("=" * 75)
    print(res_df[["model", "accuracy", "precision", "recall", "f1_pothole", "f1_macro", "roc_auc", "pr_auc"]].to_string(index=False))
    log.info("Saved E1 results table to %s", out_table_path)


if __name__ == "__main__":
    main()
