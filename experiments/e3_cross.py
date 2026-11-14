#!/usr/bin/env python3
"""
e3_cross.py – Cross-Domain Generalization Benchmark (Zero-Leakage Group-Aware).
Evaluates domain transfer across geographical regions:
  - In-Domain baseline (Group-aware partition of source countries, e.g. US + Czech)
  - Cross-Domain target (Unseen target domain, e.g. India)
  - Multiple scenarios from configs/experiments.yaml
  - Multiple architectures: Logistic Regression, SVM, Random Forest, XGBoost
  - Multiple random seeds to report Mean ± Std deviation.

Calculates Generalization Drop: Delta = Metric_in - Metric_out.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def evaluate_preds(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    pr = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    return {
        "f1": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "pr_auc": round(float(pr), 4),
    }


def get_model(name: str, seed: int, pos_weight: float):
    if name == "Logistic_Regression":
        return LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=seed)
    elif name == "SVM":
        from sklearn.linear_model import SGDClassifier
        return SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, class_weight="balanced", random_state=seed, max_iter=2000)
    elif name == "Random_Forest":
        return RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=seed)
    elif name == "XGBoost":
        return XGBClassifier(n_estimators=100, learning_rate=0.05, scale_pos_weight=pos_weight, random_state=seed, n_jobs=-1)
    raise ValueError(f"Unknown model name: {name}")


def plot_cross_domain_gap(summary_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    scenarios = summary_df["scenario"].unique()
    models = summary_df["model"].unique()

    x = np.arange(len(scenarios))
    width = 0.8 / len(models)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, m in enumerate(models):
        m_df = summary_df[summary_df["model"] == m]
        drop_vals = [m_df[m_df["scenario"] == sc]["delta_f1_mean"].values[0] if sc in m_df["scenario"].values else 0 for sc in scenarios]
        drop_errs = [m_df[m_df["scenario"] == sc]["delta_f1_std"].values[0] if sc in m_df["scenario"].values else 0 for sc in scenarios]
        plt.bar(x + i * width, drop_vals, width, yerr=drop_errs, capsize=4, label=m, color=colors[i % len(colors)], alpha=0.9)

    plt.xticks(x + width * (len(models) - 1) / 2, [s.replace("_", " ").title() for s in scenarios], fontsize=10)
    plt.ylabel("Cross-Domain Generalization Drop (Delta F1)", fontsize=11)
    plt.title("Cross-Domain Generalization Drop by Scenario and Model (Mean ± Std)", fontsize=12, fontweight="bold")
    plt.axhline(0, color="gray", linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    log.info("Saved cross-domain drop plot to %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 3: Cross-Domain Generalization.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--features-dir", type=Path, default=Path("data/interim/features"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    args = parser.parse_args()

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    scenarios = cfg.get("e3", {}).get("scenarios", [])

    log.info("Loading feature manifest and raw features...")
    manifest_path = args.features_dir / "feature_manifest.csv"
    X_raw_path = args.features_dir / "X_raw.npy"

    df = pd.read_csv(manifest_path)
    X_raw = np.load(X_raw_path)
    y_raw = df["has_pothole"].values

    models = ["Random_Forest", "XGBoost", "Logistic_Regression", "SVM"]
    all_run_records = []

    for sc in scenarios:
        sc_name = sc["name"]
        train_countries = sc["train_countries"]
        test_country = sc["test_country"]

        # Filter countries that actually exist in the dataset
        available_src_countries = [c for c in train_countries if c in df["country"].values]
        if not available_src_countries:
            log.warning("Scenario [%s] skipped: none of source countries %s exist.", sc_name, train_countries)
            continue

        target_mask = (df["country"] == test_country).values
        if not np.any(target_mask):
            log.warning("Scenario [%s] skipped: target country %s has 0 samples.", sc_name, test_country)
            continue

        log.info("=" * 65)
        log.info("E3 Scenario: %s | Source: %s -> Target: %s", sc_name, available_src_countries, test_country)
        log.info("=" * 65)

        src_mask = df["country"].isin(available_src_countries).values
        src_df = df[src_mask].reset_index(drop=True)
        X_src_raw = X_raw[src_mask]
        y_src = y_raw[src_mask]

        X_target_raw = X_raw[target_mask]
        y_target = y_raw[target_mask]

        for seed in args.seeds:
            # Group-aware splitting of source data with zero leakage
            gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
            train_idx, val_idx = next(gss.split(src_df, groups=src_df["group_id"]))

            # Verify zero group leakage between source train and source in-domain test
            train_groups = set(src_df.iloc[train_idx]["group_id"])
            val_groups = set(src_df.iloc[val_idx]["group_id"])
            leak = train_groups.intersection(val_groups)
            if leak:
                raise ValueError(f"Leakage detected in E3 split: {leak}")

            # Fit scaler strictly on source train partition
            scaler = StandardScaler()
            X_src_tr = scaler.fit_transform(X_src_raw[train_idx])
            X_src_te = scaler.transform(X_src_raw[val_idx])  # In-Domain test
            X_tgt = scaler.transform(X_target_raw)           # Cross-Domain test

            y_src_tr = y_src[train_idx]
            y_src_te = y_src[val_idx]

            neg_cnt = np.sum(y_src_tr == 0)
            pos_cnt = np.sum(y_src_tr == 1)
            pos_weight = float(neg_cnt / max(1, pos_cnt))

            for m_name in models:
                clf = get_model(m_name, seed, pos_weight)
                clf.fit(X_src_tr, y_src_tr)

                # 1. In-domain evaluation
                in_pred = clf.predict(X_src_te)
                if hasattr(clf, "predict_proba"):
                    in_prob = clf.predict_proba(X_src_te)[:, 1]
                elif hasattr(clf, "decision_function"):
                    df_v = clf.decision_function(X_src_te)
                    in_prob = 1.0 / (1.0 + np.exp(-np.clip(df_v, -25, 25)))
                else:
                    in_prob = in_pred.astype(float)
                in_metrics = evaluate_preds(y_src_te, in_pred, in_prob)

                # 2. Cross-domain target evaluation
                out_pred = clf.predict(X_tgt)
                if hasattr(clf, "predict_proba"):
                    out_prob = clf.predict_proba(X_tgt)[:, 1]
                elif hasattr(clf, "decision_function"):
                    df_v = clf.decision_function(X_tgt)
                    out_prob = 1.0 / (1.0 + np.exp(-np.clip(df_v, -25, 25)))
                else:
                    out_prob = out_pred.astype(float)
                out_metrics = evaluate_preds(y_target, out_pred, out_prob)

                delta_f1 = round(in_metrics["f1"] - out_metrics["f1"], 4)
                delta_pr = round(in_metrics["pr_auc"] - out_metrics["pr_auc"], 4)

                all_run_records.append({
                    "scenario": sc_name,
                    "model": m_name,
                    "seed": seed,
                    "source_train_samples": len(y_src_tr),
                    "source_test_samples": len(y_src_te),
                    "target_samples": len(y_target),
                    "in_f1": in_metrics["f1"],
                    "out_f1": out_metrics["f1"],
                    "delta_f1": delta_f1,
                    "in_pr_auc": in_metrics["pr_auc"],
                    "out_pr_auc": out_metrics["pr_auc"],
                    "delta_pr_auc": delta_pr,
                    "in_recall": in_metrics["recall"],
                    "out_recall": out_metrics["recall"],
                })

    runs_df = pd.DataFrame(all_run_records)
    runs_df.to_csv(args.tables_dir / "e3_cross_domain_raw_runs.csv", index=False)

    # Compute summary Mean ± Std per scenario and model
    summary = runs_df.groupby(["scenario", "model"]).agg(
        in_f1_mean=("in_f1", "mean"),
        in_f1_std=("in_f1", "std"),
        out_f1_mean=("out_f1", "mean"),
        out_f1_std=("out_f1", "std"),
        delta_f1_mean=("delta_f1", "mean"),
        delta_f1_std=("delta_f1", "std"),
        in_pr_mean=("in_pr_auc", "mean"),
        in_pr_std=("in_pr_auc", "std"),
        out_pr_mean=("out_pr_auc", "mean"),
        out_pr_std=("out_pr_auc", "std"),
        delta_pr_mean=("delta_pr_auc", "mean"),
        delta_pr_std=("delta_pr_auc", "std"),
    ).reset_index().round(4)

    summary_path = args.tables_dir / "e3_cross_domain_drop.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Saved E3 cross-domain summary to %s", summary_path)

    plot_cross_domain_gap(summary, args.figures_dir / "e3_domain_gap.png")

    log.info("=" * 80)
    log.info("EXPERIMENT 3: CROSS-DOMAIN GENERALIZATION DROP (MEAN ± STD)")
    log.info("=" * 80)
    print(summary[["scenario", "model", "in_f1_mean", "out_f1_mean", "delta_f1_mean", "delta_f1_std", "in_pr_mean", "out_pr_mean", "delta_pr_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
