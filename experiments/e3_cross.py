#!/usr/bin/env python3
"""
e3_cross.py – Cross-Domain Generalization Benchmark.
Evaluates model transferability across geographical domains:
  - In-Domain baseline (Japan + Czech + US test split)
  - Cross-Domain target (Trained on Japan + Czech + US -> Evaluated on India)
  - External Domain target (Evaluated on external pothole-only benchmark)
Computes the Domain Generalization Drop: Delta = Metric_in - Metric_out.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    feat_manifest = Path("data/interim/features/feature_manifest.csv")
    X_raw_path = Path("data/interim/features/X_raw.npy")
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not feat_manifest.exists() or not X_raw_path.exists():
        log.error("Feature cache files not found. Run src/feats/build.py first.")
        return

    df = pd.read_csv(feat_manifest)
    X_raw = np.load(X_raw_path)
    y_raw = df["has_pothole"].values

    source_mask = df["country"].isin(["Japan", "US", "Czech"]).values
    target_india_mask = (df["country"] == "India").values
    external_mask = (df["country"] == "external").values

    if not np.any(source_mask):
        log.error("No source country data (Japan/US/Czech) found in feature manifest.")
        return

    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_raw[source_mask])
    y_source = y_raw[source_mask]

    split_cutoff = int(0.8 * len(X_source_scaled))
    X_src_train, X_src_test = X_source_scaled[:split_cutoff], X_source_scaled[split_cutoff:]
    y_src_train, y_src_test = y_source[:split_cutoff], y_source[split_cutoff:]

    log.info("Training Random Forest on source countries (Japan, US, Czech)...")
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42)
    clf.fit(X_src_train, y_src_train)

    in_pred = clf.predict(X_src_test)
    in_prob = clf.predict_proba(X_src_test)[:, 1]
    in_f1 = float(f1_score(y_src_test, in_pred, zero_division=0))
    in_pr = float(average_precision_score(y_src_test, in_prob))

    records = [{
        "evaluation_domain": "In-Domain (Source Test)",
        "samples": len(y_src_test),
        "f1_pothole": round(in_f1, 4),
        "pr_auc": round(in_pr, 4),
        "generalization_drop_f1": 0.0,
    }]

    if np.any(target_india_mask):
        X_india = scaler.transform(X_raw[target_india_mask])
        y_india = y_raw[target_india_mask]
        out_pred = clf.predict(X_india)
        out_prob = clf.predict_proba(X_india)[:, 1]
        out_f1 = float(f1_score(y_india, out_pred, zero_division=0))
        out_pr = float(average_precision_score(y_india, out_prob))

        records.append({
            "evaluation_domain": "Cross-Domain Target (India)",
            "samples": len(y_india),
            "f1_pothole": round(out_f1, 4),
            "pr_auc": round(out_pr, 4),
            "generalization_drop_f1": round(in_f1 - out_f1, 4),
        })

    if np.any(external_mask):
        X_ext = scaler.transform(X_raw[external_mask])
        y_ext = y_raw[external_mask]
        ext_pred = clf.predict(X_ext)
        ext_prob = clf.predict_proba(X_ext)[:, 1]
        ext_f1 = float(f1_score(y_ext, ext_pred, zero_division=0))
        ext_pr = float(average_precision_score(y_ext, ext_prob))

        records.append({
            "evaluation_domain": "External Target (Roboflow)",
            "samples": len(y_ext),
            "f1_pothole": round(ext_f1, 4),
            "pr_auc": round(ext_pr, 4),
            "generalization_drop_f1": round(in_f1 - ext_f1, 4),
        })

    res_df = pd.DataFrame(records)
    res_df.to_csv(out_dir / "e3_cross_domain_drop.csv", index=False)
    log.info("Saved E3 cross-domain results to %s", out_dir / "e3_cross_domain_drop.csv")
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
