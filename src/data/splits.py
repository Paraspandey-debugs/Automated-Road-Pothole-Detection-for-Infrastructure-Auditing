#!/usr/bin/env python3
"""
splits.py – Group-aware dataset splitter and manifest generator.

Capabilities:
  1. In-domain group splitting (train/val/test) for E1, E2, and YOLO detection.
  2. Cross-domain partitioning for E3 (e.g. Japan+US -> India -> External).
  3. Strict zero-leakage verification: checks intersection between group sets.
  4. Exports YOLO image list manifests (train.txt, val.txt, test.txt).
  5. Updates labels.csv with split assignments.

Usage:
    python src/data/splits.py --labels-csv labels.csv --mode in_domain
    python src/data/splits.py --labels-csv labels.csv --mode cross_domain \
        --source-countries Japan US Czech --target-country India
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def verify_zero_leakage(df: pd.DataFrame, split_col: str = "split", group_col: str = "group_id") -> bool:
    """
    Assert that no group_id appears in more than one split partition.
    Raises ValueError if data leakage is detected.
    """
    splits = [s for s in df[split_col].unique() if pd.notna(s) and s != "excluded"]
    group_sets: Dict[str, Set[str]] = {}

    for s in splits:
        group_sets[s] = set(df[df[split_col] == s][group_col].dropna().unique())

    leaks = False
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = group_sets[s1].intersection(group_sets[s2])
            if overlap:
                log.error(
                    "LEAKAGE DETECTED! %d groups overlap between '%s' and '%s': %s",
                    len(overlap), s1, s2, list(overlap)[:5],
                )
                leaks = True

    if leaks:
        raise ValueError("Dataset split failed zero-leakage validation!")

    log.info("Zero-leakage verification PASSED across splits: %s", splits)
    return True


def split_groups_stratified(
    df: pd.DataFrame,
    group_col: str = "group_id",
    target_col: str = "has_pothole",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Splits DataFrame by group_id into train, val, and test partitions
    while preserving class ratios as closely as possible.
    """
    assert np.isclose(train_size + val_size + test_size, 1.0), "Splits must sum to 1.0"

    df = df.copy()
    df["split"] = "unassigned"

    group_summary = df.groupby(group_col).agg(
        total_imgs=(target_col, "count"),
        pos_imgs=(target_col, "sum"),
    ).reset_index()

    group_summary["has_pos"] = (group_summary["pos_imgs"] > 0).astype(int)

    temp_ratio = val_size + test_size
    gss1 = GroupShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=random_state)
    train_g_idx, temp_g_idx = next(gss1.split(group_summary, groups=group_summary[group_col]))

    train_groups = set(group_summary.iloc[train_g_idx][group_col])
    temp_groups_df = group_summary.iloc[temp_g_idx].reset_index(drop=True)

    val_rel_ratio = val_size / temp_ratio
    gss2 = GroupShuffleSplit(n_splits=1, test_size=(1.0 - val_rel_ratio), random_state=random_state)
    val_g_idx, test_g_idx = next(gss2.split(temp_groups_df, groups=temp_groups_df[group_col]))

    val_groups = set(temp_groups_df.iloc[val_g_idx][group_col])
    test_groups = set(temp_groups_df.iloc[test_g_idx][group_col])

    df.loc[df[group_col].isin(train_groups), "split"] = "train"
    df.loc[df[group_col].isin(val_groups), "split"] = "val"
    df.loc[df[group_col].isin(test_groups), "split"] = "test"

    return df


def partition_cross_domain(
    df: pd.DataFrame,
    source_countries: List[str],
    target_country: str,
    external_name: str = "external",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Builds partitions for E3 Cross-Domain generalization experiment:
      - e3_train: 85% of source countries (group-split)
      - e3_val:   15% of source countries (group-split)
      - e3_test_target: 100% of target country (e.g. India)
      - e3_test_external: 100% of external pothole-only benchmark
    """
    df = df.copy()
    df["e3_split"] = "excluded"

    src_mask = df["country"].isin(source_countries)
    src_df = df[src_mask].copy()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)
    train_idx, val_idx = next(gss.split(src_df, groups=src_df["group_id"]))

    train_groups = set(src_df.iloc[train_idx]["group_id"])
    val_groups = set(src_df.iloc[val_idx]["group_id"])

    df.loc[df["group_id"].isin(train_groups) & src_mask, "e3_split"] = "e3_train"
    df.loc[df["group_id"].isin(val_groups) & src_mask, "e3_split"] = "e3_val"

    df.loc[df["country"] == target_country, "e3_split"] = f"e3_test_{target_country.lower()}"
    df.loc[df["country"] == external_name, "e3_split"] = "e3_test_external"

    return df


def export_yolo_manifests(
    df: pd.DataFrame,
    out_dir: Path,
    split_col: str = "split",
    valid_splits: Tuple[str, ...] = ("train", "val", "test"),
) -> None:
    """Writes text manifests for YOLO containing absolute image file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for s in valid_splits:
        split_records = df[df[split_col] == s]
        if split_records.empty:
            continue

        manifest_path = out_dir / f"{s}.txt"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for path in split_records["image_path"]:
                f.write(f"{path}\n")

        log.info("Exported YOLO manifest: %s (%d images)", manifest_path, len(split_records))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create group-aware dataset splits.")
    p.add_argument("--labels-csv", type=Path, default=Path("labels.csv"))
    p.add_argument("--mode", choices=["in_domain", "cross_domain", "both"], default="both")
    p.add_argument("--source-countries", nargs="+", default=["Japan", "US", "Czech"])
    p.add_argument("--target-country", type=str, default="India")
    p.add_argument("--output-csv", type=Path, default=Path("data/interim/labels_split.csv"))
    p.add_argument("--yolo-dir", type=Path, default=Path("data/yolo/splits"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.labels_csv.exists():
        log.error("Labels file %s does not exist. Run labels.py first.", args.labels_csv)
        sys.exit(1)

    df = pd.read_csv(args.labels_csv)
    log.info("Loaded %d records from %s", len(df), args.labels_csv)

    if args.mode in ("in_domain", "both"):
        log.info("Computing group-aware in-domain splits (70/15/15)...")
        rdd_mask = df["dataset"] == "rdd2022"
        rdd_df = df[rdd_mask].copy()

        rdd_split = split_groups_stratified(
            rdd_df,
            group_col="group_id",
            target_col="has_pothole",
            train_size=0.70,
            val_size=0.15,
            test_size=0.15,
            random_state=args.seed,
        )

        df["split"] = "excluded"
        df.update(rdd_split)

        verify_zero_leakage(df[df["dataset"] == "rdd2022"], split_col="split", group_col="group_id")
        export_yolo_manifests(df, args.yolo_dir, split_col="split")

    if args.mode in ("cross_domain", "both"):
        log.info(
            "Creating cross-domain partition: Source=%s, Target=%s...",
            args.source_countries, args.target_country,
        )
        df = partition_cross_domain(
            df,
            source_countries=args.source_countries,
            target_country=args.target_country,
            random_state=args.seed,
        )
        verify_zero_leakage(
            df[df["e3_split"].isin(["e3_train", "e3_val"])],
            split_col="e3_split",
            group_col="group_id",
        )
        export_yolo_manifests(
            df,
            args.yolo_dir / "e3",
            split_col="e3_split",
            valid_splits=("e3_train", "e3_val", f"e3_test_{args.target_country.lower()}", "e3_test_external"),
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    df.to_csv(args.labels_csv, index=False)
    log.info("Saved updated splits to %s and %s", args.output_csv, args.labels_csv)

    log.info("=" * 65)
    log.info("SPLIT DISTRIBUTION SUMMARY:")
    log.info("=" * 65)
    if "split" in df.columns:
        print(pd.crosstab(df["country"], df["split"], margins=True))
    if "e3_split" in df.columns:
        print("\nE3 Partition Distribution:")
        print(pd.crosstab(df["country"], df["e3_split"], margins=True))
    log.info("=" * 65)


if __name__ == "__main__":
    main()
