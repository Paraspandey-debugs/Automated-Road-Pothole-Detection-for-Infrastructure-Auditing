#!/usr/bin/env python3
"""
labels.py – Parse annotations, audit dataset integrity, and generate labels.csv.

Outputs:
    data/interim/labels.csv  (and symlink/copy to labels.csv)

Columns in labels.csv:
    - image_id: Filename stem
    - image_path: Path to the image file
    - dataset: Source dataset ('rdd2022', 'roboflow_pothole_v1', etc.)
    - country: Country/domain ('Japan', 'India', 'US', 'Czech', 'external', etc.)
    - group_id: Sequence / municipality / session prefix for leakage-free splitting
    - width: Image width
    - height: Image height
    - num_d00: Longitudinal cracks count
    - num_d10: Transverse cracks count
    - num_d20: Alligator cracks count
    - num_d40: Pothole count
    - total_boxes: Total damage boxes
    - has_pothole: Binary classification target (1 if num_d40 >= 1 else 0)
    - has_crack: Crack flag (1 if num_d00 + num_d10 + num_d20 >= 1 else 0)
    - is_hard_negative: Only crack(s), no potholes (1 if has_crack and not has_pothole else 0)
    - is_normal: Completely unblemished/unannotated road (1 if total_boxes == 0 else 0)
"""

import argparse
import logging
import multiprocessing as mp
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DAMAGE_CLASSES = ("D00", "D10", "D20", "D40")


def extract_group_id(filename: str, country: str) -> str:
    """
    Derive a sequence / session / municipality group ID from the filename.
    Prevents spatial/temporal data leakage between train and test splits.
    """
    stem = Path(filename).stem

    match_country_idx = re.match(r"^([a-zA-Z_\-]+?)_?(\d+)$", stem)
    if match_country_idx:
        prefix, num_str = match_country_idx.groups()
        try:
            seq_block = int(num_str) // 50
            return f"{prefix}_seq_{seq_block:04d}"
        except ValueError:
            return prefix

    match_alpha = re.match(r"^([a-zA-Z]+[-_][a-zA-Z]+)", stem)
    if match_alpha:
        return match_alpha.group(1)

    return country


def parse_voc_xml(xml_path: Path) -> Tuple[Optional[Tuple[int, int]], Dict[str, int]]:
    """Parse VOC XML file and return ((width, height), damage_counts)."""
    counts = {cls: 0 for cls in DAMAGE_CLASSES}
    counts["other"] = 0

    if not xml_path or not xml_path.exists():
        return None, counts

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return None, counts

    dims = None
    size = root.find("size")
    if size is not None:
        try:
            w = int(size.find("width").text)
            h = int(size.find("height").text)
            if w > 0 and h > 0:
                dims = (w, h)
        except Exception:
            dims = None

    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is not None and name_elem.text:
            label = name_elem.text.strip()
            if label in counts:
                counts[label] += 1
            else:
                counts["other"] += 1

    return dims, counts


def parse_yolo_txt(txt_path: Path) -> int:
    """Parse a single-class YOLO txt file and return the number of boxes."""
    if not txt_path or not txt_path.exists():
        return 0
    count = 0
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception:
        return 0
    return count


def process_image_record(args: Tuple[Path, Optional[Path], str, str, str]) -> Optional[Dict[str, Any]]:
    """Inspects a single image and annotation file."""
    img_path, annot_path, dataset_type, country, annot_format = args

    if not img_path.exists():
        return None

    w, h = 0, 0
    if annot_format == "voc" and annot_path:
        dims, counts = parse_voc_xml(annot_path)
        if dims:
            w, h = dims

    if w == 0 or h == 0:
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("Corrupted image unreadable by OpenCV: %s", img_path)
            return None
        h, w = img.shape[:2]

    if annot_format == "voc":
        if annot_path:
            _, counts = parse_voc_xml(annot_path)
        else:
            counts = {cls: 0 for cls in DAMAGE_CLASSES}
            counts["other"] = 0
    elif annot_format == "yolo":
        pothole_count = parse_yolo_txt(annot_path) if annot_path else 0
        counts = {"D00": 0, "D10": 0, "D20": 0, "D40": pothole_count, "other": 0}
    else:
        counts = {cls: 0 for cls in DAMAGE_CLASSES}
        counts["other"] = 0

    num_d00 = counts["D00"]
    num_d10 = counts["D10"]
    num_d20 = counts["D20"]
    num_d40 = counts["D40"]
    total_boxes = sum(counts.values())

    has_pothole = 1 if num_d40 > 0 else 0
    has_crack = 1 if (num_d00 + num_d10 + num_d20) > 0 else 0
    is_hard_negative = 1 if (has_crack and not has_pothole) else 0
    is_normal = 1 if total_boxes == 0 else 0

    group_id = extract_group_id(img_path.name, country)

    return {
        "image_id": img_path.stem,
        "image_path": str(img_path.resolve()),
        "dataset": dataset_type,
        "country": country,
        "group_id": group_id,
        "width": w,
        "height": h,
        "num_d00": num_d00,
        "num_d10": num_d10,
        "num_d20": num_d20,
        "num_d40": num_d40,
        "total_boxes": total_boxes,
        "has_pothole": has_pothole,
        "has_crack": has_crack,
        "is_hard_negative": is_hard_negative,
        "is_normal": is_normal,
    }


def gather_rdd2022_records(rdd_raw_dir: Path) -> List[Tuple[Path, Optional[Path], str, str, str]]:
    tasks = []
    if not rdd_raw_dir.exists():
        return tasks

    for country_dir in sorted(rdd_raw_dir.iterdir()):
        if not country_dir.is_dir():
            continue
        cname = country_dir.name
        img_dir = country_dir / "train" / "images"
        xml_dir = country_dir / "train" / "annotations" / "xmls"

        if not img_dir.exists():
            continue

        images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        for img in images:
            xml = xml_dir / f"{img.stem}.xml" if xml_dir.exists() else None
            tasks.append((img, xml, "rdd2022", cname, "voc"))
    return tasks


def gather_extra_records(extra_raw_dir: Path) -> List[Tuple[Path, Optional[Path], str, str, str]]:
    tasks = []
    if not extra_raw_dir.exists():
        return tasks

    for set_dir in sorted(extra_raw_dir.iterdir()):
        if not set_dir.is_dir():
            continue
        set_name = set_dir.name

        images = sorted(
            list(set_dir.rglob("*.jpg")) + list(set_dir.rglob("*.png")) + list(set_dir.rglob("*.jpeg"))
        )
        for img in images:
            txt_candidate = img.with_suffix(".txt")
            if not txt_candidate.exists():
                txt_candidate = img.parent.parent / "labels" / f"{img.stem}.txt"
            lbl = txt_candidate if txt_candidate.exists() else None
            tasks.append((img, lbl, set_name, "external", "yolo"))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labels.csv for pothole audit project.")
    parser.add_argument("--rdd-dir", type=Path, default=Path("data/raw/rdd2022"))
    parser.add_argument("--extra-dir", type=Path, default=Path("data/raw/extra"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/interim/labels.csv"))
    parser.add_argument("--num-workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()

    tasks = []
    tasks.extend(gather_rdd2022_records(args.rdd_dir))
    tasks.extend(gather_extra_records(args.extra_dir))

    if not tasks:
        log.warning("No image files discovered. Check data directories.")
        return

    log.info("Discovered %d candidate image records. Parsing labels...", len(tasks))

    records = []
    with mp.Pool(processes=args.num_workers) as pool:
        for res in tqdm(pool.imap_unordered(process_image_record, tasks, chunksize=100), total=len(tasks)):
            if res is not None:
                records.append(res)

    df = pd.DataFrame(records)
    df.sort_values(by=["dataset", "country", "image_id"], inplace=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    log.info("Saved %d records to %s", len(df), args.output_csv)

    root_csv = Path("labels.csv")
    df.to_csv(root_csv, index=False)
    log.info("Updated top-level copy: %s", root_csv.resolve())

    summary = df.groupby(["dataset", "country"]).agg(
        total_images=("image_id", "count"),
        pothole_images=("has_pothole", "sum"),
        total_pothole_boxes=("num_d40", "sum"),
        hard_negatives=("is_hard_negative", "sum"),
        normal_images=("is_normal", "sum"),
    ).reset_index()

    summary["pothole_img_pct"] = (
        (summary["pothole_images"] / summary["total_images"]) * 100
    ).round(2)

    log.info("=" * 82)
    log.info("DATASET & PER-COUNTRY EDA SUMMARY")
    log.info("=" * 82)
    format_str = "{:<16} {:<12} {:<10} {:<12} {:<10} {:<10} {:<8}"
    log.info(format_str.format("Dataset", "Country", "Images", "Pothole Imgs", "D40 Boxes", "Hard Neg", "Normal"))
    log.info("-" * 82)
    for _, row in summary.iterrows():
        log.info(
            format_str.format(
                row["dataset"][:15],
                row["country"][:11],
                row["total_images"],
                row["pothole_images"],
                row["total_pothole_boxes"],
                row["hard_negatives"],
                row["normal_images"],
            )
        )
    log.info("=" * 82)


if __name__ == "__main__":
    main()
