#!/usr/bin/env python3
"""
voc2yolo.py – Convert Pascal VOC XML annotations to YOLO format.

Key capabilities:
  - Filters strictly for D40 (pothole) as class 0.
  - Automatically creates empty `.txt` label files for background/negative images.
  - Normalizes and clips bounding boxes to [0.0, 1.0].
  - Checks bounding box validity (min/max coordinates, width/height > 0).
  - Uses multiprocessing for fast processing across tens of thousands of images.
  - Prepares YOLO-ready directory structure (symlinks or copies).

Usage:
    python src/data/voc2yolo.py
    python src/data/voc2yolo.py --country Japan --raw-dir data/raw/rdd2022 --out-dir data/yolo
"""

import argparse
import logging
import multiprocessing as mp
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Target class: D40 (pothole) maps to class 0
TARGET_CLASS_NAME = "D40"
TARGET_CLASS_ID = 0


def voc_to_yolo_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_w: int,
    img_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Convert VOC pixel coords to YOLO normalized coords (xc, yc, w, h)."""
    if img_w <= 0 or img_h <= 0:
        return None

    xmin = max(0.0, min(float(xmin), float(img_w)))
    xmax = max(0.0, min(float(xmax), float(img_w)))
    ymin = max(0.0, min(float(ymin), float(img_h)))
    ymax = max(0.0, min(float(ymax), float(img_h)))

    box_w = xmax - xmin
    box_h = ymax - ymin

    if box_w < 1.0 or box_h < 1.0:
        return None

    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    norm_w = box_w / img_w
    norm_h = box_h / img_h

    x_center = max(0.0, min(x_center, 1.0))
    y_center = max(0.0, min(y_center, 1.0))
    norm_w = max(0.0, min(norm_w, 1.0))
    norm_h = max(0.0, min(norm_h, 1.0))

    return x_center, y_center, norm_w, norm_h


def parse_voc_xml(
    xml_path: Path,
    target_class: str = TARGET_CLASS_NAME,
) -> Tuple[Optional[Tuple[int, int]], List[Tuple[float, float, float, float]]]:
    """Parse a VOC XML file. Returns ((img_w, img_h), list of bboxes)."""
    if not xml_path.exists():
        return None, []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        log.warning("Corrupted or unreadable XML %s: %s", xml_path, e)
        return None, []

    img_w, img_h = None, None
    size_elem = root.find("size")
    if size_elem is not None:
        try:
            w_elem = size_elem.find("width")
            h_elem = size_elem.find("height")
            if w_elem is not None and h_elem is not None:
                img_w = int(w_elem.text)
                img_h = int(h_elem.text)
        except (TypeError, ValueError):
            img_w, img_h = None, None

    boxes = []
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None or name_elem.text != target_class:
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            boxes.append((xmin, ymin, xmax, ymax))
        except (TypeError, ValueError, AttributeError):
            continue

    return (img_w, img_h), boxes


def process_single_image(args: Tuple[Path, Optional[Path], Path, Path, bool]) -> Dict[str, int]:
    """Worker task: process one image and write YOLO .txt label."""
    img_path, xml_path, out_img_dir, out_lbl_dir, copy_images = args
    stem = img_path.stem

    out_lbl_path = out_lbl_dir / f"{stem}.txt"
    out_img_path = out_img_dir / img_path.name

    stats = {"total_images": 1, "pothole_images": 0, "total_boxes": 0, "corrupted": 0}

    xml_size = None
    raw_boxes = []
    if xml_path and xml_path.exists():
        xml_size, raw_boxes = parse_voc_xml(xml_path, target_class=TARGET_CLASS_NAME)

    img_w, img_h = xml_size if xml_size and xml_size[0] and xml_size[1] else (None, None)

    if img_w is None or img_h is None or img_w <= 0 or img_h <= 0:
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("Cannot read image %s; skipping.", img_path)
            stats["corrupted"] = 1
            return stats
        img_h, img_w = img.shape[:2]

    yolo_lines = []
    for xmin, ymin, xmax, ymax in raw_boxes:
        bbox = voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h)
        if bbox:
            xc, yc, w, h = bbox
            yolo_lines.append(f"{TARGET_CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    with open(out_lbl_path, "w", encoding="utf-8") as f:
        f.writelines(yolo_lines)

    if yolo_lines:
        stats["pothole_images"] = 1
        stats["total_boxes"] = len(yolo_lines)

    if copy_images:
        if not out_img_path.exists():
            import shutil
            shutil.copy2(img_path, out_img_path)
    else:
        if not out_img_path.exists():
            try:
                rel_target = os.path.relpath(img_path, out_img_dir)
                os.symlink(rel_target, out_img_path)
            except OSError:
                import shutil
                shutil.copy2(img_path, out_img_path)

    return stats


def convert_rdd_country(
    country_dir: Path,
    out_dir: Path,
    copy_images: bool = False,
    num_workers: int = 4,
) -> Dict[str, int]:
    img_dir = country_dir / "train" / "images"
    xml_dir = country_dir / "train" / "annotations" / "xmls"

    if not img_dir.exists():
        log.warning("Image directory not found: %s", img_dir)
        return {}

    out_img_dir = out_dir / country_dir.name / "images"
    out_lbl_dir = out_dir / country_dir.name / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    log.info("Found %d images in %s", len(image_files), country_dir.name)

    tasks = []
    for img_path in image_files:
        xml_path = xml_dir / f"{img_path.stem}.xml" if xml_dir.exists() else None
        tasks.append((img_path, xml_path, out_img_dir, out_lbl_dir, copy_images))

    overall_stats = {"total_images": 0, "pothole_images": 0, "total_boxes": 0, "corrupted": 0}

    with mp.Pool(processes=num_workers) as pool:
        for res in tqdm(
            pool.imap_unordered(process_single_image, tasks, chunksize=100),
            total=len(tasks),
            desc=f"Converting {country_dir.name}",
        ):
            for k in overall_stats:
                overall_stats[k] += res[k]

    return overall_stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert RDD2022 Pascal VOC XML to YOLO format (D40 pothole class only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/rdd2022"),
        help="Root directory where RDD2022 countries are extracted.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/yolo"),
        help="Output directory for YOLO images and labels.",
    )
    p.add_argument(
        "--countries",
        nargs="+",
        default=None,
        help="Specific countries to process. If omitted, processes all found in raw-dir.",
    )
    p.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy image files instead of creating symlinks.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Number of multiprocessing workers.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir: Path = args.raw_dir.resolve()
    out_dir: Path = args.out_dir.resolve()

    if not raw_dir.exists():
        log.error("Raw directory %s does not exist. Run download.py first.", raw_dir)
        sys.exit(1)

    country_dirs = []
    if args.countries:
        for c in args.countries:
            cd = raw_dir / c
            if cd.is_dir():
                country_dirs.append(cd)
            else:
                log.warning("Country directory not found: %s", cd)
    else:
        country_dirs = [d for d in sorted(raw_dir.iterdir()) if d.is_dir()]

    if not country_dirs:
        log.error("No country directories found to process in %s.", raw_dir)
        sys.exit(1)

    log.info("Processing countries: %s", [d.name for d in country_dirs])

    summary = {}
    for cdir in country_dirs:
        stats = convert_rdd_country(
            cdir,
            out_dir,
            copy_images=args.copy_images,
            num_workers=args.num_workers,
        )
        summary[cdir.name] = stats

    log.info("=" * 60)
    log.info("CONVERSION SUMMARY (D40 Potholes Only):")
    log.info(f"{'Country':<18} {'Images':<10} {'Pothole Imgs':<14} {'Boxes':<10}")
    log.info("-" * 60)
    for cname, st in summary.items():
        log.info(
            f"{cname:<18} {st.get('total_images', 0):<10} {st.get('pothole_images', 0):<14} {st.get('total_boxes', 0):<10}"
        )
    log.info("=" * 60)
    log.info("YOLO formatted data created in: %s", out_dir)


if __name__ == "__main__":
    main()
