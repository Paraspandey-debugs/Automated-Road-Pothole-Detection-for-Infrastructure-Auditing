#!/usr/bin/env python3
"""
download.py – High-speed direct range downloader for RDD2022 datasets.
Streams individual country zips directly from Figshare CRDDC2022 (DOI: 10.6084/m9.figshare.21431547).

Skips the 10.1 GB Norway zip by default to save bandwidth and storage.

Usage:
    python src/data/download.py                     # Downloads Czech, India, US, Japan
    python src/data/download.py --countries Czech   # Test run on smallest country (245 MB)
    python src/data/download.py --include-norway    # Also includes Norway (~10 GB)
"""

import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Master Figshare direct URL
FIGSHARE_URL = "https://ndownloader.figshare.com/files/38030910"

# Exact byte ranges and sizes of country zips stored uncompressed inside the master archive
# Computed from ZIP32/64 local file headers
COUNTRY_OFFSETS: Dict[str, Tuple[int, int, int]] = {
    # Name: (data_start_byte, data_end_byte, size_bytes)
    "Czech": (352223814, 609341031, 257117218),           # ~245.2 MB
    "India": (609341079, 1136050223, 526709145),          # ~502.3 MB
    "US": (12819836065, 13264172022, 444335958),          # ~423.8 MB
    "United_States": (12819836065, 13264172022, 444335958),
    "Japan": (1136050271, 2208649127, 1072598857),        # ~1022.9 MB
    "China_Drone": (53, 160193593, 160193541),            # ~152.8 MB
    "China_MotorBike": (160193651, 352223766, 192030116),  # ~183.1 MB
    "Norway": (2208649196, 12819836009, 10611186814),     # ~10119.6 MB
}

DEFAULT_COUNTRIES = ["Czech", "India", "US", "Japan"]


def stream_country_zip(country_name: str, dest_zip_path: Path) -> Path:
    """
    Downloads country zip using single-stream HTTP Range request with resume support.
    """
    canon_name = "US" if country_name in ("United_States", "US") else country_name
    entry = COUNTRY_OFFSETS.get(country_name)
    if not entry:
        raise ValueError(f"Unknown country '{country_name}'")

    start_byte, end_byte, total_size = entry
    dest_zip_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    if dest_zip_path.exists():
        downloaded = dest_zip_path.stat().st_size
        if downloaded == total_size:
            log.info("[%s] Zip already fully downloaded (%.1f MB).", canon_name, total_size / (1024 * 1024))
            return dest_zip_path
        if downloaded > total_size:
            dest_zip_path.unlink()
            downloaded = 0
        else:
            log.info("[%s] Resuming download from %.1f MB...", canon_name, downloaded / (1024 * 1024))

    req_start = start_byte + downloaded
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Range": f"bytes={req_start}-{end_byte}",
    }

    log.info(
        "[%s] Streaming %.1f MB (Range: %d-%d)...",
        canon_name,
        (end_byte - req_start + 1) / (1024 * 1024),
        req_start,
        end_byte,
    )

    with requests.get(FIGSHARE_URL, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        mode = "ab" if downloaded else "wb"
        with open(dest_zip_path, mode) as f, tqdm(
            desc=f"Download {canon_name}",
            total=total_size,
            initial=downloaded,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=True,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    return dest_zip_path


def normalise_extracted_dir(dest_dir: Path) -> None:
    """
    Flattens any redundant nested folders so train/images and train/annotations are at root.
    """
    for child in list(dest_dir.iterdir()):
        if child.is_dir() and child.name in (dest_dir.name, "RDD2022", "United_States", "US"):
            for item in child.iterdir():
                target = dest_dir / item.name
                if not target.exists():
                    shutil.move(str(item), str(dest_dir))
            try:
                child.rmdir()
            except OSError:
                pass


def extract_and_verify(zip_path: Path, dest_dir: Path, remove_zip: bool = False) -> None:
    """Unpacks zip archive and normalizes structure."""
    log.info("Extracting %s -> %s...", zip_path.name, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    normalise_extracted_dir(dest_dir)

    train_imgs = dest_dir / "train" / "images"
    train_xmls = dest_dir / "train" / "annotations" / "xmls"
    n_imgs = len(list(train_imgs.glob("*.jpg"))) if train_imgs.exists() else 0
    n_xmls = len(list(train_xmls.glob("*.xml"))) if train_xmls.exists() else 0

    log.info("Finished %s: %d images, %d XML annotations.", dest_dir.name, n_imgs, n_xmls)

    if remove_zip:
        zip_path.unlink()
        log.info("Removed %s to save disk space.", zip_path.name)


def main() -> None:
    p = argparse.ArgumentParser(description="Download RDD2022 datasets.")
    p.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES, choices=list(COUNTRY_OFFSETS.keys()))
    p.add_argument("--include-norway", action="store_true", help="Download Norway (~10 GB).")
    p.add_argument("--data-dir", type=Path, default=Path("data/raw/rdd2022"))
    p.add_argument("--remove-zips", action="store_true", help="Delete zip archives after extraction.")
    args = p.parse_args()

    raw_dir = args.data_dir.resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    countries = list(args.countries)
    if args.include_norway and "Norway" not in countries:
        countries.append("Norway")

    log.info("Target countries to download: %s", countries)

    for country in countries:
        canon_name = "US" if country in ("United_States", "US") else country
        target_dir = raw_dir / canon_name
        train_imgs = target_dir / "train" / "images"

        if train_imgs.exists() and len(list(train_imgs.glob("*.jpg"))) > 100:
            log.info("[%s] Already extracted (%d images found). Skipping.", canon_name, len(list(train_imgs.glob("*.jpg"))))
            continue

        zip_file = raw_dir / f"{canon_name}.zip"
        stream_country_zip(country, zip_file)
        extract_and_verify(zip_file, target_dir, remove_zip=args.remove_zips)

    log.info("=" * 60)
    log.info("All downloads completed. Raw data ready in: %s", raw_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
