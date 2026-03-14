from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

from deepfake_detector.utils.io import read_json, write_json


def _filter_train_rows(rows: List[Dict[str, str | int]]) -> List[Dict[str, str | int]]:
    filtered = []
    for row in rows:
        path = str(row.get("path", ""))
        if "\\train\\" in path.lower() or "/train/" in path.lower():
            filtered.append(row)
    return filtered


def _sample_per_class(rows: List[Dict[str, str | int]], per_class: int, seed: int) -> List[Dict[str, str | int]]:
    random.seed(seed)
    by_label: Dict[int, List[Dict[str, str | int]]] = {0: [], 1: []}
    for row in rows:
        label = int(row.get("label", 0))
        if label in by_label:
            by_label[label].append(row)
    sampled: List[Dict[str, str | int]] = []
    for label, items in by_label.items():
        if not items:
            continue
        random.shuffle(items)
        sampled.extend(items[:per_class])
    random.shuffle(sampled)
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stratified image sample JSON from Kaggle manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest.json")
    parser.add_argument("--out", type=Path, required=True, help="Path to output image_video_samples.json")
    parser.add_argument("--per-class", type=int, default=500, help="Samples per class (real/fake)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-only", action="store_true", help="Only include images under Train/ folder")
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    rows = manifest.get("images", [])
    if args.train_only:
        rows = _filter_train_rows(rows)

    samples = _sample_per_class(rows, per_class=args.per_class, seed=args.seed)
    write_json({"samples": samples}, args.out)
    print(f"Saved {len(samples)} image samples -> {args.out}")


if __name__ == "__main__":
    main()
