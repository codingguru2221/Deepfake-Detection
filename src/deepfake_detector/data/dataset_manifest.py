from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from deepfake_detector.utils.io import write_json

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _label_from_path(path: Path) -> int:
    text = str(path).lower()
    fake_tags = ["fake", "deepfake", "manipulated", "synthesis"]
    return int(any(tag in text for tag in fake_tags))


def _scan_files(root: Path, extensions: set[str]) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions]


def build_manifest(raw_root: Path) -> Dict[str, List[Dict[str, str | int]]]:
    items = {"images": [], "videos": [], "audio": []}
    for image_path in _scan_files(raw_root, IMAGE_EXTS):
        items["images"].append({"path": str(image_path), "label": _label_from_path(image_path)})
    for video_path in _scan_files(raw_root, VIDEO_EXTS):
        items["videos"].append({"path": str(video_path), "label": _label_from_path(video_path)})
    for audio_path in _scan_files(raw_root, AUDIO_EXTS):
        items["audio"].append({"path": str(audio_path), "label": _label_from_path(audio_path)})
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifest from FaceForensics++ / DFDC style folders.")
    parser.add_argument("--raw-root", type=Path, required=True, help="Path to dataset root.")
    parser.add_argument("--out", type=Path, required=True, help="Path to save JSON manifest.")
    args = parser.parse_args()

    manifest = build_manifest(args.raw_root)
    write_json(manifest, args.out)
    print(f"Saved manifest: {args.out}")
    print({k: len(v) for k, v in manifest.items()})


if __name__ == "__main__":
    main()
