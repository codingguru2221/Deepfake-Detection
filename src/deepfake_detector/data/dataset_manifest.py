from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import re

from deepfake_detector.utils.io import write_json

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


_FAKE_TOKENS = {"fake", "deepfake", "manipulated", "synthesis", "synthetic", "generated", "ai", "spoof", "forged"}
_REAL_TOKENS = {"real", "original", "authentic", "bonafide", "genuine", "live"}


def _tokenize_path(path: Path) -> list[str]:
    parts = []
    for part in path.parts:
        for tok in re.split(r"[^a-zA-Z0-9]+", part.lower()):
            if tok:
                parts.append(tok)
    return parts


def _label_from_path(path: Path) -> int:
    tokens = _tokenize_path(path)
    last_real = max((i for i, t in enumerate(tokens) if t in _REAL_TOKENS), default=-1)
    last_fake = max((i for i, t in enumerate(tokens) if t in _FAKE_TOKENS), default=-1)
    if last_real == -1 and last_fake == -1:
        return 0
    if last_fake > last_real:
        return 1
    return 0


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
