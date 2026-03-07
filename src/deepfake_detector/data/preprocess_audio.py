from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from pydub import AudioSegment

from deepfake_detector.config import AUDIO
from deepfake_detector.utils.io import ensure_dir, read_json, write_json

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def extract_audio_from_video(video_path: Path, out_path: Path) -> bool:
    try:
        audio = AudioSegment.from_file(video_path)
        audio = audio.set_channels(1).set_frame_rate(AUDIO.sample_rate)
        audio.export(out_path, format="wav")
        return True
    except Exception:
        return False


def normalize_audio(audio_path: Path, out_path: Path) -> bool:
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(AUDIO.sample_rate)
        audio.export(out_path, format="wav")
        return True
    except Exception:
        return False


def run(manifest_path: Path, out_root: Path) -> None:
    manifest = read_json(manifest_path)
    ensure_dir(out_root / "audio" / "real")
    ensure_dir(out_root / "audio" / "fake")
    processed: Dict[str, List[Dict[str, int | str]]] = {"samples": []}

    for row in manifest.get("audio", []):
        label_name = "fake" if row["label"] else "real"
        src = Path(row["path"])
        dst = out_root / "audio" / label_name / f"{src.stem}.wav"
        if normalize_audio(src, dst):
            processed["samples"].append({"path": str(dst), "label": row["label"]})

    for row in manifest.get("videos", []):
        label_name = "fake" if row["label"] else "real"
        src = Path(row["path"])
        if src.suffix.lower() not in VIDEO_EXTS:
            continue
        dst = out_root / "audio" / label_name / f"{src.stem}.wav"
        if extract_audio_from_video(src, dst):
            processed["samples"].append({"path": str(dst), "label": row["label"]})

    write_json(processed, out_root / "audio_samples.json")
    print(f"Saved {len(processed['samples'])} processed audio samples.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess audio for deepfake detection.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.out_root)


if __name__ == "__main__":
    main()
