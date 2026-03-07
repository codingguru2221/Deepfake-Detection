from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
VIDEO_FRAME_EXTS = IMAGE_EXTS

POSITIVE_LABEL_HINTS = {"deepfake", "fake", "spoof", "forged", "tampered", "manipulated", "synthetic"}
NEGATIVE_LABEL_HINTS = {"real", "original", "authentic", "bonafide", "genuine", "live"}


@dataclass
class FullTrainResult:
    status: str
    reason: str | None
    downloaded_datasets: int
    manifests: dict[str, str]
    trained_modalities: dict[str, dict[str, Any]]


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip()).strip("-")
    return cleaned.lower() or "dataset"


def _run_command(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode, output.strip()


def _extract_kaggle_slug(url: str) -> str | None:
    # Example: https://www.kaggle.com/datasets/owner/name
    m = re.search(r"kaggle\.com/datasets/([^/?#]+/[^/?#]+)", url, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1)


def _infer_label(path: Path) -> int | None:
    text = str(path).lower()
    if any(tok in text for tok in POSITIVE_LABEL_HINTS):
        return 1
    if any(tok in text for tok in NEGATIVE_LABEL_HINTS):
        return 0
    return None


def _is_video_frame_name(path: Path) -> bool:
    # Existing video trainer expects frame files grouped by "<videoid>_f<num>.jpg"
    return bool(re.search(r"_f\d+$", path.stem))


def _discover_samples(dataset_root: Path) -> dict[str, list[dict[str, Any]]]:
    image_samples: list[dict[str, Any]] = []
    audio_samples: list[dict[str, Any]] = []
    video_frame_samples: list[dict[str, Any]] = []

    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        label = _infer_label(path)
        if label is None:
            continue

        if ext in IMAGE_EXTS:
            row = {"path": str(path.resolve()), "label": int(label)}
            image_samples.append(row)
            if _is_video_frame_name(path):
                video_frame_samples.append(row)
        elif ext in AUDIO_EXTS:
            audio_samples.append({"path": str(path.resolve()), "label": int(label)})

    return {
        "image": image_samples,
        "audio": audio_samples,
        "video": video_frame_samples,
    }


def _balanced(samples: list[dict[str, Any]], min_total: int = 60) -> bool:
    if len(samples) < min_total:
        return False
    labels = {int(s["label"]) for s in samples}
    return labels == {0, 1}


def _write_manifest(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"samples": samples}, ensure_ascii=True, indent=2), encoding="utf-8")


def _parse_auc(output: str) -> float | None:
    m = re.search(r"val_auc=([0-9]*\.?[0-9]+)", output)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def run_full_auto_training(
    project_root: Path,
    catalog_path: Path,
    max_kaggle_datasets: int = 3,
    epochs_image: int = 3,
    epochs_video: int = 2,
) -> FullTrainResult:
    if not catalog_path.exists():
        return FullTrainResult("skipped", "catalog_missing", 0, {}, {})

    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    items = payload.get("items", []) if isinstance(payload, dict) else []
    if not isinstance(items, list):
        return FullTrainResult("skipped", "invalid_catalog", 0, {}, {})

    kaggle_slugs: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        slug = _extract_kaggle_slug(str(item.get("url", "")))
        if slug and slug not in kaggle_slugs:
            kaggle_slugs.append(slug)
        if len(kaggle_slugs) >= max_kaggle_datasets:
            break
    if not kaggle_slugs:
        return FullTrainResult("skipped", "no_kaggle_datasets_found", 0, {}, {})

    kaggle_cmd = shutil.which("kaggle")
    if not kaggle_cmd:
        return FullTrainResult("skipped", "kaggle_cli_not_installed", 0, {}, {})

    download_root = project_root / "data" / "external" / "downloads"
    download_root.mkdir(parents=True, exist_ok=True)

    downloaded_dirs: list[Path] = []
    for slug in kaggle_slugs:
        dest = download_root / _slugify(slug)
        dest.mkdir(parents=True, exist_ok=True)
        cmd = [kaggle_cmd, "datasets", "download", "-d", slug, "-p", str(dest), "--unzip", "-q"]
        code, _ = _run_command(cmd, project_root)
        if code == 0:
            downloaded_dirs.append(dest)

    if not downloaded_dirs:
        return FullTrainResult("skipped", "kaggle_download_failed", 0, {}, {})

    combined: dict[str, list[dict[str, Any]]] = {"image": [], "audio": [], "video": []}
    for ds in downloaded_dirs:
        part = _discover_samples(ds)
        for modality in combined:
            combined[modality].extend(part[modality])

    manifests: dict[str, str] = {}
    trainable: dict[str, Path] = {}
    manifest_root = project_root / "data" / "processed" / "auto"
    for modality, samples in combined.items():
        if not _balanced(samples):
            continue
        manifest_path = manifest_root / f"{modality}_samples.json"
        _write_manifest(manifest_path, samples)
        manifests[modality] = str(manifest_path)
        trainable[modality] = manifest_path

    if not trainable:
        return FullTrainResult("skipped", "no_balanced_labeled_samples_found", len(downloaded_dirs), manifests, {})

    trained_modalities: dict[str, dict[str, Any]] = {}
    for modality, manifest_path in trainable.items():
        if modality == "image":
            out_path = project_root / "models" / "exports" / "image_tf_model.keras"
            epochs = str(epochs_image)
        elif modality == "audio":
            out_path = project_root / "models" / "exports" / "audio_rf.joblib"
            epochs = "1"
        else:
            out_path = project_root / "models" / "checkpoints" / "video_gru.pt"
            epochs = str(epochs_video)

        cmd = [
            sys.executable,
            str(project_root / "src" / "deepfake_detector" / "train.py"),
            "--modality",
            modality,
            "--samples-json",
            str(manifest_path),
            "--out",
            str(out_path),
            "--epochs",
            epochs,
        ]
        code, output = _run_command(cmd, project_root)
        trained_modalities[modality] = {
            "status": "completed" if code == 0 else "failed",
            "out_model_path": str(out_path),
            "val_auc": _parse_auc(output),
            "samples_used": len(combined.get(modality, [])),
        }

    overall = "completed" if any(v.get("status") == "completed" for v in trained_modalities.values()) else "failed"
    reason = None if overall == "completed" else "training_commands_failed"
    return FullTrainResult(overall, reason, len(downloaded_dirs), manifests, trained_modalities)
