from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
from PIL import Image


_MODEL = None
_PROCESSOR = None


def is_enabled() -> bool:
    return os.getenv("HF_DEEPFAKE_ENABLED", "0").lower() not in {"0", "false", "no"}


def _repo_id() -> str:
    return os.getenv("HF_DEEPFAKE_MODEL_ID", "prithivMLmods/deepfake-detector-model-v1")


def _sample_frames() -> int:
    return max(1, int(os.getenv("HF_DEEPFAKE_VIDEO_FRAMES", "8")))


def _load():
    global _MODEL, _PROCESSOR
    if _MODEL is not None and _PROCESSOR is not None:
        return _MODEL, _PROCESSOR

    from transformers import AutoImageProcessor, SiglipForImageClassification

    model_id = _repo_id()
    _MODEL = SiglipForImageClassification.from_pretrained(model_id)
    _PROCESSOR = AutoImageProcessor.from_pretrained(model_id)
    _MODEL.eval()
    return _MODEL, _PROCESSOR


def _predict_pil(image: Image.Image) -> dict[str, Any]:
    import torch

    model, processor = _load()
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()

    fake_prob = float(probs[0])
    real_prob = float(probs[1]) if len(probs) > 1 else 1.0 - fake_prob
    return {
        "prob_fake": fake_prob,
        "prob_real": real_prob,
        "provider": "huggingface",
        "model_id": _repo_id(),
    }


def detect_image_file(image_path: Path) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    return _predict_pil(image)


def detect_image_bytes(image_bytes: bytes) -> dict[str, Any]:
    from io import BytesIO

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return _predict_pil(image)


def _frame_positions(total_frames: int, sample_count: int) -> list[int]:
    if total_frames <= 0:
        return []
    sample_count = min(sample_count, total_frames)
    if sample_count == 1:
        return [0]
    step = max(1, total_frames // sample_count)
    positions = list(range(0, total_frames, step))[:sample_count]
    return positions


def detect_video_file(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    positions = _frame_positions(total_frames, _sample_frames())
    if not positions:
        cap.release()
        raise ValueError("Could not determine video frames for Hugging Face inference.")

    frame_results: list[dict[str, Any]] = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)
        frame_pred = _predict_pil(frame_image)
        frame_results.append(
            {
                "frame_index": pos,
                "prob_fake": frame_pred["prob_fake"],
                "prob_real": frame_pred["prob_real"],
            }
        )
    cap.release()

    if not frame_results:
        raise ValueError("No video frames could be classified by the Hugging Face detector.")

    avg_prob_fake = float(sum(item["prob_fake"] for item in frame_results) / len(frame_results))
    return {
        "prob_fake": avg_prob_fake,
        "frames_sampled": len(frame_results),
        "frame_results": frame_results,
        "provider": "huggingface",
        "model_id": _repo_id(),
    }
