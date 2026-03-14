from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def is_enabled() -> bool:
    return bool(os.getenv("AWS_REKOGNITION_PROJECT_VERSION_ARN"))


def _region() -> str:
    return os.getenv("AWS_REGION", "us-east-1")


def _min_confidence() -> float:
    return float(os.getenv("AWS_REKOGNITION_MIN_CONFIDENCE", "50"))


def _sample_frames() -> int:
    return max(1, int(os.getenv("AWS_REKOGNITION_VIDEO_FRAMES", "8")))


def _label_aliases(kind: str, defaults: str) -> set[str]:
    raw = os.getenv(kind, defaults)
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def _deepfake_aliases() -> set[str]:
    return _label_aliases(
        "AWS_REKOGNITION_DEEPFAKE_LABELS",
        "deepfake,fake,manipulated,synthetic,generated,ai_generated",
    )


def _real_aliases() -> set[str]:
    return _label_aliases("AWS_REKOGNITION_REAL_LABELS", "real,authentic,genuine,human")


def _normalize_label(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _client():
    import boto3

    return boto3.client("rekognition", region_name=_region())


def _classify_labels(labels: list[dict[str, Any]]) -> dict[str, Any]:
    deepfake_aliases = _deepfake_aliases()
    real_aliases = _real_aliases()
    top_label = None
    top_conf = 0.0
    deepfake_conf = 0.0
    real_conf = 0.0

    for item in labels:
        name = str(item.get("Name") or "")
        normalized = _normalize_label(name)
        confidence = float(item.get("Confidence") or 0.0)
        if confidence > top_conf:
            top_conf = confidence
            top_label = name
        if normalized in deepfake_aliases:
            deepfake_conf = max(deepfake_conf, confidence)
        if normalized in real_aliases:
            real_conf = max(real_conf, confidence)

    if deepfake_conf > 0.0:
        prob_fake = deepfake_conf / 100.0
    elif real_conf > 0.0:
        prob_fake = 1.0 - (real_conf / 100.0)
    elif top_label:
        normalized = _normalize_label(top_label)
        if normalized in deepfake_aliases:
            prob_fake = top_conf / 100.0
        elif normalized in real_aliases:
            prob_fake = 1.0 - (top_conf / 100.0)
        else:
            prob_fake = 0.5
    else:
        prob_fake = 0.5

    return {
        "prob_fake": float(max(0.0, min(1.0, prob_fake))),
        "top_label": top_label,
        "top_confidence": float(top_conf / 100.0),
        "deepfake_confidence": float(deepfake_conf / 100.0),
        "real_confidence": float(real_conf / 100.0),
    }


def detect_image_bytes(image_bytes: bytes) -> dict[str, Any]:
    response = _client().detect_custom_labels(
        ProjectVersionArn=os.environ["AWS_REKOGNITION_PROJECT_VERSION_ARN"],
        Image={"Bytes": image_bytes},
        MinConfidence=_min_confidence(),
    )
    labels = response.get("CustomLabels", [])
    summary = _classify_labels(labels)
    return {
        **summary,
        "labels": labels,
        "provider": "aws_rekognition",
        "region": _region(),
    }


def _frame_positions(total_frames: int, sample_count: int) -> list[int]:
    if total_frames <= 0:
        return []
    sample_count = min(sample_count, total_frames)
    positions = np.linspace(0, total_frames - 1, num=sample_count, dtype=int)
    return sorted({int(pos) for pos in positions.tolist()})


def detect_video_file(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    positions = _frame_positions(total_frames, _sample_frames())
    if not positions:
        cap.release()
        raise ValueError("Could not determine video frames for AWS inference.")

    frame_results: list[dict[str, Any]] = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        frame_result = detect_image_bytes(encoded.tobytes())
        frame_results.append(
            {
                "frame_index": pos,
                "prob_fake": frame_result["prob_fake"],
                "top_label": frame_result.get("top_label"),
                "top_confidence": frame_result.get("top_confidence"),
            }
        )
    cap.release()

    if not frame_results:
        raise ValueError("No video frames could be classified by AWS Rekognition.")

    avg_prob = float(sum(item["prob_fake"] for item in frame_results) / len(frame_results))
    return {
        "prob_fake": avg_prob,
        "frames_sampled": len(frame_results),
        "frame_results": frame_results,
        "provider": "aws_rekognition",
        "region": _region(),
    }
