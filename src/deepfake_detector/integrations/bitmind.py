from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx


DEFAULT_BASE_URL = "https://api.bitmind.ai/oracle/v1"


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "x-bitmind-application": "oracle-api",
    }


def _base_url() -> str:
    return os.getenv("BITMIND_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def is_enabled() -> bool:
    return bool(os.getenv("BITMIND_API_KEY"))


def detect_image_bytes(
    image_bytes: bytes,
    *,
    source: str | None = None,
    rich: bool = False,
    timeout_seconds: float = 25.0,
) -> dict[str, Any]:
    api_key = os.getenv("BITMIND_API_KEY")
    if not api_key:
        raise RuntimeError("BITMIND_API_KEY is not set.")

    b64 = base64.b64encode(image_bytes).decode("ascii")
    payload: dict[str, Any] = {"image": b64, "rich": bool(rich)}
    if source:
        payload["source"] = source

    url = f"{_base_url()}/34/detect-image"
    with httpx.Client(timeout=timeout_seconds) as client:
        resp = client.post(url, headers=_headers(api_key), json=payload)
        resp.raise_for_status()
        return resp.json()


def detect_video_file(
    video_path: Path,
    *,
    source: str | None = None,
    rich: bool = False,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    api_key = os.getenv("BITMIND_API_KEY")
    if not api_key:
        raise RuntimeError("BITMIND_API_KEY is not set.")

    url = f"{_base_url()}/34/detect-video"
    size_mb = video_path.stat().st_size / (1024 * 1024)
    content_type, _ = mimetypes.guess_type(video_path.name)
    if content_type is None:
        content_type = "video/mp4"

    if size_mb <= 6:
        with httpx.Client(timeout=timeout_seconds) as client:
            with video_path.open("rb") as f:
                files = {"video": (video_path.name, f, content_type)}
                data = {"rich": str(bool(rich)).lower()}
                if source:
                    data["source"] = source
                resp = client.post(url, headers=_headers(api_key), files=files, data=data)
                resp.raise_for_status()
                return resp.json()

    upload = _get_video_upload_url(video_path.name, content_type)
    video_url = upload.get("videoUrl")
    if not video_url:
        raise RuntimeError("BitMind upload did not return videoUrl.")

    _upload_video_to_s3(upload, video_path)

    payload: dict[str, Any] = {"video": video_url, "rich": bool(rich)}
    if source:
        payload["source"] = source
    with httpx.Client(timeout=timeout_seconds) as client:
        resp = client.post(url, headers=_headers(api_key), json=payload)
        resp.raise_for_status()
        return resp.json()


def extract_verdict(payload: dict[str, Any]) -> dict[str, Any] | None:
    # Normalize common fields from BitMind responses.
    if not isinstance(payload, dict):
        return None
    is_ai = payload.get("isAI", payload.get("is_ai"))
    prediction = payload.get("prediction")
    confidence = payload.get("confidence", payload.get("score"))
    if is_ai is None and prediction is None:
        return None
    if prediction is None and isinstance(is_ai, bool):
        prediction = "deepfake" if is_ai else "real"
    elif isinstance(prediction, str):
        prediction = "deepfake" if prediction.lower() in {"ai", "deepfake", "fake"} else "real"
    return {"prediction": prediction, "confidence": confidence}


def _get_video_upload_url(filename: str, content_type: str | None) -> dict[str, Any]:
    api_key = os.getenv("BITMIND_API_KEY")
    if not api_key:
        raise RuntimeError("BITMIND_API_KEY is not set.")

    url = f"{_base_url()}/34/get-video-upload-url"
    payload: dict[str, Any] = {"filename": filename}
    if content_type:
        payload["contentType"] = content_type
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(url, headers=_headers(api_key), json=payload)
        resp.raise_for_status()
        return resp.json()


def _upload_video_to_s3(upload_payload: dict[str, Any], video_path: Path) -> None:
    url = upload_payload.get("url")
    fields = upload_payload.get("fields") or {}
    if not url or not isinstance(fields, dict):
        raise RuntimeError("Invalid BitMind upload payload.")

    with httpx.Client(timeout=60.0) as client:
        with video_path.open("rb") as f:
            files = {"file": (video_path.name, f)}
            resp = client.post(url, data=fields, files=files)
            resp.raise_for_status()
