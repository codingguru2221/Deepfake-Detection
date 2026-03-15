from __future__ import annotations

import base64
import mimetypes
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image


DEFAULT_BASE_URL = "https://api.bitmind.ai/oracle/v1"


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "x-bitmind-application": "oracle-api",
    }


def _base_url() -> str:
    return os.getenv("BITMIND_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _oracle_id() -> str:
    return os.getenv("BITMIND_ORACLE_ID", "34").strip() or "34"


def is_enabled() -> bool:
    return bool(os.getenv("BITMIND_API_KEY"))


def detect_image_bytes(
    image_bytes: bytes,
    *,
    filename: str | None = None,
    mime_type: str | None = None,
    source: str | None = None,
    rich: bool = False,
    timeout_seconds: float = 25.0,
) -> dict[str, Any]:
    api_key = os.getenv("BITMIND_API_KEY")
    if not api_key:
        raise RuntimeError("BITMIND_API_KEY is not set.")

    normalized_bytes, normalized_mime = _normalize_image_payload(image_bytes, mime_type=mime_type)
    b64 = base64.b64encode(normalized_bytes).decode("ascii")
    mime = normalized_mime or mime_type or _guess_mime_type(filename) or "image/jpeg"
    payload: dict[str, Any] = {"image": f"data:{mime};base64,{b64}", "rich": bool(rich)}
    if source:
        payload["source"] = source

    url = f"{_base_url()}/{_oracle_id()}/detect-image"
    with httpx.Client(timeout=timeout_seconds) as client:
        try:
            resp = client.post(url, headers=_headers(api_key), json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 400:
                raise RuntimeError(_format_http_error("BitMind image request failed", exc)) from exc

            # Some BitMind deployments reject data URLs and expect raw base64.
            fallback_payload = dict(payload)
            fallback_payload["image"] = b64
            try:
                resp = client.post(url, headers=_headers(api_key), json=fallback_payload)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as raw_exc:
                if raw_exc.response.status_code != 400:
                    raise RuntimeError(_format_http_error("BitMind image request failed", raw_exc)) from raw_exc

            # Final fallback: send the image as multipart form-data.
            files = {"image": (filename or "upload.jpg", normalized_bytes, mime)}
            data = {"rich": str(bool(rich)).lower()}
            if source:
                data["source"] = source
            try:
                resp = client.post(url, headers=_headers(api_key), files=files, data=data)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as multipart_exc:
                raise RuntimeError(_format_http_error("BitMind image request failed", multipart_exc)) from multipart_exc


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

    url = f"{_base_url()}/{_oracle_id()}/detect-video"
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

    url = f"{_base_url()}/{_oracle_id()}/get-video-upload-url"
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


def _guess_mime_type(filename: str | None) -> str | None:
    if not filename:
        return None
    mime, _ = mimetypes.guess_type(filename)
    return mime


def _normalize_image_payload(image_bytes: bytes, *, mime_type: str | None = None) -> tuple[bytes, str]:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            largest_edge = max(image.size)
            if largest_edge > 1600:
                scale = 1600 / float(largest_edge)
                resized = (
                    max(1, int(image.size[0] * scale)),
                    max(1, int(image.size[1] * scale)),
                )
                image = image.resize(resized)

            buffer = BytesIO()
            target_format = "PNG" if (mime_type or "").lower() == "image/png" else "JPEG"
            save_kwargs = {"optimize": True}
            if target_format == "JPEG":
                save_kwargs["quality"] = 90
            image.save(buffer, format=target_format, **save_kwargs)
            normalized = buffer.getvalue()
            normalized_mime = "image/png" if target_format == "PNG" else "image/jpeg"
            return normalized, normalized_mime
    except Exception:
        return image_bytes, mime_type or "image/jpeg"


def _format_http_error(prefix: str, exc: httpx.HTTPStatusError) -> str:
    body = ""
    try:
        body = exc.response.text.strip()
    except Exception:
        body = ""
    if body:
        body = body[:400]
        return f"{prefix}: {exc} | response={body}"
    return f"{prefix}: {exc}"
