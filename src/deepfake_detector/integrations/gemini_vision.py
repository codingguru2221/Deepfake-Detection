from __future__ import annotations

import base64
import json
import os
import time
from typing import Any

import httpx


DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-2.0-flash"

_PROMPT = (
    "Classify this image as real, deepfake, ai_generated, or uncertain. "
    "Return strict JSON with keys prediction, prob_fake, confidence, rationale. "
    "prob_fake and confidence must be numbers between 0 and 1. "
    "Treat deepfake and ai_generated as fake/manipulated content."
)


def is_enabled() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


def _base_url() -> str:
    return os.getenv("GEMINI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _model() -> str:
    return os.getenv("GEMINI_VISION_MODEL", DEFAULT_MODEL)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _normalize_prediction(value: str) -> str:
    value = value.strip().lower()
    if value in {"deepfake", "fake", "manipulated", "synthetic", "ai_generated", "ai-generated"}:
        return "deepfake"
    if value in {"real", "authentic", "genuine"}:
        return "real"
    return "uncertain"


def detect_image_bytes(
    image_bytes: bytes,
    *,
    filename: str | None = None,
    mime_type: str = "image/jpeg",
    timeout_seconds: float = 45.0,
) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": _PROMPT},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
        },
    }

    last_error: Exception | None = None
    with httpx.Client(timeout=timeout_seconds) as client:
        for attempt in range(3):
            try:
                resp = client.post(
                    f"{_base_url()}/models/{_model()}:generateContent",
                    params={"key": api_key},
                    json=payload,
                )
                resp.raise_for_status()
                raw = resp.json()
                break
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code != 429 or attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        else:
            raise last_error or RuntimeError("Gemini vision request failed.")

    candidates = raw.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini vision response did not contain candidates.")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(str(part.get("text", "")) for part in parts if part.get("text"))
    if not text:
        raise RuntimeError("Gemini vision response did not contain text output.")

    parsed = _extract_json(text)
    prediction = _normalize_prediction(str(parsed.get("prediction", "uncertain")))
    prob_fake = float(parsed.get("prob_fake", 0.5))
    confidence = float(parsed.get("confidence", max(prob_fake, 1.0 - prob_fake)))

    return {
        "provider": "gemini",
        "model": _model(),
        "filename": filename,
        "prediction": prediction,
        "prob_fake": max(0.0, min(1.0, prob_fake)),
        "confidence": max(0.0, min(1.0, confidence)),
        "rationale": parsed.get("rationale"),
        "raw_text": text,
    }
