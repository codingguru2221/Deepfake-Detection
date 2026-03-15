from __future__ import annotations

import base64
import json
import os
import time
from typing import Any

import httpx


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4.1-mini"

_PROMPT = (
    "You are an image forensics assistant. "
    "Classify the provided image as one of: real, deepfake, ai_generated, uncertain. "
    "Return strict JSON with keys prediction, prob_fake, confidence, rationale. "
    "prob_fake and confidence must be numbers between 0 and 1. "
    "Treat deepfake and ai_generated as fake/manipulated content."
)


def is_enabled() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _model() -> str:
    return os.getenv("OPENAI_VISION_MODEL", DEFAULT_MODEL)


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
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    payload = {
        "model": _model(),
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": _PROMPT},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    with httpx.Client(timeout=timeout_seconds) as client:
        for attempt in range(3):
            try:
                resp = client.post(f"{_base_url()}/responses", headers=headers, json=payload)
                resp.raise_for_status()
                raw = resp.json()
                break
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code != 429 or attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        else:
            raise last_error or RuntimeError("OpenAI vision request failed.")

    output_text = raw.get("output_text")
    if not output_text:
        pieces: list[str] = []
        for item in raw.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    pieces.append(str(content["text"]))
        output_text = "\n".join(pieces)
    if not output_text:
        raise RuntimeError("OpenAI vision response did not contain text output.")

    parsed = _extract_json(output_text)
    prediction = _normalize_prediction(str(parsed.get("prediction", "uncertain")))
    prob_fake = float(parsed.get("prob_fake", 0.5))
    confidence = float(parsed.get("confidence", max(prob_fake, 1.0 - prob_fake)))

    return {
        "provider": "openai",
        "model": _model(),
        "filename": filename,
        "prediction": prediction,
        "prob_fake": max(0.0, min(1.0, prob_fake)),
        "confidence": max(0.0, min(1.0, confidence)),
        "rationale": parsed.get("rationale"),
        "raw_text": output_text,
    }
