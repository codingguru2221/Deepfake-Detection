from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from deepfake_detector.features.audio_features import AudioFeatureExtractor
from deepfake_detector.features.image_features import ImageFeatureExtractor
from deepfake_detector.models.audio_model import load_audio_model
from deepfake_detector.models.video_model_torch import VideoGRUClassifier, load_torch_checkpoint
from deepfake_detector.data.calibration import load_thresholds


def predict_image(image_path: Path, model_path: Path) -> float:
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype(np.float32)
    prob = float(model.predict(np.expand_dims(img, axis=0), verbose=0).ravel()[0])
    return prob


def predict_video(video_path: Path, model_path: Path) -> float:
    extractor = ImageFeatureExtractor()
    cap = cv2.VideoCapture(str(video_path))
    seq = []
    frame_idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % 10 == 0:
            seq.append(extractor.deep_embedding(frame))
        frame_idx += 1
        if len(seq) >= 30:
            break
    cap.release()
    if not seq:
        raise RuntimeError("No frames extracted from video.")
    X = np.expand_dims(np.stack(seq, axis=0), axis=0).astype(np.float32)
    model = VideoGRUClassifier(input_dim=X.shape[-1])
    model = load_torch_checkpoint(model, model_path)
    with torch.no_grad():
        logit = model(torch.tensor(X))
        prob = torch.sigmoid(logit).item()
    return float(prob)


def predict_audio(audio_path: Path, model_path: Path) -> float:
    extractor = AudioFeatureExtractor()
    feats = extractor.extract(audio_path).reshape(1, -1)
    model = load_audio_model(model_path)
    return float(model.predict_proba(feats)[0, 1])


def fuse(image_prob: Optional[float], video_prob: Optional[float], audio_prob: Optional[float]) -> float:
    probs = [p for p in [image_prob, video_prob, audio_prob] if p is not None]
    if not probs:
        raise ValueError("At least one modality prediction is required.")
    return float(np.mean(probs))


def _resolve_modality(image_prob: Optional[float], video_prob: Optional[float], audio_prob: Optional[float]) -> str:
    present = [p is not None for p in (image_prob, video_prob, audio_prob)]
    if sum(present) > 1:
        return "multimodal"
    if image_prob is not None:
        return "image"
    if video_prob is not None:
        return "video"
    if audio_prob is not None:
        return "audio"
    return "unknown"


def _get_thresholds(modality: str) -> tuple[float, float, str]:
    stored = load_thresholds()
    if modality in stored:
        return stored[modality]["real"], stored[modality]["fake"], "calibrated"
    defaults = {
        "image": (float(os.getenv("DF_IMAGE_REAL_THRESHOLD", "0.4")), float(os.getenv("DF_IMAGE_FAKE_THRESHOLD", "0.6"))),
        "video": (float(os.getenv("DF_VIDEO_REAL_THRESHOLD", "0.4")), float(os.getenv("DF_VIDEO_FAKE_THRESHOLD", "0.6"))),
        "audio": (float(os.getenv("DF_AUDIO_REAL_THRESHOLD", "0.4")), float(os.getenv("DF_AUDIO_FAKE_THRESHOLD", "0.6"))),
        "multimodal": (
            float(os.getenv("DF_MULTIMODAL_REAL_THRESHOLD", "0.4")),
            float(os.getenv("DF_MULTIMODAL_FAKE_THRESHOLD", "0.6")),
        ),
    }
    return defaults.get(modality, (0.4, 0.6)) + ("env",)


def _maybe_invert_prob(prob_fake: float, modality: str) -> tuple[float, bool]:
    invert_any = os.getenv("DF_INVERT_PROB", "0").lower() in {"1", "true", "yes"}
    per_modality = {
        "image": "DF_INVERT_IMAGE_PROB",
        "video": "DF_INVERT_VIDEO_PROB",
        "audio": "DF_INVERT_AUDIO_PROB",
        "multimodal": "DF_INVERT_MULTIMODAL_PROB",
    }
    flag = per_modality.get(modality)
    invert = invert_any or (flag and os.getenv(flag, "0").lower() in {"1", "true", "yes"})
    if invert:
        return 1.0 - prob_fake, True
    return prob_fake, False


def _as_result(prob_fake: float, modality: str) -> dict:
    real_th, fake_th, source = _get_thresholds(modality)
    if prob_fake >= fake_th:
        prediction = "deepfake"
        confidence = prob_fake
    elif prob_fake <= real_th:
        prediction = "real"
        confidence = 1.0 - prob_fake
    else:
        prediction = "uncertain"
        confidence = max(prob_fake, 1.0 - prob_fake)
    return {
        "prediction": prediction,
        "prob_fake": float(prob_fake),
        "confidence": float(confidence),
        "thresholds": {"real": float(real_th), "fake": float(fake_th)},
        "thresholds_source": source,
        "modality": modality,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multimodal deepfake inference.")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--image-model", type=Path, default=None)
    parser.add_argument("--video-model", type=Path, default=None)
    parser.add_argument("--audio-model", type=Path, default=None)
    args = parser.parse_args()

    image_prob = predict_image(args.image, args.image_model) if args.image and args.image_model else None
    video_prob = predict_video(args.video, args.video_model) if args.video and args.video_model else None
    audio_prob = predict_audio(args.audio, args.audio_model) if args.audio and args.audio_model else None
    final_raw = fuse(image_prob, video_prob, audio_prob)
    modality = _resolve_modality(image_prob, video_prob, audio_prob)
    final_prob, inverted = _maybe_invert_prob(final_raw, modality)
    result = _as_result(final_prob, modality)

    print(
        {
            "image_prob_fake": image_prob,
            "video_prob_fake": video_prob,
            "audio_prob_fake": audio_prob,
            "fused_prob_fake": final_prob,
            "fused_prob_fake_raw": final_raw,
            "prob_inverted": inverted,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "thresholds": result["thresholds"],
            "thresholds_source": result["thresholds_source"],
            "modality": result["modality"],
        }
    )


if __name__ == "__main__":
    main()
