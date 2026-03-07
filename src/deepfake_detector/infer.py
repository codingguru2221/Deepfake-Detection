from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from deepfake_detector.features.audio_features import AudioFeatureExtractor
from deepfake_detector.features.image_features import ImageFeatureExtractor
from deepfake_detector.models.audio_model import load_audio_model
from deepfake_detector.models.video_model_torch import VideoGRUClassifier, load_torch_checkpoint


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
    final_prob = fuse(image_prob, video_prob, audio_prob)
    pred = "deepfake" if final_prob > 0.5 else "real"

    print(
        {
            "image_prob_fake": image_prob,
            "video_prob_fake": video_prob,
            "audio_prob_fake": audio_prob,
            "fused_prob_fake": final_prob,
            "prediction": pred,
        }
    )


if __name__ == "__main__":
    main()
