from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from deepfake_detector.features.audio_features import batch_extract as audio_batch_extract
from deepfake_detector.features.image_features import ImageFeatureExtractor
from deepfake_detector.models.audio_model import load_audio_model
from deepfake_detector.models.video_model_torch import VideoGRUClassifier, load_torch_checkpoint
from deepfake_detector.utils.io import read_json


def evaluate_image_tf(samples_json: Path, model_path: Path) -> None:
    import tensorflow as tf

    rows = read_json(samples_json)["samples"]
    X, y = [], []
    for r in rows:
        img = cv2.imread(r["path"])
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        X.append(img.astype(np.float32))
        y.append(int(r["label"]))
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    model = tf.keras.models.load_model(model_path)
    probs = model.predict(X).ravel()
    preds = (probs > 0.5).astype(np.int32)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    print(f"[image] acc={accuracy_score(y, preds):.4f} auc={auc:.4f}")
    print(confusion_matrix(y, preds))
    print(classification_report(y, preds, zero_division=0))


def evaluate_audio(samples_json: Path, model_path: Path) -> None:
    rows = read_json(samples_json)["samples"]
    paths = [Path(r["path"]) for r in rows]
    y = np.asarray([int(r["label"]) for r in rows], dtype=np.int32)
    X = audio_batch_extract(paths)
    model = load_audio_model(model_path)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(np.int32)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    print(f"[audio] acc={accuracy_score(y, preds):.4f} auc={auc:.4f}")
    print(confusion_matrix(y, preds))
    print(classification_report(y, preds, zero_division=0))


def evaluate_video(samples_json: Path, ckpt_path: Path) -> None:
    rows = read_json(samples_json)["samples"]
    groups = {}
    for row in rows:
        key = Path(row["path"]).stem.split("_f")[0]
        groups.setdefault(key, {"paths": [], "label": int(row["label"])})
        groups[key]["paths"].append(row["path"])

    extractor = ImageFeatureExtractor()
    X, y = [], []
    for payload in groups.values():
        seq = []
        for p in sorted(payload["paths"])[:30]:
            img = cv2.imread(p)
            if img is None:
                continue
            seq.append(extractor.deep_embedding(img))
        if len(seq) < 3:
            continue
        X.append(np.stack(seq))
        y.append(payload["label"])

    max_len = max(s.shape[0] for s in X)
    feat_dim = X[0].shape[1]
    X_pad = np.zeros((len(X), max_len, feat_dim), dtype=np.float32)
    for i, seq in enumerate(X):
        X_pad[i, : seq.shape[0]] = seq

    model = VideoGRUClassifier(input_dim=feat_dim)
    model = load_torch_checkpoint(model, ckpt_path)
    with torch.no_grad():
        logits = model(torch.tensor(X_pad))
        probs = torch.sigmoid(logits).numpy()
    y_np = np.asarray(y, dtype=np.int32)
    preds = (probs > 0.5).astype(np.int32)
    auc = roc_auc_score(y_np, probs) if len(np.unique(y_np)) > 1 else float("nan")
    print(f"[video] acc={accuracy_score(y_np, preds):.4f} auc={auc:.4f}")
    print(confusion_matrix(y_np, preds))
    print(classification_report(y_np, preds, zero_division=0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deepfake models.")
    parser.add_argument("--modality", choices=["image", "video", "audio"], required=True)
    parser.add_argument("--samples-json", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    if args.modality == "image":
        evaluate_image_tf(args.samples_json, args.model)
    elif args.modality == "video":
        evaluate_video(args.samples_json, args.model)
    else:
        evaluate_audio(args.samples_json, args.model)


if __name__ == "__main__":
    main()
