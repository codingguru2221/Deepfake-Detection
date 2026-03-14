from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from deepfake_detector.models.audio_model import build_audio_model, save_audio_model
from deepfake_detector.models.image_model_tf import build_image_model
from deepfake_detector.models.video_model_torch import VideoGRUClassifier, save_torch_checkpoint
from deepfake_detector.utils.io import read_json, set_seed


def _load_images_from_manifest(samples_json: Path) -> Tuple[np.ndarray, np.ndarray]:
    rows = read_json(samples_json)["samples"]
    X, y = [], []
    for row in rows:
        img = cv2.imread(row["path"])
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        X.append(img.astype(np.float32))
        y.append(int(row["label"]))
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int64)
    return X_arr, y_arr


def train_image_tf(samples_json: Path, out_model: Path, epochs: int, batch_size: int) -> None:
    import tensorflow as tf

    X, y = _load_images_from_manifest(samples_json)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = build_image_model()
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=2, mode="max", restore_best_weights=True)]
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )
    preds = model.predict(X_val).ravel()
    auc = roc_auc_score(y_val, preds) if len(np.unique(y_val)) > 1 else float("nan")
    print(f"[image] val_auc={auc:.4f}")
    out_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_model)


def _group_video_frames(rows: List[Dict[str, str | int]]) -> Dict[str, Dict[str, List[str] | int]]:
    groups: Dict[str, Dict[str, List[str] | int]] = {}
    for row in rows:
        stem = Path(str(row["path"])).stem
        key = stem.split("_f")[0]
        if key not in groups:
            groups[key] = {"paths": [], "label": int(row["label"])}
        groups[key]["paths"].append(str(row["path"]))
    return groups


def train_video_torch(samples_json: Path, out_ckpt: Path, epochs: int) -> None:
    from deepfake_detector.features.image_features import ImageFeatureExtractor

    rows = read_json(samples_json)["samples"]
    groups = _group_video_frames(rows)
    extractor = ImageFeatureExtractor()
    X_seq, y = [], []
    for key, payload in groups.items():
        _ = key
        paths = sorted(payload["paths"])[:30]  # type: ignore[index]
        seq = []
        for p in paths:
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            emb = extractor.deep_embedding(frame)
            seq.append(emb)
        if len(seq) < 3:
            continue
        seq_np = np.stack(seq, axis=0)
        X_seq.append(seq_np)
        y.append(int(payload["label"]))

    if not X_seq:
        raise RuntimeError("No valid video sequences found.")

    max_len = max(s.shape[0] for s in X_seq)
    feat_dim = X_seq[0].shape[1]
    X_padded = np.zeros((len(X_seq), max_len, feat_dim), dtype=np.float32)
    for i, seq in enumerate(X_seq):
        X_padded[i, : seq.shape[0], :] = seq

    X = torch.tensor(X_padded, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

    model = VideoGRUClassifier(input_dim=feat_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_pred = (val_probs > 0.5).astype(np.int32)
            y_val_np = y_val.cpu().numpy().astype(np.int32)
            auc = roc_auc_score(y_val_np, val_probs) if len(np.unique(y_val_np)) > 1 else float("nan")
            print(f"[video] epoch={epoch+1} loss={loss.item():.4f} val_auc={auc:.4f}")
            print(classification_report(y_val_np, val_pred, zero_division=0))

    save_torch_checkpoint(model, out_ckpt)


def train_audio(samples_json: Path, out_model: Path) -> None:
    from deepfake_detector.features.audio_features import batch_extract as audio_batch_extract

    rows = read_json(samples_json)["samples"]
    paths = [Path(r["path"]) for r in rows]
    y = np.asarray([int(r["label"]) for r in rows], dtype=np.int32)
    X = audio_batch_extract(paths)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = build_audio_model()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs > 0.5).astype(np.int32)
    auc = roc_auc_score(y_val, probs) if len(np.unique(y_val)) > 1 else float("nan")
    print(f"[audio] val_auc={auc:.4f}")
    print(classification_report(y_val, preds, zero_division=0))
    save_audio_model(model, out_model)


def train_image_features_baseline(samples_json: Path, out_npz: Path) -> None:
    from deepfake_detector.features.image_features import batch_extract as image_batch_extract

    rows = read_json(samples_json)["samples"]
    paths = [Path(r["path"]) for r in rows]
    y = np.asarray([int(r["label"]) for r in rows], dtype=np.int32)
    X = image_batch_extract(paths)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, X=X, y=y)
    print(f"Saved feature matrix: {out_npz} with shape={X.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deepfake detection models.")
    parser.add_argument("--modality", choices=["image", "video", "audio", "image_features"], required=True)
    parser.add_argument("--samples-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    set_seed(42)
    if args.modality == "image":
        train_image_tf(args.samples_json, args.out, args.epochs, args.batch_size)
    elif args.modality == "video":
        train_video_torch(args.samples_json, args.out, args.epochs)
    elif args.modality == "audio":
        train_audio(args.samples_json, args.out)
    elif args.modality == "image_features":
        train_image_features_baseline(args.samples_json, args.out)


if __name__ == "__main__":
    main()
