from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from deepfake_detector.config import IMGVID
from deepfake_detector.utils.io import ensure_dir, read_json, write_json

try:
    import dlib

    DLIB_AVAILABLE = True
except Exception:
    DLIB_AVAILABLE = False
    dlib = None  # type: ignore


class FaceDetector:
    def __init__(self) -> None:
        self.cv2_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.dlib_detector = dlib.get_frontal_face_detector() if DLIB_AVAILABLE else None

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes: List[Tuple[int, int, int, int]] = []
        if self.dlib_detector is not None:
            dets = self.dlib_detector(gray, 1)
            for det in dets:
                boxes.append((det.left(), det.top(), det.right(), det.bottom()))
        if not boxes:
            dets = self.cv2_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for x, y, w, h in dets:
                boxes.append((x, y, x + w, y + h))
        return boxes


def _safe_crop(frame: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = cv2.resize(frame, (IMGVID.image_size, IMGVID.image_size))
    return crop


def preprocess_image(path: Path, detector: FaceDetector, out_dir: Path) -> List[str]:
    frame = cv2.imread(str(path))
    if frame is None:
        return []
    boxes = detector.detect(frame)
    outputs = []
    if not boxes:
        boxes = [(0, 0, frame.shape[1], frame.shape[0])]
    for i, box in enumerate(boxes[:1]):
        face = _safe_crop(frame, box)
        face = cv2.resize(face, (IMGVID.image_size, IMGVID.image_size))
        out_path = out_dir / f"{path.stem}_face{i}.jpg"
        cv2.imwrite(str(out_path), face)
        outputs.append(str(out_path))
    return outputs


def preprocess_video(path: Path, detector: FaceDetector, out_dir: Path) -> List[str]:
    cap = cv2.VideoCapture(str(path))
    outputs = []
    frame_idx = 0
    saved = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % IMGVID.frame_stride != 0:
            frame_idx += 1
            continue
        boxes = detector.detect(frame)
        if not boxes:
            boxes = [(0, 0, frame.shape[1], frame.shape[0])]
        face = _safe_crop(frame, boxes[0])
        face = cv2.resize(face, (IMGVID.image_size, IMGVID.image_size))
        out_path = out_dir / f"{path.stem}_f{frame_idx:05d}.jpg"
        cv2.imwrite(str(out_path), face)
        outputs.append(str(out_path))
        saved += 1
        frame_idx += 1
        if saved >= IMGVID.max_frames_per_video:
            break
    cap.release()
    return outputs


def run(manifest_path: Path, out_root: Path) -> None:
    manifest = read_json(manifest_path)
    detector = FaceDetector()
    ensure_dir(out_root / "faces" / "real")
    ensure_dir(out_root / "faces" / "fake")

    processed: Dict[str, List[Dict[str, int | str]]] = {"samples": []}
    for image_row in manifest.get("images", []):
        label_name = "fake" if image_row["label"] else "real"
        dest = out_root / "faces" / label_name
        outputs = preprocess_image(Path(image_row["path"]), detector, dest)
        processed["samples"].extend({"path": p, "label": image_row["label"]} for p in outputs)

    for video_row in manifest.get("videos", []):
        label_name = "fake" if video_row["label"] else "real"
        dest = out_root / "faces" / label_name
        outputs = preprocess_video(Path(video_row["path"]), detector, dest)
        processed["samples"].extend({"path": p, "label": video_row["label"]} for p in outputs)

    write_json(processed, out_root / "image_video_samples.json")
    print(f"Saved {len(processed['samples'])} preprocessed face samples.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess image/video for deepfake detection.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.out_root)


if __name__ == "__main__":
    main()
