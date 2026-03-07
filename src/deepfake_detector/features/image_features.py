from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torchvision import models, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ImageFeatureExtractor:
    def __init__(self) -> None:
        try:
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            # Offline-safe fallback when pretrained weights cannot be downloaded.
            backbone = models.resnet18(weights=None)
        self.model = torch.nn.Sequential(*(list(backbone.children())[:-1])).to(DEVICE).eval()
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def handcrafted(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        fft = np.fft.fft2(gray)
        fft_mag = np.mean(np.abs(np.fft.fftshift(fft)))
        color_means = frame.mean(axis=(0, 1))
        return np.array([lap_var, fft_mag, *color_means], dtype=np.float32)

    def deep_embedding(self, frame: np.ndarray) -> np.ndarray:
        x = self.preprocess(frame).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = self.model(x).flatten(1).squeeze(0).cpu().numpy().astype(np.float32)
        return emb

    def extract(self, image_path: Path) -> np.ndarray:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        hand = self.handcrafted(frame)
        deep = self.deep_embedding(frame)
        return np.concatenate([hand, deep], axis=0)


def batch_extract(paths: List[Path]) -> np.ndarray:
    extractor = ImageFeatureExtractor()
    feats = [extractor.extract(p) for p in paths]
    return np.stack(feats, axis=0)
