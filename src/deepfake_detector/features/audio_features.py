from __future__ import annotations

from pathlib import Path
from typing import List

import librosa
import numpy as np

try:
    from speechbrain.inference.classifiers import EncoderClassifier

    SPEECHBRAIN_AVAILABLE = True
except Exception:
    SPEECHBRAIN_AVAILABLE = False
    EncoderClassifier = None  # type: ignore

try:
    from resemblyzer import VoiceEncoder, preprocess_wav

    RESSEMBLYZER_AVAILABLE = True
except Exception:
    RESSEMBLYZER_AVAILABLE = False
    VoiceEncoder = None  # type: ignore
    preprocess_wav = None  # type: ignore


class AudioFeatureExtractor:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.sb_encoder = None
        self.resemblyzer_encoder = None
        if SPEECHBRAIN_AVAILABLE:
            try:
                self.sb_encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/speechbrain_ecapa"
                )
            except Exception:
                # Offline-safe fallback: keep zero-vector embedding path.
                self.sb_encoder = None
        if RESSEMBLYZER_AVAILABLE:
            try:
                self.resemblyzer_encoder = VoiceEncoder()
            except Exception:
                self.resemblyzer_encoder = None

    @staticmethod
    def handcrafted(y: np.ndarray, sr: int) -> np.ndarray:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        features = np.concatenate(
            [
                mfcc.mean(axis=1),
                mfcc.std(axis=1),
                delta.mean(axis=1),
                spec_cent.mean(axis=1),
                spec_rolloff.mean(axis=1),
                zcr.mean(axis=1),
            ]
        )
        return features.astype(np.float32)

    def speechbrain_embedding(self, wav_path: Path) -> np.ndarray:
        if self.sb_encoder is None:
            return np.zeros(192, dtype=np.float32)
        emb = self.sb_encoder.encode_file(str(wav_path))
        return emb.squeeze().detach().cpu().numpy().astype(np.float32)

    def resemblyzer_embedding(self, wav_path: Path) -> np.ndarray:
        if self.resemblyzer_encoder is None or preprocess_wav is None:
            return np.zeros(256, dtype=np.float32)
        wav = preprocess_wav(str(wav_path))
        emb = self.resemblyzer_encoder.embed_utterance(wav)
        return emb.astype(np.float32)

    def extract(self, wav_path: Path) -> np.ndarray:
        y, sr = librosa.load(wav_path, sr=self.sample_rate, mono=True)
        hand = self.handcrafted(y, sr)
        sb = self.speechbrain_embedding(wav_path)
        re = self.resemblyzer_embedding(wav_path)
        return np.concatenate([hand, sb, re], axis=0)


def batch_extract(paths: List[Path], sample_rate: int = 16000) -> np.ndarray:
    extractor = AudioFeatureExtractor(sample_rate=sample_rate)
    feats = [extractor.extract(p) for p in paths]
    return np.stack(feats, axis=0)
