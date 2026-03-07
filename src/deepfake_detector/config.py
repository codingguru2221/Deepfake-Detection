from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = Path(__file__).resolve().parents[2]
    raw_data: Path = root / "data" / "raw"
    processed_data: Path = root / "data" / "processed"
    checkpoints: Path = root / "models" / "checkpoints"
    exports: Path = root / "models" / "exports"
    logs: Path = root / "logs"


@dataclass(frozen=True)
class ImageVideoParams:
    image_size: int = 224
    frame_stride: int = 10
    max_frames_per_video: int = 30
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 1e-4


@dataclass(frozen=True)
class AudioParams:
    sample_rate: int = 16000
    n_mfcc: int = 40
    hop_length: int = 256
    n_fft: int = 1024


PATHS = ProjectPaths()
IMGVID = ImageVideoParams()
AUDIO = AudioParams()
