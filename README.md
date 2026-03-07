# Multimodal Deepfake Detection (Image + Video + Audio)

A complete Python machine learning project for detecting whether content is real or AI-generated/deepfake across:
- Images (face artifacts and visual inconsistencies)
- Videos (frame-level facial analysis + temporal inconsistency modeling)
- Audio (spectral/prosodic patterns + speaker embedding anomalies)

The project supports datasets such as FaceForensics++ and DFDC (Deepfake Detection Challenge).

## Tech Stack

- Core: Python, NumPy, SciPy, scikit-learn
- Computer Vision: OpenCV, Dlib
- Deep Learning: TensorFlow, PyTorch, Torchvision
- Audio: LibROSA, SpeechBrain, Resemblyzer, PyDub
- Notebook: Jupyter

## Project Structure

```text
JupyterProject/
├── configs/
│   └── project.yaml
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── checkpoints/
│   └── exports/
├── notebooks/
│   └── deepfake_pipeline.ipynb
├── scripts/
│   ├── run_pipeline.ps1
│   └── start_notebook.ps1
├── src/
│   └── deepfake_detector/
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── utils/
│       ├── config.py
│       ├── train.py
│       ├── evaluate.py
│       └── infer.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Dataset Setup

Place datasets under `data/raw/`.
Labeling is inferred from path names:
- `real`, `original` -> label `0`
- `fake`, `deepfake`, `manipulated`, `synthesis` -> label `1`

Example:

```text
data/raw/
├── faceforensicspp/
│   ├── original_sequences/
│   └── manipulated_sequences/
└── dfdc/
    ├── real/
    └── fake/
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

### Optional Dependencies (Advanced)

`dlib` and `resemblyzer` can fail on Windows without Visual C++ Build Tools.
Project default pipeline still works without them (OpenCV face detector + non-Resemblyzer audio features).

Install optional packages only if your system supports native builds:

```powershell
pip install -r requirements-optional.txt
```

If installation fails, continue with base `requirements.txt` only.

## Standard CLI Workflow

```powershell
python -m deepfake_detector.data.dataset_manifest --raw-root data/raw --out data/processed/manifest.json
python -m deepfake_detector.data.preprocess_image_video --manifest data/processed/manifest.json --out-root data/processed
python -m deepfake_detector.data.preprocess_audio --manifest data/processed/manifest.json --out-root data/processed

python -m deepfake_detector.train --modality image --samples-json data/processed/image_video_samples.json --out models/exports/image_tf_model.keras --epochs 5
python -m deepfake_detector.train --modality video --samples-json data/processed/image_video_samples.json --out models/checkpoints/video_gru.pt --epochs 5
python -m deepfake_detector.train --modality audio --samples-json data/processed/audio_samples.json --out models/exports/audio_rf.joblib

python -m deepfake_detector.evaluate --modality image --samples-json data/processed/image_video_samples.json --model models/exports/image_tf_model.keras
python -m deepfake_detector.evaluate --modality video --samples-json data/processed/image_video_samples.json --model models/checkpoints/video_gru.pt
python -m deepfake_detector.evaluate --modality audio --samples-json data/processed/audio_samples.json --model models/exports/audio_rf.joblib
```

## Jupyter Notebook (Easy Execute)

Use `notebooks/deepfake_pipeline.ipynb`.
Notebook me parameter cell hai jaha se aap path, epochs, aur modality toggles set kar sakte ho.

Quick start (recommended):

```powershell
.\scripts\start_notebook.ps1
```

This script:
- Activates `.venv`
- Sets `PYTHONPATH=src`
- Registers Jupyter kernel `Python (deepfake-env)`
- Opens the project notebook directly

## EXE Runner (Full Procedure)

You can build one Windows EXE that runs the full pipeline (manifest -> preprocess -> train -> evaluate -> optional inference).

Build EXE:

```powershell
.\scripts\build_exe.ps1
```

Output:
- `dist/DeepfakePipelineRunner.exe`

Run full pipeline:

```powershell
.\dist\DeepfakePipelineRunner.exe --project-root . --raw-root data/raw --processed-root data/processed --epochs 5 --run-image --run-video --run-audio
```

Run only image+audio:

```powershell
.\dist\DeepfakePipelineRunner.exe --project-root . --raw-root data/raw --processed-root data/processed --epochs 5 --run-image --run-audio
```

Run with inference at end:

```powershell
.\dist\DeepfakePipelineRunner.exe --project-root . --raw-root data/raw --processed-root data/processed --epochs 5 --run-image --run-audio --infer-image path/to/test.jpg --infer-audio path/to/test.wav
```

## Inference

```powershell
python -m deepfake_detector.infer \
  --image path/to/test.jpg --image-model models/exports/image_tf_model.keras \
  --video path/to/test.mp4 --video-model models/checkpoints/video_gru.pt \
  --audio path/to/test.wav --audio-model models/exports/audio_rf.joblib
```

## License

MIT License.

## Run Frontend + Backend Together

Single command (recommended):

```powershell
.\run_fullstack.bat
```

This starts:
- FastAPI backend on `http://localhost:8000`
- Vite frontend on `http://localhost:5000`

Health check:

```text
http://localhost:8000/health
```

## Auto Dataset Crawler (Low RAM)

Backend startup par ek lightweight metadata crawler background me run karta hai aur catalog save karta hai:
- Output: `data/external/dataset_catalog.json`
- Sources: Kaggle, Figshare, Zenodo, arXiv
- RAM usage low rakhne ke liye hard item limits aur short network timeouts use hote hain.

Environment variables:

```powershell
$env:DF_CRAWLER_ENABLED="1"            # 0 to disable
$env:DF_CRAWLER_MAX_ITEMS="60"         # total records cap
$env:DF_CRAWLER_TIMEOUT_SECONDS="8"    # per source request timeout
$env:DF_CRAWLER_REFRESH_HOURS="24"     # startup refresh interval
$env:DF_CRAWLER_QUERY="deepfake dataset"
```
