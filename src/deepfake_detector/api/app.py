from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMAGE_MODEL_PATH = Path(os.getenv("DF_IMAGE_MODEL", PROJECT_ROOT / "models" / "exports" / "image_tf_model.keras"))
VIDEO_MODEL_PATH = Path(os.getenv("DF_VIDEO_MODEL", PROJECT_ROOT / "models" / "checkpoints" / "video_gru.pt"))
AUDIO_MODEL_PATH = Path(os.getenv("DF_AUDIO_MODEL", PROJECT_ROOT / "models" / "exports" / "audio_rf.joblib"))


app = FastAPI(title="Deepfake Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _as_result(prob_fake: float, details: dict) -> dict:
    prediction = "deepfake" if prob_fake > 0.5 else "real"
    confidence = prob_fake if prediction == "deepfake" else 1.0 - prob_fake
    return {
        "prediction": prediction,
        "prob_fake": float(prob_fake),
        "confidence": float(confidence),
        "details": details,
    }


def _save_upload(file: UploadFile, suffix: Optional[str] = None) -> Path:
    ext = suffix or Path(file.filename or "").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        data = file.file.read()
        tmp.write(data)
        return Path(tmp.name)


def _infer_funcs():
    # Lazy import so API can boot even if heavy ML deps are not installed yet.
    from deepfake_detector.infer import fuse as _fuse
    from deepfake_detector.infer import predict_audio as _predict_audio
    from deepfake_detector.infer import predict_image as _predict_image
    from deepfake_detector.infer import predict_video as _predict_video

    return _fuse, _predict_audio, _predict_image, _predict_video


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "models": {
            "image": IMAGE_MODEL_PATH.exists(),
            "video": VIDEO_MODEL_PATH.exists(),
            "audio": AUDIO_MODEL_PATH.exists(),
        },
    }


@app.post("/infer/image")
def infer_image(file: UploadFile = File(...)) -> dict:
    _, _, predict_image, _ = _infer_funcs()
    if not IMAGE_MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Image model not found: {IMAGE_MODEL_PATH}")
    tmp = _save_upload(file)
    try:
        prob = predict_image(tmp, IMAGE_MODEL_PATH)
        return _as_result(prob, {"modality": "image", "filename": file.filename})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image inference failed: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/infer/video")
def infer_video(file: UploadFile = File(...)) -> dict:
    _, _, _, predict_video = _infer_funcs()
    if not VIDEO_MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Video model not found: {VIDEO_MODEL_PATH}")
    tmp = _save_upload(file)
    try:
        prob = predict_video(tmp, VIDEO_MODEL_PATH)
        return _as_result(prob, {"modality": "video", "filename": file.filename})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video inference failed: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/infer/audio")
def infer_audio(file: UploadFile = File(...)) -> dict:
    _, predict_audio, _, _ = _infer_funcs()
    if not AUDIO_MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Audio model not found: {AUDIO_MODEL_PATH}")
    tmp = _save_upload(file, suffix=".wav")
    try:
        prob = predict_audio(tmp, AUDIO_MODEL_PATH)
        return _as_result(prob, {"modality": "audio", "filename": file.filename})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio inference failed: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/infer/multimodal")
def infer_multimodal(file: UploadFile = File(...)) -> dict:
    fuse, predict_audio, predict_image, predict_video = _infer_funcs()
    filename = (file.filename or "").lower()
    suffix = Path(filename).suffix or ".bin"
    tmp = _save_upload(file, suffix=suffix)
    image_prob: Optional[float] = None
    video_prob: Optional[float] = None
    audio_prob: Optional[float] = None
    details: dict = {"modality": "multimodal", "filename": file.filename}

    try:
        if suffix in {".jpg", ".jpeg", ".png", ".bmp"} and IMAGE_MODEL_PATH.exists():
            image_prob = predict_image(tmp, IMAGE_MODEL_PATH)
        elif suffix in {".wav", ".mp3", ".flac", ".m4a"} and AUDIO_MODEL_PATH.exists():
            audio_prob = predict_audio(tmp, AUDIO_MODEL_PATH)
        elif suffix in {".mp4", ".avi", ".mov", ".mkv"}:
            if VIDEO_MODEL_PATH.exists():
                video_prob = predict_video(tmp, VIDEO_MODEL_PATH)

            # Attempt audio branch for video if audio model exists.
            if AUDIO_MODEL_PATH.exists():
                try:
                    from pydub import AudioSegment

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_tmp:
                        AudioSegment.from_file(tmp).set_channels(1).set_frame_rate(16000).export(
                            wav_tmp.name, format="wav"
                        )
                        audio_prob = predict_audio(Path(wav_tmp.name), AUDIO_MODEL_PATH)
                    Path(wav_tmp.name).unlink(missing_ok=True)
                except Exception:
                    details["audio_from_video"] = "failed"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file for multimodal inference.")

        final_prob = fuse(image_prob, video_prob, audio_prob)
        result = _as_result(final_prob, details)
        result["modalityScores"] = {
            "image": image_prob,
            "video": video_prob,
            "audio": audio_prob,
        }
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Multimodal inference failed: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)
