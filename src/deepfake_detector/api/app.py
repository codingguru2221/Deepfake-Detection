from __future__ import annotations

import os
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from deepfake_detector.data.runtime_learning import RuntimeLearningManager

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMAGE_MODEL_PATH = Path(os.getenv("DF_IMAGE_MODEL", PROJECT_ROOT / "models" / "exports" / "image_tf_model.keras"))
VIDEO_MODEL_PATH = Path(os.getenv("DF_VIDEO_MODEL", PROJECT_ROOT / "models" / "checkpoints" / "video_gru.pt"))
AUDIO_MODEL_PATH = Path(os.getenv("DF_AUDIO_MODEL", PROJECT_ROOT / "models" / "exports" / "audio_rf.joblib"))

CRAWLER_ENABLED = os.getenv("DF_CRAWLER_ENABLED", "1").lower() not in {"0", "false", "no"}
CRAWLER_MAX_ITEMS = int(os.getenv("DF_CRAWLER_MAX_ITEMS", "60"))
CRAWLER_TIMEOUT_SECONDS = float(os.getenv("DF_CRAWLER_TIMEOUT_SECONDS", "8"))
CRAWLER_REFRESH_HOURS = int(os.getenv("DF_CRAWLER_REFRESH_HOURS", "24"))
CRAWLER_QUERY = os.getenv("DF_CRAWLER_QUERY", "deepfake dataset")
CRAWLER_OUTPUT = Path(os.getenv("DF_CRAWLER_OUTPUT", PROJECT_ROOT / "data" / "external" / "dataset_catalog.json"))
CRAWLER_LOG = Path(os.getenv("DF_CRAWLER_LOG", PROJECT_ROOT / "data" / "external" / "crawler.log"))
RUNTIME_LEARNING_ENABLED = os.getenv("DF_RUNTIME_LEARNING_ENABLED", "1").lower() not in {"0", "false", "no"}
MAX_RUNTIME_SAMPLE_MB = float(os.getenv("DF_RUNTIME_MAX_SAMPLE_MB", "50"))


app = FastAPI(title="Deepfake Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_crawler_status: dict = {
    "enabled": CRAWLER_ENABLED,
    "running": False,
    "last_run": None,
    "last_error": None,
    "records": 0,
    "output": str(CRAWLER_OUTPUT),
    "log": str(CRAWLER_LOG),
}
_crawler_lock = threading.Lock()
_crawler_stop_event = threading.Event()
_crawler_thread: Optional[threading.Thread] = None
_runtime_trainer = RuntimeLearningManager(PROJECT_ROOT)
_training_status: dict = {"running": False, "last_run": None, "last_error": None, "last_result": None}
_training_lock = threading.Lock()
_calibrator_model = None
_calibrator_mtime: Optional[float] = None


class CrawlerControlRequest(BaseModel):
    enabled: bool


class FeedbackRequest(BaseModel):
    sample_id: str = Field(min_length=6)
    actual_label: str = Field(pattern="^(real|deepfake)$")
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    comment: Optional[str] = Field(default=None, max_length=500)


class RuntimeTrainRequest(BaseModel):
    include_pseudo: bool = True


def _append_crawler_log(message: str) -> None:
    CRAWLER_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = f"{datetime.now(timezone.utc).replace(microsecond=0).isoformat()} {message}"
    with _crawler_lock:
        with CRAWLER_LOG.open("a", encoding="utf-8") as f:
            f.write(row + "\n")


def _crawler_log_tail(limit: int = 100) -> list[str]:
    if not CRAWLER_LOG.exists():
        return []
    with CRAWLER_LOG.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    return lines[-max(1, limit) :]


def _load_calibrator():
    global _calibrator_model, _calibrator_mtime
    path = _runtime_trainer.calibrator_file
    if not path.exists():
        _calibrator_model = None
        _calibrator_mtime = None
        return None
    mtime = path.stat().st_mtime
    if _calibrator_model is not None and _calibrator_mtime == mtime:
        return _calibrator_model
    try:
        from joblib import load

        _calibrator_model = load(path)
        _calibrator_mtime = mtime
        return _calibrator_model
    except Exception:
        _calibrator_model = None
        _calibrator_mtime = None
        return None


def _apply_calibration(prob_fake: float, modality: str) -> float:
    model = _load_calibrator()
    if model is None:
        return prob_fake
    features = [[
        float(prob_fake),
        1.0 if modality == "image" else 0.0,
        1.0 if modality == "video" else 0.0,
        1.0 if modality == "audio" else 0.0,
        1.0 if modality == "multimodal" else 0.0,
    ]]
    try:
        calibrated = float(model.predict_proba(features)[0][1])
        return max(0.0, min(1.0, calibrated))
    except Exception:
        return prob_fake


def _start_dataset_crawler(force: bool = False) -> bool:
    if not _crawler_status["enabled"]:
        return False

    from deepfake_detector.data.web_crawler import DatasetCrawler

    crawler = DatasetCrawler(
        output_file=CRAWLER_OUTPUT,
        max_items=CRAWLER_MAX_ITEMS,
        timeout_seconds=CRAWLER_TIMEOUT_SECONDS,
        refresh_hours=CRAWLER_REFRESH_HOURS,
        search_query=CRAWLER_QUERY,
        stop_event=_crawler_stop_event,
    )

    if not force and not crawler.should_refresh():
        _append_crawler_log("skip reason=fresh_catalog")
        return False

    def _worker() -> None:
        _crawler_status["running"] = True
        _crawler_status["last_error"] = None
        _append_crawler_log("run_started")
        try:
            count = crawler.crawl_once()
            _crawler_status["records"] = count
            _append_crawler_log(f"run_completed records={count}")
        except Exception as exc:
            _crawler_status["last_error"] = str(exc)
            _append_crawler_log(f"run_failed error={exc}")
        finally:
            _crawler_status["running"] = False
            if CRAWLER_OUTPUT.exists():
                mtime = datetime.fromtimestamp(Path(CRAWLER_OUTPUT).stat().st_mtime, tz=timezone.utc)
                _crawler_status["last_run"] = mtime.replace(microsecond=0).isoformat()
            else:
                _crawler_status["last_run"] = None

    global _crawler_thread
    _crawler_stop_event.clear()
    _crawler_thread = threading.Thread(target=_worker, name="dataset-crawler", daemon=True)
    _crawler_thread.start()
    return True


def _run_runtime_training(include_pseudo: bool) -> None:
    _training_status["running"] = True
    _training_status["last_error"] = None
    try:
        result = _runtime_trainer.run_training(CRAWLER_OUTPUT, include_pseudo=include_pseudo)
        _training_status["last_result"] = {
            "status": result.status,
            "manifest_path": result.manifest_path,
            "calibrator_path": result.calibrator_path,
            "user_labeled_count": result.user_labeled_count,
            "pseudo_count": result.pseudo_count,
            "crawler_refs_count": result.crawler_refs_count,
        }
    except Exception as exc:
        _training_status["last_error"] = str(exc)
    finally:
        _training_status["running"] = False
        _training_status["last_run"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@app.on_event("startup")
def _app_startup() -> None:
    _start_dataset_crawler()


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
        file.file.seek(0)
        shutil.copyfileobj(file.file, tmp, length=1024 * 1024)
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
        "datasetCrawler": _crawler_status,
        "runtimeLearning": {
            "enabled": RUNTIME_LEARNING_ENABLED,
            "training": _training_status,
        },
    }


@app.get("/crawler/status")
def crawler_status() -> dict:
    return _crawler_status


@app.get("/crawler/logs")
def crawler_logs(limit: int = 100) -> dict:
    return {"lines": _crawler_log_tail(limit), "runtimeLearning": _runtime_trainer.get_recent_logs(limit)}


@app.post("/crawler/run")
def crawler_run() -> dict:
    if _crawler_status["running"]:
        return {"started": False, "message": "Crawler already running."}
    started = _start_dataset_crawler(force=True)
    return {"started": started, "message": "Crawler run triggered." if started else "Crawler is disabled."}


@app.post("/crawler/control")
def crawler_control(payload: CrawlerControlRequest) -> dict:
    _crawler_status["enabled"] = payload.enabled
    if not payload.enabled:
        _crawler_stop_event.set()
        _append_crawler_log("disabled_by_user")
    else:
        _append_crawler_log("enabled_by_user")
    return {"enabled": _crawler_status["enabled"]}


@app.post("/feedback/accuracy")
def feedback_accuracy(payload: FeedbackRequest) -> dict:
    if not RUNTIME_LEARNING_ENABLED:
        raise HTTPException(status_code=400, detail="Runtime learning is disabled.")
    _runtime_trainer.save_feedback(
        sample_id=payload.sample_id,
        actual_label=payload.actual_label,
        rating=payload.rating,
        comment=payload.comment,
    )
    return {"status": "saved"}


@app.get("/train/runtime/status")
def runtime_train_status() -> dict:
    return _training_status


@app.post("/train/runtime")
def runtime_train(payload: RuntimeTrainRequest) -> dict:
    if not RUNTIME_LEARNING_ENABLED:
        raise HTTPException(status_code=400, detail="Runtime learning is disabled.")
    with _training_lock:
        if _training_status["running"]:
            return {"started": False, "message": "Training already running."}
        worker = threading.Thread(
            target=_run_runtime_training,
            kwargs={"include_pseudo": payload.include_pseudo},
            name="runtime-learning-train",
            daemon=True,
        )
        worker.start()
    return {"started": True}


@app.post("/infer/image")
def infer_image(file: UploadFile = File(...)) -> dict:
    _, _, predict_image, _ = _infer_funcs()
    if not IMAGE_MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Image model not found: {IMAGE_MODEL_PATH}")
    tmp = _save_upload(file)
    try:
        raw_prob = float(predict_image(tmp, IMAGE_MODEL_PATH))
        prob = _apply_calibration(raw_prob, "image")
        result = _as_result(prob, {"modality": "image", "filename": file.filename, "raw_prob_fake": raw_prob})
        if RUNTIME_LEARNING_ENABLED and tmp.exists() and (tmp.stat().st_size / (1024 * 1024)) <= MAX_RUNTIME_SAMPLE_MB:
            sample = _runtime_trainer.save_inference_sample(
                source_path=tmp,
                modality="image",
                prediction=result["prediction"],
                prob_fake=result["prob_fake"],
                confidence=result["confidence"],
            )
            result["details"]["sample_id"] = sample["sample_id"]
        return result
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
        raw_prob = float(predict_video(tmp, VIDEO_MODEL_PATH))
        prob = _apply_calibration(raw_prob, "video")
        result = _as_result(prob, {"modality": "video", "filename": file.filename, "raw_prob_fake": raw_prob})
        if RUNTIME_LEARNING_ENABLED and tmp.exists() and (tmp.stat().st_size / (1024 * 1024)) <= MAX_RUNTIME_SAMPLE_MB:
            sample = _runtime_trainer.save_inference_sample(
                source_path=tmp,
                modality="video",
                prediction=result["prediction"],
                prob_fake=result["prob_fake"],
                confidence=result["confidence"],
            )
            result["details"]["sample_id"] = sample["sample_id"]
        return result
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
        raw_prob = float(predict_audio(tmp, AUDIO_MODEL_PATH))
        prob = _apply_calibration(raw_prob, "audio")
        result = _as_result(prob, {"modality": "audio", "filename": file.filename, "raw_prob_fake": raw_prob})
        if RUNTIME_LEARNING_ENABLED and tmp.exists() and (tmp.stat().st_size / (1024 * 1024)) <= MAX_RUNTIME_SAMPLE_MB:
            sample = _runtime_trainer.save_inference_sample(
                source_path=tmp,
                modality="audio",
                prediction=result["prediction"],
                prob_fake=result["prob_fake"],
                confidence=result["confidence"],
            )
            result["details"]["sample_id"] = sample["sample_id"]
        return result
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

        fused_prob = float(fuse(image_prob, video_prob, audio_prob))
        final_prob = _apply_calibration(fused_prob, "multimodal")
        details["raw_prob_fake"] = fused_prob
        result = _as_result(final_prob, details)
        result["modalityScores"] = {
            "image": image_prob,
            "video": video_prob,
            "audio": audio_prob,
        }
        if RUNTIME_LEARNING_ENABLED and tmp.exists() and (tmp.stat().st_size / (1024 * 1024)) <= MAX_RUNTIME_SAMPLE_MB:
            sample = _runtime_trainer.save_inference_sample(
                source_path=tmp,
                modality="multimodal",
                prediction=result["prediction"],
                prob_fake=result["prob_fake"],
                confidence=result["confidence"],
            )
            result["details"]["sample_id"] = sample["sample_id"]
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Multimodal inference failed: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)
