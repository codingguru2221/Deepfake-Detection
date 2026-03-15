from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from deepfake_detector.data.auto_dataset_training import run_full_auto_training
from deepfake_detector.data.runtime_learning import RuntimeLearningManager
from deepfake_detector.integrations import aws_rekognition
from deepfake_detector.integrations import gemini_vision
from deepfake_detector.integrations import hf_deepfake
from deepfake_detector.integrations import bitmind
from deepfake_detector.integrations import openai_vision
from deepfake_detector.secrets_loader import apply_secret_env
from deepfake_detector.utils.timezone import IST, now_ist_iso

logger = logging.getLogger(__name__)

AWS_SECRET_NAME = os.getenv("AWS_SECRET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
if AWS_SECRET_NAME:
    try:
        apply_secret_env(AWS_SECRET_NAME, AWS_REGION)
    except Exception as exc:
        logger.warning("Failed to load AWS secret %s: %s", AWS_SECRET_NAME, exc)

try:
    from deepfake_detector.data.calibration import run_threshold_calibration, load_thresholds
except ModuleNotFoundError:
    logger.warning("Calibration module unavailable; using environment thresholds only.")

    def load_thresholds() -> dict:
        return {}

    def run_threshold_calibration(_modality: str = "all") -> dict:
        raise RuntimeError("Calibration module is unavailable in this deployment.")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMAGE_MODEL_PATH = Path(os.getenv("DF_IMAGE_MODEL", PROJECT_ROOT / "models" / "exports" / "image_tf_model.keras"))
VIDEO_MODEL_PATH = Path(os.getenv("DF_VIDEO_MODEL", PROJECT_ROOT / "models" / "checkpoints" / "video_gru.pt"))
AUDIO_MODEL_PATH = Path(os.getenv("DF_AUDIO_MODEL", PROJECT_ROOT / "models" / "exports" / "audio_rf.joblib"))

CRAWLER_ENABLED = os.getenv("DF_CRAWLER_ENABLED", "1").lower() not in {"0", "false", "no"}
CRAWLER_MAX_ITEMS = int(os.getenv("DF_CRAWLER_MAX_ITEMS", "100"))
CRAWLER_TIMEOUT_SECONDS = float(os.getenv("DF_CRAWLER_TIMEOUT_SECONDS", "8"))
CRAWLER_REFRESH_HOURS = int(os.getenv("DF_CRAWLER_REFRESH_HOURS", "24"))
CRAWLER_POLL_SECONDS = int(os.getenv("DF_CRAWLER_POLL_SECONDS", "300"))
CRAWLER_QUERY = os.getenv("DF_CRAWLER_QUERY", "deepfake dataset")
CRAWLER_OUTPUT = Path(os.getenv("DF_CRAWLER_OUTPUT", PROJECT_ROOT / "data" / "external" / "dataset_catalog.json"))
CRAWLER_LOG = Path(os.getenv("DF_CRAWLER_LOG", PROJECT_ROOT / "data" / "external" / "crawler.log"))
RUNTIME_LEARNING_ENABLED = os.getenv("DF_RUNTIME_LEARNING_ENABLED", "1").lower() not in {"0", "false", "no"}
AUTO_TRAIN_ON_CRAWLER = os.getenv("DF_AUTO_TRAIN_ON_CRAWLER", "1").lower() not in {"0", "false", "no"}
AUTO_TRAIN_MIN_RECORDS = int(os.getenv("DF_AUTO_TRAIN_MIN_RECORDS", "100"))
AUTO_FULL_MODEL_TRAIN_ENABLED = os.getenv("DF_AUTO_FULL_MODEL_TRAIN_ENABLED", "1").lower() not in {
    "0",
    "false",
    "no",
}
MAX_RUNTIME_SAMPLE_MB = float(os.getenv("DF_RUNTIME_MAX_SAMPLE_MB", "50"))
BITMIND_ENABLED = os.getenv("BITMIND_ENABLED", "0").lower() not in {"0", "false", "no"} and bitmind.is_enabled()
BITMIND_VERIFY_ON_INFER = os.getenv("BITMIND_VERIFY_ON_INFER", "1").lower() not in {"0", "false", "no"}
BITMIND_RICH = os.getenv("BITMIND_RICH", "0").lower() not in {"0", "false", "no"}
AWS_REKOGNITION_ENABLED = os.getenv("AWS_REKOGNITION_ENABLED", "0").lower() not in {"0", "false", "no"} and aws_rekognition.is_enabled()
HF_DEEPFAKE_ENABLED = hf_deepfake.is_enabled()
OPENAI_VISION_ENABLED = os.getenv("OPENAI_VISION_ENABLED", "1").lower() not in {"0", "false", "no"} and openai_vision.is_enabled()
GEMINI_VISION_ENABLED = os.getenv("GEMINI_VISION_ENABLED", "1").lower() not in {"0", "false", "no"} and gemini_vision.is_enabled()

# Conservative thresholds to reduce false positives.
DF_IMAGE_FAKE_THRESHOLD = float(os.getenv("DF_IMAGE_FAKE_THRESHOLD", "0.6"))
DF_IMAGE_REAL_THRESHOLD = float(os.getenv("DF_IMAGE_REAL_THRESHOLD", "0.4"))
DF_VIDEO_FAKE_THRESHOLD = float(os.getenv("DF_VIDEO_FAKE_THRESHOLD", "0.6"))
DF_VIDEO_REAL_THRESHOLD = float(os.getenv("DF_VIDEO_REAL_THRESHOLD", "0.4"))
DF_AUDIO_FAKE_THRESHOLD = float(os.getenv("DF_AUDIO_FAKE_THRESHOLD", "0.6"))
DF_AUDIO_REAL_THRESHOLD = float(os.getenv("DF_AUDIO_REAL_THRESHOLD", "0.4"))
DF_MULTIMODAL_FAKE_THRESHOLD = float(os.getenv("DF_MULTIMODAL_FAKE_THRESHOLD", "0.6"))
DF_MULTIMODAL_REAL_THRESHOLD = float(os.getenv("DF_MULTIMODAL_REAL_THRESHOLD", "0.4"))


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
    "crawl_cycle_running": False,
    "last_run": None,
    "last_error": None,
    "records": 0,
    "genuine_records": 0,
    "output": str(CRAWLER_OUTPUT),
    "log": str(CRAWLER_LOG),
    "auto_train_enabled": AUTO_TRAIN_ON_CRAWLER,
    "auto_train_min_records": AUTO_TRAIN_MIN_RECORDS,
}
_crawler_lock = threading.Lock()
_crawler_stop_event = threading.Event()
_crawler_thread: Optional[threading.Thread] = None
_crawler_scheduler_thread: Optional[threading.Thread] = None
_service_stop_event = threading.Event()
_runtime_trainer = RuntimeLearningManager(PROJECT_ROOT)
_training_status: dict = {"running": False, "last_run": None, "last_error": None, "last_result": None}
_training_lock = threading.Lock()
_full_train_status: dict = {"running": False, "last_run": None, "last_error": None, "last_result": None}
_full_train_lock = threading.Lock()
_calibrator_model = None
_calibrator_mtime: Optional[float] = None
_auto_train_state: dict[str, Optional[float]] = {"last_triggered_catalog_mtime": None}


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
    row = f"{now_ist_iso()} {message}"
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


def _maybe_invert_prob(prob_fake: float, modality: str) -> tuple[float, bool]:
    invert_any = os.getenv("DF_INVERT_PROB", "0").lower() in {"1", "true", "yes"}
    per_modality = {
        "image": "DF_INVERT_IMAGE_PROB",
        "video": "DF_INVERT_VIDEO_PROB",
        "audio": "DF_INVERT_AUDIO_PROB",
        "multimodal": "DF_INVERT_MULTIMODAL_PROB",
    }
    flag = per_modality.get(modality)
    invert = invert_any or (flag and os.getenv(flag, "0").lower() in {"1", "true", "yes"})
    if invert:
        return 1.0 - prob_fake, True
    return prob_fake, False


def _source_prob_fake(source: dict | None) -> float | None:
    if not source:
        return None
    value = source.get("prob_fake")
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    prediction = str(source.get("prediction") or "").lower()
    confidence = source.get("confidence")
    conf = float(confidence) if isinstance(confidence, (int, float)) else 0.5
    if prediction == "deepfake":
        return max(0.0, min(1.0, conf))
    if prediction == "real":
        return max(0.0, min(1.0, 1.0 - conf))
    return None


def _pair_average(*values: float | None) -> float | None:
    usable = [float(v) for v in values if isinstance(v, (int, float))]
    if not usable:
        return None
    return sum(usable) / len(usable)


def _find_source(details: dict, provider: str) -> dict | None:
    for source in details.get("sources", []) or []:
        if str(source.get("provider") or "").lower() == provider:
            return source
    return None


def _bitmind_prob_fake(details: dict) -> float | None:
    verdict = str(details.get("bitmind_verdict") or "").lower()
    confidence = details.get("bitmind_confidence")
    conf = float(confidence) if isinstance(confidence, (int, float)) else 0.5
    if verdict == "deepfake":
        return max(0.0, min(1.0, conf))
    if verdict == "real":
        return max(0.0, min(1.0, 1.0 - conf))
    return None


def _blend_image_video_probability(prob_fake: float, details: dict) -> float:
    modality = str(details.get("modality") or "").lower()
    if modality not in {"image", "video"}:
        return prob_fake

    local_prob = _source_prob_fake(_find_source(details, "local"))
    openai_prob = _source_prob_fake(_find_source(details, "openai"))
    gemini_prob = _source_prob_fake(_find_source(details, "gemini"))
    bitmind_prob = _bitmind_prob_fake(details)

    pair_local_openai = _pair_average(local_prob, openai_prob)
    pair_gemini_bitmind = _pair_average(gemini_prob, bitmind_prob)

    weights: list[tuple[float, float]] = [(float(prob_fake), 0.20)]
    if pair_local_openai is not None:
        weights.append((pair_local_openai, 0.50))
    if pair_gemini_bitmind is not None:
        weights.append((pair_gemini_bitmind, 0.30))

    adjusted = sum(value * weight for value, weight in weights) / sum(weight for _, weight in weights)

    bitmind_verdict = str(details.get("bitmind_verdict") or "").lower()
    if bitmind_verdict == "deepfake" and pair_gemini_bitmind is not None and pair_gemini_bitmind >= 0.75:
        adjusted = max(adjusted, pair_gemini_bitmind, 0.68)
        details["bitmind_priority_applied"] = True
    elif bitmind_verdict == "real" and pair_gemini_bitmind is not None and pair_gemini_bitmind <= 0.25:
        adjusted = min(adjusted, pair_gemini_bitmind, 0.32)

    details["fusion"] = {
        "base_prob_fake": float(prob_fake),
        "local_openai_prob_fake": pair_local_openai,
        "gemini_bitmind_prob_fake": pair_gemini_bitmind,
        "final_blended_prob_fake": float(adjusted),
    }
    return max(0.0, min(1.0, adjusted))


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
        _crawler_status["crawl_cycle_running"] = True
        _crawler_status["last_error"] = None
        _append_crawler_log("run_started")
        try:
            count = crawler.crawl_once()
            _crawler_status["records"] = count
            _crawler_status["genuine_records"] = _load_catalog_genuine_count()
            _append_crawler_log(f"run_completed records={count}")
            _trigger_auto_training_if_ready(int(_crawler_status.get("genuine_records") or count))
        except Exception as exc:
            _crawler_status["last_error"] = str(exc)
            _append_crawler_log(f"run_failed error={exc}")
        finally:
            _crawler_status["running"] = False
            _crawler_status["crawl_cycle_running"] = False
            if CRAWLER_OUTPUT.exists():
                mtime = datetime.fromtimestamp(Path(CRAWLER_OUTPUT).stat().st_mtime, tz=IST)
                _crawler_status["last_run"] = mtime.replace(microsecond=0).isoformat()
            else:
                _crawler_status["last_run"] = None

    global _crawler_thread
    _crawler_stop_event.clear()
    _crawler_thread = threading.Thread(target=_worker, name="dataset-crawler", daemon=True)
    _crawler_thread.start()
    return True


def _load_catalog_count() -> int:
    if not CRAWLER_OUTPUT.exists():
        return 0
    try:
        import json

        payload = json.loads(CRAWLER_OUTPUT.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            count = payload.get("count")
            if isinstance(count, int):
                return max(0, count)
            items = payload.get("items")
            if isinstance(items, list):
                return len(items)
    except Exception:
        return 0
    return 0


def _load_catalog_genuine_count() -> int:
    if not CRAWLER_OUTPUT.exists():
        return 0
    try:
        import json

        payload = json.loads(CRAWLER_OUTPUT.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            value = payload.get("genuine_count")
            if isinstance(value, int):
                return max(0, value)
            count = payload.get("count")
            if isinstance(count, int):
                return max(0, count)
    except Exception:
        return 0
    return 0


def _trigger_auto_training_if_ready(crawler_records: int) -> None:
    if not RUNTIME_LEARNING_ENABLED or not AUTO_TRAIN_ON_CRAWLER:
        return
    if crawler_records < AUTO_TRAIN_MIN_RECORDS:
        return
    if not CRAWLER_OUTPUT.exists():
        return
    catalog_mtime = CRAWLER_OUTPUT.stat().st_mtime
    if _auto_train_state.get("last_triggered_catalog_mtime") == catalog_mtime:
        return
    with _training_lock:
        if _training_status["running"]:
            return
        worker = threading.Thread(
            target=_run_runtime_training,
            kwargs={"include_pseudo": True},
            name="runtime-learning-auto-train",
            daemon=True,
        )
        _training_status["running"] = True
        _auto_train_state["last_triggered_catalog_mtime"] = catalog_mtime
        worker.start()
    _append_crawler_log(
        f"auto_training_triggered records={crawler_records} threshold={AUTO_TRAIN_MIN_RECORDS}"
    )
    if AUTO_FULL_MODEL_TRAIN_ENABLED:
        with _full_train_lock:
            if not _full_train_status["running"]:
                worker = threading.Thread(
                    target=_run_full_model_training,
                    name="crawler-full-model-train",
                    daemon=True,
                )
                _full_train_status["running"] = True
                worker.start()


def _crawler_scheduler_loop() -> None:
    while not _service_stop_event.wait(max(30, CRAWLER_POLL_SECONDS)):
        if not _crawler_status["enabled"] or _crawler_status["running"]:
            continue
        current = int(_crawler_status.get("genuine_records") or _crawler_status.get("records") or 0)
        should_force = AUTO_TRAIN_ON_CRAWLER and current < AUTO_TRAIN_MIN_RECORDS
        _start_dataset_crawler(force=should_force)


def _crawler_status_snapshot() -> dict:
    scheduler_alive = _crawler_scheduler_thread is not None and _crawler_scheduler_thread.is_alive()
    status = dict(_crawler_status)
    status["service_running"] = bool(status.get("enabled")) and scheduler_alive
    status["running"] = bool(status.get("enabled")) and (scheduler_alive or bool(status.get("crawl_cycle_running")))
    return status


def _run_runtime_training(include_pseudo: bool) -> None:
    _training_status["running"] = True
    _training_status["last_error"] = None
    try:
        result = _runtime_trainer.run_training(CRAWLER_OUTPUT, include_pseudo=include_pseudo)
        _runtime_trainer.apply_training_success(result)
        _runtime_trainer.refresh_model_accuracies()
        _training_status["last_result"] = {
            "status": result.status,
            "reason": result.reason,
            "manifest_path": result.manifest_path,
            "calibrator_path": result.calibrator_path,
            "user_labeled_count": result.user_labeled_count,
            "pseudo_count": result.pseudo_count,
            "crawler_refs_count": result.crawler_refs_count,
            "trainable_samples": result.trainable_samples,
            "calibrator_accuracy": result.calibrator_accuracy,
            "calibrator_auc": result.calibrator_auc,
        }
    except Exception as exc:
        _training_status["last_error"] = str(exc)
    finally:
        _training_status["running"] = False
        _training_status["last_run"] = now_ist_iso()


def _run_full_model_training() -> None:
    _full_train_status["running"] = True
    _full_train_status["last_error"] = None
    try:
        result = run_full_auto_training(project_root=PROJECT_ROOT, catalog_path=CRAWLER_OUTPUT)
        _full_train_status["last_result"] = {
            "status": result.status,
            "reason": result.reason,
            "downloaded_datasets": result.downloaded_datasets,
            "manifests": result.manifests,
            "trained_modalities": result.trained_modalities,
        }
    except Exception as exc:
        _full_train_status["last_error"] = str(exc)
    finally:
        _full_train_status["running"] = False
        _full_train_status["last_run"] = now_ist_iso()


@app.on_event("startup")
def _app_startup() -> None:
    _runtime_trainer.refresh_model_accuracies()
    _crawler_status["records"] = _load_catalog_count()
    _crawler_status["genuine_records"] = _load_catalog_genuine_count()
    if CRAWLER_OUTPUT.exists():
        mtime = datetime.fromtimestamp(Path(CRAWLER_OUTPUT).stat().st_mtime, tz=IST)
        _crawler_status["last_run"] = mtime.replace(microsecond=0).isoformat()
    _trigger_auto_training_if_ready(int(_crawler_status.get("genuine_records") or _crawler_status.get("records") or 0))
    _start_dataset_crawler()
    global _crawler_scheduler_thread
    if _crawler_scheduler_thread is None or not _crawler_scheduler_thread.is_alive():
        _service_stop_event.clear()
        _crawler_scheduler_thread = threading.Thread(
            target=_crawler_scheduler_loop,
            name="dataset-crawler-scheduler",
            daemon=True,
        )
        _crawler_scheduler_thread.start()


@app.on_event("shutdown")
def _app_shutdown() -> None:
    _service_stop_event.set()
    _crawler_stop_event.set()


def _as_result(prob_fake: float, details: dict) -> dict:
    modality = str(details.get("modality") or "").lower()
    prob_fake = _blend_image_video_probability(prob_fake, details)
    stored = load_thresholds()
    if modality in stored:
        real_th, fake_th = stored[modality]["real"], stored[modality]["fake"]
        details["thresholds_source"] = "calibrated"
    else:
        thresholds = {
            "image": (DF_IMAGE_REAL_THRESHOLD, DF_IMAGE_FAKE_THRESHOLD),
            "video": (DF_VIDEO_REAL_THRESHOLD, DF_VIDEO_FAKE_THRESHOLD),
            "audio": (DF_AUDIO_REAL_THRESHOLD, DF_AUDIO_FAKE_THRESHOLD),
            "multimodal": (DF_MULTIMODAL_REAL_THRESHOLD, DF_MULTIMODAL_FAKE_THRESHOLD),
        }
        real_th, fake_th = thresholds.get(modality, (0.4, 0.6))
        details["thresholds_source"] = "env"
    if prob_fake >= fake_th:
        prediction = "deepfake"
        confidence = prob_fake
    elif prob_fake <= real_th:
        prediction = "real"
        confidence = 1.0 - prob_fake
    else:
        prediction = "uncertain"
        confidence = max(prob_fake, 1.0 - prob_fake)
    details["thresholds"] = {"real": real_th, "fake": fake_th}

    bitmind_verdict = str(details.get("bitmind_verdict") or "").lower()
    original_prediction = prediction
    if bitmind_verdict in {"real", "deepfake"}:
        if bitmind_verdict == "real":
            prediction = "real"
            confidence = max(float(details.get("bitmind_confidence") or 0.0), 1.0 - prob_fake)
        elif prediction == "deepfake" and bitmind_verdict == "deepfake":
            prediction = "deepfake"
            confidence = max(float(details.get("bitmind_confidence") or 0.0), prob_fake)
        else:
            prediction = "uncertain"
            confidence = max(prob_fake, 1.0 - prob_fake)
        details["bitmind_disagree"] = bitmind_verdict != original_prediction
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


def _attach_runtime_sample_if_possible(result: dict, tmp: Path, modality: str) -> None:
    if not RUNTIME_LEARNING_ENABLED or not tmp.exists():
        return
    if (tmp.stat().st_size / (1024 * 1024)) > MAX_RUNTIME_SAMPLE_MB:
        return
    try:
        sample = _runtime_trainer.save_inference_sample(
            source_path=tmp,
            modality=modality,
            prediction=result["prediction"],
            prob_fake=result["prob_fake"],
            confidence=result["confidence"],
        )
        result.setdefault("details", {})["sample_id"] = sample["sample_id"]
    except Exception as exc:
        result.setdefault("details", {})["runtime_sample_error"] = str(exc)


def _provider_weight(name: str) -> float:
    weights = {
        "local": float(os.getenv("DF_WEIGHT_LOCAL", "0.25")),
        "huggingface": float(os.getenv("DF_WEIGHT_HUGGINGFACE", "0.2")),
        "aws_rekognition": float(os.getenv("DF_WEIGHT_AWS_REKOGNITION", "0.2")),
        "openai": float(os.getenv("DF_WEIGHT_OPENAI", "0.3")),
        "gemini": float(os.getenv("DF_WEIGHT_GEMINI", "0.3")),
    }
    return max(0.0, weights.get(name, 0.0))


def _classify_provider_error(message: str) -> str:
    text = message.lower()
    if "429" in text or "too many requests" in text or "rate limit" in text or "quota" in text:
        return "rate_limited"
    if "400" in text or "bad request" in text:
        return "bad_request"
    if "getaddrinfo" in text or "name or service not known" in text or "forbidden by its access permissions" in text:
        return "network"
    return "other"


def _summarize_provider_error(provider: str, message: str) -> str:
    error_type = _classify_provider_error(message)
    labels = {
        "rate_limited": "temporarily rate limited",
        "bad_request": "request rejected",
        "network": "network unavailable",
        "other": "temporarily unavailable",
    }
    return f"{provider}={labels.get(error_type, 'temporarily unavailable')}"


def _ensemble_image_sources(sources: list[dict]) -> tuple[float, list[dict]]:
    usable = [src for src in sources if isinstance(src.get("prob_fake"), (int, float))]
    if not usable:
        raise RuntimeError("No usable image inference sources were available.")

    total_weight = 0.0
    weighted_sum = 0.0
    normalized_sources: list[dict] = []
    for src in usable:
        name = str(src.get("provider") or "unknown")
        weight = _provider_weight(name)
        prob_fake = float(src["prob_fake"])
        weighted_sum += prob_fake * weight
        total_weight += weight
        normalized_sources.append(
            {
                "provider": name,
                "prob_fake": prob_fake,
                "confidence": src.get("confidence"),
                "prediction": src.get("prediction"),
                "weight": weight,
                "model": src.get("model"),
                "backend": src.get("backend"),
                "available": True,
            }
        )

    if total_weight <= 0:
        avg = sum(float(src["prob_fake"]) for src in usable) / len(usable)
        return avg, normalized_sources
    return weighted_sum / total_weight, normalized_sources


def _run_image_inference(tmp: Path, filename: str | None = None) -> tuple[float, dict]:
    _, _, predict_image, _ = _infer_funcs()
    errors: list[str] = []
    sources: list[dict] = []

    if IMAGE_MODEL_PATH.exists():
        try:
            sources.append(
                {
                    "provider": "local",
                    "backend": "local",
                    "model": str(IMAGE_MODEL_PATH),
                    "prob_fake": float(predict_image(tmp, IMAGE_MODEL_PATH)),
                    "prediction": None,
                    "confidence": None,
                }
            )
        except Exception as exc:
            logger.warning("Local image model failed, trying fallback providers: %s", exc)
            errors.append(f"local={exc}")

    if HF_DEEPFAKE_ENABLED:
        try:
            hf_result = hf_deepfake.detect_image_file(tmp)
            hf_result["provider"] = "huggingface"
            hf_result["backend"] = "huggingface"
            sources.append(hf_result)
        except Exception as exc:
            errors.append(f"huggingface={exc}")

    if AWS_REKOGNITION_ENABLED:
        try:
            aws_result = aws_rekognition.detect_image_bytes(tmp.read_bytes())
            aws_result["provider"] = "aws_rekognition"
            aws_result["backend"] = "aws_rekognition"
            sources.append(aws_result)
        except Exception as exc:
            errors.append(f"aws_rekognition={exc}")

    image_bytes = tmp.read_bytes()
    suffix = tmp.suffix.lower()
    mime_type = {
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix, "image/jpeg")

    if OPENAI_VISION_ENABLED:
        try:
            openai_result = openai_vision.detect_image_bytes(image_bytes, filename=filename, mime_type=mime_type)
            sources.append(openai_result)
        except Exception as exc:
            errors.append(_summarize_provider_error("openai", str(exc)))
            sources.append(
                {
                    "provider": "openai",
                    "prediction": "unavailable",
                    "prob_fake": None,
                    "confidence": None,
                    "model": os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini"),
                    "backend": "openai",
                    "available": False,
                    "error": str(exc),
                    "error_type": _classify_provider_error(str(exc)),
                }
            )

    if GEMINI_VISION_ENABLED:
        try:
            gemini_result = gemini_vision.detect_image_bytes(image_bytes, filename=filename, mime_type=mime_type)
            sources.append(gemini_result)
        except Exception as exc:
            errors.append(_summarize_provider_error("gemini", str(exc)))
            sources.append(
                {
                    "provider": "gemini",
                    "prediction": "unavailable",
                    "prob_fake": None,
                    "confidence": None,
                    "model": os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash"),
                    "backend": "gemini",
                    "available": False,
                    "error": str(exc),
                    "error_type": _classify_provider_error(str(exc)),
                }
            )

    if sources:
        prob_fake, ensemble_sources = _ensemble_image_sources([src for src in sources if src.get("available", True)])
        unavailable = [src for src in sources if not src.get("available", True)]
        ensemble_sources.extend(unavailable)
        return prob_fake, {"sources": ensemble_sources, "errors": errors}
    if errors:
        raise RuntimeError("; ".join(errors))
    raise RuntimeError(f"Image model not found: {IMAGE_MODEL_PATH}")


def _run_video_inference(tmp: Path) -> tuple[float, dict | None, dict | None, str]:
    _, _, _, predict_video = _infer_funcs()
    errors: list[str] = []

    if VIDEO_MODEL_PATH.exists():
        try:
            return float(predict_video(tmp, VIDEO_MODEL_PATH)), None, None, "local"
        except Exception as exc:
            logger.warning("Local video model failed, trying fallback providers: %s", exc)
            errors.append(f"local={exc}")

    if HF_DEEPFAKE_ENABLED:
        try:
            hf_result = hf_deepfake.detect_video_file(tmp)
            return float(hf_result["prob_fake"]), hf_result, None, "huggingface"
        except Exception as exc:
            errors.append(f"huggingface={exc}")

    if AWS_REKOGNITION_ENABLED:
        try:
            aws_result = aws_rekognition.detect_video_file(tmp)
            return float(aws_result["prob_fake"]), None, aws_result, "aws_rekognition"
        except Exception as exc:
            errors.append(f"aws_rekognition={exc}")

    if errors:
        raise RuntimeError("; ".join(errors))
    raise RuntimeError(f"Video model not found: {VIDEO_MODEL_PATH}")


@app.get("/health")
def health() -> dict:
    availability = {
        "image": IMAGE_MODEL_PATH.exists() or AWS_REKOGNITION_ENABLED or HF_DEEPFAKE_ENABLED,
        "video": VIDEO_MODEL_PATH.exists() or AWS_REKOGNITION_ENABLED or HF_DEEPFAKE_ENABLED,
        "audio": AUDIO_MODEL_PATH.exists(),
        "multimodal": True,
    }
    return {
        "status": "ok",
        "modelFiles": availability,
        "models": _runtime_trainer.get_model_status(availability),
        "datasetCrawler": _crawler_status_snapshot(),
        "runtimeLearning": {
            "enabled": RUNTIME_LEARNING_ENABLED,
            "training": _training_status,
            "fullTraining": {
                "enabled": AUTO_FULL_MODEL_TRAIN_ENABLED,
                "status": _full_train_status,
            },
        },
        "awsRekognition": {
            "enabled": AWS_REKOGNITION_ENABLED,
            "region": os.getenv("AWS_REGION", "us-east-1"),
            "projectVersionArn": os.getenv("AWS_REKOGNITION_PROJECT_VERSION_ARN"),
        },
        "hfDeepfake": {
            "enabled": HF_DEEPFAKE_ENABLED,
            "modelId": os.getenv("HF_DEEPFAKE_MODEL_ID", "prithivMLmods/deepfake-detector-model-v1"),
        },
        "bitmind": {
            "enabled": BITMIND_ENABLED,
            "verifyOnInfer": BITMIND_VERIFY_ON_INFER,
            "rich": BITMIND_RICH,
            "oracleId": os.getenv("BITMIND_ORACLE_ID", "34"),
        },
        "openaiVision": {
            "enabled": OPENAI_VISION_ENABLED,
            "model": os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini"),
        },
        "geminiVision": {
            "enabled": GEMINI_VISION_ENABLED,
            "model": os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash"),
        },
    }


@app.get("/crawler/status")
def crawler_status() -> dict:
    return _crawler_status_snapshot()


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
    _runtime_trainer.refresh_model_accuracies()
    return {"status": "saved"}


@app.get("/train/runtime/status")
def runtime_train_status() -> dict:
    return _training_status


@app.get("/train/full/status")
def full_train_status() -> dict:
    return _full_train_status


@app.post("/train/runtime")
def runtime_train(payload: RuntimeTrainRequest) -> dict:
    if not RUNTIME_LEARNING_ENABLED:
        raise HTTPException(status_code=400, detail="Runtime learning is disabled.")
    with _training_lock:
        if _training_status["running"]:
            return {"started": False, "message": "Training already running."}
        _training_status["running"] = True
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
    if not HF_DEEPFAKE_ENABLED and not AWS_REKOGNITION_ENABLED and not IMAGE_MODEL_PATH.exists():
        if not OPENAI_VISION_ENABLED and not GEMINI_VISION_ENABLED:
            raise HTTPException(status_code=503, detail=f"Image model not found: {IMAGE_MODEL_PATH}")
    tmp = _save_upload(file)
    try:
        raw_prob, provider_meta = _run_image_inference(tmp, filename=file.filename)
        prob, inverted = _maybe_invert_prob(raw_prob, "image")
        prob = _apply_calibration(prob, "image")
        details = {
            "modality": "image",
            "filename": file.filename,
            "backend": "ensemble",
            "raw_prob_fake": prob,
            "raw_prob_fake_raw": raw_prob,
            "prob_inverted": inverted,
        }
        if provider_meta.get("sources"):
            details["sources"] = provider_meta["sources"]
        if provider_meta.get("errors"):
            details["provider_errors"] = provider_meta["errors"]
        if BITMIND_ENABLED and BITMIND_VERIFY_ON_INFER and tmp.exists():
            try:
                bm = bitmind.detect_image_bytes(
                    tmp.read_bytes(),
                    filename=file.filename,
                    mime_type=file.content_type,
                    source=file.filename,
                    rich=BITMIND_RICH,
                )
                details["bitmind"] = bm
                verdict = bitmind.extract_verdict(bm)
                if verdict:
                    details["bitmind_verdict"] = verdict["prediction"]
                    details["bitmind_confidence"] = verdict.get("confidence")
            except Exception as exc:
                details["bitmind_error"] = str(exc)
                details["bitmind_error_type"] = _classify_provider_error(str(exc))
        result = _as_result(prob, details)
        _attach_runtime_sample_if_possible(result, tmp, "image")
        return result
    except Exception as exc:
        logger.exception("Image inference failed for file=%s", file.filename)
        raise HTTPException(status_code=500, detail=f"Image inference failed: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/infer/video")
def infer_video(file: UploadFile = File(...)) -> dict:
    if not HF_DEEPFAKE_ENABLED and not AWS_REKOGNITION_ENABLED and not VIDEO_MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Video model not found: {VIDEO_MODEL_PATH}")
    tmp = _save_upload(file)
    try:
        raw_prob, hf_result, aws_result, backend = _run_video_inference(tmp)
        prob, inverted = _maybe_invert_prob(raw_prob, "video")
        prob = _apply_calibration(prob, "video")
        details = {
            "modality": "video",
            "filename": file.filename,
            "backend": backend,
            "raw_prob_fake": prob,
            "raw_prob_fake_raw": raw_prob,
            "prob_inverted": inverted,
        }
        if hf_result:
            details["hf_deepfake"] = hf_result
        if aws_result:
            details["aws_rekognition"] = aws_result
        if BITMIND_ENABLED and BITMIND_VERIFY_ON_INFER and tmp.exists():
            try:
                bm = bitmind.detect_video_file(
                    tmp,
                    source=file.filename,
                    rich=BITMIND_RICH,
                )
                details["bitmind"] = bm
                verdict = bitmind.extract_verdict(bm)
                if verdict:
                    details["bitmind_verdict"] = verdict["prediction"]
                    details["bitmind_confidence"] = verdict.get("confidence")
            except Exception as exc:
                details["bitmind_error"] = str(exc)
                details["bitmind_error_type"] = _classify_provider_error(str(exc))
        result = _as_result(prob, details)
        _attach_runtime_sample_if_possible(result, tmp, "video")
        return result
    except Exception as exc:
        logger.exception("Video inference failed for file=%s", file.filename)
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
        prob, inverted = _maybe_invert_prob(raw_prob, "audio")
        prob = _apply_calibration(prob, "audio")
        result = _as_result(
            prob,
            {
                "modality": "audio",
                "filename": file.filename,
                "raw_prob_fake": prob,
                "raw_prob_fake_raw": raw_prob,
                "prob_inverted": inverted,
            },
        )
        _attach_runtime_sample_if_possible(result, tmp, "audio")
        return result
    except Exception as exc:
        logger.exception("Audio inference failed for file=%s", file.filename)
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
            image_raw = predict_image(tmp, IMAGE_MODEL_PATH)
            image_prob, inv = _maybe_invert_prob(float(image_raw), "image")
            details["image_prob_inverted"] = inv
        elif suffix in {".wav", ".mp3", ".flac", ".m4a"} and AUDIO_MODEL_PATH.exists():
            audio_raw = predict_audio(tmp, AUDIO_MODEL_PATH)
            audio_prob, inv = _maybe_invert_prob(float(audio_raw), "audio")
            details["audio_prob_inverted"] = inv
        elif suffix in {".mp4", ".avi", ".mov", ".mkv"}:
            if VIDEO_MODEL_PATH.exists():
                video_raw = predict_video(tmp, VIDEO_MODEL_PATH)
                video_prob, inv = _maybe_invert_prob(float(video_raw), "video")
                details["video_prob_inverted"] = inv

            # Attempt audio branch for video if audio model exists.
            if AUDIO_MODEL_PATH.exists():
                try:
                    from pydub import AudioSegment

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_tmp:
                        AudioSegment.from_file(tmp).set_channels(1).set_frame_rate(16000).export(
                            wav_tmp.name, format="wav"
                        )
                        audio_raw = predict_audio(Path(wav_tmp.name), AUDIO_MODEL_PATH)
                        audio_prob, inv = _maybe_invert_prob(float(audio_raw), "audio")
                        details["audio_prob_inverted"] = inv
                    Path(wav_tmp.name).unlink(missing_ok=True)
                except Exception:
                    details["audio_from_video"] = "failed"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file for multimodal inference.")

        fused_raw = float(fuse(image_prob, video_prob, audio_prob))
        fused_prob, fused_inverted = _maybe_invert_prob(fused_raw, "multimodal")
        final_prob = _apply_calibration(fused_prob, "multimodal")
        details["raw_prob_fake"] = fused_prob
        details["raw_prob_fake_raw"] = fused_raw
        details["prob_inverted"] = fused_inverted
        result = _as_result(final_prob, details)
        result["modalityScores"] = {
            "image": image_prob,
            "video": video_prob,
            "audio": audio_prob,
        }
        _attach_runtime_sample_if_possible(result, tmp, "multimodal")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Multimodal inference failed for file=%s", file.filename)
        raise HTTPException(status_code=500, detail=f"Multimodal inference failed: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/calibration/run")
def run_calibration(modality: str = "all") -> dict:
    if modality not in {"all", "image", "video", "audio"}:
        raise HTTPException(status_code=400, detail="modality must be one of: all, image, video, audio")
    try:
        result = run_threshold_calibration(modality)
        return {"status": "ok", "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Calibration failed: {exc}") from exc


@app.get("/calibration")
def get_calibration() -> dict:
    return {"thresholds": load_thresholds()}
