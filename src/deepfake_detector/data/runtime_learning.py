from __future__ import annotations

import json
import hashlib
import shutil
import threading
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from deepfake_detector.utils.timezone import IST, now_ist_iso


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "image": {
        "label": "Image CNN",
        "parameter_count": 105729,
        "parameter_summary": "CNN weights",
    },
    "video": {
        "label": "Video GRU",
        "parameter_count": 1248513,
        "parameter_summary": "GRU + dense weights",
    },
    "audio": {
        "label": "Audio RandomForest",
        "parameter_count": 300,
        "parameter_summary": "Decision trees",
    },
    "multimodal": {
        "label": "Fusion Calibrator",
        "parameter_count": 5,
        "parameter_summary": "Calibration features",
    },
}


def _now() -> str:
    return now_ist_iso()


def _append_jsonl(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=True)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return rows


@dataclass
class RuntimeTrainResult:
    status: str
    reason: str | None
    manifest_path: str
    calibrator_path: str | None
    user_labeled_count: int
    pseudo_count: int
    crawler_refs_count: int
    trainable_samples: int
    calibrator_accuracy: float | None
    calibrator_auc: float | None


class RuntimeLearningManager:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.runtime_dir = self.project_root / "data" / "runtime"
        self.upload_dir = self.runtime_dir / "uploads"
        self.samples_file = self.runtime_dir / "user_samples.jsonl"
        self.feedback_file = self.runtime_dir / "feedback.jsonl"
        self.jobs_file = self.runtime_dir / "training_jobs.jsonl"
        self.manifest_file = self.runtime_dir / "runtime_training_manifest.json"
        self.model_metrics_file = self.runtime_dir / "model_metrics.json"
        self.calibrator_file = self.project_root / "models" / "exports" / "runtime_calibrator.joblib"
        self.log_file = self.runtime_dir / "runtime_learning.log"
        self._lock = threading.Lock()
        self._ensure_model_metrics()

    def save_inference_sample(
        self,
        source_path: Path,
        modality: str,
        prediction: str,
        prob_fake: float,
        confidence: float,
    ) -> dict[str, str]:
        sample_id = uuid4().hex
        ext = source_path.suffix or ".bin"
        dest = self.upload_dir / f"{sample_id}{ext}"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)

        row = {
            "sample_id": sample_id,
            "path": str(dest),
            "modality": modality,
            "prediction": prediction,
            "prob_fake": float(prob_fake),
            "confidence": float(confidence),
            "created_at": _now(),
        }
        _append_jsonl(self.samples_file, row, self._lock)
        self._log(f"sample_saved id={sample_id} modality={modality} prediction={prediction}")
        return {"sample_id": sample_id, "path": str(dest)}

    def save_feedback(
        self,
        sample_id: str,
        actual_label: str,
        rating: int | None = None,
        comment: str | None = None,
    ) -> None:
        row = {
            "sample_id": sample_id,
            "actual_label": actual_label,
            "rating": rating,
            "comment": comment,
            "created_at": _now(),
        }
        _append_jsonl(self.feedback_file, row, self._lock)
        self._log(f"feedback_saved id={sample_id} label={actual_label} rating={rating}")

    def run_training(self, crawler_catalog_path: Path, include_pseudo: bool = True) -> RuntimeTrainResult:
        samples = _read_jsonl(self.samples_file)
        feedback_rows = _read_jsonl(self.feedback_file)
        feedback_by_id = {str(r.get("sample_id")): r for r in feedback_rows if r.get("sample_id")}

        labeled: list[dict[str, Any]] = []
        pseudo: list[dict[str, Any]] = []
        for sample in samples:
            sid = str(sample.get("sample_id", ""))
            path = str(sample.get("path", ""))
            modality = str(sample.get("modality", ""))
            if not sid or not path or not modality:
                continue

            fb = feedback_by_id.get(sid)
            if fb:
                label_name = str(fb.get("actual_label", "")).strip().lower()
                if label_name in {"real", "deepfake"}:
                    labeled.append(
                        {
                            "sample_id": sid,
                            "path": path,
                            "modality": modality,
                            "label": 1 if label_name == "deepfake" else 0,
                            "source": "user_feedback",
                        }
                    )
                continue

            if include_pseudo:
                confidence = float(sample.get("confidence", 0.0))
                prediction = str(sample.get("prediction", "")).strip().lower()
                if confidence >= 0.95 and prediction in {"real", "deepfake"}:
                    pseudo.append(
                        {
                            "sample_id": sid,
                            "path": path,
                            "modality": modality,
                            "label": 1 if prediction == "deepfake" else 0,
                            "source": "high_confidence_pseudo",
                        }
                    )

        crawler_refs: list[dict[str, Any]] = []
        if crawler_catalog_path.exists():
            try:
                payload = json.loads(crawler_catalog_path.read_text(encoding="utf-8"))
                items = payload.get("items", []) if isinstance(payload, dict) else []
                if isinstance(items, list):
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        crawler_refs.append(
                            {
                                "source": str(it.get("source", "")),
                                "title": str(it.get("title", "")),
                                "url": str(it.get("url", "")),
                            }
                        )
            except (OSError, json.JSONDecodeError):
                crawler_refs = []

        manifest = {
            "generated_at": _now(),
            "user_labeled_samples": labeled,
            "pseudo_labeled_samples": pseudo,
            "crawler_dataset_references": crawler_refs,
            "counts": {
                "user_labeled": len(labeled),
                "pseudo_labeled": len(pseudo),
                "crawler_refs": len(crawler_refs),
            },
        }
        self.manifest_file.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_file.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

        calibrator, calibrator_info = self._train_calibrator(samples, feedback_rows)
        status = "completed" if calibrator else "skipped"
        reason = None if calibrator else str((calibrator_info or {}).get("reason", "insufficient_trainable_data"))
        result = RuntimeTrainResult(
            status=status,
            reason=reason,
            manifest_path=str(self.manifest_file),
            calibrator_path=str(self.calibrator_file) if calibrator else None,
            user_labeled_count=len(labeled),
            pseudo_count=len(pseudo),
            crawler_refs_count=len(crawler_refs),
            trainable_samples=int((calibrator_info or {}).get("train_rows", 0)),
            calibrator_accuracy=(calibrator_info or {}).get("accuracy"),
            calibrator_auc=(calibrator_info or {}).get("auc"),
        )
        _append_jsonl(
            self.jobs_file,
            {
                "created_at": _now(),
                "status": result.status,
                "reason": result.reason,
                "manifest_path": result.manifest_path,
                "calibrator_path": result.calibrator_path,
                "counts": {
                    "user_labeled": result.user_labeled_count,
                    "pseudo_labeled": result.pseudo_count,
                    "crawler_refs": result.crawler_refs_count,
                    "trainable_samples": result.trainable_samples,
                },
                "metrics": {
                    "calibrator_accuracy": result.calibrator_accuracy,
                    "calibrator_auc": result.calibrator_auc,
                },
            },
            self._lock,
        )
        self._log(
            "training_done "
            f"status={result.status} reason={result.reason} "
            f"user_labeled={result.user_labeled_count} pseudo={result.pseudo_count} crawler_refs={result.crawler_refs_count}"
        )
        return result

    def get_recent_logs(self, limit: int = 100) -> list[str]:
        if not self.log_file.exists():
            return []
        with self.log_file.open("r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        return lines[-max(1, limit) :]

    def apply_training_success(self, result: RuntimeTrainResult) -> dict[str, Any]:
        payload = self._load_model_metrics()
        if result.status != "completed":
            return payload
        manifest = self._read_training_manifest()
        fingerprint = self._training_data_fingerprint(manifest)
        if fingerprint and payload.get("last_training_data_fingerprint") == fingerprint:
            self._log("model_metrics_skipped reason=duplicate_training_data")
            return payload
        increment = int(result.user_labeled_count + result.pseudo_count + result.crawler_refs_count)
        if increment <= 0:
            return payload

        models = payload.get("models", {})
        if not isinstance(models, dict):
            models = {}
        for modality in MODEL_SPECS:
            row = models.get(modality, {})
            if not isinstance(row, dict):
                row = {}
            current = int(row.get("trained_data_points", 0))
            row["trained_data_points"] = current + increment
            row["last_trained_at"] = _now()
            row["successful_training_runs"] = int(row.get("successful_training_runs", 0)) + 1
            models[modality] = row
        payload["models"] = models
        payload["updated_at"] = _now()
        payload["last_training_data_fingerprint"] = fingerprint
        self._save_model_metrics(payload)
        self._log(f"model_metrics_updated trained_increment={increment}")
        return payload

    def refresh_model_accuracies(self) -> dict[str, Any]:
        samples = _read_jsonl(self.samples_file)
        feedback_rows = _read_jsonl(self.feedback_file)
        sample_by_id = {str(r.get("sample_id")): r for r in samples if r.get("sample_id")}

        correct_by_modality: dict[str, int] = {}
        total_by_modality: dict[str, int] = {}
        for fb in feedback_rows:
            sid = str(fb.get("sample_id", "")).strip()
            actual = str(fb.get("actual_label", "")).strip().lower()
            sample = sample_by_id.get(sid)
            if not sample or actual not in {"real", "deepfake"}:
                continue
            modality = str(sample.get("modality", "")).strip().lower()
            predicted = str(sample.get("prediction", "")).strip().lower()
            if modality not in MODEL_SPECS:
                continue
            total_by_modality[modality] = total_by_modality.get(modality, 0) + 1
            if predicted == actual:
                correct_by_modality[modality] = correct_by_modality.get(modality, 0) + 1

        payload = self._load_model_metrics()
        models = payload.get("models", {})
        if not isinstance(models, dict):
            models = {}
        for modality in MODEL_SPECS:
            row = models.get(modality, {})
            if not isinstance(row, dict):
                row = {}
            total = int(total_by_modality.get(modality, 0))
            correct = int(correct_by_modality.get(modality, 0))
            row["accuracy"] = (correct / total) if total > 0 else None
            row["evaluated_samples"] = total
            models[modality] = row
        payload["models"] = models
        payload["updated_at"] = _now()
        self._save_model_metrics(payload)
        return payload

    def get_model_status(self, availability: dict[str, bool] | None = None) -> dict[str, dict[str, Any]]:
        payload = self._load_model_metrics()
        models = payload.get("models", {})
        file_times = self._model_file_times()
        status: dict[str, dict[str, Any]] = {}
        for modality, spec in MODEL_SPECS.items():
            row = models.get(modality, {})
            if not isinstance(row, dict):
                row = {}
            last_trained_at = row.get("last_trained_at")
            file_time = file_times.get(modality)
            if file_time is not None:
                if last_trained_at is None:
                    last_trained_at = file_time
                else:
                    parsed = self._parse_iso(last_trained_at)
                    if parsed is None or file_time > parsed:
                        last_trained_at = file_time
            status[modality] = {
                "label": spec["label"],
                "parameter_count": spec["parameter_count"],
                "parameter_summary": spec["parameter_summary"],
                "trained_data_points": int(row.get("trained_data_points", 0)),
                "accuracy": row.get("accuracy"),
                "evaluated_samples": int(row.get("evaluated_samples", 0)),
                "last_trained_at": last_trained_at,
                "successful_training_runs": int(row.get("successful_training_runs", 0)),
                "model_available": bool((availability or {}).get(modality, False)),
            }
        return status

    def _train_calibrator(
        self, samples: list[dict[str, Any]], feedback_rows: list[dict[str, Any]]
    ) -> tuple[bool, dict[str, Any] | None]:
        feedback_by_id = {str(r.get("sample_id")): r for r in feedback_rows if r.get("sample_id")}
        rows: list[tuple[list[float], int]] = []
        for sample in samples:
            sid = str(sample.get("sample_id", ""))
            fb = feedback_by_id.get(sid)
            if not fb:
                continue
            actual = str(fb.get("actual_label", "")).strip().lower()
            if actual not in {"real", "deepfake"}:
                continue
            modality = str(sample.get("modality", "")).strip().lower()
            prob_fake = float(sample.get("prob_fake", 0.5))
            target = 1 if actual == "deepfake" else 0
            features = [
                prob_fake,
                1.0 if modality == "image" else 0.0,
                1.0 if modality == "video" else 0.0,
                1.0 if modality == "audio" else 0.0,
                1.0 if modality == "multimodal" else 0.0,
            ]
            rows.append((features, target))

        # Need at least two classes for a usable calibrator.
        classes = {target for _, target in rows}
        if len(rows) < 8 or len(classes) < 2:
            self._log("calibrator_skipped reason=insufficient_labeled_data")
            return False, {"reason": "insufficient_labeled_data", "train_rows": len(rows)}

        try:
            from joblib import dump
            from sklearn.linear_model import LogisticRegression
        except Exception:
            self._log("calibrator_skipped reason=dependencies_missing")
            return False, {"reason": "dependencies_missing", "train_rows": len(rows)}

        x = [r[0] for r in rows]
        y = [r[1] for r in rows]
        model = LogisticRegression(max_iter=200)
        model.fit(x, y)
        accuracy = None
        auc = None
        try:
            from sklearn.metrics import accuracy_score, roc_auc_score

            probs = model.predict_proba(x)[:, 1]
            preds = [1 if p >= 0.5 else 0 for p in probs]
            accuracy = float(accuracy_score(y, preds))
            auc = float(roc_auc_score(y, probs)) if len(set(y)) > 1 else None
        except Exception:
            accuracy = None
            auc = None
        self.calibrator_file.parent.mkdir(parents=True, exist_ok=True)
        dump(model, self.calibrator_file)
        self._log("calibrator_trained samples=%d accuracy=%s auc=%s" % (len(rows), accuracy, auc))
        return True, {"reason": None, "train_rows": len(rows), "accuracy": accuracy, "auc": auc}

    def _log(self, message: str) -> None:
        row = f"{_now()} {message}"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(row + "\n")

    def _ensure_model_metrics(self) -> None:
        if self.model_metrics_file.exists():
            return
        payload = {
            "updated_at": _now(),
            "last_training_data_fingerprint": None,
            "models": {
                modality: {
                    "trained_data_points": 0,
                    "accuracy": None,
                    "evaluated_samples": 0,
                    "last_trained_at": None,
                    "successful_training_runs": 0,
                }
                for modality in MODEL_SPECS
            },
        }
        self._save_model_metrics(payload)

    def _load_model_metrics(self) -> dict[str, Any]:
        self._ensure_model_metrics()
        try:
            payload = json.loads(self.model_metrics_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except (OSError, json.JSONDecodeError):
            pass
        return {"updated_at": _now(), "models": {}}

    def _save_model_metrics(self, payload: dict[str, Any]) -> None:
        self.model_metrics_file.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, ensure_ascii=True, indent=2)
        with self._lock:
            self.model_metrics_file.write_text(text, encoding="utf-8")

    def _read_training_manifest(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.manifest_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    def _training_data_fingerprint(self, manifest: dict[str, Any]) -> str | None:
        if not manifest:
            return None
        labeled_ids = sorted(
            str(row.get("sample_id", ""))
            for row in manifest.get("user_labeled_samples", [])
            if isinstance(row, dict) and row.get("sample_id")
        )
        pseudo_ids = sorted(
            str(row.get("sample_id", ""))
            for row in manifest.get("pseudo_labeled_samples", [])
            if isinstance(row, dict) and row.get("sample_id")
        )
        crawler_urls = sorted(
            str(row.get("url", ""))
            for row in manifest.get("crawler_dataset_references", [])
            if isinstance(row, dict) and row.get("url")
        )
        blob = json.dumps(
            {
                "labeled": labeled_ids,
                "pseudo": pseudo_ids,
                "crawler_urls": crawler_urls,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _model_file_times(self) -> dict[str, datetime]:
        model_paths = {
            "image": self.project_root / "models" / "exports" / "image_tf_model.keras",
            "video": self.project_root / "models" / "checkpoints" / "video_gru.pt",
            "audio": self.project_root / "models" / "exports" / "audio_rf.joblib",
            "multimodal": self.project_root / "models" / "exports" / "runtime_calibrator.joblib",
        }
        times: dict[str, datetime] = {}
        for modality, path in model_paths.items():
            if not path.exists():
                continue
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=IST).replace(microsecond=0)
                times[modality] = mtime
            except OSError:
                continue
        return times

    def _parse_iso(self, value: str) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=IST)
            return parsed
        except (TypeError, ValueError):
            return None
