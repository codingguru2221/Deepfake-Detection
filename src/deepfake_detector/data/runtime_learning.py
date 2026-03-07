from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
    manifest_path: str
    calibrator_path: str | None
    user_labeled_count: int
    pseudo_count: int
    crawler_refs_count: int


class RuntimeLearningManager:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.runtime_dir = self.project_root / "data" / "runtime"
        self.upload_dir = self.runtime_dir / "uploads"
        self.samples_file = self.runtime_dir / "user_samples.jsonl"
        self.feedback_file = self.runtime_dir / "feedback.jsonl"
        self.jobs_file = self.runtime_dir / "training_jobs.jsonl"
        self.manifest_file = self.runtime_dir / "runtime_training_manifest.json"
        self.calibrator_file = self.project_root / "models" / "exports" / "runtime_calibrator.joblib"
        self.log_file = self.runtime_dir / "runtime_learning.log"
        self._lock = threading.Lock()

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
            "created_at": _utc_now(),
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
            "created_at": _utc_now(),
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
            "generated_at": _utc_now(),
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

        calibrator = self._train_calibrator(samples, feedback_rows)
        result = RuntimeTrainResult(
            status="completed",
            manifest_path=str(self.manifest_file),
            calibrator_path=str(self.calibrator_file) if calibrator else None,
            user_labeled_count=len(labeled),
            pseudo_count=len(pseudo),
            crawler_refs_count=len(crawler_refs),
        )
        _append_jsonl(
            self.jobs_file,
            {
                "created_at": _utc_now(),
                "status": result.status,
                "manifest_path": result.manifest_path,
                "calibrator_path": result.calibrator_path,
                "counts": {
                    "user_labeled": result.user_labeled_count,
                    "pseudo_labeled": result.pseudo_count,
                    "crawler_refs": result.crawler_refs_count,
                },
            },
            self._lock,
        )
        self._log(
            "training_done "
            f"user_labeled={result.user_labeled_count} pseudo={result.pseudo_count} crawler_refs={result.crawler_refs_count}"
        )
        return result

    def get_recent_logs(self, limit: int = 100) -> list[str]:
        if not self.log_file.exists():
            return []
        with self.log_file.open("r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        return lines[-max(1, limit) :]

    def _train_calibrator(self, samples: list[dict[str, Any]], feedback_rows: list[dict[str, Any]]) -> bool:
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
            return False

        try:
            from joblib import dump
            from sklearn.linear_model import LogisticRegression
        except Exception:
            self._log("calibrator_skipped reason=dependencies_missing")
            return False

        x = [r[0] for r in rows]
        y = [r[1] for r in rows]
        model = LogisticRegression(max_iter=200)
        model.fit(x, y)
        self.calibrator_file.parent.mkdir(parents=True, exist_ok=True)
        dump(model, self.calibrator_file)
        self._log("calibrator_trained samples=%d" % len(rows))
        return True

    def _log(self, message: str) -> None:
        row = f"{_utc_now()} {message}"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(row + "\n")

