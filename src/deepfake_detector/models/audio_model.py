from __future__ import annotations

from pathlib import Path

from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_audio_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300, max_depth=16, n_jobs=1, random_state=random_state, class_weight="balanced"
                ),
            ),
        ]
    )


def save_audio_model(model: Pipeline, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, out_path)


def load_audio_model(path: Path) -> Pipeline:
    return load(path)
