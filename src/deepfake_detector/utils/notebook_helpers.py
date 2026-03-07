from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Iterable


def run_cmd(command: str, cwd: str | None = None) -> int:
    print(f"$ {command}")
    proc = subprocess.run(shlex.split(command), cwd=cwd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {command}")
    return proc.returncode


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def show_manifest_counts(path: str | Path) -> None:
    data = load_json(path)
    counts = {k: len(v) if isinstance(v, list) else "n/a" for k, v in data.items()}
    print("Manifest counts:", counts)


def show_sample_counts(path: str | Path) -> None:
    data = load_json(path)
    rows = data.get("samples", [])
    total = len(rows)
    fake = sum(int(r.get("label", 0)) for r in rows)
    real = total - fake
    print({"total": total, "real": real, "fake": fake})


def ensure_paths(paths: Iterable[str | Path]) -> None:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required paths: {missing}")
