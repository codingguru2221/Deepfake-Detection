from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def detect_python(project_root: Path, python_path: str | None) -> str:
    if python_path:
        return python_path
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return "python"


def main() -> None:
    parser = argparse.ArgumentParser(description="Deepfake full pipeline launcher")
    parser.add_argument("--project-root", default=".", help="Project root path")
    parser.add_argument("--python-path", default=None, help="Python executable path")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--run-image", action="store_true")
    parser.add_argument("--run-video", action="store_true")
    parser.add_argument("--run-audio", action="store_true")
    parser.add_argument("--infer-image", default=None)
    parser.add_argument("--infer-video", default=None)
    parser.add_argument("--infer-audio", default=None)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    py = detect_python(root, args.python_path)

    run_image = args.run_image or (not args.run_video and not args.run_audio)
    run_video = args.run_video
    run_audio = args.run_audio or (not args.run_image and not args.run_video)

    env = dict(**os.environ)
    src_path = str(root / "src")
    env["PYTHONPATH"] = src_path

    def run_module(module_args: list[str]) -> None:
        cmd = [py, "-m", *module_args]
        print(f"[RUN] {' '.join(cmd)}")
        res = subprocess.run(cmd, cwd=str(root), env=env, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"Failed ({res.returncode}): {' '.join(cmd)}")

    manifest = f"{args.processed_root}/manifest.json"
    iv_samples = f"{args.processed_root}/image_video_samples.json"
    au_samples = f"{args.processed_root}/audio_samples.json"
    image_model = "models/exports/image_tf_model.keras"
    video_model = "models/checkpoints/video_gru.pt"
    audio_model = "models/exports/audio_rf.joblib"

    print("[INFO] Step 1/5: Manifest")
    run_module(["deepfake_detector.data.dataset_manifest", "--raw-root", args.raw_root, "--out", manifest])

    print("[INFO] Step 2/5: Preprocess")
    run_module(
        [
            "deepfake_detector.data.preprocess_image_video",
            "--manifest",
            manifest,
            "--out-root",
            args.processed_root,
        ]
    )
    run_module(
        ["deepfake_detector.data.preprocess_audio", "--manifest", manifest, "--out-root", args.processed_root]
    )

    if not args.skip_train:
        print("[INFO] Step 3/5: Train")
        if run_image:
            run_module(
                [
                    "deepfake_detector.train",
                    "--modality",
                    "image",
                    "--samples-json",
                    iv_samples,
                    "--out",
                    image_model,
                    "--epochs",
                    str(args.epochs),
                ]
            )
        if run_video:
            run_module(
                [
                    "deepfake_detector.train",
                    "--modality",
                    "video",
                    "--samples-json",
                    iv_samples,
                    "--out",
                    video_model,
                    "--epochs",
                    str(args.epochs),
                ]
            )
        if run_audio:
            run_module(
                [
                    "deepfake_detector.train",
                    "--modality",
                    "audio",
                    "--samples-json",
                    au_samples,
                    "--out",
                    audio_model,
                ]
            )

    if not args.skip_eval:
        print("[INFO] Step 4/5: Evaluate")
        if run_image:
            run_module(
                [
                    "deepfake_detector.evaluate",
                    "--modality",
                    "image",
                    "--samples-json",
                    iv_samples,
                    "--model",
                    image_model,
                ]
            )
        if run_video:
            run_module(
                [
                    "deepfake_detector.evaluate",
                    "--modality",
                    "video",
                    "--samples-json",
                    iv_samples,
                    "--model",
                    video_model,
                ]
            )
        if run_audio:
            run_module(
                [
                    "deepfake_detector.evaluate",
                    "--modality",
                    "audio",
                    "--samples-json",
                    au_samples,
                    "--model",
                    audio_model,
                ]
            )

    if args.infer_image or args.infer_video or args.infer_audio:
        print("[INFO] Step 5/5: Inference")
        infer_cmd = ["deepfake_detector.infer"]
        if args.infer_image:
            infer_cmd += ["--image", args.infer_image, "--image-model", image_model]
        if args.infer_video:
            infer_cmd += ["--video", args.infer_video, "--video-model", video_model]
        if args.infer_audio:
            infer_cmd += ["--audio", args.infer_audio, "--audio-model", audio_model]
        run_module(infer_cmd)

    print("[DONE] Pipeline completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
