param(
  [string]$RawRoot = 'data/raw',
  [string]$ProcessedRoot = 'data/processed'
)

$env:PYTHONPATH = "src"
python -m deepfake_detector.data.dataset_manifest --raw-root $RawRoot --out "$ProcessedRoot/manifest.json"
python -m deepfake_detector.data.preprocess_image_video --manifest "$ProcessedRoot/manifest.json" --out-root $ProcessedRoot
python -m deepfake_detector.data.preprocess_audio --manifest "$ProcessedRoot/manifest.json" --out-root $ProcessedRoot
