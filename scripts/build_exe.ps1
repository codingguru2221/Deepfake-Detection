param(
  [string]$VenvPath = ".venv"
)

if (!(Test-Path "$VenvPath/Scripts/Activate.ps1")) {
  throw "Virtual environment not found at $VenvPath"
}

. "$VenvPath/Scripts/Activate.ps1"

python -m pip install --upgrade pip
python -m pip install pyinstaller

pyinstaller `
  --noconfirm `
  --clean `
  --onefile `
  --name DeepfakePipelineRunner `
  scripts/pipeline_launcher.py

Write-Host ""
Write-Host "Build complete."
Write-Host "EXE path: dist/DeepfakePipelineRunner.exe"
