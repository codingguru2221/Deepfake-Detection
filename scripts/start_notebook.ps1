param(
  [string]$VenvPath = '.venv',
  [string]$NotebookPath = 'notebooks/deepfake_pipeline.ipynb'
)

if (!(Test-Path "$VenvPath/Scripts/Activate.ps1")) {
  throw "Virtual environment not found at $VenvPath. Create it first: python -m venv .venv"
}

. "$VenvPath/Scripts/Activate.ps1"
$env:PYTHONPATH = "src"

python -m ipykernel install --user --name deepfake-env --display-name "Python (deepfake-env)"
Write-Host "Kernel registered: Python (deepfake-env)"

jupyter notebook $NotebookPath
