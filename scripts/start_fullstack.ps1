param(
  [string]$VenvPath = ".venv",
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5000
)

$ErrorActionPreference = "Stop"

if (!(Test-Path "$VenvPath/Scripts/Activate.ps1")) {
  throw "Virtual environment not found at $VenvPath"
}

. "$VenvPath/Scripts/Activate.ps1"
$env:PYTHONPATH = "src"

Write-Host "[1/4] Checking Python API dependencies..."
python -m pip install fastapi uvicorn python-multipart | Out-Null

$frontendDir = Join-Path $PSScriptRoot "..\Deepfake-Frontend"
if (!(Test-Path $frontendDir)) {
  throw "Frontend directory not found: $frontendDir"
}

Write-Host "[2/4] Installing frontend dependencies (if needed)..."
if (!(Test-Path (Join-Path $frontendDir "node_modules"))) {
  Push-Location $frontendDir
  npm install
  Pop-Location
}

Write-Host "[3/4] Starting backend at http://localhost:$BackendPort ..."
$backendCmd = "cd `"$((Resolve-Path (Join-Path $PSScriptRoot '..')).Path)`"; `$env:PYTHONPATH='src'; python -m uvicorn deepfake_detector.api.app:app --host 0.0.0.0 --port $BackendPort"
$backend = Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd -PassThru

Start-Sleep -Seconds 3

Write-Host "[4/4] Starting frontend at http://localhost:$FrontendPort ..."
$frontendCmd = "cd `"$((Resolve-Path $frontendDir).Path)`"; `$env:VITE_API_BASE_URL='http://localhost:$BackendPort'; npm run dev:client"
$frontend = Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd -PassThru

Start-Sleep -Seconds 3

$backendUrl = "http://localhost:$BackendPort/health"
$frontendUrl = "http://localhost:$FrontendPort"

try {
  Invoke-WebRequest -UseBasicParsing -Uri $backendUrl -TimeoutSec 5 | Out-Null
  Write-Host "Backend health check passed: $backendUrl"
} catch {
  Write-Host "Backend health check failed. Check backend window logs."
}

try {
  Invoke-WebRequest -UseBasicParsing -Uri $frontendUrl -TimeoutSec 5 | Out-Null
  Write-Host "Frontend check passed: $frontendUrl"
  Start-Process $frontendUrl | Out-Null
} catch {
  Write-Host "Frontend check failed. Possible port conflict or npm error. Check frontend window logs."
}

Write-Host ""
Write-Host "Full stack started."
Write-Host "Frontend: $frontendUrl"
Write-Host "Backend:  $backendUrl"
Write-Host ""
Write-Host "Press Enter to stop both..."
[void][System.Console]::ReadLine()

foreach ($p in @($frontend, $backend)) {
  if ($null -ne $p -and -not $p.HasExited) {
    Stop-Process -Id $p.Id -Force
  }
}

Write-Host "Stopped frontend and backend."
