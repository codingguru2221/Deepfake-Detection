param(
  [string]$VenvPath = ".venv",
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5000,
  [string]$BitmindApiKey = "",
  [string]$OpenAiApiKey = "",
  [string]$GeminiApiKey = ""
)

$ErrorActionPreference = "Stop"

function Import-DotEnvFile {
  param(
    [string]$Path
  )

  if (!(Test-Path $Path)) {
    return
  }

  Get-Content $Path | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq "" -or $line.StartsWith("#")) {
      return
    }

    $parts = $line -split "=", 2
    if ($parts.Count -ne 2) {
      return
    }

    $name = $parts[0].Trim()
    $value = $parts[1].Trim()
    if (
      ($value.StartsWith('"') -and $value.EndsWith('"')) -or
      ($value.StartsWith("'") -and $value.EndsWith("'"))
    ) {
      $value = $value.Substring(1, $value.Length - 2)
    }

    if ($name) {
      Set-Item -Path "Env:$name" -Value $value
    }
  }
}

if (!(Test-Path "$VenvPath/Scripts/Activate.ps1")) {
  throw "Virtual environment not found at $VenvPath"
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$frontendDir = Join-Path $PSScriptRoot "..\Deepfake-Frontend"

Import-DotEnvFile (Join-Path $projectRoot ".env")
Import-DotEnvFile (Join-Path $projectRoot ".env.local")
Import-DotEnvFile (Join-Path $frontendDir ".env")
Import-DotEnvFile (Join-Path $frontendDir ".env.local")

. "$VenvPath/Scripts/Activate.ps1"
$env:PYTHONPATH = "src"

if ($BitmindApiKey -ne "") {
  $env:BITMIND_API_KEY = $BitmindApiKey
}
if ($env:BITMIND_API_KEY) {
  $env:BITMIND_ENABLED = "1"
  $env:BITMIND_VERIFY_ON_INFER = "1"
}

if ($OpenAiApiKey -ne "") {
  $env:OPENAI_API_KEY = $OpenAiApiKey
}
if ($env:OPENAI_API_KEY) {
  $env:OPENAI_VISION_ENABLED = "1"
}

if ($GeminiApiKey -ne "") {
  $env:GEMINI_API_KEY = $GeminiApiKey
}
if ($env:GEMINI_API_KEY) {
  $env:GEMINI_VISION_ENABLED = "1"
}

$env:DF_RUNTIME_LEARNING_ENABLED = "0"
$env:DF_CRAWLER_ENABLED = "0"
$env:HF_DEEPFAKE_ENABLED = "0"
$env:AWS_REKOGNITION_ENABLED = "0"
$env:BITMIND_ENABLED = if ($env:BITMIND_API_KEY) { "1" } else { "0" }
$env:OPENAI_VISION_ENABLED = if ($env:OPENAI_API_KEY) { "1" } else { "0" }
$env:GEMINI_VISION_ENABLED = if ($env:GEMINI_API_KEY) { "1" } else { "0" }

Write-Host "[1/4] Checking Python API dependencies..."
python -m pip install fastapi uvicorn python-multipart | Out-Null

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
$backendCmd = @"
cd "$projectRoot"
`$env:PYTHONPATH='src'
`$env:DF_RUNTIME_LEARNING_ENABLED='$($env:DF_RUNTIME_LEARNING_ENABLED)'
`$env:DF_CRAWLER_ENABLED='$($env:DF_CRAWLER_ENABLED)'
`$env:HF_DEEPFAKE_ENABLED='$($env:HF_DEEPFAKE_ENABLED)'
`$env:AWS_REKOGNITION_ENABLED='$($env:AWS_REKOGNITION_ENABLED)'
`$env:BITMIND_ENABLED='$($env:BITMIND_ENABLED)'
`$env:OPENAI_VISION_ENABLED='$($env:OPENAI_VISION_ENABLED)'
`$env:GEMINI_VISION_ENABLED='$($env:GEMINI_VISION_ENABLED)'
`$env:BITMIND_API_KEY='$($env:BITMIND_API_KEY)'
`$env:OPENAI_API_KEY='$($env:OPENAI_API_KEY)'
`$env:GEMINI_API_KEY='$($env:GEMINI_API_KEY)'
python -m uvicorn deepfake_detector.api.app:app --host 0.0.0.0 --port $BackendPort
"@
$backend = Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd -PassThru

Start-Sleep -Seconds 3

Write-Host "[4/4] Starting frontend at http://localhost:$FrontendPort ..."
$frontendCmd = @"
cd "$((Resolve-Path $frontendDir).Path)"
`$env:VITE_API_BASE_URL='http://localhost:$BackendPort'
`$env:VITE_DEV_API_PROXY_TARGET='http://127.0.0.1:$BackendPort'
npm run dev:client
"@
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
Write-Host "BitMind:  $($env:BITMIND_ENABLED)"
Write-Host "OpenAI:   $($env:OPENAI_VISION_ENABLED)"
Write-Host "Gemini:   $($env:GEMINI_VISION_ENABLED)"
Write-Host ""
Write-Host "Press Enter to stop both..."
[void][System.Console]::ReadLine()

foreach ($p in @($frontend, $backend)) {
  if ($null -ne $p -and -not $p.HasExited) {
    Stop-Process -Id $p.Id -Force
  }
}

Write-Host "Stopped frontend and backend."
