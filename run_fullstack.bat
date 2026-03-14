@echo off
set BITMIND_KEY_ARG=%~1
if not "%BITMIND_KEY_ARG%"=="" (
  powershell -ExecutionPolicy Bypass -File scripts\start_fullstack.ps1 -BitmindApiKey "%BITMIND_KEY_ARG%"
) else (
  powershell -ExecutionPolicy Bypass -File scripts\start_fullstack.ps1
)
