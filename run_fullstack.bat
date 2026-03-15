@echo off
set BITMIND_KEY_ARG=%~1
set OPENAI_KEY_ARG=%~2
set GEMINI_KEY_ARG=%~3

powershell -ExecutionPolicy Bypass -File scripts\start_fullstack.ps1 -BitmindApiKey "%BITMIND_KEY_ARG%" -OpenAiApiKey "%OPENAI_KEY_ARG%" -GeminiApiKey "%GEMINI_KEY_ARG%"
