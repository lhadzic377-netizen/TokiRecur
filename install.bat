@echo off
rem
rem Toki Recur installer (Windows) — run from a git checkout.
rem
rem   git clone https://github.com/lhadzic377-netizen/TokiRecur.git
rem   cd TokiRecur
rem   install.bat
rem
rem Installs the `toki` CLI with pipx (installs pipx first if missing).
rem No tokens needed — the clone already authenticated you.

setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo error: python not found on PATH. Install Python 3.10+ and tick "Add to PATH".
  exit /b 1
)

where pipx >nul 2>nul
if errorlevel 1 (
  echo pipx not found - installing it to the user site first...
  python -m pip install --user pipx
  if errorlevel 1 exit /b 1
)

echo Installing tokirecur (embeddings extra)...
pipx install ".[embeddings]"
if errorlevel 1 exit /b 1

echo.
echo Done. Next steps:
echo   toki model install-llama
echo   toki model pull Qwen/Qwen3-8B-GGUF --name qwen3-8b
echo   toki model use qwen3-8b ^&^& toki tui
