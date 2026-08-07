@echo off
rem
rem Toki Recur installer (Windows) - run from a git checkout.
rem
rem   git clone https://github.com/lhadzic377-netizen/TokiRecur.git
rem   cd TokiRecur
rem   install.bat
rem
rem Installs the `toki` CLI with pipx (auto-installs pipx first if
rem missing). No tokens needed - the clone already authenticated you.
rem
rem Extras: TOKI_EXTRAS unset  -> embeddings (sentence-transformers+torch)
rem         TOKI_EXTRAS set     -> those extras (e.g. "dev")
rem         TOKI_EXTRAS empty   -> no extras (fast install)
rem         TOKI_VENV_DIR set   -> install into that venv instead of pipx

setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo error: python not found on PATH. Install Python 3.10+ and tick "Add to PATH".
  exit /b 1
)

if defined TOKI_VENV_DIR (
  echo Installing tokirecur into %TOKI_VENV_DIR% ...
  python -m venv "%TOKI_VENV_DIR%"
  if errorlevel 1 exit /b 1
  "%TOKI_VENV_DIR%\Scripts\pip" install --upgrade pip >nul
  if defined TOKI_EXTRAS (
    "%TOKI_VENV_DIR%\Scripts\pip" install ".[%TOKI_EXTRAS%]"
  ) else (
    "%TOKI_VENV_DIR%\Scripts\pip" install ".[embeddings]"
  )
  if errorlevel 1 exit /b 1
  echo.
  echo Done. toki is at "%TOKI_VENV_DIR%\Scripts\toki.exe"
  goto :next
)

where pipx >nul 2>nul
if errorlevel 1 (
  echo pipx not found - installing it to the user site first...
  python -m pip install --user pipx
  if errorlevel 1 exit /b 1
)

if defined TOKI_EXTRAS (
  if "%TOKI_EXTRAS%"=="" (
    set "EXTRA=."
  ) else (
    set "EXTRA=.[%TOKI_EXTRAS%]"
  )
) else (
  set "EXTRA=.[embeddings]"
)

echo Installing tokirecur (%EXTRA%)...
python -m pipx install "%EXTRA%"
if errorlevel 1 exit /b 1

echo.
echo Done. Next steps:
echo   toki model install-llama
echo   toki model pull Qwen/Qwen3-8B-GGUF --name qwen3-8b
echo   toki model use qwen3-8b ^&^& toki tui

:next
