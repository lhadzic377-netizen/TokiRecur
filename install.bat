@echo off
rem
rem Toki Recur installer (Windows) - full setup from a git checkout.
rem
rem   git clone https://github.com/lhadzic377-netizen/TokiRecur.git
rem   cd TokiRecur
rem   install.bat
rem
rem Does everything:
rem   1. python 3.10+ check
rem   2. venv at %%LOCALAPPDATA%%\tokirecur\venv + package (embeddings extra)
rem   3. idempotent user-PATH hook so `toki` works from any terminal
rem   4. toki model install-llama (official llama-server.exe for this machine)
rem   5. verification: toki --version + llama-server --version
rem
rem No tokens needed - the clone already authenticated you. Re-runnable.
rem
rem Switches (all optional):
rem   TOKI_EXTRAS=         install without extras (faster, no torch)
rem   TOKI_EXTRAS=dev      install dev/test extras instead
rem   TOKI_SKIP_LLAMA=1    skip the llama-server download
rem   TOKI_BASE=<dir>      install location instead of %%LOCALAPPDATA%%\tokirecur

setlocal enabledelayedexpansion
cd /d "%~dp0"

rem ── 1. python check ─────────────────────────────────────────────
where python >nul 2>nul
if errorlevel 1 (
  echo error: python not found on PATH. Install Python 3.10+ and tick "Add to PATH".
  exit /b 1
)

rem ── 2. venv + package ───────────────────────────────────────────
if defined TOKI_BASE (set "BASE=%TOKI_BASE%") else (set "BASE=%LOCALAPPDATA%\tokirecur")
set "VENV=%BASE%\venv"
set "PIP=%VENV%\Scripts\pip.exe"
set "TOKI=%VENV%\Scripts\toki.exe"

if not exist "%VENV%\Scripts\python.exe" (
  echo [+] creating venv at %VENV%
  python -m venv "%VENV%"
  if errorlevel 1 (echo error: venv creation failed & exit /b 1)
)
if not exist "%PIP%" (
  echo error: venv missing pip at %PIP% & exit /b 1
)

set "EXTRA=.[embeddings]"
if defined TOKI_EXTRAS (
  if not "%TOKI_EXTRAS%"=="" set "EXTRA=.[%TOKI_EXTRAS%]"
  if "%TOKI_EXTRAS%"=="" set "EXTRA=."
)

echo [+] installing tokirecur (%EXTRA%) into the venv...
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip >nul
"%PIP%" install "%EXTRA%"
if errorlevel 1 (echo error: package install failed & exit /b 1)
if not exist "%TOKI%" (
  echo error: toki.exe missing after install at %TOKI% & exit /b 1
)

rem ── 3. user-PATH hook (idempotent, before llama so a fetch
rem    failure never leaves toki off PATH) ─────────────────────────
set "SCRIPTS=%VENV%\Scripts"
echo %PATH% | findstr /i /c:"%SCRIPTS%" >nul
if errorlevel 1 (
  for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v PATH 2^>nul') do set "USERPATH=%%b"
  echo [+] adding %SCRIPTS% to your user PATH
  setx PATH "%SCRIPTS%;!USERPATH!" >nul
)

rem ── 4. llama-server (non-fatal: warn and continue on failure) ───
if "%TOKI_SKIP_LLAMA%"=="1" goto :verify
echo [+] fetching llama-server for this machine (toki model install-llama)...
"%TOKI%" model install-llama
if errorlevel 1 (
  echo.
  echo warning: toki model install-llama failed. toki itself is installed and
  echo on PATH; retry the binary later from a new terminal with:
  echo     toki model install-llama
)

rem ── 5. verification ─────────────────────────────────────────────
:verify
echo [+] verifying...
"%TOKI%" --version
set "LLAMA=%USERPROFILE%\.local\state\tokirecur\bin\llama-server.exe"
if exist "%LLAMA%" (
  "%LLAMA%" --version
) else (
  if "%TOKI_SKIP_LLAMA%"=="1" (
    echo     llama-server skipped, TOKI_SKIP_LLAMA=1. Install later: toki model install-llama
  ) else (
    echo warning: llama-server not found at %LLAMA%
  )
)

rem ── 6. starter model + agent config wiring (0.2.17: plug-and-play) ─
rem    Windows never wired the four agent configs at install time, so
rem    direct agent launches hit stale/absent configs (hermes "model
rem    '<name>' not found"). Pull a starter model when the registry is
rem    empty, then sync opencode/openclaw/hermes/codex (--sync-only
rem    writes configs without booting a model).
set "HAS_MODELS=0"
for /f "delims=" %%a in ('"%TOKI%" model list') do set "HAS_MODELS=1"
if "%HAS_MODELS%"=="0" (
  echo [+] no models registered - pulling a starter model...
  set "DEFAULT_REPO=Qwen/Qwen3-8B-GGUF"
  set "DEFAULT_FILE=Qwen3-8B-Q4_K_M.gguf"
  set "DEFAULT_NAME=qwen3-8b"
  for /f "delims=" %%a in ('powershell -NoProfile -Command "try{[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB)}catch{8}"') do set "RAM_GB=%%a"
  if defined RAM_GB if !RAM_GB! LSS 16 (
    set "DEFAULT_REPO=Qwen/Qwen3-0.6B-GGUF"
    set "DEFAULT_FILE=Qwen3-0.6B-Q4_K_M.gguf"
    set "DEFAULT_NAME=qwen3-0.6b"
  )
  "%TOKI%" model pull "!DEFAULT_REPO!" --name "!DEFAULT_NAME!" --file "!DEFAULT_FILE!"
  if not errorlevel 1 (
    echo [+] wiring opencode/openclaw/hermes/codex to "!DEFAULT_NAME!"...
    "%TOKI%" model use "!DEFAULT_NAME!" --sync-only
  ) else (
    echo warning: starter model pull failed. Pull one later:
    echo     toki model pull Qwen/Qwen3-8B-GGUF --name qwen3-8b
  )
) else (
  for /f "tokens=3" %%a in ('"%TOKI%" model status ^| findstr /b /c:"active model:"') do set "ACTIVE_MODEL=%%a"
  if defined ACTIVE_MODEL (
    if not "!ACTIVE_MODEL!"=="none" (
      echo [+] re-syncing agent configs for "!ACTIVE_MODEL!"...
      "%TOKI%" model use "!ACTIVE_MODEL!" --sync-only
    )
  )
)

echo.
echo Done.
echo   toki CLI:        %TOKI%
echo   llama-server:    %LLAMA%
echo.
echo Agents are wired automatically; start the stack with: toki model use ^<name^> ^&^& toki tui
