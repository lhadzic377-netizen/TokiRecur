@echo off
rem
rem Toki Recur installer (Windows) - full setup from a git checkout.
rem
rem   git clone https://github.com/lhadzic377-netizen/TokiRecur.git
rem   cd TokiRecur
rem   install.bat
rem
rem Does everything:
rem   1. python 3.10+ check - INSTALLS Python itself when missing/too old
rem      (winget first, python.org installer as fallback, both silent)
rem   2. venv at %%LOCALAPPDATA%%\tokirecur\venv + package (embeddings extra)
rem      (ensurepip fallback if the venv was created without pip)
rem   3. idempotent user-PATH hook so `toki` works from any terminal
rem   4. toki model install-llama (official llama-server.exe for this machine)
rem   5. Microsoft Visual C++ runtime (llama-server.exe hard dependency) -
rem      best-effort: needs an elevated prompt to install
rem   6. verification: toki --version + llama-server --version
rem   7. starter model pull + agent config wiring (opencode/openclaw/hermes/codex)
rem
rem No tokens needed - the clone already authenticated you. Re-runnable.
rem
rem Switches (all optional):
rem   TOKI_EXTRAS=         install without extras (faster, no torch)
rem   TOKI_EXTRAS=dev      install dev/test extras instead
rem   TOKI_SKIP_LLAMA=1    skip the llama-server download
rem   TOKI_BASE=<dir>      install location instead of %%LOCALAPPDATA%%\tokirecur
rem   TOKI_SKIP_PYTHON=1   never install Python (error instead)

setlocal enabledelayedexpansion
cd /d "%~dp0"

rem ── 1. python check + install ─────────────────────────────
rem The classic "python not found" failure: instead of erroring out,
rem install Python 3.12 (winget first; python.org silent installer as
rem fallback). PATH is re-read from the registry afterwards so this
rem same session sees the new python.
:ensure_python
where python >nul 2>nul
if errorlevel 1 goto :no_python
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if errorlevel 1 (
  echo warning: installed python is older than 3.10 - installing 3.12 alongside
  goto :install_python
)
goto :python_ok

:install_python
if "%TOKI_SKIP_PYTHON%"=="1" (
  echo error: python not found and TOKI_SKIP_PYTHON=1. Install Python 3.10+ and tick "Add to PATH".
  exit /b 1
)
where winget >nul 2>nul
if not errorlevel 1 (
  echo [+] installing Python 3.12 via winget (silent)...
  winget install --id Python.Python.3.12 -e --silent --accept-package-agreements --accept-source-agreements >nul 2>nul
  if not errorlevel 1 (
    call :refresh_path
    where python >nul 2>nul
    if not errorlevel 1 goto :python_ok
  )
)
echo [+] winget unavailable or failed - downloading the python.org installer...
set "PYEXE=%TEMP%\python-3.12.10-amd64.exe"
curl -L --fail --silent --show-error -o "%PYEXE%" "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
if errorlevel 1 (
  echo error: could not download Python (curl failed). Install Python 3.10+ manually and tick "Add to PATH".
  exit /b 1
)
start /wait "" "%PYEXE%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_launcher=1 Include_pip=1
call :refresh_path
where python >nul 2>nul
if errorlevel 1 (
  echo error: python still not found after installing. Open a NEW terminal and re-run install.bat.
  exit /b 1
)
:python_ok
for /f "tokens=*" %%v in ('python -c "import sys;print('%%d.%%d'%%sys.version_info[:2])"') do set "PYVER=%%v"
echo [+] python: %PYVER%
goto :proceed

:no_python
echo [i] python not on PATH - installing it...
goto :install_python

:refresh_path
rem Re-read user + system PATH from the registry so a freshly
rem installed python is visible to THIS batch session (setx only
rem affects new processes).
for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v PATH 2^>nul') do set "USERPATH=%%b"
for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH 2^>nul') do set "SYS_PATH=%%b"
set "PATH=%USERPATH%;%SYS_PATH%;%PATH%"
exit /b 0

:proceed

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
  rem A venv from a Store/truncated python can lack pip; ensurepip fixes it.
  echo [+] venv missing pip - bootstrapping with ensurepip...
  "%VENV%\Scripts\python.exe" -m ensurepip --upgrade
  if errorlevel 1 (echo error: ensurepip failed & exit /b 1)
)
if not exist "%PIP%" (
  echo error: venv still missing pip at %PIP% & exit /b 1
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

rem ── 5. MSVC runtime (llama-server.exe hard dependency) ────────
rem llama-server.exe is built against the Microsoft Visual C++
rem runtime; a missing vcruntime140.dll makes it fail with a dialog
rem right after install. Best-effort: the redist wants an elevated
rem prompt, so a non-admin run warns instead of hard-failing.
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed >nul 2>nul
if not errorlevel 1 goto :vcredist_ok
echo [+] installing Microsoft Visual C++ 2015-2022 x64 runtime (silent)...
set "VCRED=%TEMP%\vc_redist.x64.exe"
curl -L --fail --silent --show-error -o "%VCRED%" "https://aka.ms/vs/17/release/vc_redist.x64.exe"
if errorlevel 1 (
  echo warning: could not download the VC++ runtime. llama-server needs it;
  echo         install https://aka.ms/vs/17/release/vc_redist.x64.exe manually
) else (
  start /wait "" "%VCRED%" /quiet /norestart
  reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed >nul 2>nul
  if errorlevel 1 (
    echo warning: VC++ runtime did not register - re-run install.bat from an
    echo         ADMIN terminal, or install vc_redist.x64.exe manually
  )
)
:vcredist_ok

rem ── 6. verification ─────────────────────────────────────────────
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

rem ── 7. starter model + agent config wiring (0.2.17: plug-and-play) ─
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
