@echo off
rem Portable toki launcher for a default install.
rem The installer writes the same file into %LOCALAPPDATA%\TokiRecur\bin.
setlocal
set "TOKI_ROOT=%LOCALAPPDATA%\TokiRecur"
set "TOKI_PY=%TOKI_ROOT%\venv\Scripts\python.exe"
if not exist "%TOKI_PY%" (
  echo TokiRecur is not installed yet.
  echo Run this from the unzipped folder:
  echo   powershell -ExecutionPolicy Bypass -File "%~dp0install.ps1"
  exit /b 1
)
"%TOKI_PY%" -m tokirecur.cli %*
exit /b %ERRORLEVEL%
