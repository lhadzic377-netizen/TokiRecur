#!/usr/bin/env pwsh
# Portable toki launcher for a default install.
$root = Join-Path $env:LOCALAPPDATA "TokiRecur"
$python = Join-Path $root "venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "TokiRecur is not installed. Run install.ps1 from the unzipped folder first."
    exit 1
}
& $python -m tokirecur.cli @args
exit $LASTEXITCODE
