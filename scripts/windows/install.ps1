<#
.SYNOPSIS
    Install TokiRecur for the current Windows user from this folder.

.DESCRIPTION
    Creates a private virtual environment under
    %LOCALAPPDATA%\TokiRecur, installs the bundled wheel into it, and
    writes a `toki` command into %LOCALAPPDATA%\TokiRecur\bin.

    No administrator rights are needed. Nothing is written outside the
    install folder except the user PATH entry (skip it with -NoPath).

.PARAMETER Python
    Python launcher to use, e.g. "py -3.12" or "C:\Python312\python.exe".
    Default: the first of py -3.13 ... py -3.10, py -3, python that is
    3.10 or newer.

.PARAMETER InstallDir
    Where to install. Default: %LOCALAPPDATA%\TokiRecur

.PARAMETER Wheelhouse
    Folder with the dependency wheels for an offline install. When a
    `wheelhouse` folder sits next to this script it is used
    automatically.

.PARAMETER Online
    Ignore a bundled `wheelhouse` folder and fetch dependencies from
    PyPI instead. Needed when the wheelhouse was built for a different
    Python version and you cannot install that version.

.PARAMETER NoPath
    Do not add the bin folder to the user PATH.

.PARAMETER NoDoctor
    Do not run `toki doctor` at the end.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\install.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\install.ps1 -Python "py -3.12" -NoDoctor
#>
[CmdletBinding()]
param(
    [string]$Python = "",
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "TokiRecur"),
    [string]$Wheelhouse = "",
    [switch]$Online,
    [switch]$NoPath,
    [switch]$NoDoctor
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Find-Python {
    param([string]$Requested)

    $candidates = @()
    if ($Requested) { $candidates += $Requested }
    $candidates += @(
        "py -3.13", "py -3.12", "py -3.11", "py -3.10", "py -3", "python"
    )

    foreach ($candidate in $candidates) {
        $parts = $candidate.Trim() -split "\s+"
        $exe = $parts[0]
        $pre = @()
        if ($parts.Length -gt 1) { $pre = $parts[1..($parts.Length - 1)] }

        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }

        try {
            $version = & $exe @pre -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -ne 0 -or -not $version) { continue }
            if ([version]$version -ge [version]"3.10") {
                return @{ Exe = $exe; Pre = $pre; Version = $version.Trim() }
            }
        } catch {
            continue
        }
    }
    throw "Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/windows/ (tick 'Add python.exe to PATH') and run this script again."
}

Write-Host "TokiRecur - Windows installer" -ForegroundColor Green
Write-Host "install folder: $InstallDir"

$py = Find-Python $Python
Write-Host "python: $($py.Exe) $($py.Pre -join ' ') (version $($py.Version))"

$venv = Join-Path $InstallDir "venv"
$bin = Join-Path $InstallDir "bin"

function Get-VenvPython([string]$Root) {
    $candidate = Join-Path $Root "Scripts\python.exe"
    if (Test-Path $candidate) { return $candidate }
    # POSIX layout, only so this installer can be smoke-tested on CI;
    # on Windows the first path always wins.
    $posix = Join-Path $Root "bin/python"
    if (Test-Path $posix) { return $posix }
    return $candidate
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
New-Item -ItemType Directory -Force -Path $bin | Out-Null

$venvPython = Get-VenvPython $venv
if (-not (Test-Path $venvPython)) {
    Write-Step "Creating virtual environment"
    & $py.Exe @($py.Pre) -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "python -m venv failed." }
    $venvPython = Get-VenvPython $venv
    if (-not (Test-Path $venvPython)) { throw "The virtual environment was created but has no python." }
}

$wheel = Get-ChildItem -Path (Join-Path $PSScriptRoot "wheel") -Filter "tokirecur-*.whl" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $wheel) { throw "No tokirecur wheel found in $(Join-Path $PSScriptRoot 'wheel')." }

if (-not $Wheelhouse -and -not $Online) {
    $localWheelhouse = Join-Path $PSScriptRoot "wheelhouse"
    if (Test-Path $localWheelhouse) { $Wheelhouse = $localWheelhouse }
}

Write-Step "Installing TokiRecur"
if (-not $Wheelhouse) {
    & $venvPython -m pip install --upgrade pip --quiet
    if ($LASTEXITCODE -ne 0) { throw "pip could not be upgraded (is there network access?)." }
}

if ($Wheelhouse) {
    # A wheelhouse built for one CPython version cannot install into
    # another: numpy/pydantic-core/pyyaml ship cpXY wheels. Fail with
    # instructions instead of a wall of pip resolver errors.
    $tag = "cp" + ($py.Version -replace "\.", "")
    $bad = Get-ChildItem $Wheelhouse -Filter *.whl | Where-Object {
        $_.Name -match "-(cp\d+)-(cp\d+|abi3|none)-win_amd64\.whl$" -and
        $_.Name -notmatch "-cp\d+-abi3-" -and
        $_.Name -notmatch "-$tag-"
    }
    if ($bad) {
        throw ("This wheelhouse does not match Python $($py.Version). " +
            "Install a matching Python (the package was built for a different version), " +
            "or run install.ps1 with -Online to fetch dependencies from PyPI. " +
            "First mismatch: $($bad[0].Name)")
    }
}

$pipArgs = @("-m", "pip", "install", "--force-reinstall", "--no-warn-script-location", $wheel.FullName)
if ($Wheelhouse) {
    Write-Host "offline wheelhouse: $Wheelhouse"
    $pipArgs += @("--no-index", "--find-links", $Wheelhouse)
}
& $venvPython @pipArgs
if ($LASTEXITCODE -ne 0) { throw "pip install failed. See the messages above." }

Write-Step "Writing the toki command"
$cmdPath = Join-Path $bin "toki.cmd"
$cmdBody = @"
@echo off
setlocal
"$venvPython" -m tokirecur.cli %*
"@
Set-Content -Path $cmdPath -Value $cmdBody -Encoding ASCII

$ps1Path = Join-Path $bin "toki.ps1"
$ps1Body = @"
#!/usr/bin/env pwsh
& "$venvPython" -m tokirecur.cli @args
exit `$LASTEXITCODE
"@
Set-Content -Path $ps1Path -Value $ps1Body -Encoding UTF8

if (-not $NoPath) {
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $entries = @()
    if ($userPath) { $entries = $userPath -split ";" }
    if (-not ($entries | Where-Object { $_ -and ($_.TrimEnd('\') -ieq $bin.TrimEnd('\')) })) {
        $newPath = if ($userPath) { "$userPath;$bin" } else { $bin }
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        Write-Host "added $bin to the user PATH - open a new terminal to use 'toki'"
    } else {
        Write-Host "$bin is already on the user PATH"
    }
}

Write-Step "Verifying"
& $venvPython -m tokirecur.cli --version
if ($LASTEXITCODE -ne 0) { throw "The install finished but 'toki --version' failed." }

if (-not $NoDoctor) {
    Write-Step "Doctor"
    & $venvPython -m tokirecur.cli doctor
}

Write-Host ""
Write-Host "TokiRecur is installed." -ForegroundColor Green
Write-Host ""
Write-Host "Open a NEW terminal, then:"
Write-Host "  toki                          # full-screen launcher"
Write-Host "  toki model install-llama      # fetch the llama-server build"
Write-Host "  toki model pull <repo> --name <name>"
Write-Host "  toki model use <name>"
Write-Host "  toki tui                      # chat"
Write-Host ""
Write-Host "No full-screen terminal? Set TOKI_CLI=1 for the plain menu,"
Write-Host "or install Windows Terminal (winget install Microsoft.WindowsTerminal)."
