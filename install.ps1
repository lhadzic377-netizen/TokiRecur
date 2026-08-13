# Toki Recur bootstrap installer (Windows).
#
# The primary install path is pip (see README "Quick start": pipx install
# from the private git URL, then `toki model install-llama`). This script
# is the optional standalone bootstrap: source under %LOCALAPPDATA%,
# venv + package, llama-server binary, user-PATH hook.
#
# Full install from zero to a working CLI + llama-server:
#   1. require Python 3.10+
#   2. materialise the app source under %LOCALAPPDATA%\tokirecur\src
#      (fetched as a tarball from the private GitHub repo using
#      TOKI_INSTALL_TOKEN, a Contents:read PAT; no git required)
#   3. venv at %LOCALAPPDATA%\tokirecur\venv + pip install .[embeddings]
#   4. toki model install-llama (official llama-server.exe for this arch)
#   5. add the venv Scripts dir to the user PATH (idempotent)
#
# Windows has no shell service manager: run `toki serve` yourself, or wrap
# it in NSSM / a scheduled task:
#   nssm install TokiRecur "<venv>\Scripts\toki.exe" serve
#
# One-liner (private repo - token required):
#   $env:TOKI_INSTALL_TOKEN='<PAT>'; $j=irm -Headers @{Authorization='Bearer '+$env:TOKI_INSTALL_TOKEN} 'https://api.github.com/repos/lhadzic377-netizen/TokiRecur/contents/install.ps1?ref=master'; iex ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($j.content)))
#
# Re-runnable: re-runs re-fetch the source, reinstall the package, and
# re-download the binary.

$ErrorActionPreference = 'Stop'

# ---- Python 3.10+ requirement ----

$Py = $null
foreach ($cand in @('python', 'py')) {
    $cmd = Get-Command $cand -ErrorAction SilentlyContinue
    if ($cmd) { $Py = $cmd.Source; break }
}
if (-not $Py) {
    Write-Host "error: Python 3.10+ not found." -ForegroundColor Red
    Write-Host "       Install it from https://www.python.org/downloads and re-run." -ForegroundColor Red
    exit 1
}
$PyVer = (& $Py -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>$null)
if ($LASTEXITCODE -ne 0 -or $PyVer -notmatch '^3\.(1[0-9]|[2-9][0-9])$') {
    Write-Host "error: Python $PyVer found; Toki Recur requires 3.10+." -ForegroundColor Red
    exit 1
}

if (-not $env:TOKI_INSTALL_TOKEN) {
    Write-Host "error: TOKI_INSTALL_TOKEN is required (fine-grained PAT with" -ForegroundColor Red
    Write-Host "       Contents: read on lhadzic377-netizen/TokiRecur only)." -ForegroundColor Red
    exit 1
}

# ---- Materialise the source tree ----

$Base = Join-Path $env:LOCALAPPDATA 'tokirecur'
$Src = Join-Path $Base 'src'
New-Item -ItemType Directory -Force -Path $Base | Out-Null

Write-Host "[+] fetching TokiRecur source from GitHub"
$Zip = Join-Path $Base 'src.zip'
# zipball API endpoint: redirects to a signed codeload URL, so no basic
# auth on codeload is needed (WebClient challenge-response never fires
# because GitHub answers 404 instead of 401 without auth).
irm -Headers @{Authorization = 'Bearer ' + $env:TOKI_INSTALL_TOKEN} `
    -OutFile $Zip `
    'https://api.github.com/repos/lhadzic377-netizen/TokiRecur/zipball/refs/heads/master'
if (Test-Path $Src) { Remove-Item -Recurse -Force $Src }
Expand-Archive -Path $Zip -DestinationPath $Src
Remove-Item -Force $Zip
# Expand-Archive nests a single TokiRecur-master/ folder; flatten it.
$Nested = Get-ChildItem -Directory $Src | Select-Object -First 1
if ($Nested) {
    Get-ChildItem -Force $Nested.FullName | Move-Item -Destination $Src -Force
    Remove-Item -Recurse -Force $Nested.FullName
}
if (-not (Test-Path (Join-Path $Src 'pyproject.toml'))) {
    Write-Host "error: source tree incomplete at $Src" -ForegroundColor Red
    exit 1
}

# ---- Venv + package ----

$Venv = Join-Path $Base 'venv'
if (-not (Test-Path (Join-Path $Venv 'Scripts\python.exe'))) {
    Write-Host "[+] creating venv at $Venv"
    & $Py -m venv $Venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "error: could not create the venv" -ForegroundColor Red
        exit 1
    }
}
$Pip = Join-Path $Venv 'Scripts\pip.exe'
$Toki = Join-Path $Venv 'Scripts\toki.exe'
if (-not (Test-Path $Pip)) {
    Write-Host "error: venv missing pip at $Pip" -ForegroundColor Red
    exit 1
}
Write-Host "[+] installing tokirecur (with embeddings extras) into the venv"
& $Pip install --upgrade --quiet "$Src[embeddings]"
if ($LASTEXITCODE -ne 0) {
    Write-Host "error: pip install failed" -ForegroundColor Red
    exit 1
}

# ---- llama-server binary ----

# Reuse the install token as GITHUB_TOKEN so the GitHub API lookup in
# install-llama is authenticated (unauthenticated lookups can hit the
# shared-IP rate limit when several machines install concurrently).
$env:GITHUB_TOKEN = $env:TOKI_INSTALL_TOKEN
Write-Host "[+] fetching llama-server for this OS/arch (toki model install-llama)"
& $Toki model install-llama
if ($LASTEXITCODE -ne 0) {
    Write-Host "error: toki model install-llama failed" -ForegroundColor Red
    exit 1
}

# ---- Add venv Scripts to the user PATH (idempotent) ----

$Scripts = Join-Path $Venv 'Scripts'
$UserPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($UserPath -notmatch [regex]::Escape($Scripts)) {
    Write-Host "[+] adding $Scripts to your user PATH"
    [Environment]::SetEnvironmentVariable(
        'Path',
        ($Scripts + ';' + $UserPath),
        'User'
    )
    $env:Path = $Scripts + ';' + $env:Path
}

# ---- Agent config wiring (0.2.17: plug-and-play) ----
#
# Windows never wired the four agent configs at install time, so direct
# agent launches hit stale/absent configs (hermes "model '<name>' not
# found"). Toki Recur never downloads models — the user installs their
# own. This block only re-syncs opencode/openclaw/hermes/codex configs
# (--sync-only writes configs without booting a model) when a model is
# already registered.

$Models = & $Toki model list
if (-not $Models) {
    Write-Host "[+] no models registered - install your own, e.g.:" -ForegroundColor Yellow
    Write-Host "    toki model pull Qwen/Qwen3-8B-GGUF --name qwen3-8b --file Qwen3-8B-Q4_K_M.gguf"
    Write-Host "    toki model use qwen3-8b"
} else {
    $ActiveLine = (& $Toki model status | Select-String '^active model: (.+)$')
    if ($ActiveLine -and $ActiveLine.Matches[0].Groups[1].Value.Trim() -ne 'none') {
        $ActiveModel = $ActiveLine.Matches[0].Groups[1].Value.Trim()
        Write-Host "[+] re-syncing agent configs for active model '$ActiveModel'..."
        & $Toki model use $ActiveModel --sync-only
    }
}

Write-Host ""
Write-Host "Toki Recur installed." -ForegroundColor Green
Write-Host "  toki CLI:        $Toki"
Write-Host "  llama-server:    $env:USERPROFILE\.local\state\tokirecur\bin\llama-server.exe"
Write-Host ""
Write-Host "Agents are wired automatically; start the stack with:"
Write-Host "  toki model use <name> --sync-only   # or: toki model use <name> ^&^& toki tui"
Write-Host ""
Write-Host "To run as a service instead of a terminal window:"
Write-Host "  nssm install TokiRecur `"$Toki`" serve   (or a scheduled task at logon)"
