# Toki Recur irm installer (Windows PowerShell) — pipx install from the public repo.
#
# The repository is public, so no token is needed. If you set
# $env:TOKI_REPO_TOKEN it is used instead, which only raises the GitHub API
# rate limit. Never hardcode PATs in scripts or workflow files.
#
# Usage:
#   pwsh scripts/install-irm.ps1
#
#   # or fetch it first and run it locally:
#   $installer = irm -Uri "https://raw.githubusercontent.com/lhadzic377-netizen/TokiRecur/main/scripts/install-irm.ps1" -ErrorAction Stop
#   $installer | Out-File -Encoding utf8 "$env:TEMP\toki-install.ps1"
#   pwsh -File "$env:TEMP\toki-install.ps1"
$ErrorActionPreference = "Stop"

$repo = "lhadzic377-netizen/TokiRecur"
$token = $env:TOKI_REPO_TOKEN

# The embeddings extra pulls torch. Skip it with TOKI_EXTRAS= (CI does).
$extras = if ($env:TOKI_EXTRAS) { $env:TOKI_EXTRAS } else { "embeddings" }
$spec = if ($extras) { "tokirecur[$extras]" } else { "tokirecur" }

if (-not (Get-Command pipx -ErrorAction SilentlyContinue)) {
    Write-Host "pipx not found — installing it to the user site first…"
    python -m pip install --user pipx
    if ($LASTEXITCODE -ne 0) {
        py -m pip install --user pipx
    }
}

if ([string]::IsNullOrEmpty($token)) {
    Write-Host "Installing tokirecur from $repo (public)…"
    pipx install "$spec @ git+https://github.com/$repo.git"
} else {
    Write-Host "Installing tokirecur from $repo using the token in TOKI_REPO_TOKEN…"
    pipx install "$spec @ git+https://x-access-token:$token@github.com/$repo.git"
}

Write-Host ""
Write-Host "Done. Next steps:"
Write-Host "  toki model install-llama                  # fetch the llama-server binary"
Write-Host "  toki model pull Qwen/Qwen3-8B-GGUF --name qwen3-8b"
Write-Host "  toki model use qwen3-8b; toki tui         # first cached chat"
Write-Host ""
Write-Host "Reinstall/upgrade: pipx reinstall tokirecur"
