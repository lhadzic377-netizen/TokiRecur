# Toki Recur irm installer (Windows PowerShell) — pipx install from the private repo.
#
# The token is read from the environment ONLY ($env:TOKI_REPO_TOKEN) —
# never hardcode PATs in scripts or workflow files. Scope it to the
# minimum: a fine-grained PAT with Contents: read on lhadzic377-netizen/TokiRecur.
#
# Usage:
#   $env:TOKI_REPO_TOKEN = "<pat>"
#   pwsh scripts/install-irm.ps1
#
#   # or fetch it securely from the private repo first:
#   $env:TOKI_REPO_TOKEN = "<pat>"
#   $installer = irm -Uri "https://api.github.com/repos/lhadzic377-netizen/TokiRecur/contents/scripts/install-irm.ps1" `
#     -Headers @{ Authorization = "Bearer $env:TOKI_REPO_TOKEN"; Accept = "application/vnd.github.raw" } -ErrorAction Stop
#   $installer | Out-File -Encoding utf8 "$env:TEMP\toki-install.ps1"
#   pwsh -File "$env:TEMP\toki-install.ps1"
$ErrorActionPreference = "Stop"

$repo = "lhadzic377-netizen/TokiRecur"
$token = $env:TOKI_REPO_TOKEN
if ([string]::IsNullOrEmpty($token)) {
    throw "TOKI_REPO_TOKEN is not set. Create a fine-grained PAT with Contents: read on $repo, then set env:TOKI_REPO_TOKEN."
}

if (-not (Get-Command pipx -ErrorAction SilentlyContinue)) {
    Write-Host "pipx not found — installing it to the user site first…"
    python -m pip install --user pipx
    if ($LASTEXITCODE -ne 0) {
        py -m pip install --user pipx
    }
}

Write-Host "Installing tokirecur from $repo (private)…"
pipx install "tokirecur[embeddings] @ git+https://x-access-token:$token@github.com/$repo.git"

Write-Host ""
Write-Host "Done. Next steps:"
Write-Host "  toki model install-llama                  # fetch the llama-server binary"
Write-Host "  toki model pull Qwen/Qwen3-8B-GGUF --name qwen3-8b"
Write-Host "  toki model use qwen3-8b; toki tui         # first cached chat"
Write-Host ""
Write-Host "Reinstall/upgrade: pipx reinstall tokirecur   (token still needed)"
