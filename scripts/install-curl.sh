#!/usr/bin/env bash
# Toki Recur curl installer (Linux/macOS) — pipx install from the private repo.
#
# The token is read from the environment ONLY (TOKI_REPO_TOKEN or
# GITHUB_TOKEN) — never hardcode PATs in scripts or workflow files.
# Scope it to the minimum: a fine-grained PAT with Contents: read on
# lhadzic377-netizen/TokiRecur.
#
# Usage:
#   export TOKI_REPO_TOKEN=<pat>
#   bash scripts/install-curl.sh
#
#   # or fetch it securely from the private repo first:
#   curl -sf -H "Authorization: Bearer $TOKI_REPO_TOKEN" \
#     -H "Accept: application/vnd.github.raw" \
#     "https://api.github.com/repos/lhadzic377-netizen/TokiRecur/contents/scripts/install-curl.sh" \
#     -o /tmp/toki-install.sh && bash /tmp/toki-install.sh
set -euo pipefail

REPO="lhadzic377-netizen/TokiRecur"
TOKEN="${TOKI_REPO_TOKEN:-${GITHUB_TOKEN:-}}"

if [[ -z "$TOKEN" ]]; then
  echo "error: TOKI_REPO_TOKEN is not set" >&2
  echo "  create a fine-grained PAT with Contents: read on ${REPO}, then:" >&2
  echo "  export TOKI_REPO_TOKEN=<pat>" >&2
  exit 1
fi

INSTALLER="pipx"
if ! command -v pipx >/dev/null 2>&1; then
  echo "pipx not found — installing it to the user site first…"
  python3 -m pip install --user pipx
fi

echo "Installing tokirecur from ${REPO} (private)…"
pipx install "tokirecur[embeddings] @ git+https://x-access-token:${TOKEN}@github.com/${REPO}.git"

echo ""
echo "Done. Next steps:"
echo "  toki model install-llama                  # fetch the llama-server binary"
echo "  toki model pull Qwen/Qwen3-8B-GGUF --name qwen3-8b"
echo "  toki model use qwen3-8b && toki tui       # first cached chat"
echo ""
echo "Reinstall/upgrade: pipx reinstall tokirecur   (token still needed)"
