#!/usr/bin/env bash
# Toki Recur curl installer (Linux/macOS) — pipx install from the public repo.
#
# The repository is public, so no token is needed. If you set
# TOKI_REPO_TOKEN (or GITHUB_TOKEN) it is used instead, which only raises
# the GitHub API rate limit. Never hardcode PATs in scripts or workflow
# files.
#
# Usage:
#   bash scripts/install-curl.sh
#
#   # or fetch it first and run it locally:
#   curl -sf "https://raw.githubusercontent.com/lhadzic377-netizen/TokiRecur/main/scripts/install-curl.sh" \
#     -o /tmp/toki-install.sh && bash /tmp/toki-install.sh
set -euo pipefail

REPO="lhadzic377-netizen/TokiRecur"
TOKEN="${TOKI_REPO_TOKEN:-${GITHUB_TOKEN:-}}"

# The embeddings extra pulls torch. Skip it with TOKI_EXTRAS= (CI does).
EXTRAS="${TOKI_EXTRAS-embeddings}"
if [[ -z "$EXTRAS" ]]; then
    SPEC="tokirecur"
else
    SPEC="tokirecur[${EXTRAS}]"
fi

INSTALLER="pipx"
if ! command -v pipx >/dev/null 2>&1; then
  echo "pipx not found — installing it to the user site first…"
  python3 -m pip install --user pipx
fi

if [[ -z "$TOKEN" ]]; then
  echo "Installing tokirecur from ${REPO} (public)…"
  pipx install "${SPEC} @ git+https://github.com/${REPO}.git"
else
  echo "Installing tokirecur from ${REPO} using the token in TOKI_REPO_TOKEN…"
  pipx install "${SPEC} @ git+https://x-access-token:${TOKEN}@github.com/${REPO}.git"
fi

echo ""
echo "Done. Next steps:"
echo "  toki model install-llama                  # fetch the llama-server binary"
echo "  toki model pull Qwen/Qwen3-8B-GGUF --name qwen3-8b"
echo "  toki model use qwen3-8b && toki tui       # first cached chat"
echo ""
echo "Reinstall/upgrade: pipx reinstall tokirecur"
