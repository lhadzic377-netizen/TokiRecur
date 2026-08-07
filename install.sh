#!/usr/bin/env bash
#
# Toki Recur installer (Linux / macOS) — run from a git checkout.
#
#   git clone https://github.com/lhadzic377-netizen/TokiRecur.git
#   cd TokiRecur
#   ./install.sh
#
# Installs the `toki` CLI into its own pipx environment (or a venv
# fallback). No tokens needed — the clone already authenticated you.
# The embeddings extra (sentence-transformers + torch) is optional and
# only used by the similarity index; drop it to keep the install light:
#
#   TOKI_EXTRAS= pipx install .
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
if [[ -z "$PYTHON" ]]; then
  echo "error: python3 not found — install Python 3.10+ first" >&2
  exit 1
fi

EXTRA="."
if [[ -n "${TOKI_EXTRAS:-}" ]]; then
  EXTRA=".[${TOKI_EXTRAS}]"
fi

if command -v pipx >/dev/null 2>&1; then
  echo "Installing tokirecur with pipx ($EXTRA)…"
  pipx install "$EXTRA"
elif [[ -n "${TOKI_VENV_DIR:-}" ]]; then
  echo "Installing tokirecur into $TOKI_VENV_DIR …"
  "$PYTHON" -m venv "$TOKI_VENV_DIR"
  "$TOKI_VENV_DIR/bin/pip" install --upgrade pip >/dev/null
  "$TOKI_VENV_DIR/bin/pip" install "$EXTRA"
  TOKI="$TOKI_VENV_DIR/bin/toki"
else
  echo "pipx not found — installing it to the user site first…"
  "$PYTHON" -m pip install --user pipx
  "$PYTHON" -m pipx install "$EXTRA"
  TOKI="$(command -v toki || "$HOME/.local/bin/toki")"
fi

TOKI="${TOKI:-$(command -v toki || echo "$HOME/.local/bin/toki")}"
if [[ -x "$TOKI" ]]; then
  echo ""
  echo "Installed. Next steps:"
  echo "  $TOKI model install-llama               # fetch the llama-server binary"
  echo "  $TOKI model pull Qwen/Qwen3-8B-GGUF --name qwen3-8b"
  echo "  $TOKI model use qwen3-8b && $TOKI tui   # first cached chat"
else
  echo "Install finished, but the 'toki' binary was not found on PATH." >&2
  echo "Add $HOME/.local/bin to PATH, or re-run with TOKI_VENV_DIR=<dir>." >&2
  exit 1
fi
