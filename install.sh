#!/usr/bin/env bash
#
# Toki Recur installer — two modes in one script:
#
#   Full user install (normal user, from a git checkout):
#     git clone https://github.com/lhadzic377-netizen/TokiRecur.git
#     cd TokiRecur && ./install.sh
#   Installs the `toki` CLI with pipx (or a venv fallback) and fetches
#   the llama-server binary. No tokens needed — the clone already
#   authenticated you. Re-runnable.
#
#   System-daemon install (root, Linux systemd / macOS launchd):
#     sudo ./install.sh          # from a checkout
#     TOKI_INSTALL_TOKEN=<PAT> curl -fsSL ... | sudo bash   # one-liner
#   Bootstraps the full stack: /opt/toki-recur venv + package,
#   llama-server binary, dedicated user, env file, service unit.
#
# Switches (all optional):
#   TOKI_EXTRAS=         install without extras (faster, no torch)
#   TOKI_EXTRAS=dev      install dev/test extras instead
#   TOKI_SKIP_LLAMA=1    skip the llama-server download
#   TOKI_VENV_DIR=<dir>  install into a venv at <dir> instead of pipx
#
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    # ── full user install ───────────────────────────────────────────
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
      # --force so re-running ./install.sh overwrites an older venv
      # (e.g. upgrading 0.1.0 → 0.2.10); without it pipx refuses to
      # modify an existing install and leaves the user on the old
      # version.
      pipx install --force "$EXTRA"
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
    if [[ ! -x "$TOKI" ]]; then
      echo "Install finished, but the 'toki' binary was not found on PATH." >&2
      echo "Add $HOME/.local/bin to PATH, or re-run with TOKI_VENV_DIR=<dir>." >&2
      exit 1
    fi

    if [[ "${TOKI_SKIP_LLAMA:-0}" != "1" ]]; then
      echo ""
      echo "Fetching llama-server for this machine ($TOKI model install-llama)…"
      "$TOKI" model install-llama
      if [[ $? -ne 0 ]]; then
        echo "error: toki model install-llama failed. Fix the error above, or" >&2
        echo "re-run with TOKI_SKIP_LLAMA=1 and retry later with:" >&2
        echo "  $TOKI model install-llama" >&2
        exit 1
      fi
    fi

    echo ""
    echo "Installed."
    echo "  toki CLI:        $TOKI"
    if [[ "${TOKI_SKIP_LLAMA:-0}" != "1" && -x "$HOME/.local/state/tokirecur/bin/llama-server" ]]; then
      echo "  llama-server:    $HOME/.local/state/tokirecur/bin/llama-server"
    fi

    # Smoke test: catch permission / config-path issues now instead of
    # on first `toki` launch. `toki model list` writes the registry
    # (or honors the writable fallback in ctl._default_paths), so a
    # broken ~/.config surfaces here as a non-zero exit.
    echo ""
    echo "Smoke test: $TOKI --version"
    if ! "$TOKI" --version; then
      echo "warning: 'toki --version' failed; the install landed but the CLI is broken." >&2
    fi
    echo "Smoke test: $TOKI model list"
    if ! "$TOKI" model list; then
      echo "" >&2
      echo "error: 'toki model list' failed. This usually means" >&2
      echo "  ~/.config/tokirecur isn't writable. Toki Recur auto-falls" >&2
      echo "  back to ~/Library/Application Support/tokirecur on macOS" >&2
      echo "  when the XDG path is locked down, but if that also failed" >&2
      echo "  set TOKI_MODEL_REGISTRY to a writable file:" >&2
      echo "    export TOKI_MODEL_REGISTRY=\"\$HOME/.config/tokirecur/models.yaml\"" >&2
      echo "    echo 'export TOKI_MODEL_REGISTRY=\"...\"' >> \$HOME/.zshrc" >&2
      exit 1
    fi

    # If the registry is empty, auto-pull a small default model so the
    # user can `toki` immediately. A 0.8B Q4_K_M is ~500 MB and runs in
    # a couple of minutes on a normal connection; pick a bigger default
    # on machines that look like they have headroom (>= 16 GB RAM).
    if "$TOKI" model list | grep -q "no models"; then
      if [[ "$(uname -s)" == "Darwin" ]]; then
        # macOS: hard to read total RAM portably; default to the
        # 0.8B model which fits any Apple Silicon Mac.
        DEFAULT_REPO="Qwen/Qwen3-0.6B-GGUF"
        DEFAULT_FILE="Qwen3-0.6B-Q4_K_M.gguf"
        DEFAULT_NAME="qwen3-0.6b"
      else
        TOTAL_RAM_GB=$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo 2>/dev/null || echo 8)
        if [[ "$TOTAL_RAM_GB" -ge 16 ]]; then
          DEFAULT_REPO="Qwen/Qwen3-8B-GGUF"
          DEFAULT_FILE="Qwen3-8B-Q4_K_M.gguf"
          DEFAULT_NAME="qwen3-8b"
        else
          DEFAULT_REPO="Qwen/Qwen3-0.6B-GGUF"
          DEFAULT_FILE="Qwen3-0.6B-Q4_K_M.gguf"
          DEFAULT_NAME="qwen3-0.6b"
        fi
      fi
      echo ""
      echo "Registry is empty — pulling a starter model ($DEFAULT_REPO)..."
      echo "(set TOKI_SKIP_DEFAULT_MODEL=1 to skip this)"
      if [[ "${TOKI_SKIP_DEFAULT_MODEL:-0}" != "1" ]]; then
        if "$TOKI" model pull "$DEFAULT_REPO" --name "$DEFAULT_NAME" \
              --file "$DEFAULT_FILE"; then
          echo ""
          echo "Activating $DEFAULT_NAME (writes opencode/openclaw/hermes/codex configs)..."
          "$TOKI" model use "$DEFAULT_NAME" || true
        else
          echo "warning: starter model pull failed. Pull one manually:" >&2
          echo "  $TOKI model pull Qwen/Qwen3-0.6B-GGUF --name qwen3-0.6b" >&2
        fi
      fi
    fi

    echo ""
    echo "Installed."
    echo "  toki CLI:        $TOKI"
    if [[ "${TOKI_SKIP_LLAMA:-0}" != "1" && -x "$HOME/.local/state/tokirecur/bin/llama-server" ]]; then
      echo "  llama-server:    $HOME/.local/state/tokirecur/bin/llama-server"
    fi
    echo ""
    echo "Run 'toki' to open the launcher (or 'toki tui' for chat-only)."
    exit 0
fi

# ── daemon mode (systemd / launchd) ────────────────────────────────


set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "error: must run as root (use sudo)" >&2
    exit 1
fi

OS="$(uname -s)"
case "$OS" in
    Linux)  OS_FLAVOR="linux" ;;
    Darwin) OS_FLAVOR="darwin" ;;
    MINGW*|MSYS*|CYGWIN*)
        echo "error: Windows has no shell service installer." >&2
        echo "       Use the PowerShell one-liner (install.ps1) instead." >&2
        exit 1
        ;;
    *)
        echo "error: unsupported platform: $OS" >&2
        exit 1
        ;;
esac

# ---- Python 3.10+ requirement ----

PY="$(command -v python3 || true)"
if [[ -z "$PY" ]]; then
    echo "error: python3 not found; Toki Recur requires Python 3.10+." >&2
    echo "       Install Python 3.10+ and re-run (Debian/Ubuntu: apt install python3 python3-venv)." >&2
    exit 1
fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "error: python3 is $("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'); Toki Recur requires 3.10+." >&2
    exit 1
fi

# ---- Materialise the source tree at /opt/toki-recur/src ----

ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC=/opt/toki-recur/src
mkdir -p /opt/toki-recur
if [[ -d "$SRC/packaging" ]]; then
    echo "[+] using existing source at $SRC"
elif [[ "$ROOT" != "$SRC" && -f "$ROOT/packaging/systemd/toki.env.example" ]]; then
    echo "[+] copying source from $ROOT"
    rm -rf "$SRC"
    mkdir -p "$SRC"
    cp -a "$ROOT"/. "$SRC"/
    rm -rf "$SRC/.git" "$SRC/.venv"
else
    if [[ -z "${TOKI_INSTALL_TOKEN:-}" ]]; then
        echo "error: TOKI_INSTALL_TOKEN is required for the curl-piped install." >&2
        echo "       Create a fine-grained PAT with Contents: read on" >&2
        echo "       lhadzic377-netizen/TokiRecur and pass it via" >&2
        echo "       TOKI_INSTALL_TOKEN=<PAT> before the one-liner." >&2
        exit 1
    fi
    echo "[+] fetching TokiRecur source from GitHub"
    rm -rf "$SRC"
    mkdir -p "$SRC"
    curl -fsSL -o /tmp/tokirecur-src.tar.gz \
        "https://x-access-token:$TOKI_INSTALL_TOKEN@codeload.github.com/lhadzic377-netizen/TokiRecur/tar.gz/refs/heads/master"
    tar xzf /tmp/tokirecur-src.tar.gz -C "$SRC" --strip-components=1
    rm -f /tmp/tokirecur-src.tar.gz
fi

ENV_SRC="$SRC/packaging/systemd/toki.env.example"
UNIT_SRC="$SRC/packaging/systemd/toki-recur.service"
PLIST_SRC="$SRC/packaging/launchd/com.tokirecur.plist"
WRAPPER_SRC="$SRC/packaging/launchd/toki-serve.sh"
LOGROTATE_SRC="$SRC/packaging/logrotate/toki-recur"

# ---- Sanity checks ----

if [[ ! -f "$ENV_SRC" ]]; then
    echo "error: packaging files not found in $SRC" >&2
    exit 1
fi
if [[ "$OS_FLAVOR" == "linux" && ! -f "$UNIT_SRC" ]] || \
   [[ "$OS_FLAVOR" == "darwin" && ( ! -f "$PLIST_SRC" || ! -f "$WRAPPER_SRC" ) ]]; then
    echo "error: packaging files not found in $SRC" >&2
    exit 1
fi

# ---- Python venv + package ----

VENV=/opt/toki-recur/venv
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "[+] creating venv at $VENV"
    if ! "$PY" -m venv "$VENV"; then
        echo "error: could not create a venv." >&2
        echo "       Debian/Ubuntu: sudo apt install python3-venv and re-run." >&2
        exit 1
    fi
fi
echo "[+] installing tokirecur (with embeddings extras) into the venv"
"$VENV/bin/pip" install --upgrade --quiet "$SRC[embeddings]"

echo "[+] linking /usr/local/bin/toki"
ln -sf "$VENV/bin/toki" /usr/local/bin/toki
TOKI=/usr/local/bin/toki

# ---- Create system user ----

if ! id toki &>/dev/null; then
    echo "[+] creating user 'toki'"
    if [[ "$OS_FLAVOR" == "darwin" ]]; then
        sysadminctl -addUser toki -fullName "Toki Recur daemon" \
            -home /var/lib/toki-recur -shell /usr/sbin/nologin 2>/dev/null \
            || sysadminctl -addUser toki -fullName "Toki Recur daemon" \
                -home /var/lib/toki-recur -shell /usr/sbin/nologin
    else
        useradd --system --home /var/lib/toki-recur --shell /usr/sbin/nologin toki
    fi
fi

# ---- Create directories ----

mkdir -p /etc/toki-recur
mkdir -p /var/lib/toki-recur
mkdir -p /var/log/toki-recur
# Numeric GID: macOS's sysadminctl creates no user-private group (unlike
# Linux useradd), so the name would be an "illegal group name" there.
TOKI_GID="$(id -g toki)"
chown -R "toki:$TOKI_GID" /var/lib/toki-recur /var/log/toki-recur

# ---- Install llama-server binary (as the daemon user) ----
#
# Reuse the install token as GITHUB_TOKEN so the GitHub API lookup is
# authenticated: several machines installing concurrently can otherwise
# trip the unauthenticated API rate limit.

echo "[+] fetching llama-server for this OS/arch (toki model install-llama)"
sudo -u toki env HOME=/var/lib/toki-recur \
    GITHUB_TOKEN="${TOKI_INSTALL_TOKEN:-}" \
    "$TOKI" model install-llama

# ---- Install env file (shared by both platforms) ----

if [[ ! -f /etc/toki-recur/toki.env ]]; then
    echo "[+] creating env file /etc/toki-recur/toki.env"
    install -m 0600 -o root -g "$TOKI_GID" "$ENV_SRC" /etc/toki-recur/toki.env
fi

# If caller passed TOKI_AUTH_TOKEN, write it to the env file.
if [[ -n "${TOKI_AUTH_TOKEN:-}" ]]; then
    if grep -q '^TOKI_AUTH_TOKEN=' /etc/toki-recur/toki.env; then
        sed -i "s|^TOKI_AUTH_TOKEN=.*|TOKI_AUTH_TOKEN=$TOKI_AUTH_TOKEN|" /etc/toki-recur/toki.env
    else
        echo "TOKI_AUTH_TOKEN=$TOKI_AUTH_TOKEN" >> /etc/toki-recur/toki.env
    fi
    chmod 600 /etc/toki-recur/toki.env
    echo "[+] TOKI_AUTH_TOKEN written to /etc/toki-recur/toki.env"
fi

# ---- Install + enable the service ----

if [[ "$OS_FLAVOR" == "linux" ]]; then
    echo "[+] installing systemd unit"
    install -m 0644 "$UNIT_SRC" /etc/systemd/system/toki-recur.service

    if [[ -d /etc/logrotate.d ]]; then
        echo "[+] installing logrotate config"
        install -m 0644 "$LOGROTATE_SRC" /etc/logrotate.d/toki-recur
    fi

    echo "[+] reloading systemd"
    systemctl daemon-reload
    systemctl enable toki-recur.service
else
    echo "[+] installing launchd plist + wrapper"
    install -m 0755 "$WRAPPER_SRC" /etc/toki-recur/toki-serve.sh
    install -m 0644 "$PLIST_SRC" /Library/LaunchDaemons/com.tokirecur.plist

    echo "[+] bootstrapping launchd service"
    launchctl bootout system/com.tokirecur 2>/dev/null || true
    launchctl bootstrap system /Library/LaunchDaemons/com.tokirecur.plist
    launchctl enable system/com.tokirecur
fi

echo
echo "Toki Recur installed."
echo
echo "  toki CLI:        /usr/local/bin/toki (venv: $VENV)"
echo "  llama-server:    /var/lib/toki-recur/.local/state/tokirecur/bin/llama-server"
echo "  env file:        /etc/toki-recur/toki.env"
if [[ "$OS_FLAVOR" == "linux" ]]; then
    echo
    echo "Next steps:"
    echo "  1. (optional) edit /etc/toki-recur/toki.env to customise settings"
    echo "  2. systemctl start toki-recur.service"
    echo "  3. systemctl status toki-recur.service"
    echo
    echo "Logs:    journalctl -u toki-recur -f"
else
    echo
    echo "Next steps:"
    echo "  1. (optional) edit /etc/toki-recur/toki.env to customise settings"
    echo "  2. sudo launchctl kickstart -k system/com.tokirecur   # restart"
    echo "  3. sudo launchctl print system/com.tokirecur          # status"
    echo
    echo "Logs:    /var/log/toki-recur/stdout.log, stderr.log"
fi
echo "Metrics: curl http://127.0.0.1:8787/metrics (requires TOKI_AUTH_TOKEN)"
