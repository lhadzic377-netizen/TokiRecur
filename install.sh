#!/usr/bin/env bash
# Toki Recur one-shot installer.
#
# Sets up a system-wide installation: dedicated user, env file, and a
# service manager unit (systemd on Linux, launchd on macOS). Does NOT
# install llama-server — point TOKI_BACKEND_URL in the env file at your
# existing llama-server (or run `toki model install-llama` first).
#
# Usage:
#   sudo ./install.sh
#   sudo TOKI_AUTH_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))') ./install.sh
#
# Re-runnable: subsequent runs update files in place.
#
# Windows has no shell service manager: install the Python package with
# `pip install .` and fetch a server binary with `toki model install-llama`;
# run `toki serve` yourself or wrap it in a scheduled task / NSSM service.

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
        echo "error: Windows has no shell-based service installer." >&2
        echo "       Use: pip install .  then  toki model install-llama" >&2
        echo "       and run 'toki serve' yourself (or wrap in NSSM)." >&2
        exit 1
        ;;
    *)
        echo "error: unsupported platform: $OS" >&2
        exit 1
        ;;
esac

ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_SRC="$ROOT/packaging/systemd/toki.env.example"
UNIT_SRC="$ROOT/packaging/systemd/toki-recur.service"
PLIST_SRC="$ROOT/packaging/launchd/com.tokirecur.plist"
WRAPPER_SRC="$ROOT/packaging/launchd/toki-serve.sh"
LOGROTATE_SRC="$ROOT/packaging/logrotate/toki-recur"

# ---- Sanity checks ----

if [[ ! -f "$ENV_SRC" ]]; then
    echo "error: packaging files not found" >&2
    exit 1
fi
if [[ "$OS_FLAVOR" == "linux" && ! -f "$UNIT_SRC" ]] || \
   [[ "$OS_FLAVOR" == "darwin" && ( ! -f "$PLIST_SRC" || ! -f "$WRAPPER_SRC" ) ]]; then
    echo "error: packaging files not found" >&2
    exit 1
fi

if ! command -v toki &>/dev/null; then
    echo "warning: 'toki' binary not in PATH; install with: pip install ." >&2
    echo "         the installer will still configure the service, but it will fail to start." >&2
fi

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
if [[ "$OS_FLAVOR" == "linux" ]]; then
    echo "Next steps:"
    echo "  1. (optional) edit /etc/toki-recur/toki.env to customise settings"
    echo "  2. systemctl start toki-recur.service"
    echo "  3. systemctl status toki-recur.service"
    echo
    echo "Logs:    journalctl -u toki-recur -f"
else
    echo "Next steps:"
    echo "  1. (optional) edit /etc/toki-recur/toki.env to customise settings"
    echo "  2. sudo launchctl kickstart -k system/com.tokirecur   # restart"
    echo "  3. sudo launchctl print system/com.tokirecur          # status"
    echo
    echo "Logs:    /var/log/toki-recur/stdout.log, stderr.log"
fi
echo "Metrics: curl http://127.0.0.1:8787/metrics (requires TOKI_AUTH_TOKEN)"
