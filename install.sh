#!/usr/bin/env bash
# Toki Recur one-shot installer.
#
# Sets up a system-wide installation: dedicated user, env file,
# systemd unit, and log rotation. Does NOT install llama-server —
# point TOKI_BACKEND_URL in the env file at your existing llama-server.
#
# Usage:
#   sudo ./install.sh
#   sudo TOKI_AUTH_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))') ./install.sh
#
# Re-runnable: subsequent runs update files in place.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "error: must run as root (use sudo)" >&2
    exit 1
fi

PREFIX="${PREFIX:-/usr/local}"
UNIT_SRC="$(cd "$(dirname "$0")" && pwd)/packaging/systemd/toki-recur.service"
ENV_SRC="$(cd "$(dirname "$0")" && pwd)/packaging/systemd/toki.env.example"
LOGROTATE_SRC="$(cd "$(dirname "$0")" && pwd)/packaging/logrotate/toki-recur"

# ---- Sanity checks ----

if [[ ! -f "$UNIT_SRC" ]] || [[ ! -f "$ENV_SRC" ]]; then
    echo "error: packaging files not found" >&2
    exit 1
fi

if ! command -v toki &>/dev/null; then
    echo "warning: 'toki' binary not in PATH; install with: pip install ." >&2
    echo "         the installer will still configure systemd, but the service will fail to start." >&2
fi

# ---- Create system user ----

if ! id toki &>/dev/null; then
    echo "[+] creating user 'toki'"
    useradd --system --home /var/lib/toki-recur --shell /usr/sbin/nologin toki
fi

# ---- Create directories ----

mkdir -p /etc/toki-recur
mkdir -p /var/lib/toki-recur
mkdir -p /var/log/toki-recur
chown -R toki:toki /var/lib/toki-recur /var/log/toki-recur

# ---- Install systemd unit ----

echo "[+] installing systemd unit"
install -m 0644 "$UNIT_SRC" /etc/systemd/system/toki-recur.service

# ---- Install env file ----

if [[ ! -f /etc/toki-recur/toki.env ]]; then
    echo "[+] creating env file /etc/toki-recur/toki.env"
    install -m 0600 -o root -g toki "$ENV_SRC" /etc/toki-recur/toki.env
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

# ---- Install logrotate config ----

if [[ -d /etc/logrotate.d ]]; then
    echo "[+] installing logrotate config"
    install -m 0644 "$LOGROTATE_SRC" /etc/logrotate.d/toki-recur
fi

# ---- Reload systemd ----

echo "[+] reloading systemd"
systemctl daemon-reload
systemctl enable toki-recur.service

echo
echo "Toki Recur installed."
echo
echo "Next steps:"
echo "  1. (optional) edit /etc/toki-recur/toki.env to customise settings"
echo "  2. systemctl start toki-recur.service"
echo "  3. systemctl status toki-recur.service"
echo
echo "Logs:    journalctl -u toki-recur -f"
echo "Metrics: curl http://127.0.0.1:8787/metrics (requires TOKI_AUTH_TOKEN)"
