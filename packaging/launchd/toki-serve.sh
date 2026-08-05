#!/bin/sh
# Launchd wrapper for the Toki Recur daemon (macOS).
#
# launchd does not support EnvironmentFile, so this wrapper sources
# /etc/toki-recur/toki.env and then execs the daemon. The launchd PATH
# is minimal, so the toki binary is located explicitly.
set -a
[ -f /etc/toki-recur/toki.env ] && . /etc/toki-recur/toki.env
set +a

TOKI_BIN="$(command -v toki 2>/dev/null || true)"
if [ -z "$TOKI_BIN" ] && [ -x /usr/local/bin/toki ]; then
    TOKI_BIN=/usr/local/bin/toki
fi
if [ -z "$TOKI_BIN" ]; then
    echo "error: 'toki' binary not found; install with: pip install ." >&2
    exit 1
fi

exec "$TOKI_BIN" serve \
    --host 0.0.0.0 \
    --port 8787 \
    --backend-url "${TOKI_BACKEND_URL:-http://127.0.0.1:8088}"
