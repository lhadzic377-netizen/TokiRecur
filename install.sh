#!/usr/bin/env bash
# Toki Recur one-shot bootstrap installer (Linux systemd / macOS launchd).
#
# Full install from zero to a running daemon:
#   1. require Python 3.10+
#   2. materialise the app source at /opt/toki-recur/src
#      - one-liner (curl-piped): fetched as a tarball from the private
#        GitHub repo using TOKI_INSTALL_TOKEN (Contents:read PAT); no git
#        binary required on the target machine
#      - from a checkout (./install.sh): copied from the repo directory
#   3. venv at /opt/toki-recur/venv + pip install .[embeddings]
#   4. symlink /usr/local/bin/toki -> venv binary
#   5. toki model install-llama (official llama-server binary for this OS/arch)
#   6. dedicated user, env file, service unit (systemd / launchd) + enable
#
# Usage:
#   One-liner (private repo — token required):
#     TOKI_INSTALL_TOKEN=<PAT> bash -c 'curl -fsSL -H "Accept: application/vnd.github.raw" "https://api.github.com/repos/lhadzic377-netizen/TokiRecur/contents/install.sh?ref=master" -u "x-access-token:$TOKI_INSTALL_TOKEN" | sudo TOKI_INSTALL_TOKEN=$TOKI_INSTALL_TOKEN bash'
#   From a checkout:
#     sudo ./install.sh
#     sudo TOKI_AUTH_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))') ./install.sh
#
# Re-runnable: re-runs re-fetch/re-copy the source, reinstall the package,
# and update all files in place.
#
# Windows: use the PowerShell one-liner instead (install.ps1):
#   $env:TOKI_INSTALL_TOKEN='<PAT>'; $wc=New-Object System.Net.WebClient; $wc.Headers.Add('Accept','application/vnd.github.raw'); $wc.Credentials=New-Object System.Net.NetworkCredential($env:TOKI_INSTALL_TOKEN,''); iex ($wc.DownloadString('https://api.github.com/repos/lhadzic377-netizen/TokiRecur/contents/install.ps1?ref=master'))

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
