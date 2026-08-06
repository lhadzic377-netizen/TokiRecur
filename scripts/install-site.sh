#!/usr/bin/env bash
#
# Toki Recur site installer — Debian / Ubuntu / Linux Mint
#
# Turns a laptop into the server for your domain: installs Caddy (automatic
# HTTPS via Let's Encrypt), deploys the landing page, installs the tokirecur
# stack, and exposes the live web chat on a subdomain.
#
#   Usage:  sudo ./install.sh [options]
#
#   Options:
#     --domain <name>    Main domain (default: tokirecur.com)
#     --chat <name>      Chat subdomain (default: chat.<domain>)
#     --email <addr>     ACME account email for Let's Encrypt (recommended)
#     --no-chat          Skip the chat subdomain + web UI service
#     --no-toki          Install Caddy + site only, skip the tokirecur stack
#     --git-token <pat>  Install tokirecur from the private repo using this
#                        read-only PAT (used only if PyPI install fails)
#     -h, --help         Show this help
#
# Before running: point DNS A records at this machine's public IP:
#     <domain>      A  <public-ip>
#     chat.<domain> A  <public-ip>
# and forward TCP 80 + 443 to this machine if it sits behind a router.
#
set -euo pipefail

DOMAIN="tokirecur.com"
CHAT_DOMAIN=""
EMAIL=""
WITH_CHAT=1
WITH_TOKI=1
GIT_TOKEN="${TOKI_GIT_PAT:-}"

SITE_ROOT="/opt/toki-site"
WWW_ROOT="$SITE_ROOT/www"
VENV_DIR="$SITE_ROOT/venv"
TOKI_USER="toki"
TOKI_HOME="/var/lib/toki"
WEB_PORT=7777

log()  { printf '\033[1;33m[*]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[+] \033[0m%s\n' "$*"; }
fail() { printf '\033[1;31m[!] \033[0m%s\n' "$*" >&2; exit 1; }

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain)    DOMAIN="${2:?--domain needs a value}"; shift 2 ;;
    --chat)      CHAT_DOMAIN="${2:?--chat needs a value}"; shift 2 ;;
    --email)     EMAIL="${2:?--email needs a value}"; shift 2 ;;
    --no-chat)   WITH_CHAT=0; shift ;;
    --no-toki)   WITH_TOKI=0; shift ;;
    --git-token) GIT_TOKEN="$2"; shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    *)           fail "unknown option: $1 (see --help)" ;;
  esac
done

[[ "${WITH_CHAT:-0}" == "0" && "${WITH_TOKI:-0}" == "0" ]] && fail "refusing to install nothing: drop --no-chat and/or --no-toki"

[[ $EUID -eq 0 ]] || fail "run with sudo:  sudo ./install-site.sh"
command -v apt-get >/dev/null || fail "this installer targets Debian / Ubuntu / Linux Mint (apt)"
command -v curl >/dev/null || apt-get install -y curl >/dev/null

[[ -z "$CHAT_DOMAIN" ]] && CHAT_DOMAIN="chat.$DOMAIN"

log "Domain:      $DOMAIN"
log "Chat:        $CHAT_DOMAIN"
log "ACME email:  ${EMAIL:-<none — recommended to pass --email>}"

# ── 1. Caddy (official apt repo) ──────────────────────────────────────
if ! command -v caddy >/dev/null; then
  log "Installing Caddy from the official apt repository..."
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key \
    | gpg --dearmor --yes -o /etc/apt/keyrings/caddy-stable-archive-keyring.gpg
  echo "deb [signed-by=/etc/apt/keyrings/caddy-stable-archive-keyring.gpg] https://dl.cloudsmith.io/public/caddy/stable/deb/debian any-version main" \
    > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq
  apt-get install -y caddy
else
  ok "Caddy already installed"
fi

# ── 2. tokirecur stack (own venv, not pipx, so systemd can use it) ───
if [[ "$WITH_TOKI" == "1" ]]; then
  command -v python3 >/dev/null || apt-get install -y python3 python3-venv
  if [[ ! -x "$VENV_DIR/bin/toki" ]]; then
    log "Installing tokirecur into $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    if ! "$VENV_DIR/bin/pip" install --quiet tokirecur 2>/dev/null; then
      if [[ -n "$GIT_TOKEN" ]]; then
        log "PyPI install failed — falling back to git+https with the provided PAT"
        "$VENV_DIR/bin/pip" install --quiet "tokirecur @ git+https://x-access-token:${GIT_TOKEN}@github.com/lhadzic377-netizen/TokiRecur.git"
      else
        fail "PyPI install failed and no --git-token was given. Pass --git-token <PAT> to install from the private repo, or fix your network/PyPI access."
      fi
    fi
  else
    ok "tokirecur already installed in $VENV_DIR"
  fi

  if ! id -u "$TOKI_USER" >/dev/null 2>&1; then
    log "Creating service user '$TOKI_USER' ..."
    useradd --system --home "$TOKI_HOME" --shell /usr/sbin/nologin "$TOKI_USER"
  fi
  mkdir -p "$TOKI_HOME/.config" "$TOKI_HOME/.local/state" "$TOKI_HOME/.local/bin"
  chown -R "$TOKI_USER:$TOKI_USER" "$TOKI_HOME"
fi

# ── 3. Landing page ───────────────────────────────────────────────────
log "Deploying the landing page to $WWW_ROOT ..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_SRC="$SCRIPT_DIR/../site"
mkdir -p "$WWW_ROOT"
cp "$SITE_SRC/index.html" "$SITE_SRC/style.css" "$SITE_SRC/app.js" "$SITE_SRC/favicon.svg" "$WWW_ROOT/"
mkdir -p "$WWW_ROOT/docs"
cp "$SITE_SRC/docs/"*.html "$SITE_SRC/docs/"*.css "$SITE_SRC/docs/"*.js "$WWW_ROOT/docs/"
chown -R root:root "$WWW_ROOT"

# ── 4. Caddyfile ──────────────────────────────────────────────────────
log "Writing /etc/caddy/Caddyfile ..."
if [[ -f /etc/caddy/Caddyfile ]]; then
  cp /etc/caddy/Caddyfile "/etc/caddy/Caddyfile.bak.$(date +%s)"
fi
sed -e "s/{{DOMAIN}}/$DOMAIN/g" \
    -e "s/{{CHAT_DOMAIN}}/$CHAT_DOMAIN/g" \
    -e "s/{{EMAIL}}/${EMAIL:-admin@$DOMAIN}/g" \
    "$SCRIPT_DIR/Caddyfile.template" > /etc/caddy/Caddyfile
caddy validate --config /etc/caddy/Caddyfile >/dev/null || fail "generated Caddyfile failed validation"
systemctl enable --now caddy
systemctl restart caddy

# ── 5. Web chat service ───────────────────────────────────────────────
if [[ "$WITH_CHAT" == "1" ]]; then
  log "Installing toki-web systemd service (127.0.0.1:$WEB_PORT) ..."
  cat > /etc/systemd/system/toki-web.service <<UNIT
[Unit]
Description=Toki Recur web chat UI
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$TOKI_USER
Group=$TOKI_USER
Environment=HOME=$TOKI_HOME
Environment=XDG_CONFIG_HOME=$TOKI_HOME/.config
Environment=XDG_STATE_HOME=$TOKI_HOME/.local/state
ExecStart=$VENV_DIR/bin/toki web --host 127.0.0.1 --port $WEB_PORT
Restart=on-failure
RestartSec=3
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload
  systemctl enable toki-web
  systemctl start toki-web || warn_chat=1
fi

# ── 6. Summary ────────────────────────────────────────────────────────
cat > "$SITE_ROOT/README" <<DOC
Toki Recur site — installed $(date -u +%Y-%m-%dT%H:%MZ)
  Landing page:  $DOMAIN        -> $WWW_ROOT (Caddy)
  Web chat:      $CHAT_DOMAIN   -> 127.0.0.1:$WEB_PORT (toki-web.service)

Bring the live chat online (needs a model + GPU/CPU):
  sudo -u $TOKI_USER $VENV_DIR/bin/toki model install-llama
  sudo -u $TOKI_USER $VENV_DIR/bin/toki model pull <name>
  sudo -u $TOKI_USER $VENV_DIR/bin/toki model use <name>
  sudo systemctl restart toki-web

Logs:  journalctl -u toki-web -f    |    journalctl -u caddy -f
DOC

echo
ok "Install complete."
echo
echo "  Landing page : https://$DOMAIN"
echo "  Web chat     : https://$CHAT_DOMAIN"
echo "  Config       : /etc/caddy/Caddyfile   (backed up if it existed)"
echo "  Site files   : $WWW_ROOT"
echo "  Notes        : $SITE_ROOT/README"
echo
echo "Still to do on your side:"
echo "  1. DNS  A  $DOMAIN       -> public IP of this machine"
echo "     DNS  A  $CHAT_DOMAIN  -> public IP of this machine"
echo "     (or a wildcard A *.$DOMAIN covering both)"
echo "  2. Forward TCP 80 + 443 to this machine if it's behind a router."
echo "  3. Caddy already holds the certificates; first https visit takes ~30s."
if [[ "${warn_chat:-0}" == "1" ]]; then
  echo "  4. NOTE: toki-web failed to start — see: journalctl -u toki-web"
fi
if [[ "$WITH_TOKI" == "1" ]]; then
  echo "  4. Optional: install a model for the live chat (see $SITE_ROOT/README)."
fi
