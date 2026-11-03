#!/usr/bin/env bash
#
# Toki Recur site installer — Debian / Ubuntu / Linux Mint
#
# Installs Caddy (automatic HTTPS via Let's Encrypt), deploys a built static
# site as an atomic release, and optionally installs the tokirecur stack and
# the web chat subdomain.
#
#   Usage:  sudo ./install.sh [options]
#
#   Options:
#     --domain <name>    Main domain (default: tokirecur.com)
#     --chat <name>      Chat subdomain (default: chat.<domain>)
#     --email <addr>     ACME account email for Let's Encrypt (recommended)
#     --no-chat          Skip the chat subdomain + web UI service
#     --no-toki          Skip the tokirecur stack; site only
#     --dry-run          Build the site and render the Caddyfile into a
#                        temporary directory. Changes nothing on the machine.
#     --git-token <pat>  Install tokirecur from the repository with this
#                        read-only PAT. Optional: the repository is public and
#                        the token only raises the GitHub API rate limit.
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
DRY_RUN=0
GIT_TOKEN="${TOKI_GIT_PAT:-}"

SITE_ROOT="${TOKI_SITE_ROOT:-/opt/toki-site}"
CADDY_CONFIG="${TOKI_CADDY_CONFIG:-/etc/caddy/Caddyfile}"
UNIT_DIR="${TOKI_SYSTEMD_DIR:-/etc/systemd/system}"
RELEASES_DIR="$SITE_ROOT/releases"
CURRENT_LINK="$SITE_ROOT/current"
VENV_DIR="$SITE_ROOT/venv"
TOKI_USER="toki"
TOKI_HOME="/var/lib/toki"
WEB_PORT=7777

log()  { printf '\033[1;33m[*]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[+] \033[0m%s\n' "$*"; }
fail() { printf '\033[1;31m[!]\033[0m %s\n' "$*" >&2; exit 1; }

usage() { sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain)    DOMAIN="${2:?--domain needs a value}"; shift 2 ;;
    --chat)      CHAT_DOMAIN="${2:?--chat needs a value}"; shift 2 ;;
    --email)     EMAIL="${2:?--email needs a value}"; shift 2 ;;
    --no-chat)   WITH_CHAT=0; shift ;;
    --no-toki)   WITH_TOKI=0; shift ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --git-token) GIT_TOKEN="$2"; shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    *)           fail "unknown option: $1 (see --help)" ;;
  esac
done

if [[ "$WITH_CHAT" == "1" && "$WITH_TOKI" == "0" ]]; then
  log "--no-toki given; skipping the web chat service too"
  WITH_CHAT=0
fi

if [[ "$DRY_RUN" == "1" ]]; then
  DRY_ROOT="$(mktemp -d /tmp/toki-site-dryrun.XXXXXX)"
  SITE_ROOT="$DRY_ROOT/site"
  RELEASES_DIR="$SITE_ROOT/releases"
  CURRENT_LINK="$SITE_ROOT/current"
  VENV_DIR="$SITE_ROOT/venv"
  CADDY_CONFIG="$DRY_ROOT/Caddyfile"
  UNIT_DIR="$DRY_ROOT/systemd"
  mkdir -p "$UNIT_DIR"
  log "Dry run: nothing is installed, written to /etc, or started."
  log "Dry run root: $DRY_ROOT"
else
  [[ $EUID -eq 0 ]] || fail "run with sudo:  sudo ./install.sh"
fi

command -v apt-get >/dev/null || { [[ "$DRY_RUN" == "1" ]] || fail "this installer targets Debian / Ubuntu / Linux Mint (apt)"; }
command -v curl >/dev/null || { [[ "$DRY_RUN" == "1" ]] || apt-get install -y curl >/dev/null; }
command -v python3 >/dev/null || { [[ "$DRY_RUN" == "1" ]] || apt-get install -y python3 >/dev/null; }

# A chat host of chat.www.example.com is not a thing anybody points DNS at.
BASE_DOMAIN="${DOMAIN#www.}"
[[ -z "$CHAT_DOMAIN" ]] && CHAT_DOMAIN="chat.$BASE_DOMAIN"
if [[ "$DOMAIN" == www.* ]]; then
  WWW_DOMAIN=""
else
  WWW_DOMAIN="www.$DOMAIN"
fi

log "Domain:      $DOMAIN"
log "WWW:         ${WWW_DOMAIN:-<none>}"
log "Chat:        $CHAT_DOMAIN"
log "ACME email:  ${EMAIL:-<none — recommended to pass --email>}"

# ── 1. Caddy (official apt repo) ──────────────────────────────────────
if command -v caddy >/dev/null; then
  ok "Caddy already installed ($(caddy version 2>/dev/null | head -1))"
elif [[ "$DRY_RUN" == "1" ]]; then
  log "Dry run: would install Caddy from the official apt repository"
else
  log "Installing Caddy from the official apt repository..."
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key \
    | gpg --dearmor --yes -o /etc/apt/keyrings/caddy-stable-archive-keyring.gpg
  echo "deb [signed-by=/etc/apt/keyrings/caddy-stable-archive-keyring.gpg] https://dl.cloudsmith.io/public/caddy/stable/deb/debian any-version main" \
    > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq
  apt-get install -y caddy
fi

# ── 2. tokirecur stack (own venv, not pipx, so systemd can use it) ───
if [[ "$WITH_TOKI" == "1" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    log "Dry run: would install tokirecur into $VENV_DIR"
  else
    command -v python3 >/dev/null || apt-get install -y python3 python3-venv
    if [[ ! -x "$VENV_DIR/bin/toki" ]]; then
      log "Installing tokirecur into $VENV_DIR ..."
      python3 -m venv "$VENV_DIR"
      if ! "$VENV_DIR/bin/pip" install --quiet tokirecur 2>/dev/null; then
        # Not on PyPI yet: install from the public repository. A token is
        # optional and only raises the GitHub API rate limit.
        if [[ -n "$GIT_TOKEN" ]]; then
          log "Installing from the repository with the provided token"
          "$VENV_DIR/bin/pip" install --quiet "tokirecur @ git+https://x-access-token:${GIT_TOKEN}@github.com/lhadzic377-netizen/TokiRecur.git"
        else
          log "Installing from the public repository"
          "$VENV_DIR/bin/pip" install --quiet "tokirecur @ git+https://github.com/lhadzic377-netizen/TokiRecur.git"
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
fi

# ── 3. Build the public site ──────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/site/index.html" ]]; then
  # Built zip layout: install.sh + site/.
  SITE_SRC="$SCRIPT_DIR/site"
  log "Using the prebuilt site from $SITE_SRC"
elif [[ -f "$SCRIPT_DIR/../site/index.html" && -f "$SCRIPT_DIR/build-site.py" ]]; then
  # Repo layout: build from source so the deploy matches the source tree.
  REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
  log "Building the public site ..."
  python3 "$SCRIPT_DIR/build-site.py" --site-url "https://$DOMAIN"
  SITE_SRC="$REPO_DIR/dist/site"
else
  fail "cannot find site/ or build-site.py next to this installer"
fi

[[ -f "$SITE_SRC/index.html" ]] || fail "site build is missing index.html"
[[ -f "$SITE_SRC/404.html" ]] || fail "site build is missing 404.html"

if [[ -f "$SCRIPT_DIR/check-site-build.py" ]]; then
  python3 "$SCRIPT_DIR/check-site-build.py" || fail "the built site failed the pre-deploy check"
fi

# ── 4. Atomic release ─────────────────────────────────────────────────
log "Deploying the site as an immutable release ..."
mkdir -p "$RELEASES_DIR"
RELEASE="$RELEASES_DIR/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RELEASE"
cp -a "$SITE_SRC/." "$RELEASE/"
if [[ "$DRY_RUN" == "0" ]]; then
  chown -R root:root "$RELEASE"
  chmod -R u=rwX,go=rX "$RELEASE"
fi

# Migrate an older in-place install rather than replacing a real directory
# with a symlink in one step.
if [[ -e "$CURRENT_LINK" && ! -L "$CURRENT_LINK" ]]; then
  mv "$CURRENT_LINK" "$SITE_ROOT/www.legacy.$(date +%s)"
fi
ln -sfn "$RELEASE" "$CURRENT_LINK"

# Keep three releases; the previous one is the rollback.
mapfile -t OLD_RELEASES < <(
  find "$RELEASES_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
    | sort -nr | tail -n +4 | cut -d' ' -f2-
)
for old in "${OLD_RELEASES[@]:-}"; do
  [[ -n "$old" ]] && rm -rf "$old"
done

# ── 5. Caddyfile ──────────────────────────────────────────────────────
log "Writing $CADDY_CONFIG ..."
TMP_CADDY="$(mktemp)"
sed -e "s|{{DOMAIN}}|$DOMAIN|g" \
    -e "s|{{WWW_DOMAIN}}|$WWW_DOMAIN|g" \
    -e "s|{{CHAT_DOMAIN}}|$CHAT_DOMAIN|g" \
    -e "s|{{EMAIL}}|${EMAIL:-admin@$DOMAIN}|g" \
    -e "s|{{WWW_ROOT}}|$CURRENT_LINK|g" \
    "$SCRIPT_DIR/Caddyfile.template" > "$TMP_CADDY"

if [[ -z "$WWW_DOMAIN" ]]; then
  awk '/^# WWW_BEGIN$/{skip=1;next} /^# WWW_END$/{skip=0;next} !skip' "$TMP_CADDY" > "$TMP_CADDY.2"
  mv "$TMP_CADDY.2" "$TMP_CADDY"
fi
if [[ "$WITH_CHAT" == "0" ]]; then
  awk '/^# CHAT_BEGIN$/{skip=1;next} /^# CHAT_END$/{skip=0;next} !skip' "$TMP_CADDY" > "$TMP_CADDY.3"
  mv "$TMP_CADDY.3" "$TMP_CADDY"
fi

if [[ -f "$CADDY_CONFIG" ]]; then
  cp "$CADDY_CONFIG" "$CADDY_CONFIG.bak.$(date +%s)"
fi
install -m 0644 "$TMP_CADDY" "$CADDY_CONFIG"
rm -f "$TMP_CADDY"

if command -v caddy >/dev/null; then
  caddy validate --config "$CADDY_CONFIG" >/dev/null || fail "generated Caddyfile failed validation"
  ok "Caddyfile validated"
else
  log "Caddy is not installed here, so the Caddyfile was not validated"
fi

if [[ "$DRY_RUN" == "0" ]]; then
  systemctl enable --now caddy
  systemctl restart caddy

  # Remove the old wishlist service from previous installs. The site no longer
  # ships a server-side counter; the public release is static.
  if systemctl cat toki-site-api.service >/dev/null 2>&1; then
    log "Removing the retired wishlist API service"
    systemctl disable --now toki-site-api 2>/dev/null || true
    rm -f "$UNIT_DIR/toki-site-api.service"
    systemctl daemon-reload
  fi
fi

# ── 6. Web chat service ───────────────────────────────────────────────
if [[ "$WITH_CHAT" == "1" ]]; then
  log "Installing toki-web systemd unit (127.0.0.1:$WEB_PORT) ..."
  cat > "$UNIT_DIR/toki-web.service" <<UNIT
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
  if [[ "$DRY_RUN" == "0" ]]; then
    systemctl daemon-reload
    systemctl enable toki-web
    systemctl start toki-web || warn_chat=1
  fi
elif [[ "$DRY_RUN" == "0" ]]; then
  if systemctl cat toki-web.service >/dev/null 2>&1; then
    log "Disabling the web chat service (--no-chat)"
    systemctl disable --now toki-web 2>/dev/null || true
  fi
fi

# ── 7. Summary ────────────────────────────────────────────────────────
cat > "$SITE_ROOT/README" <<DOC
Toki Recur site — installed $(date -u +%Y-%m-%dT%H:%MZ)
  Site:         $DOMAIN -> $CURRENT_LINK
  Release:      $RELEASE
  Rollback:     ln -sfn \$(ls -1dt $RELEASES_DIR/* | sed -n 2p) $CURRENT_LINK && systemctl reload caddy
  Site files:   $RELEASE
  Caddyfile:    $CADDY_CONFIG

$(if [[ "$WITH_CHAT" == "1" ]]; then echo "Web chat:     $CHAT_DOMAIN -> 127.0.0.1:$WEB_PORT (toki-web.service)"; else echo "Web chat:     not installed (--no-chat)"; fi)

Bring the live chat online (needs a model + GPU/CPU):
  sudo -u $TOKI_USER $VENV_DIR/bin/toki model install-llama
  sudo -u $TOKI_USER $VENV_DIR/bin/toki model pull <name>
  sudo -u $TOKI_USER $VENV_DIR/bin/toki model use <name>
  sudo systemctl restart toki-web

Logs:  journalctl -u toki-web -f    |    journalctl -u caddy -f
DOC

echo
ok "$([[ "$DRY_RUN" == "1" ]] && echo 'Dry run complete.' || echo 'Install complete.')"
echo
echo "  Landing page : https://$DOMAIN"
if [[ "$WITH_CHAT" == "1" ]]; then
  echo "  Web chat     : https://$CHAT_DOMAIN"
fi
echo "  Config       : $CADDY_CONFIG   (backed up if it existed)"
echo "  Site files   : $CURRENT_LINK -> $RELEASE"
echo "  Notes        : $SITE_ROOT/README"
echo
echo "Still to do on your side:"
echo "  1. DNS  A  $DOMAIN       -> public IP of this machine"
if [[ "$WITH_CHAT" == "1" ]]; then
  echo "     DNS  A  $CHAT_DOMAIN  -> public IP of this machine"
fi
echo "     (or a wildcard A *.$DOMAIN covering both)"
echo "  2. Forward TCP 80 + 443 to this machine if it's behind a router."
echo "  3. Caddy already holds the certificates; first https visit takes ~30s."
if [[ "${warn_chat:-0}" == "1" ]]; then
  echo "  4. NOTE: toki-web failed to start — see: journalctl -u toki-web"
fi
if [[ "$WITH_TOKI" == "1" ]]; then
  echo "  4. Optional: install a model for the live chat (see $SITE_ROOT/README)."
fi
