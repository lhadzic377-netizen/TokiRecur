#!/usr/bin/env bash
#
# Build the standalone toki-site distribution zip.
#
#   Usage:  ./scripts/build-site-zip.sh
#
# Produces dist/toki-site.zip containing:
#   toki-site/
#     install.sh              # self-contained server installer (sudo ./install.sh)
#     Caddyfile.template
#     site/                   # index.html, style.css, app.js, favicon.svg, docs/
#
# On the server laptop: unzip, then `sudo ./install.sh --domain ... --email ...`
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/dist/toki-site.zip"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

DEST="$BUILD/toki-site"
mkdir -p "$DEST/site/docs" "$DEST/site/img" "$REPO/dist"

cp "$REPO/site/index.html" "$REPO/site/style.css" \
   "$REPO/site/app.js" "$REPO/site/favicon.png" "$DEST/site/"
mkdir -p "$DEST/site/img"
cp "$REPO/site/img/"*.png "$DEST/site/img/"
cp -r "$REPO/site/docs/"*.html "$REPO/site/docs/"*.css "$REPO/site/docs/"*.js \
   "$DEST/site/docs/"
cp "$REPO/scripts/install-site.sh" "$DEST/install.sh"
cp "$REPO/scripts/Caddyfile.template" "$DEST/"
chmod +x "$DEST/install.sh"

rm -f "$OUT"
( cd "$BUILD" && zip -qr "$OUT" toki-site )

echo "built $OUT"
echo "  size   : $(du -h "$OUT" | cut -f1)"
echo "  sha256 : $(sha256sum "$OUT" | cut -d' ' -f1)"
echo
echo "deploy on the server laptop:"
echo "  unzip toki-site.zip && sudo ./toki-site/install.sh --domain tokirecur.com --email you@example.com"
