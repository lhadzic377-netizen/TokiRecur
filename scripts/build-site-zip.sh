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
#     site/                   # built public site (no server.py, no data)
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/dist/toki-site.zip"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

DEST="$BUILD/toki-site"
mkdir -p "$DEST" "$REPO/dist"

"$REPO/scripts/build-site.py"

cp "$REPO/scripts/install-site.sh" "$DEST/install.sh"
cp "$REPO/scripts/Caddyfile.template" "$DEST/"
cp -a "$REPO/dist/site" "$DEST/site"
chmod +x "$DEST/install.sh"

rm -f "$OUT"
( cd "$BUILD" && zip -qr "$OUT" toki-site )

echo "built $OUT"
echo "  size   : $(du -h "$OUT" | cut -f1)"
echo "  sha256 : $(sha256sum "$OUT" | cut -d' ' -f1)"
echo
echo "deploy on the server laptop:"
echo "  unzip toki-site.zip && sudo ./toki-site/install.sh --domain tokirecur.com --email you@example.com"
