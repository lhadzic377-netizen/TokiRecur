#!/usr/bin/env python3
"""Serve the built site the way the production server does.

The site is deployed behind Caddy, which resolves a URL by trying the path
itself, then the path with ``.html``, then the path as a directory index.
That is what makes ``/privacy``, ``/terms`` and ``/docs/install`` work.
A plain static server does not do that, so opening the site through one
gives you a 404 for pages that exist.

This script serves ``dist/site`` with the same rules, so what you look at
locally is the same set of addresses the server hands out.

Usage:
    python3 scripts/serve-site.py                    # http://127.0.0.1:8903
    python3 scripts/serve-site.py --port 9000
    python3 scripts/serve-site.py --root dist/site --no-browser
"""

from __future__ import annotations

import argparse
import http.server
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import unquote, urlparse

TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".avif": "image/avif",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
    ".xml": "application/xml; charset=utf-8",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    root: Path
    quiet: bool = False

    def translate_path(self, path: str) -> str:
        """Resolve like Caddy: path, path.html, path/index.html."""
        parsed = urlparse(path)
        clean = unquote(parsed.path).lstrip("/")
        candidate = self.root / clean
        if candidate.is_file():
            return str(candidate)
        if not Path(clean).suffix:
            for sibling in (candidate.with_suffix(".html"), candidate / "index.html"):
                if sibling.is_file():
                    return str(sibling)
        return str(candidate)

    def send_head(self):
        path = self.translate_path(self.path)
        file = Path(path)
        if file.is_file():
            return super().send_head()
        # A real 404: the page exists, the status does not.
        not_found = self.root / "404.html"
        if not_found.is_file():
            body = not_found.read_bytes()
            self.send_response(404)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return __import__("io").BytesIO(body)
        return super().send_head()

    def guess_type(self, path):
        ext = Path(path).suffix.lower()
        if ext in TYPES:
            return TYPES[ext]
        return super().guess_type(path)

    def log_message(self, fmt, *args):  # noqa: A003 - stdlib signature
        if not self.quiet:
            sys.stderr.write("  %s\n" % (fmt % args))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dist/site")
    ap.add_argument("--port", type=int, default=8903)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not (root / "index.html").is_file():
        print(f"error: {root} has no index.html. Run: python3 scripts/build-site.py")
        return 1

    Handler.root = root
    Handler.quiet = args.quiet

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with Server((args.host, args.port), Handler) as httpd:
        url = f"http://{args.host}:{args.port}/"
        print(f"serving {root}")
        print(f"  {url}")
        print("  clean URLs work here the way they do on the server")
        if not args.no_browser:
            threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
