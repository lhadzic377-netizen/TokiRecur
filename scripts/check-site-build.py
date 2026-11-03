#!/usr/bin/env python3
"""Check the built site for the mistakes a static deploy can hide.

Run after ``scripts/build-site.py``.  This is deliberately independent of the
builder: it reads only ``dist/site`` and fails on a missing page, a reference
that escaped the build, duplicated ids, or a source file that should never be
public.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

DIST = Path(__file__).resolve().parents[1] / "dist" / "site"
REQUIRED = (
    "index.html",
    "pricing.html",
    "404.html",
    "style.css",
    "fonts.css",
    "head.js",
    "app.js",
    "tui.js",
    "sitemap.xml",
    "robots.txt",
    "fonts/geist-latin.woff2",
    "fonts/geist-mono-latin.woff2",
    "docs/index.html",
    "docs/install.html",
    "docs/quickstart.html",
)
FORBIDDEN = (
    "server.py",
    "wishlist",
    "/data/",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
)


def main() -> int:
    if not DIST.is_dir():
        print(f"error: {DIST} does not exist; run scripts/build-site.py first", file=sys.stderr)
        return 2

    errors: list[str] = []

    for name in REQUIRED:
        if not (DIST / name).is_file():
            errors.append(f"missing required file: {name}")

    for path in DIST.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(DIST).as_posix()
        if path.suffix.lower() in {".html", ".css", ".js", ".txt", ".xml"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for bad in FORBIDDEN:
                if bad in text:
                    errors.append(f"{rel}: contains forbidden text {bad!r}")

    pages = sorted(DIST.rglob("*.html"))
    total_ids = 0
    for page in pages:
        rel = page.relative_to(DIST).as_posix()
        text = page.read_text(encoding="utf-8")
        if '<html lang="en">' not in text:
            errors.append(f"{rel}: missing <html lang=\"en\">")
        if "<script>" in text or re.search(r"<script\s+[^>]*>[^<]", text):
            errors.append(f"{rel}: has an inline script (CSP blocks it)")
        if "rel=\"canonical\"" not in text and page.name != "404.html":
            errors.append(f"{rel}: missing canonical link")

        page_ids: set[str] = set()
        for match in re.finditer(r'\bid="([^"]+)"', text):
            value = match.group(1)
            if value in page_ids:
                errors.append(f"{rel}: duplicate id {value!r}")
            page_ids.add(value)
            total_ids += 1

        for match in re.finditer(r'(?:href|src)="(/(?!/)[^"]*)"', text):
            raw = match.group(1).split("#", 1)[0].split("?", 1)[0]
            target = DIST / raw.lstrip("/")
            if raw.endswith("/"):
                target = target / "index.html"
            if target.is_dir():
                target = target / "index.html"
            if not target.exists() and not target.suffix:
                target = target.with_suffix(".html")
            if not target.exists():
                errors.append(f"{rel}: broken local reference {match.group(1)}")

    sitemap = (DIST / "sitemap.xml").read_text(encoding="utf-8")
    locations = re.findall(r"<loc>([^<]+)</loc>", sitemap)
    if not locations:
        errors.append("sitemap.xml: no <loc> entries")
    else:
        parts = urlsplit(locations[0])
        origin = f"{parts.scheme}://{parts.netloc}"
        paths = {urlsplit(loc).path for loc in locations}
        for path in ("/", "/pricing", "/docs/"):
            if path not in paths:
                errors.append(f"sitemap.xml: missing {origin}{path}")
        for loc in locations:
            if not loc.startswith(origin):
                errors.append(f"sitemap.xml: {loc} is not on {origin}")

    if errors:
        print("site build check failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    print(f"site build check ok: {len(pages)} pages, {total_ids} ids")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
