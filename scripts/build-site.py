#!/usr/bin/env python3
"""Build the public Toki Recur site.

The source lives in ``site/``.  This build produces ``dist/site/`` with only
the files that may be served, content-addressed asset URLs, a sitemap, and a
robots file.  It also fails when an internal link or asset reference does not
resolve, so the installer cannot ship a page with a missing image again.

Usage:
    python3 scripts/build-site.py
    python3 scripts/build-site.py --site-url https://example.com
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "site"
DIST = REPO / "dist" / "site"

PUBLIC_FILES = (
    "index.html",
    "pricing.html",
    "404.html",
    "privacy.html",
    "terms.html",
    "style.css",
    "landing.css",
    "pricing.css",
    "fonts.css",
    "head.js",
    "app.js",
    "mechanism.js",
    "motion.js",
    "tui.js",
    "favicon.png",
    "favicon.ico",
    "favicon.svg",
    "robots.txt",
)

PUBLIC_DIRS = ("img", "docs", "assets", "fonts")

DIAGRAMS = SITE / "designs" / "diagrams"
DIAGRAM_RE = re.compile(r"<!--\s*DIAGRAM:(?P<name>[a-z0-9-]+)\s*-->")

ASSET_SUFFIXES = {
    ".css",
    ".js",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".avif",
    ".gif",
    ".svg",
    ".ico",
    ".woff",
    ".woff2",
    ".txt",
}

HTML_SUFFIXES = {".html"}
LOCAL_URL_RE = re.compile(
    r"""(?P<attr>\b(?:href|src|poster)=")(?P<url>/[^"#?]*)(?:\?v=[^"]*)?(?P<close>")"""
)
CSS_URL_RE = re.compile(
    r"""url\((?P<quote>["']?)(?P<url>/[^)"']+)(?P=quote)\)"""
)
EXTERNAL_PREFIXES = (
    "http://",
    "https://",
    "//",
    "mailto:",
    "tel:",
    "data:",
    "javascript:",
    "#",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


def copy_tree() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)

    for name in PUBLIC_FILES:
        src = SITE / name
        if src.is_file():
            shutil.copy2(src, DIST / name)

    for name in PUBLIC_DIRS:
        src = SITE / name
        if not src.is_dir():
            continue
        for item in src.rglob("*"):
            if item.is_dir() or "__pycache__" in item.parts:
                continue
            if item.suffix.lower() not in ASSET_SUFFIXES | HTML_SUFFIXES:
                continue
            if item.name.startswith("."):
                continue
            rel = item.relative_to(SITE)
            dest = DIST / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)


def inline_diagrams() -> None:
    """Replace ``<!--DIAGRAM:name-->`` placeholders with the authored SVG.

    The diagrams live in the design source, not in the served tree: they are
    inlined so they inherit the page's colour, scale with the text, and cost
    no extra requests.
    """
    index = DIST / "index.html"
    if not index.is_file():
        return
    text = index.read_text(encoding="utf-8")

    def repl(match: re.Match[str]) -> str:
        name = match.group("name")
        path = DIAGRAMS / f"{name}.svg"
        if not path.is_file():
            raise SystemExit(f"error: missing diagram {path.relative_to(REPO)}")
        svg = path.read_text(encoding="utf-8").strip()
        svg = re.sub(r"^<\?xml[^>]*>\s*", "", svg)
        return svg

    text, count = DIAGRAM_RE.subn(repl, text)
    leftover = DIAGRAM_RE.findall(text)
    if leftover:
        raise SystemExit(f"error: unknown diagram placeholder(s): {leftover}")
    index.write_text(text, encoding="utf-8")
    if count:
        print(f"  inlined: {count} diagrams")


def asset_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in DIST.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in ASSET_SUFFIXES:
            continue
        web = "/" + path.relative_to(DIST).as_posix()
        mapping[web] = sha(path)
    return mapping


def rewrite_css(mapping: dict[str, str]) -> None:
    for path in DIST.rglob("*.css"):
        text = path.read_text(encoding="utf-8")
        changed = False

        def repl(match: re.Match[str]) -> str:
            nonlocal changed
            url = match.group("url")
            digest = mapping.get(url)
            if not digest:
                return match.group(0)
            changed = True
            return f'url("{url}?v={digest}")'

        text = CSS_URL_RE.sub(repl, text)
        if changed:
            path.write_text(text, encoding="utf-8")


def page_url(path: Path) -> str:
    """The public URL of a built page (clean URLs where the server has them)."""
    rel = path.relative_to(DIST).as_posix()
    if rel == "index.html":
        return "/"
    if rel == "pricing.html":
        return "/pricing"
    if rel == "docs/index.html":
        return "/docs/"
    return "/" + rel


def rewrite_html(mapping: dict[str, str], site_url: str) -> None:
    base = site_url.rstrip("/")
    for path in DIST.rglob("*.html"):
        text = path.read_text(encoding="utf-8")

        def repl(match: re.Match[str]) -> str:
            url = match.group("url")
            digest = mapping.get(url)
            if not digest:
                return match.group(0)
            return f'{match.group("attr")}{url}?v={digest}{match.group("close")}'

        text = LOCAL_URL_RE.sub(repl, text)

        # The canonical URL, the share URL and the share image all follow the
        # domain the artifact is being deployed to. A build that says
        # tokirecur.com while being served from a staging host tells every
        # crawler and every share card the wrong thing.
        canonical = f"{base}{page_url(path)}"
        text = re.sub(
            r'<link rel="canonical" href="[^"]*">',
            f'<link rel="canonical" href="{canonical}">',
            text,
        )
        text = re.sub(
            r'<meta property="og:url" content="[^"]*">',
            f'<meta property="og:url" content="{canonical}">',
            text,
        )
        text = re.sub(
            r'<meta property="og:image" content="[^"]*">',
            f'<meta property="og:image" content="{base}/img/toki-bear.jpg">',
            text,
        )
        text = re.sub(
            r'<meta name="twitter:image" content="[^"]*">',
            f'<meta name="twitter:image" content="{base}/img/toki-bear.jpg">',
            text,
        )

        path.write_text(text, encoding="utf-8")


def resolve_target(page: Path, raw: str) -> Path | None:
    url = raw.split("#", 1)[0].split("?", 1)[0]
    if not url:
        return page
    if url.startswith(EXTERNAL_PREFIXES):
        return page
    if url.startswith("/"):
        target = DIST / url.lstrip("/")
    else:
        target = page.parent / url
    if url.endswith("/"):
        target = target / "index.html"
    if target.is_dir():
        target = target / "index.html"
    if not target.exists() and not target.suffix:
        html_target = target.with_suffix(".html")
        if html_target.exists():
            return html_target
    return target


def validate_links() -> list[str]:
    missing: list[str] = []
    for page in sorted(DIST.rglob("*.html")):
        text = page.read_text(encoding="utf-8")
        for match in re.finditer(r'(?:href|src)="([^"]+)"', text):
            raw = match.group(1).strip()
            if not raw or raw.startswith(EXTERNAL_PREFIXES):
                continue
            target = resolve_target(page, raw)
            if target is None or not target.exists():
                missing.append(f"{page.relative_to(DIST)}: {raw}")
    return missing


def write_discovery(site_url: str) -> None:
    base = site_url.rstrip("/")
    pages = ["/", "/pricing", "/docs/"]
    docs_dir = DIST / "docs"
    if docs_dir.is_dir():
        for page in sorted(docs_dir.glob("*.html")):
            if page.name == "index.html":
                continue
            pages.append(f"/docs/{page.name}")
    for legal in ("privacy.html", "terms.html"):
        if (DIST / legal).is_file():
            pages.append(f"/{legal}")

    sitemap = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for page in pages:
        sitemap.append(f"  <url><loc>{base}{page}</loc></url>")
    sitemap.append("</urlset>")
    (DIST / "sitemap.xml").write_text("\n".join(sitemap) + "\n", encoding="utf-8")

    robots = [
        "User-agent: *",
        "Allow: /",
        "Disallow: /api/",
        f"Sitemap: {base}/sitemap.xml",
        "",
    ]
    (DIST / "robots.txt").write_text("\n".join(robots), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--site-url",
        default="https://tokirecur.com",
        help="canonical site URL used in robots.txt and sitemap.xml",
    )
    args = parser.parse_args()

    if not (SITE / "index.html").is_file():
        print(f"error: no site source at {SITE}", file=sys.stderr)
        return 2

    copy_tree()
    inline_diagrams()

    # CSS can reference images.  Resolve those first, then hash the rewritten
    # CSS so every HTML reference points at the exact bytes that were built.
    initial = asset_map()
    rewrite_css(initial)
    mapping = asset_map()
    rewrite_html(mapping, args.site_url)

    missing = validate_links()
    if missing:
        print("error: unresolved internal references:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        return 1

    write_discovery(args.site_url)

    files = [p for p in DIST.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    print(f"built {DIST}")
    print(f"  files : {len(files)}")
    print(f"  size  : {total / 1024:.1f} KiB")
    print(f"  assets: {len(mapping)} fingerprinted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
