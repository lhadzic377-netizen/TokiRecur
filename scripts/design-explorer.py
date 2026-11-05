#!/usr/bin/env python3
"""Assemble the design candidates.

Twenty visual directions, one shared content. Every candidate is a real
page: the production ``site/index.html`` with a different visual system
layered on top, so the twenty can be rendered, measured and compared on
equal terms — same words, same claims, same accessibility.

    python3 scripts/design-explorer.py            # build dist/designs/<slug>/
    python3 scripts/design-explorer.py --list     # show the candidates

The output is a static site per candidate: ``dist/designs/<slug>/index.html``
plus the assets it needs. Serve ``dist/designs/<slug>`` and open index.html.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "site"
DESIGNS = SITE / "designs"
DIST = REPO / "dist" / "designs"
BUILD = REPO / "dist" / "site"

# Each candidate: slug -> (title, one-line thesis)
CANDIDATES: dict[str, tuple[str, str]] = {
    "terminal-native": ("Terminal native", "The page is a TUI. Mono, amber, a window frame around everything."),
    "instrument-editorial": ("Instrument editorial", "A magazine about a system: serif display, ink and oxblood."),
    "swiss-graphite": ("Swiss graphite", "Twelve columns, one red, no decoration."),
    "dashboard-console": ("Dashboard console", "The landing page as an instrument panel."),
    "aurora-depth": ("Aurora depth", "A rendered field of light behind frosted glass."),
    "blueprint-cyan": ("Blueprint cyan", "Technical drawing: measured, dimensioned, annotated."),
    "bauhaus-primary": ("Bauhaus primary", "Primary colours, blocks and circles as structure."),
    "paper-light": ("Paper light", "The site in daylight: warm paper, serif text, green ink."),
    "brutal-raw": ("Brutal raw", "No radius, no shadow, no politeness."),
    "cinema-dark": ("Cinema dark", "One enormous image, one enormous line of type."),
    "dataviz-hero": ("Data-viz hero", "The hero is the measurement: what is skipped, drawn."),
    "split-console": ("Split console", "Statement on the left, live terminal on the right."),
    "swiss-minimal": ("Swiss minimal", "One column, enormous quiet, a single accent."),
    "crt-phosphor": ("CRT phosphor", "Amber phosphor, scanlines, a screen you can lean into."),
    "docs-first": ("Docs first", "The landing page as an index into the documentation."),
    "bento-modular": ("Bento modular", "Tiles of different sizes, each with its own colour."),
    "newspaper": ("Newspaper", "Columns, rules, a masthead and a dateline."),
    "neon-arcade": ("Neon arcade", "Saturated light on a purple-black field."),
    "riso-print": ("Riso print", "Two spot inks on warm stock, misregistered on purpose."),
    "ink-quiet": ("Ink quiet", "Black, Geist, one yellow — the current direction, finished."),
}


def build_one(slug: str, title: str) -> Path:
    css_path = DESIGNS / f"{slug}.css"
    if not css_path.is_file():
        raise SystemExit(f"missing design stylesheet: {css_path}")

    out = DIST / slug
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(BUILD, out)

    html = (SITE / "index.html").read_text(encoding="utf-8")

    # The candidate's own stylesheet, loaded after the production one.
    html = html.replace(
        '<link rel="stylesheet" href="/style.css?v=65">',
        '<link rel="stylesheet" href="/style.css?v=65">\n'
        '<link rel="stylesheet" href="/design.css">\n'
        '<link rel="stylesheet" href="/fonts-designs.css">',
    )
    html = html.replace('<html lang="en">', f'<html lang="en" data-design="{slug}">')
    html = html.replace(
        "<title>Toki Recur: KV cache reuse for LLM inference</title>",
        f"<title>{title} — Toki Recur</title>",
    )

    # A candidate may add its own markup: a marquee, a chart, an extra panel.
    partial = DESIGNS / f"{slug}.html"
    if partial.is_file():
        block = partial.read_text(encoding="utf-8").strip()
        html = html.replace("<!-- DESIGN:EXTRA -->", block)
        html = html.replace("<body>", f"<body>\n{block}", 1)

    (out / "index.html").write_text(html, encoding="utf-8")
    shutil.copy2(css_path, out / "design.css")

    design_fonts = DESIGNS / "fonts-designs.css"
    if design_fonts.is_file():
        shutil.copy2(design_fonts, out / "fonts-designs.css")

    extra_fonts = DESIGNS / "fonts"
    if extra_fonts.is_dir():
        target = out / "fonts"
        target.mkdir(exist_ok=True)
        for font in extra_fonts.glob("*.woff2"):
            shutil.copy2(font, target / font.name)

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list the candidates")
    parser.add_argument("--only", action="append", default=[], help="build only these slugs")
    args = parser.parse_args()

    if args.list:
        for i, (slug, (title, thesis)) in enumerate(CANDIDATES.items(), 1):
            print(f"{i:2}. {slug:22} {title:22} {thesis}")
        return 0

    if not BUILD.is_dir():
        print("error: dist/site is missing; run scripts/build-site.py first", file=sys.stderr)
        return 2

    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)

    slugs = args.only or list(CANDIDATES)
    for slug in slugs:
        if slug not in CANDIDATES:
            print(f"error: unknown candidate {slug!r}", file=sys.stderr)
            return 2
        title = CANDIDATES[slug][0]
        out = build_one(slug, title)
        size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        print(f"built {out.relative_to(REPO)}  ({size / 1024:.0f} KiB)")

    print(f"\n{len(slugs)} candidates in {DIST.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
