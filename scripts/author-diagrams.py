#!/usr/bin/env python3
"""Author the six mechanism diagrams for the feature panels.

These are real diagrams of how the product works, drawn in the same visual
language as the page: hairline structure, the brand yellow for the cache
path, the teal for what the model computed, muted ink for labels. They are
written into ``site/designs/diagrams/*.svg`` and inlined into the landing
page by ``scripts/build-site.py`` so they can be themed and animated.

    python3 scripts/author-diagrams.py
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "site" / "designs" / "diagrams"

STRUCT = "#3a4450"        # hairline structure
INK = "#9199a3"           # labels
BRAND = "#ffd60a"         # the cache path
HIT = "#47e0e2"           # what the model computed
DIM = "#5b6672"           # inactive structure

MONO = "ui-monospace, SFMono-Regular, Menlo, monospace"


def wrap(body: str, w: int = 248, h: int = 150) -> str:
    return (
        f'<svg class="dg" viewBox="0 0 {w} {h}" fill="none" '
        f'aria-hidden="true" focusable="false" '
        f'xmlns="http://www.w3.org/2000/svg">{body}</svg>'
    )


def label(x: float, y: float, text: str, colour: str = INK, size: float = 8.5, anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" fill="{colour}" font-family="{MONO}" '
        f'font-size="{size}" text-anchor="{anchor}">{text}</text>'
    )


def blocks(x: float, y: float, n: int, w: float = 15, h: float = 20, gap: float = 3,
            hot: int = 0, cool: int = 0, from_end: bool = False) -> str:
    """A row of token blocks. `hot` blocks are the cache path, `cool` computed."""
    out = []
    for i in range(n):
        bx = x + i * (w + gap)
        idx = (n - 1 - i) if from_end else i
        if idx < hot:
            fill, stroke = "rgba(255,214,10,0.22)", BRAND
        elif idx < hot + cool:
            fill, stroke = "rgba(71,224,226,0.16)", HIT
        else:
            fill, stroke = "rgba(255,255,255,0.05)", STRUCT
        out.append(
            f'<rect x="{bx:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="2.5" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
        )
    return "".join(out)


def arrow(x1: float, y1: float, x2: float, y2: float, colour: str = STRUCT, dash: str = "") -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{colour}" stroke-width="1.2" '
        f'stroke-linecap="round"{d}/>'
        f'<path d="M{x2} {y2} l-4 -2.4 v4.8 z" fill="{colour}"/>'
    )


# ── 1. drop-in cache ────────────────────────────────────────────────────────
def dg_cache() -> str:
    body = f'''
    {label(6, 14, "same prompt, twice")}
    {blocks(6, 24, 10, hot=7, cool=2)}
    {arrow(86, 52, 86, 62, BRAND)}
    {label(86, 74, "1st computed", BRAND, 8, "middle")}
    {arrow(158, 52, 158, 62, DIM, dash="3 3")}
    {label(158, 74, "2nd replayed", DIM, 8, "middle")}
    <rect x="34" y="78" width="180" height="52" rx="10" fill="rgba(255,214,10,0.05)" stroke="{BRAND}" stroke-width="1"/>
    {label(44, 96, "one living KV pool", BRAND)}
    {label(44, 112, "shared by every client", INK)}
    <circle cx="196" cy="100" r="4" fill="{HIT}"/>
    <circle cx="196" cy="116" r="4" fill="{BRAND}"/>
    '''
    return wrap(body)


# ── 2. live reuse ──────────────────────────────────────────────────────────
def dg_fanout() -> str:
    body = f'''
    {label(6, 14, "three clients, one cache")}
    <rect x="6" y="26" width="70" height="24" rx="6" fill="rgba(255,255,255,0.04)" stroke="{STRUCT}"/>
    {label(41, 42, "opencode", INK, 8.5, "middle")}
    <rect x="6" y="58" width="70" height="24" rx="6" fill="rgba(255,255,255,0.04)" stroke="{STRUCT}"/>
    {label(41, 74, "claude code", INK, 8.5, "middle")}
    <rect x="6" y="90" width="70" height="24" rx="6" fill="rgba(255,255,255,0.04)" stroke="{STRUCT}"/>
    {label(41, 106, "hermes", INK, 8.5, "middle")}
    {arrow(78, 38, 122, 62, STRUCT)}
    {arrow(78, 70, 122, 70, BRAND)}
    {arrow(78, 102, 122, 78, STRUCT)}
    <rect x="124" y="44" width="112" height="46" rx="10" fill="rgba(255,214,10,0.05)" stroke="{BRAND}" stroke-width="1"/>
    {label(134, 62, "KV pool", BRAND)}
    {label(134, 78, "prefix written once", INK, 8)}
    {arrow(180, 92, 180, 112, DIM, dash="3 3")}
    {label(180, 124, "replayed next", DIM, 8, "middle")}
    '''
    return wrap(body)


# ── 3. save & restore ──────────────────────────────────────────────────────
def dg_pools() -> str:
    body = f'''
    {label(6, 14, "hot prefixes keep a slot")}
    {blocks(6, 24, 9, w=8, h=10, gap=2, hot=6)}
    {label(6, 46, "incoming", DIM, 7.5)}
    {arrow(98, 34, 108, 34, BRAND)}
    <rect x="110" y="20" width="124" height="24" rx="5" fill="rgba(255,214,10,0.10)" stroke="{BRAND}" stroke-width="1"/>
    {label(118, 35, "slot 0 · reserved", BRAND, 8)}
    <rect x="110" y="52" width="124" height="24" rx="5" fill="rgba(255,255,255,0.03)" stroke="{STRUCT}"/>
    {label(118, 67, "slot 1", INK, 8)}
    <rect x="110" y="84" width="124" height="24" rx="5" fill="rgba(255,255,255,0.03)" stroke="{STRUCT}"/>
    {label(118, 99, "slot 2", INK, 8)}
    {label(6, 76, "never spilled", HIT, 8)}
    {label(6, 88, "while it stays hot", DIM, 7.5)}
    '''
    return wrap(body)


# ── 4. ram-aware ───────────────────────────────────────────────────────────
def dg_ram() -> str:
    body = f'''
    {label(6, 14, "the context that actually fits")}
    <rect x="6" y="28" width="34" height="46" rx="5" fill="rgba(255,255,255,0.05)" stroke="{STRUCT}"/>
    {label(23, 48, "GGUF", INK, 8, "middle")}
    {label(23, 60, "header", INK, 7.5, "middle")}
    {arrow(42, 51, 70, 51, STRUCT)}
    <rect x="72" y="28" width="164" height="26" rx="6" fill="rgba(255,255,255,0.04)" stroke="{STRUCT}"/>
    <rect x="72" y="28" width="118" height="26" rx="6" fill="rgba(255,214,10,0.16)" stroke="{BRAND}"/>
    {label(80, 45, "65k ctx", BRAND)}
    {label(206, 45, "free", DIM, 8, "middle")}
    <rect x="72" y="64" width="164" height="10" rx="5" fill="rgba(255,255,255,0.05)"/>
    <rect x="72" y="64" width="118" height="10" rx="5" fill="rgba(255,214,10,0.5)"/>
    {label(72, 96, "KV pool sized to this machine,", INK)}
    {label(72, 108, "not to the model's wish", INK)}
    <path d="M72 122 h118" stroke="{BRAND}" stroke-width="1.2"/>
    <path d="M72 118 v8 M190 118 v8" stroke="{BRAND}" stroke-width="1.2"/>
    '''
    return wrap(body)


# ── 5. benchmark harness ───────────────────────────────────────────────────
def dg_bench() -> str:
    rows = [("realistic_mixed", 0.88), ("summarization", 0.95), ("code_review", 0.97), ("agent_100", 0.99)]
    body = [label(6, 14, "cache hit rate, measured")]
    y = 30
    for name, rate in rows:
        body.append(label(6, y + 8, name, INK, 7.5))
        body.append(f'<rect x="104" y="{y}" width="116" height="11" rx="3" fill="rgba(255,255,255,0.05)"/>')
        body.append(
            f'<rect x="104" y="{y}" width="{116 * rate:.1f}" height="11" rx="3" '
            f'fill="rgba(71,224,226,0.35)" stroke="{HIT}" stroke-width="1"/>'
        )
        body.append(label(240, y + 9, f"{int(rate * 100)}%", HIT, 8, "end"))
        y += 26
    body.append(label(104, 138, "hit rate per workload", DIM, 7.5))
    return wrap("".join(body))


# ── 6. tui + web ───────────────────────────────────────────────────────────
def dg_interfaces() -> str:
    body = f'''
    {label(6, 14, "two interfaces, one daemon")}
    <rect x="6" y="26" width="104" height="76" rx="7" fill="rgba(255,255,255,0.03)" stroke="{STRUCT}"/>
    <rect x="6" y="26" width="104" height="15" rx="7" fill="rgba(255,255,255,0.05)"/>
    <circle cx="16" cy="33.5" r="2.4" fill="{BRAND}"/>
    {label(26, 37, "toki", INK, 7.5)}
    {blocks(14, 48, 9, w=8, h=9, gap=2, hot=5)}
    {label(14, 74, "full-screen TUI", INK, 7.5)}
    {label(14, 86, "toki tui", DIM, 7)}
    <rect x="132" y="26" width="104" height="82" rx="7" fill="rgba(255,255,255,0.03)" stroke="{STRUCT}"/>
    <rect x="132" y="26" width="104" height="15" rx="7" fill="rgba(255,255,255,0.05)"/>
    <circle cx="142" cy="33.5" r="2.4" fill="{HIT}"/>
    {label(152, 37, "api", INK, 7.5)}
    {label(142, 56, "POST /v1/chat/completions", DIM, 6)}
    {label(142, 68, "POST /v1/completions", DIM, 6)}
    {label(142, 90, "OpenAI-compatible API", INK, 7.5)}
    {label(142, 103, "any client", DIM, 7)}
    {arrow(112, 64, 130, 64, BRAND)}
    '''
    return wrap(body)


DIAGRAMS = {
    "dg-cache": dg_cache,
    "dg-fanout": dg_fanout,
    "dg-pools": dg_pools,
    "dg-ram": dg_ram,
    "dg-bench": dg_bench,
    "dg-interfaces": dg_interfaces,
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in DIAGRAMS.items():
        svg = fn()
        (OUT / f"{name}.svg").write_text(svg, encoding="utf-8")
        print(f"{name:18} {len(svg):>6} bytes")
    print(f"\n{len(DIAGRAMS)} diagrams in {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
