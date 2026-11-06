#!/usr/bin/env python3
"""Drop the CSS for the sections the redesign replaced.

The landing page was rebuilt around a new set of sections.  The old ones —
the hero, the video, the feature list, the architecture diagram, the demo
terminal, the backend strip — are gone from the markup, but their rules were
still shipped, so the stylesheet carried code for a page that no longer
existed.

This script cuts those blocks out by their banner comments and removes the
stray responsive rules that pointed at the same classes.  It is idempotent:
running it twice changes nothing the second time.

Usage:
    python3 scripts/prune-css.py site/style.css site/landing.css
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# (start marker, end marker) — the text between them goes.
BLOCKS_STYLE = [
    ("/* --- hero ", "/* --- the reuse diagram "),
    ("/* --- stats ", "/* --- video "),
    ("/* --- video ", "/* --- what ships "),
    ("/* --- what ships ", "/* --- cards (pricing) "),
    ("/* --- steps ", "/* --- honesty "),
    ("/* --- honesty ", "/* --- meet toki "),
    ("/* --- meet toki ", "/* --- benchmarks "),
    ("/* --- architecture ", "/* --- final cta "),
    ("/* --- final cta ", "/* --- footer "),
    ("/* --- the demo terminal ", ".tui {"),
    ("/* ── hero: the one screen that has to land ", "/* ── the backend strip "),
    ("/* ── the backend strip ", "/* ── the mechanism: the prefix matcher "),
    ("/* ── the demo section gets a raised surface ", "/* ── benchmarks read as a printed ledger "),
]

# Single rules and rule groups that are dead but sit inside live blocks.
LOOSE_STYLE = [
    (
        """/* The status dot in the terminal bar is lit. */
.terminal-dot {
  background: radial-gradient(circle at 35% 30%, var(--brand-lift), var(--brand) 55%, var(--brand-deep));
  box-shadow: 0 0 10px oklch(0.885 0.18 95 / 0.7);
}

""",
        "",
    ),
    (
        """/* The one picture that explains the product: a prompt is computed once and
   every later request reads it back. */
.reuse-loop {
  position: relative;
  margin: 0;
  padding: var(--s5) var(--s5) var(--s4);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--panel);
}

.reuse-loop svg {
  width: 100%;
  height: auto;
  overflow: visible;
}

.reuse-loop figcaption {
  margin-top: var(--s4);
  padding-top: var(--s4);
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 0.88rem;
  line-height: 1.6;
}

""",
        "",
    ),
    (
        """@media (max-width: 1080px) {
  .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .how-grid { grid-template-columns: minmax(0, 1fr); }
  .reuse-loop { max-width: 520px; }
}

""",
        "",
    ),
    (
        """  .hero {
    grid-template-columns: minmax(0, 1fr);
    text-align: left;
  }

  .hero-visual { max-width: 560px; }

  /* A phone cannot show a whole terminal legibly, so it shows the part that
     identifies it: the wordmark and the model in use. */
  .hero-visual img {
    aspect-ratio: 16 / 6;
    object-fit: cover;
    object-position: top center;
  }

  .meet-toke,
  .honesty-body { grid-template-columns: minmax(0, 1fr); }

  ol.steps > li { grid-template-columns: 52px minmax(0, 1fr); }

""",
        "",
    ),
    (
        """  .feature-grid { grid-template-columns: minmax(0, 1fr); }
  .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s4); }
  .stat .num { font-size: var(--step-1); }
  .cta .btn { flex: 1 1 100%; justify-content: center; }
  ol.steps > li { grid-template-columns: minmax(0, 1fr); }
  ol.steps > li::before { margin-bottom: var(--s2); }
""",
        """  .cta .btn { flex: 1 1 100%; justify-content: center; }
""",
    ),
    (
        """@media (max-width: 620px) {
  .backend-strip {
    padding: var(--s3) 0;
  }
}

""",
        "",
    ),
    (
        """/* The backend strip: a slow scroll, and it stops when it is hovered. */
@media (prefers-reduced-motion: reduce) {
  .loop-arc,
  .loop-arc-back,
  .arch-arrow path {
    animation: none;
  }
  .backend-track {
    animation: none;
    transform: none;
  }
  .feature::after {
    display: none;
  }
}
""",
        """@media (prefers-reduced-motion: reduce) {
  .loop-arc,
  .loop-arc-back {
    animation: none;
  }
}
""",
    ),
    (
        """/* The backend strip picks up the light behind it. */
.backend-strip {
  background: linear-gradient(180deg, oklch(0.185 0.02 264 / 0.9), oklch(0.155 0.018 268 / 0.9));
  border-top: 1px solid oklch(0.4 0.024 262 / 0.34);
  border-bottom: 1px solid oklch(0.4 0.024 262 / 0.34);
}

""",
        "",
    ),
    (
        """/* The demo band is a raised surface, not a tint. */
#demo::before {
  background: linear-gradient(180deg, oklch(0.185 0.02 264 / 0.92), oklch(0.165 0.019 266 / 0.92));
  border-top: 1px solid oklch(0.4 0.024 262 / 0.3);
  border-bottom: 1px solid oklch(0.4 0.024 262 / 0.3);
}

""",
        "",
    ),
    (
        """/* The architecture arrows carry traffic: a dash that moves. */
.arch-arrow path {
  stroke-dasharray: 4 5;
  animation: loop-dash 1.1s linear infinite;
}

""",
        "",
    ),
    (
        """/* The play button breathes, once, until it is clicked. */
.video-play {
  transition: transform 0.3s var(--ease), box-shadow 0.3s var(--ease);
}

.video-poster:hover .video-play {
  transform: translate(-50%, -50%) scale(1.06);
  box-shadow: 0 0 0 14px rgba(255, 214, 10, 0.08);
}

""",
        "",
    ),
    (
        """/* The cards light up where the pointer is. --mx/--my are written by
   motion.js; the gradient is painted by the compositor. */
.feature {
  position: relative;
  isolation: isolate;
}

.feature::after {
  content: "";
  position: absolute;
  inset: 0;
  z-index: -1;
  border-radius: inherit;
  background: radial-gradient(
    240px circle at var(--mx, 50%) var(--my, 50%),
    rgba(255, 214, 10, 0.07),
    transparent 62%
  );
  opacity: 0;
  transition: opacity 0.3s var(--ease);
  pointer-events: none;
}

.feature:hover::after {
  opacity: 1;
}

""",
        "",
    ),
    (
        """  .terminal-keys { display: none; }
  .demo-real { overflow-x: auto; }
  .demo-real img { min-width: 760px; }
  .demo-real .swipe { display: block; margin-top: 6px; color: var(--muted); }
  .demo-start .on-wide { display: none; }
  .demo-start .on-narrow { display: inline; }
  .demo-start { font-size: 0.6rem; }
""",
        "",
    ),
    (
        """  .js-ready .reveal { opacity: 1; transform: none; }
  .hero-mark { animation: none; }
  .loop-arc { animation: none; }""",
        """  .js-ready .reveal { opacity: 1; transform: none; }
  .loop-arc { animation: none; }""",
    ),
    (
        """  .card,
  .arch-node,
  ol.steps > li,
  .table-wrap { border-color: var(--line-strong); }""",
        """  .card,
  .table-wrap { border-color: var(--line-strong); }""",
    ),
    (
        """/* The hero sits in a pool of its own light: violet from the corner the
   terminal is in, gold from the headline. */
.hero::before {
  content: "";
  position: absolute;
  inset: -30% -12% auto -12%;
  height: 130%;
  z-index: -1;
  pointer-events: none;
  background:
    radial-gradient(40% 46% at 88% 16%, oklch(0.52 0.185 290 / 0.18), transparent 66%),
    radial-gradient(26% 30% at 14% 40%, oklch(0.885 0.18 95 / 0.05), transparent 62%);
}

""",
        "",
    ),
    (
        """.hero-visual,
.mech-stage,
.honesty,
.code,
.table-wrap,
.arch-node {
  border: 1px solid transparent;""",
        """.mech-stage,
.code,
.table-wrap {
  border: 1px solid transparent;""",
    ),
    (
        """.hero-visual,
.mech-stage,
.honesty,
.code,
.table-wrap,
.arch-node {
  box-shadow:""",
        """.mech-stage,
.code,
.table-wrap {
  box-shadow:""",
    ),
    (
        """/* The terminal is a screen: near-black glass with the brand's own light
   coming off it. */
.terminal {
  border: 1px solid transparent;
  background-image:
    linear-gradient(#080b11, #080b11),
    linear-gradient(
      180deg,
      oklch(0.75 0.07 95 / 0.42),
      oklch(0.34 0.03 262 / 0.16) 26%,
      oklch(0.24 0.02 262 / 0.06)
    );
  background-origin: border-box;
  background-clip: padding-box, border-box;
  box-shadow:
    inset 0 1px 0 oklch(1 0 0 / 0.07),
    0 34px 80px -46px oklch(0.06 0.05 280 / 0.95),
    0 0 110px -56px oklch(0.885 0.18 95 / 0.4);
}

.terminal-bar {
  background: linear-gradient(180deg, oklch(0.22 0.022 258 / 0.9), oklch(0.18 0.02 262 / 0.75));
  border-bottom: 1px solid oklch(0.34 0.024 260 / 0.5);
}

""",
        "",
    ),
]

BLOCKS_LANDING = [
    ("/* ── 6 · the counters, full bleed ", "/* ── 7 · the spec sheet "),
]

LOOSE_LANDING = [
    (
        """.l-hero,
.l-proof,
.l-statement,
.l-show,
.l-strip,
.l-package,""",
        """.l-hero,
.l-proof,
.l-statement,
.l-show,
.l-package,""",
    ),
    (
        """/* A shared measure for the contained sections. */
.l-show,
.l-strip,
.l-package,""",
        """/* A shared measure for the contained sections. */
.l-show,
.l-package,""",
    ),
    (
        """.l-show-media {
  margin: 0;
  min-width: 0;
  position: relative;
}

.l-show-media img {
  display: block;
  width: 100%;
  height: auto;
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-lg);
  background: #08090b;
  box-shadow:
    0 40px 90px -50px rgba(0, 0, 0, 0.95),
    0 0 100px -60px oklch(0.885 0.18 95 / 0.28);
}

/* The web UI is a wide plate. It stays inside the measure: the full-bleed
   moments on this page are the hero, the counters and the mascot, and a
   fourth one here would spend the contrast they buy. */
.l-show-media--wide {
  width: 100%;
}

""",
        "",
    ),
    (
        """  .l-show-media--wide {
    width: 100%;
    margin-right: 0;
  }

  .l-show-media--wide img {
    border-radius: var(--radius-lg);
    border-right: 1px solid var(--line-strong);
  }

  .l-show-media--plate { justify-self: start; }

""",
        "",
    ),
    (
        """  .l-show-copy h2,
  .l-strip-copy h2 { max-width: 22ch; }""",
        """  .l-show-copy h2 { max-width: 22ch; }""",
    ),
    (
        """  .l-strip-copy { grid-template-columns: minmax(0, 1fr); }

""",
        "",
    ),
]


def cut(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i < 0:
        return text
    j = text.find(end, i)
    if j < 0:
        return text
    return text[:i] + text[j:]


def main() -> int:
    for raw in sys.argv[1:]:
        path = Path(raw)
        text = path.read_text(encoding="utf-8")
        before = len(text)

        blocks = BLOCKS_LANDING if path.name == "landing.css" else BLOCKS_STYLE
        loose = LOOSE_LANDING if path.name == "landing.css" else LOOSE_STYLE

        for start, end in blocks:
            text = cut(text, start, end)

        for old, new in loose:
            if old in text:
                text = text.replace(old, new)

        # Two blank lines are never two blank lines.
        text = re.sub(r"\n{3,}", "\n\n", text)
        path.write_text(text, encoding="utf-8")
        print(f"{path}: {before} -> {len(text)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
