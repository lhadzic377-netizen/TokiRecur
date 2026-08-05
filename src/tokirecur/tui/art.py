"""Shared design tokens for the toki UIs.

One yellow/black design language across the launcher, the chat TUI, the
CLI and the web dashboards: pure black backgrounds, vivid yellow
accent, amber secondary, warm off-white text. Semantic status colors
(green / red / blue / purple) are kept separately so health, cache
hits and modes stay readable.

Glyphs live here too so every surface renders the same app icons.
"""

from __future__ import annotations

# ── Core palette ───────────────────────────────────────────────────

BG = "#000000"
SURFACE = "#121212"
BORDER = "#2b2b2b"
BORDER_HI = "#3d3d3d"
YELLOW = "#ffd60a"
AMBER = "#b8860b"
YELLOW_DIM = "#7a6a00"
TEXT = "#f0ece0"
MUTED = "#9a967f"

# ── Semantic colors (kept for readability) ─────────────────────────

GREEN = "#4ade80"
RED = "#ff6b8b"
BLUE = "#5a7adb"
PURPLE = "#c084fc"

# ── Brand wordmark ─────────────────────────────────────────────────

# The ASCII banner is gone; the brand is typographic. Renderers style
# this string with `YELLOW` + bold and center it.
WORDMARK = "TOKI"

TAGLINE = "recycled-prefill LLM stack · AMD Ryzen AI"
