"""Shared design tokens for the toki UIs.

One design language across the launcher, the chat TUI, the CLI and the
web dashboards: pure black backgrounds, vivid yellow accent, amber
secondary, warm off-white text. Semantic status colors (green / red /
blue / purple) are kept separately so health, cache hits and modes stay
readable.

Glyphs live here too so every surface renders the same app icons.

Themes
------
The palette above is the house style. A limited edition theme replaces
it: ``toki theme gta6`` switches the whole box to it and
``toki theme default`` switches back. ``TOKI_THEME`` overrides both for
one run. The names of the colour slots do not change with the theme -
``YELLOW`` is the primary accent whatever it looks like - so a theme is
a palette, not a rename.
"""

from __future__ import annotations

import os

# ── The house palette ───────────────────────────────────────────────

DEFAULT_THEME = "default"

DEFAULT_PALETTE: dict[str, str] = {
    "BG": "#000000",
    "SURFACE": "#121212",
    "BORDER": "#2b2b2b",
    "BORDER_HI": "#3d3d3d",
    "YELLOW": "#ffd60a",
    "AMBER": "#b8860b",
    "YELLOW_DIM": "#7a6a00",
    "TEXT": "#f0ece0",
    "MUTED": "#9a967f",
    "GREEN": "#4ade80",
    "RED": "#ff6b8b",
    "BLUE": "#5a7adb",
    "PURPLE": "#c084fc",
}

#: Limited edition themes, by name.
THEMES: dict[str, dict[str, str]] = {
    DEFAULT_THEME: DEFAULT_PALETTE,
    # GTA VI - Leonida at dusk: near-black with a blue cast, the hot
    # magenta of a Vice City sign, a sunset orange, and the cyan of
    # the water.
    "gta6": {
        "BG": "#07070c",
        "SURFACE": "#14141f",
        "BORDER": "#2a2a3d",
        "BORDER_HI": "#3e3e5a",
        "YELLOW": "#ff2e88",
        "AMBER": "#ff8a2b",
        "YELLOW_DIM": "#8c1f4c",
        "TEXT": "#f6f1e7",
        "MUTED": "#8c87a8",
        "GREEN": "#45e0a0",
        "RED": "#ff5a6e",
        "BLUE": "#35d0ff",
        "PURPLE": "#a66bff",
    },
}

#: The badge a limited edition shows under the wordmark.
EDITION_LABELS: dict[str, str] = {
    "gta6": "GTA VI · LIMITED EDITION",
}


def _active_theme_name() -> str:
    """Which theme this run uses.

    The environment wins, so a one-off ``TOKI_THEME=gta6 toki`` works;
    then the choice the user persisted with ``toki theme``.
    """
    env = (os.environ.get("TOKI_THEME") or "").strip().lower()
    if env:
        return env if env in THEMES else DEFAULT_THEME
    try:
        from ..ctl import read_theme

        name = (read_theme() or "").strip().lower()
    except Exception:
        name = ""
    return name if name in THEMES else DEFAULT_THEME


#: The theme in force.
THEME = _active_theme_name()

#: Its palette.
PALETTE = dict(THEMES[THEME])

#: The badge to show under the wordmark (empty for the house style).
EDITION_LABEL = EDITION_LABELS.get(THEME, "")

# ── Core palette ───────────────────────────────────────────────────

BG = PALETTE["BG"]
SURFACE = PALETTE["SURFACE"]
BORDER = PALETTE["BORDER"]
BORDER_HI = PALETTE["BORDER_HI"]
YELLOW = PALETTE["YELLOW"]
AMBER = PALETTE["AMBER"]
YELLOW_DIM = PALETTE["YELLOW_DIM"]
TEXT = PALETTE["TEXT"]
MUTED = PALETTE["MUTED"]

# ── Semantic colors (kept for readability) ─────────────────────────

GREEN = PALETTE["GREEN"]
RED = PALETTE["RED"]
BLUE = PALETTE["BLUE"]
PURPLE = PALETTE["PURPLE"]

# ── Brand wordmark ─────────────────────────────────────────────────

# The ASCII banner is gone; the brand is typographic. Renderers style
# this string with `YELLOW` + bold and center it.
WORDMARK = "TOKI"

TAGLINE = "recycled-prefill LLM stack · AMD Ryzen AI"


def stylesheet() -> str:
    """The Textual stylesheet for the active theme.

    ``theme.tcss`` is written in the house palette; the colours are
    swapped for the active theme's, so a theme reaches the widgets that
    are styled in CSS as well as the ones styled in Python.
    """
    from pathlib import Path

    raw = Path(__file__).with_name("theme.tcss").read_text()
    for slot, value in PALETTE.items():
        raw = raw.replace(DEFAULT_PALETTE[slot].lower(), value.lower())
    return raw
