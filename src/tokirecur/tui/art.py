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
    # The scene: a sky, a sun, a sand, a sea. The house style has no
    # scene, so these follow its single accent.
    "SKY": "#12121c",
    "SUN": "#ffd60a",
    "SAND": "#b8860b",
    "SEA": "#5a7adb",
    "CHROME": "#f0ece0",
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
        "GREEN": "#3fe0a0",
        "RED": "#ff5a6e",
        "BLUE": "#35d0ff",
        "PURPLE": "#a66bff",
        # Leonida at dusk: a deep sky, the sun still hot over the
        # water, sand between them, the sea in neon teal.
        "SKY": "#0d0b24",
        "SUN": "#ff2e88",
        "SAND": "#ff9e2c",
        "SEA": "#25e0d8",
        "CHROME": "#e8e4f2",
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

# ── The scene ────────────────────────────────────────────────────────

SKY = PALETTE["SKY"]
SUN = PALETTE["SUN"]
SAND = PALETTE["SAND"]
SEA = PALETTE["SEA"]
CHROME = PALETTE["CHROME"]

# ── Brand wordmark ─────────────────────────────────────────────────

# The ASCII banner is gone; the brand is typographic. Renderers style
# this string with `YELLOW` + bold and center it.
WORDMARK = "TOKI"

TAGLINE = "recycled-prefill LLM stack · AMD Ryzen AI"


#: The wordmark font per theme. The house style uses pyfiglet's
#: "standard"; a limited edition can pick a heavier, more poster-like
#: face - GTA VI's wordmark is a heavy upright plate.
WORDMARK_FONTS: dict[str, str] = {
    "default": "standard",
    "gta6": "doom",
}


def wordmark_font() -> str:
    """The figlet font the wordmark is drawn in."""
    return WORDMARK_FONTS.get(THEME, "standard")


def wordmark_gradient() -> tuple[str, ...]:
    """The colours the wordmark is painted in, top to bottom.

    The house style is one colour. GTA VI is a sunset: the hot sun at
    the top of the letters, the sand through the middle, the neon sea
    at the foot.
    """
    if THEME == "gta6":
        return (SUN, SAND, SEA)
    return (YELLOW,)


def horizon_gradient() -> tuple[str, ...]:
    """The colours of the horizon rule under the wordmark."""
    if THEME == "gta6":
        return (SUN, SAND, SEA)
    return (YELLOW,)


def _mix(a: str, b: str, t: float) -> str:
    """Two colours, a fraction between them."""
    ah = a.lstrip("#")
    bh = b.lstrip("#")
    out = []
    for i in (0, 2, 4):
        ca = int(ah[i:i + 2], 16)
        cb = int(bh[i:i + 2], 16)
        out.append(round(ca + (cb - ca) * t))
    return "#" + "".join(f"{c:02x}" for c in out)


def gradient(stops: tuple[str, ...], steps: int) -> list[str]:
    """``steps`` colours from the first stop to the last.

    One stop means one colour - the house style paints the wordmark in
    a single accent.
    """
    if steps <= 0:
        return []
    if len(stops) == 1 or steps == 1:
        return [stops[0]] * steps
    out: list[str] = []
    segments = len(stops) - 1
    for i in range(steps):
        pos = i / max(steps - 1, 1) * segments
        segment = min(int(pos), segments - 1)
        out.append(_mix(stops[segment], stops[segment + 1], pos - segment))
    return out


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
