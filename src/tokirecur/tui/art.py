"""Shared design tokens for the toki UIs.

One design language across the launcher, the chat TUI, the CLI and the
web dashboards: a dark background, a vivid accent, a secondary, warm
text. Semantic status colors (green / red / blue / purple) are kept
separately so health, cache hits and modes stay readable.

Glyphs live here too so every surface renders the same app icons.

Themes
------
The palette above is the house style. Every harness toki launches has a
theme of its own, taken from the colours that harness actually wears -
Claude's terracotta, opencode's cream on charcoal, Kimi's blue, Xiaomi's
orange, DeepSeek's blue, Grok's Tokyo Night purple, and so on - plus
GTA VI, which is a mood of its own.

``toki theme`` lists them, ``toki theme <name>`` switches the whole
box, and ``TOKI_THEME`` applies one for a single run.

The slots keep their names whatever the theme - ``YELLOW`` is the
primary accent - so a theme is a palette and not a rename. The palette
is read at the moment it is used, so a theme switched from the
settings pane repaints the launcher without a restart.
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

#: The semantic colours every theme shares: health, cache hits, modes.
_SEMANTIC = {
    "GREEN": "#3fb950",
    "RED": "#f85149",
    "BLUE": "#58a6ff",
    "PURPLE": "#a371f7",
}


def _palette(
    bg: str,
    surface: str,
    border: str,
    border_hi: str,
    accent: str,
    secondary: str,
    dim: str,
    text: str,
    muted: str,
    *,
    sky: str | None = None,
    sun: str | None = None,
    sand: str | None = None,
    sea: str | None = None,
    chrome: str | None = None,
) -> dict[str, str]:
    """A full palette from the colours that make a theme its own.

    The scene slots (sky, sun, sand, sea, chrome) are what the
    wordmark and the horizon are painted with; a theme that has no
    scene of its own borrows the accent.
    """
    return {
        "BG": bg,
        "SURFACE": surface,
        "BORDER": border,
        "BORDER_HI": border_hi,
        "YELLOW": accent,
        "AMBER": secondary,
        "YELLOW_DIM": dim,
        "TEXT": text,
        "MUTED": muted,
        **_SEMANTIC,
        "SKY": sky or surface,
        "SUN": sun or accent,
        "SAND": sand or secondary,
        "SEA": sea or _SEMANTIC["BLUE"],
        "CHROME": chrome or text,
    }


#: Every theme, by name. The names are the clients toki launches.
THEMES: dict[str, dict[str, str]] = {
    "default": dict(DEFAULT_PALETTE),
    # Anthropic's warm clay on a dark ground.
    "claude": _palette(
        "#1b1a18", "#262421", "#3d3833", "#57514a",
        "#d97757", "#a85a3c", "#6b3d2a",
        "#f5f1e8", "#a79c8f",
        sky="#221f1c", sun="#d97757", sand="#c96442", sea="#7aa2c4",
        chrome="#f5f1e8",
    ),
    # openclaw: amber on black with a rust and a teal.
    "openclaw": _palette(
        "#000000", "#111111", "#2b2b2b", "#454545",
        "#f4b840", "#bd4531", "#7a5a2b",
        "#f5f7fa", "#8a8f98",
        sky="#0b0b0b", sun="#f4b840", sand="#bd4531", sea="#8abeb7",
        chrome="#f5f7fa",
    ),
    # hermes: a deep petrol blue.
    "hermes": _palette(
        "#0a1418", "#10222a", "#1d3b46", "#2b5563",
        "#3fb8c4", "#1f7a8c", "#14525e",
        "#dceef2", "#7d9aa3",
        sky="#0a1418", sun="#3fb8c4", sand="#1f7a8c", sea="#2f9fb5",
        chrome="#dceef2",
    ),
    # opencode: cream on charcoal, with a lavender-blue for the quiet
    # parts.
    "opencode": _palette(
        "#1a1a1a", "#232323", "#343434", "#4c4c4c",
        "#fbf0df", "#828bb8", "#5a5f75",
        "#eeeeee", "#828bb8",
        sky="#1a1a1a", sun="#fbf0df", sand="#828bb8", sea="#7086b5",
        chrome="#fbf0df",
    ),
    # kilo is an opencode fork; it wears the same colours.
    "kilo": _palette(
        "#1a1a1a", "#232323", "#343434", "#4c4c4c",
        "#fbf0df", "#828bb8", "#5a5f75",
        "#eeeeee", "#828bb8",
        sky="#1a1a1a", sun="#fbf0df", sand="#828bb8", sea="#7086b5",
        chrome="#fbf0df",
    ),
    # Kimi: a deep black and an electric blue.
    "kimi": _palette(
        "#121212", "#1a1a1a", "#292929", "#3d3d3d",
        "#1a88ff", "#0f5faa", "#0b3d70",
        "#f5f5f5", "#8b949e",
        sky="#121212", sun="#1a88ff", sand="#d29922", sea="#1a88ff",
        chrome="#f5f5f5",
    ),
    # Qwen: violet.
    "qwen": _palette(
        "#14121c", "#1d1a2a", "#2f2a44", "#463f63",
        "#7b61ff", "#5b45c7", "#3d2f8c",
        "#ece9f5", "#9b93b8",
        sky="#14121c", sun="#7b61ff", sand="#c4b5fd", sea="#38bdf8",
        chrome="#ece9f5",
    ),
    # MiMo: Xiaomi's orange.
    "mimo": _palette(
        "#0a0a0a", "#151515", "#2b2b2b", "#434343",
        "#ff6900", "#c24f00", "#7a3300",
        "#f2f2f2", "#9a9a9a",
        sky="#0a0a0a", sun="#ff6900", sand="#ff8a33", sea="#4dc4ff",
        chrome="#f2f2f2",
    ),
    # Oh My Pi: lime.
    "omp": _palette(
        "#0b1207", "#12200c", "#234014", "#376024",
        "#a3e635", "#5f9e1f", "#3d6614",
        "#eaffea", "#8aa884",
        sky="#0b1207", sun="#a3e635", sand="#5f9e1f", sea="#34d399",
        chrome="#eaffea",
    ),
    # Pi: rose.
    "pi": _palette(
        "#140a10", "#22101a", "#3d1b2b", "#5c2b40",
        "#fb7185", "#c2445b", "#832e3f",
        "#ffeef1", "#b4879a",
        sky="#140a10", sun="#fb7185", sand="#f9a8d4", sea="#c084fc",
        chrome="#ffeef1",
    ),
    # MiniMax: indigo.
    "minimax": _palette(
        "#060a16", "#0d1428", "#1b2542", "#2c3a63",
        "#4d6bfe", "#3149c9", "#1f2f8c",
        "#eef1fb", "#8b96bd",
        sky="#060a16", sun="#4d6bfe", sand="#35d0ff", sea="#35d0ff",
        chrome="#eef1fb",
    ),
    # Codex: OpenAI's green.
    "codex": _palette(
        "#0d0d0d", "#161616", "#262626", "#3a3a3a",
        "#10a37f", "#0b7a5f", "#075440",
        "#f0f0f0", "#909090",
        sky="#0d0d0d", sun="#10a37f", sand="#0b7a5f", sea="#3fb950",
        chrome="#f0f0f0",
    ),
    # Factory Droid: periwinkle.
    "droid": _palette(
        "#0d0f14", "#161922", "#262a36", "#3b4050",
        "#7c8cff", "#515ecb", "#343d87",
        "#eef0f8", "#949ab0",
        sky="#0d0f14", sun="#7c8cff", sand="#a5b4fc", sea="#38bdf8",
        chrome="#eef0f8",
    ),
    # Cline: GitHub's greys and their light blue.
    "cline": _palette(
        "#24292e", "#2f363d", "#444c56", "#5a636e",
        "#79b8ff", "#3d7fc4", "#2a5480",
        "#e1e4e8", "#8b949e",
        sky="#24292e", sun="#79b8ff", sand="#3d7fc4", sea="#56d364",
        chrome="#e1e4e8",
    ),
    # DeepSeek Harness: their blue.
    "dsh": _palette(
        "#0b0e14", "#141822", "#232a38", "#384154",
        "#4d6bfe", "#3348c9", "#20328c",
        "#e6e9f2", "#8b93a7",
        sky="#0b0e14", sun="#4d6bfe", sand="#7d92ff", sea="#4d6bfe",
        chrome="#e6e9f2",
    ),
    # Grok Build: Tokyo Night.
    "grok": _palette(
        "#16161e", "#1a1b26", "#292e42", "#3b4261",
        "#bb9af7", "#7aa2f7", "#565f89",
        "#c0caf5", "#787fa6",
        sky="#16161e", sun="#bb9af7", sand="#f7768e", sea="#0db9d7",
        chrome="#c0caf5",
    ),
    # Goose: Block's agent - a blue on a warm near-black, with the
    # teal and the violet its binary carries.
    "goose": _palette(
        "#131010", "#1c1a19", "#332f2c", "#4b4642",
        "#5c98f9", "#47c9a2", "#2f4a75",
        "#e3e6ea", "#b7b1b1",
        sky="#131010", sun="#5c98f9", sand="#47c9a2", sea="#9b8afb",
        chrome="#f4f4f4",
    ),
    # Pool: Poolside's agent - the blue of its logo, on the dark it
    # ships with.
    "pool": _palette(
        "#1a1b26", "#24283b", "#2f344d", "#414868",
        "#0087ff", "#5bc4bf", "#0a4f80",
        "#c0caf5", "#565f89",
        sky="#1a1b26", sun="#0087ff", sand="#5bc4bf", sea="#57c7ff",
        chrome="#f8f8f2",
    ),
    # GitHub Copilot: the Primer palette.
    "copilot": _palette(
        "#0d1117", "#161b22", "#30363d", "#484f58",
        "#58a6ff", "#1f6feb", "#0d419d",
        "#e6edf3", "#8b949e",
        sky="#0d1117", sun="#58a6ff", sand="#3fb950", sea="#58a6ff",
        chrome="#e6edf3",
    ),
    # GTA VI: Leonida at dusk.
    "gta6": _palette(
        "#07070c", "#14141f", "#2a2a3d", "#3e3e5a",
        "#ff2e88", "#ff8a2b", "#8c1f4c",
        "#f6f1e7", "#8c87a8",
        sky="#0d0b24", sun="#ff2e88", sand="#ff9e2c", sea="#25e0d8",
        chrome="#e8e4f2",
    ),
}

#: The wordmark font per theme. The house style uses pyfiglet's
#: "standard"; a theme can pick a heavier, more poster-like face -
#: GTA VI's wordmark is a heavy upright plate.
WORDMARK_FONTS: dict[str, str] = {
    "gta6": "doom",
}

#: The lettering a theme shows under the wordmark. GTA VI says its
#: name - it is a mood like the others, not an edition.
THEME_LABELS: dict[str, str] = {
    "gta6": "GTA VI",
}


def _theme_file() -> os.PathLike[str] | str:
    """Where the chosen theme is remembered."""
    if os.environ.get("TOKI_THEME_FILE"):
        return os.environ["TOKI_THEME_FILE"]
    from ..ctl import DEFAULT_REGISTRY

    return DEFAULT_REGISTRY.parent / "theme"


def _active_theme_name() -> str:
    """Which theme this run uses.

    The environment wins, so a one-off ``TOKI_THEME=gta6 toki`` works;
    then the choice the user persisted with ``toki theme``.
    """
    env = (os.environ.get("TOKI_THEME") or "").strip().lower()
    if env:
        return env if env in THEMES else DEFAULT_THEME
    try:
        from pathlib import Path

        name = Path(_theme_file()).read_text().strip().lower()
    except OSError:
        return DEFAULT_THEME
    return name if name in THEMES else DEFAULT_THEME


def palette() -> dict[str, str]:
    """The active theme's palette."""
    return THEMES[_active_theme_name()]


def __getattr__(name: str) -> str:
    """The palette's colours, read at the moment they are used.

    A module-level constant would freeze the palette at import, and a
    theme switched from the settings pane would repaint only half the
    launcher. Reading it here lets the switch take effect at once.
    """
    if name == "THEME":
        return _active_theme_name()
    if name == "PALETTE":
        return palette()
    if name in DEFAULT_PALETTE:
        return palette()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ── The scene ────────────────────────────────────────────────────────

def wordmark_font() -> str:
    """The figlet font the wordmark is drawn in."""
    return WORDMARK_FONTS.get(_active_theme_name(), "standard")


def theme_label() -> str:
    """The lettering the active theme shows under the wordmark.

    Empty for a theme that does not name itself - the house style does
    not.
    """
    return THEME_LABELS.get(_active_theme_name(), "")


def wordmark_gradient() -> tuple[str, ...]:
    """The colours the wordmark is painted in, top to bottom.

    The house style is one colour. GTA VI is a sunset: the hot sun at
    the top of the letters, the sand through the middle, the neon sea
    at the foot.
    """
    if _active_theme_name() == "gta6":
        pal = palette()
        return (pal["SUN"], pal["SAND"], pal["SEA"])
    return (palette()["YELLOW"],)


def horizon_gradient() -> tuple[str, ...]:
    """The colours of the horizon rule under the wordmark."""
    return wordmark_gradient()


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

    One stop means one colour - a theme with no scene paints the
    wordmark in a single accent.
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


# ── Brand wordmark ─────────────────────────────────────────────────

# The ASCII banner is gone; the brand is typographic. Renderers style
# this string with the accent + bold and center it.
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
    for slot, value in palette().items():
        raw = raw.replace(DEFAULT_PALETTE[slot].lower(), value.lower())
    return raw


def retheme(app: object, css: str) -> None:
    """Repaint a running Textual app in a new stylesheet.

    Textual reads ``App.CSS`` once, when the app starts, and keeps the
    text it read. Replacing the class attribute is not enough - the
    stylesheet still holds the colours it was built with - so the
    source is replaced too and the stylesheet re-parsed.
    """
    import inspect

    cls = type(app)
    try:
        app_path = inspect.getfile(cls)
    except (TypeError, OSError):
        app_path = ""
    cls.CSS = css  # type: ignore[attr-defined]
    stylesheet_ = app.stylesheet  # type: ignore[attr-defined]
    stylesheet_.add_source(
        css,
        read_from=(app_path, f"{cls.__name__}.CSS"),
        is_default_css=False,
    )
    app.refresh_css()  # type: ignore[attr-defined]
