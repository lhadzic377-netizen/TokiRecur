"""The brand, painted.

One place so the launcher, the chat TUI and the CLI draw the same
wordmark: the house style in one accent, GTA VI in a sunset - the sun
at the top of the letters, the sand through the middle, the sea at the
foot.
"""

from __future__ import annotations

from functools import lru_cache

from rich.text import Text

from . import art

#: The star GTA VI is stamped with.
STAR = "★"


@lru_cache(maxsize=1)
def _poster_lines(font: str) -> tuple[str, ...]:
    """The wordmark as a poster: one string per line."""
    import pyfiglet

    raw = pyfiglet.figlet_format("TOKI", font=font).rstrip("\n")
    return tuple(ln for ln in raw.split("\n") if ln.strip())


def poster(gutter: str = "") -> list[Text]:
    """The wordmark as a poster, painted in the theme's gradient.

    The house style paints it in one accent; GTA VI paints it in a
    sunset - the hot sun at the top of the letters, the sand through
    the middle, the neon sea at the foot.
    """
    lines = _poster_lines(art.wordmark_font())
    colors = art.gradient(art.wordmark_gradient(), len(lines))
    return [
        Text(gutter + line, style=f"bold {color}")
        for line, color in zip(lines, colors, strict=False)
    ]


def wordmark(gutter: str = "") -> Text:
    """The wordmark on one line, each letter a step along the gradient.

    This is the small form - the chat TUI's header and the CLI's
    banner - where a six-line poster would take more room than it
    deserves.
    """
    text = Text(gutter)
    letters = art.WORDMARK
    colors = art.gradient(art.wordmark_gradient(), len(letters))
    for letter, color in zip(letters, colors, strict=False):
        text.append(letter, style=f"bold {color}")
    return text


def horizon(width: int, gutter: str = "") -> Text:
    """The horizon under the wordmark: a rule in the theme's gradient.

    In the house style it is a single accent line; in GTA VI it is the
    sea meeting the sky.
    """
    span = max(width - len(gutter), 1)
    colors = art.gradient(art.horizon_gradient(), span)
    text = Text(gutter)
    for color in colors:
        text.append("▀", style=color)
    return text
