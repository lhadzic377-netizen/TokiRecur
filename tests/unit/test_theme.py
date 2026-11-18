"""Themes: the palette the surfaces are painted with.

A limited edition theme replaces the palette; the slots keep their
names (``YELLOW`` is the primary accent whatever it looks like), so a
theme is a palette and not a rename. The Textual stylesheet is rendered
from the active palette, so a theme reaches the widgets that are styled
in CSS as well as the ones styled in Python.
"""

from pathlib import Path

import pytest

from tokirecur import ctl
from tokirecur.tui import art


def test_every_theme_has_the_same_slots():
    """A theme that misses a colour would leave a surface unpainted."""
    for name, palette in art.THEMES.items():
        assert set(palette) == set(art.DEFAULT_PALETTE), name
        for slot, value in palette.items():
            assert value.startswith("#") and len(value) == 7, (name, slot)


def test_the_house_style_is_the_default():
    assert art.DEFAULT_THEME in art.THEMES
    assert art.THEMES[art.DEFAULT_THEME] == art.DEFAULT_PALETTE


def test_the_stylesheet_follows_the_palette(monkeypatch):
    monkeypatch.setattr(art, "PALETTE", art.THEMES["gta6"])
    css = art.stylesheet()
    assert art.THEMES["gta6"]["YELLOW"].lower() in css
    assert art.THEMES["gta6"]["BG"].lower() in css
    # The house accent is gone.
    assert art.DEFAULT_PALETTE["YELLOW"].lower() not in css


def test_the_stylesheet_covers_the_house_palette():
    """Every colour the stylesheet names is a palette slot.

    A hex in the CSS that is not in the palette would stay house-yellow
    in a themed run.
    """
    raw = (Path(art.__file__).with_name("theme.tcss")).read_text()
    house = {v.lower() for v in art.DEFAULT_PALETTE.values()}
    for hexcode in {h.lower() for h in __import__("re").findall(r"#[0-9a-f]{6}", raw)}:
        assert hexcode in house, hexcode


def test_the_theme_choice_is_remembered(tmp_path, monkeypatch):
    marker = tmp_path / "theme"
    monkeypatch.setenv("TOKI_THEME_FILE", str(marker))
    assert ctl.read_theme() is None
    ctl.write_theme("gta6")
    assert ctl.read_theme() == "gta6"
    assert marker.read_text().strip() == "gta6"
    ctl.write_theme(None)
    assert ctl.read_theme() is None


def test_an_unknown_theme_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKI_THEME_FILE", str(tmp_path / "theme"))
    with pytest.raises(ValueError, match="unknown theme"):
        ctl.write_theme("neon-dreams")


def test_the_limited_edition_says_so_in_the_launcher(monkeypatch):
    """The badge is what makes it an edition rather than a colour."""
    import asyncio

    from tokirecur.tui.launcher import LauncherApp

    monkeypatch.setattr(art, "EDITION_LABEL", art.EDITION_LABELS["gta6"])

    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            return str(app.pane.render())

    rendered = asyncio.run(run_app())
    assert art.EDITION_LABELS["gta6"] in rendered


def test_the_house_style_shows_no_badge(monkeypatch):
    import asyncio

    from tokirecur.tui.launcher import LauncherApp

    monkeypatch.setattr(art, "EDITION_LABEL", "")

    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            return str(app.pane.render())

    rendered = asyncio.run(run_app())
    assert "LIMITED EDITION" not in rendered


def test_the_gradient_runs_from_the_first_stop_to_the_last():
    stops = ("#ff0000", "#0000ff")
    ramp = art.gradient(stops, 3)
    assert len(ramp) == 3
    assert ramp[0] == "#ff0000"
    assert ramp[-1] == "#0000ff"
    assert ramp[1] == "#800080"


def test_one_stop_is_one_colour():
    assert art.gradient(("#ffd60a",), 4) == ["#ffd60a"] * 4


def test_the_sunset_wordmark_is_a_gradient():
    """GTA VI is a sunset: the wordmark runs from the sun to the sea."""
    if art.THEME != "gta6":
        pytest.skip("the house style paints the wordmark in one colour")
    assert art.wordmark_gradient() == (art.SUN, art.SAND, art.SEA)
    assert art.horizon_gradient() == (art.SUN, art.SAND, art.SEA)


def test_the_wordmark_font_is_the_themes_own():
    assert art.wordmark_font() == art.WORDMARK_FONTS[art.THEME]
    # The limited edition is a heavier face than the house style.
    assert art.WORDMARK_FONTS["gta6"] != art.WORDMARK_FONTS["default"]


def test_the_launcher_draws_the_horizon_under_the_wordmark(monkeypatch):
    """The horizon is the sea meeting the sky - it is the edition."""
    import asyncio

    from tokirecur.tui import launcher as launcher_mod

    monkeypatch.setattr(art, "EDITION_LABEL", art.EDITION_LABELS["gta6"])

    async def run_app():
        app = launcher_mod.LauncherApp()
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            return str(app.pane.render())

    rendered = asyncio.run(run_app())
    assert "▀" in rendered
