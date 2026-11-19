"""Themes: the palette the surfaces are painted with.

Every harness toki launches has a theme of its own, taken from the
colours that harness actually wears, plus the house style and GTA VI.
The slots keep their names (``YELLOW`` is the primary accent whatever
it looks like), so a theme is a palette and not a rename. The palette
is read at the moment it is used, so a theme switched from the
settings pane repaints the launcher without a restart, and the Textual
stylesheet is rendered from it, so the widgets styled in CSS follow too.
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


def test_every_harness_toki_launches_has_a_theme():
    """A theme per harness: the launcher knows them all."""
    from tokirecur.launch_catalog import LAUNCH_APPS

    for app in LAUNCH_APPS:
        if app.interface == "web" or app.client in ("toki", "tokirecur"):
            continue
        assert app.client in art.THEMES, app.client


def test_gta_vi_is_a_theme_not_an_edition():
    """It was a limited edition once. It is a mood now."""
    assert "gta6" in art.THEMES
    assert not hasattr(art, "EDITION_LABEL")
    assert not hasattr(art, "EDITION_LABELS")


def test_the_stylesheet_follows_the_palette(monkeypatch, tmp_path):
    # The environment wins over the file, so a run with TOKI_THEME set
    # would not see the file's theme at all.
    monkeypatch.delenv("TOKI_THEME", raising=False)
    monkeypatch.setenv("TOKI_THEME_FILE", str(tmp_path / "theme"))
    (tmp_path / "theme").write_text("gta6\n")
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
    import re

    raw = (Path(art.__file__).with_name("theme.tcss")).read_text()
    house = {v.lower() for v in art.DEFAULT_PALETTE.values()}
    for hexcode in {h.lower() for h in re.findall(r"#[0-9a-f]{6}", raw)}:
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


def test_the_palette_is_read_at_the_moment_it_is_used(monkeypatch, tmp_path):
    """A theme switched from the settings pane has to repaint.

    A module-level constant would freeze the palette at import, and
    half the launcher would keep the old colours.
    """
    marker = tmp_path / "theme"
    # The environment wins over the file, so a run with TOKI_THEME set
    # would not see the file's theme at all.
    monkeypatch.delenv("TOKI_THEME", raising=False)
    monkeypatch.setenv("TOKI_THEME_FILE", str(marker))
    marker.write_text("gta6\n")
    assert art.YELLOW == art.THEMES["gta6"]["YELLOW"]
    marker.write_text("claude\n")
    assert art.YELLOW == art.THEMES["claude"]["YELLOW"]


def test_the_brand_draws_the_same_wordmark_everywhere():
    """The launcher, the chat TUI and the CLI draw one brand."""
    from tokirecur.tui import brand

    assert brand.wordmark().plain == art.WORDMARK
    poster = brand.poster()
    assert poster and len(poster) > 1


def test_gta_vi_gets_the_sunset_and_the_horizon():
    """GTA VI is a sunset: the wordmark runs from the sun to the sea."""
    from tokirecur.tui import brand

    if art.THEME != "gta6":
        pytest.skip("the house style paints the wordmark in one colour")
    assert art.wordmark_gradient() == (art.SUN, art.SAND, art.SEA)
    assert brand.horizon(20).plain.count("▀") == 20


def test_the_wordmark_font_is_the_themes_own():
    assert art.wordmark_font() == art.WORDMARK_FONTS.get(art.THEME, "standard")
    # GTA VI is a heavier face than the house style.
    assert art.WORDMARK_FONTS["gta6"] != "standard"


def test_the_dashboard_stylesheet_follows_the_theme():
    """The browser dashboard is the one surface outside the terminal."""
    from fastapi.testclient import TestClient

    from tokirecur.dashboard.server import create_dashboard_app

    client = TestClient(create_dashboard_app(reports_dir=Path("/tmp")))
    response = client.get("/static/style.css")
    assert response.status_code == 200
    css = response.text
    if art.THEME == art.DEFAULT_THEME:
        # The house style is what the file already says.
        assert art.DEFAULT_PALETTE["YELLOW"].lower() in css
    else:
        # No house colour is left in the file the browser gets.
        for slot in art.DEFAULT_PALETTE:
            assert art.DEFAULT_PALETTE[slot].lower() not in css


def test_the_settings_panel_lists_the_themes():
    """The themes are a machine-wide setting, like the recorder."""
    from tokirecur.tui import launcher as launcher_mod

    name, field, value = launcher_mod._theme_row()
    assert field == launcher_mod.THEME_FIELD
    assert value in art.THEMES


@pytest.mark.asyncio
async def test_picking_a_theme_in_the_settings_repaints_the_launcher(
    tmp_path, monkeypatch
):
    """The whole loop, driven the way a person drives it.

    Open the settings pane, find the theme row, open its dropdown,
    pick a theme - and the launcher has to be wearing it when the
    dropdown closes. A theme that only changes a file on disk changes
    nothing until the next run, which is not what the row promises.
    """
    from textual.widgets import Select

    from tokirecur.tui.launcher import (
        THEME_FIELD,
        LauncherApp,
        _settings_rows,
    )

    monkeypatch.delenv("TOKI_THEME", raising=False)
    monkeypatch.setenv("TOKI_THEME_FILE", str(tmp_path / "theme"))
    app = LauncherApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()

        rows = _settings_rows(app.specs, app=app)
        assert THEME_FIELD in {field for _, field, _ in rows}
        index = next(i for i, row in enumerate(rows) if row[1] == THEME_FIELD)
        app.settings_idx = index
        await pilot.pause()

        await pilot.press("enter")
        await pilot.pause()
        select = app.screen.query_one("#settings-choice-select", Select)
        values = {value for _, value in select._options}
        assert "gta6" in values
        assert len(values) == len(art.THEMES)

        before = app.screen.styles.background
        select.value = "gta6"
        await pilot.pause()
        await pilot.pause()

        assert ctl.read_theme() == "gta6"
        assert art.YELLOW == art.THEMES["gta6"]["YELLOW"]
        after = app.screen.styles.background
        assert after != before
        expected = art.THEMES["gta6"]["BG"].lstrip("#")
        assert after.hex.lower().endswith(expected.lower())
