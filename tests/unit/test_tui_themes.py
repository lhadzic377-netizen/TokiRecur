"""A theme has to reach everything, not half of it.

The launcher paints its own panes in Python and everything else - the
dialogs, the dropdowns, the chat panels - is painted by Textual from its
own design tokens. Those tokens used to be Textual's house colours, so a
theme switch changed the parts we paint by hand and left the rest of the
screen exactly where it was.

The stylesheets used to carry the house palette's hexes and swap them
for the theme's, one slot at a time. Two slots share a hex in the house
palette, so the first swap won and the second never landed.
"""

import pytest

from tokirecur import ctl
from tokirecur.tui import art
from tokirecur.tui.launcher import LauncherApp, SettingsEditScreen


def _rgb(value: str) -> tuple[int, int, int]:
    """A CSS colour as RGB. Textual shifts a colour by a hair when it
    derives its shades, so the comparison has to allow for that."""
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise AssertionError(f"not a plain colour: {value!r}")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def _same_colour(a: str, b: str, tolerance: int = 3) -> bool:
    """Two colours the same, to within Textual's own rounding."""
    ra, rb = _rgb(a), _rgb(b)
    return all(abs(x - y) <= tolerance for x, y in zip(ra, rb, strict=True))


@pytest.fixture(autouse=True)
def theme_file(tmp_path, monkeypatch):
    """A theme file of our own, so a test never repaints the real box."""
    import yaml

    path = tmp_path / "theme"
    path.write_text("default\n")
    monkeypatch.setenv("TOKI_THEME_FILE", str(path))
    monkeypatch.delenv("TOKI_THEME", raising=False)
    registry = tmp_path / "models.yaml"
    registry.write_text(yaml.safe_dump({"models": [{
        "name": "m", "label": "M", "gguf": str(tmp_path / "m.gguf"),
        "llama_port": 9100, "daemon_port": 9101, "ctx_size": 200000,
        "max_tokens": 100000, "model_id": "m", "parallel": 1, "kv": "f16",
    }]}))
    (tmp_path / "m.gguf").write_bytes(b"x")
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", registry)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", tmp_path / "active")
    monkeypatch.setenv("TOKI_MODEL_REGISTRY", str(registry))
    monkeypatch.setenv("TOKI_ACTIVE_MODEL", str(tmp_path / "active"))
    return path


def test_the_palette_is_the_textual_theme():
    """Textual's design tokens are the palette, not Textual's own."""
    theme = art.textual_theme()
    pal = art.palette()
    assert theme.primary == pal["YELLOW"]
    assert theme.secondary == pal["AMBER"]
    assert theme.error == pal["RED"]
    assert theme.success == pal["GREEN"]
    assert theme.background == pal["BG"]
    assert theme.surface == pal["SURFACE"]
    assert theme.foreground == pal["TEXT"]


def test_every_slot_is_a_variable_of_its_own():
    """The slots that shared a hex in the house palette stay apart.

    The accent and the sun were both #ffd60a, the text and the chrome
    both #f0ece0. Swapping hexes could not tell them apart; variables
    can.
    """
    variables = art.theme_variables()
    assert variables["sun"] == art.palette()["SUN"]
    assert variables["sand"] == art.palette()["SAND"]
    assert variables["sea"] == art.palette()["SEA"]
    assert variables["chrome"] == art.palette()["CHROME"]
    assert variables["muted"] == art.palette()["MUTED"]


def test_the_stylesheet_carries_no_colours():
    """theme.tcss is variables only: no hex can go stale in it."""
    css = art.stylesheet()
    assert "#000000" not in css
    assert "#ffd60a" not in css
    for slot in art.DEFAULT_PALETTE.values():
        assert slot.lower() not in css.lower()


async def test_a_theme_switch_moves_the_design_tokens():
    """The tokens the widgets wear follow the theme, live."""
    app = LauncherApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        before = app.get_css_variables()
        app._set_theme("gta6")
        await pilot.pause()
        await pilot.pause()
        after = app.get_css_variables()

    assert before["primary"] != after["primary"]
    assert _same_colour(before["primary"], art.THEMES["default"]["YELLOW"])
    assert _same_colour(after["primary"], art.THEMES["gta6"]["YELLOW"])
    assert _same_colour(after["background"], art.THEMES["gta6"]["BG"])
    assert _same_colour(after["muted"], art.THEMES["gta6"]["MUTED"])
    assert _same_colour(after["panel"], art.THEMES["gta6"]["SURFACE"])


async def test_a_theme_switch_reaches_a_dialog():
    """The dialogs are painted by Textual; they have to follow too."""
    app = LauncherApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.push_screen(SettingsEditScreen("m", "ctx_size", "200000"))
        await pilot.pause()
        before = str(app.screen.query_one("#settings-dialog").styles.background)
        app.screen.dismiss(None)
        await pilot.pause()
        app._set_theme("gta6")
        await pilot.pause()
        await pilot.pause()
        await app.push_screen(SettingsEditScreen("m", "ctx_size", "200000"))
        await pilot.pause()
        after = str(app.screen.query_one("#settings-dialog").styles.background)

    assert before != after


def test_the_dashboard_paints_every_slot(theme_file):
    """The browser gets the palette too - one variable at a time."""
    from tokirecur.dashboard.server import STATIC_DIR, themed_css

    theme_file.write_text("gta6\n")

    css = themed_css((STATIC_DIR / "style.css").read_text())
    pal = art.palette()
    assert f"--bg: {pal['BG']};" in css
    assert f"--accent: {pal['YELLOW']};" in css
    assert f"--text: {pal['TEXT']};" in css
    assert f"--good: {pal['GREEN']};" in css
    # The house values are gone: the block was rebuilt, not patched.
    assert "--bg: #000000;" not in css
    assert "--accent: #ffd60a;" not in css


def test_the_chat_app_wears_the_palette():
    """The chat TUI is the other Textual app; it wears the same theme."""
    from tokirecur.tui.app import TokiApp

    app = TokiApp()
    assert app.current_theme.primary == art.palette()["YELLOW"]
    assert app.current_theme.background == art.palette()["BG"]
    assert _same_colour(app.get_css_variables()["primary"], art.palette()["YELLOW"])


def test_the_web_surface_paints_every_slot(theme_file):
    """The web surface has its own variable names; they follow too."""
    from tokirecur.web.server import STATIC_DIR, web_variables

    theme_file.write_text("gta6\n")
    css = art.themed_root((STATIC_DIR / "style.css").read_text(), web_variables())
    pal = art.palette()

    assert f"--bg-primary: {pal['BG']};" in css
    assert f"--accent: {pal['YELLOW']};" in css
    assert f"--text-primary: {pal['TEXT']};" in css
    assert f"--cached: {pal['GREEN']};" in css
    # The shades the palette has no name for are mixed from it, so they
    # move with the theme as well.
    assert "--bg-tertiary: #000000;" not in css
    assert "--accent: #ffd60a;" not in css


async def test_the_chat_widgets_wear_the_palette():
    """The chat TUI's panels are painted by Textual; they follow too."""
    from tokirecur.tui.app import TokiApp

    app = TokiApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        from textual.widgets import Static

        sidebar = app.query_one("#sidebar", Static)
        background = str(sidebar.styles.background)
        assert background != "Color(0, 0, 0)"
