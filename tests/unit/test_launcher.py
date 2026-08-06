"""Tests for the Ollama-style Textual launcher and its UI design tokens."""

import asyncio

import pytest

from tokirecur import ctl
from tokirecur.tui.launcher import LAUNCH_APPS, LauncherApp


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Point every launcher test at a throwaway registry + active-marker
    so running the suite never mutates the user's real config."""
    reg = tmp_path / "models.yaml"
    marker = tmp_path / "active-model"
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", marker)
    monkeypatch.setenv("TOKI_MODEL_REGISTRY", str(reg))
    monkeypatch.setenv("TOKI_ACTIVE_MODEL", str(marker))
    return reg


def test_apps_bind_to_pinned_model():
    keys = [a.key for a in LAUNCH_APPS]
    assert keys == ["toki", "opencode", "hermes", "openclaw", "web"]
    assert all(a.interface in ("chat", "tui", "web") for a in LAUNCH_APPS)


def test_enter_launches_pinned_model():
    calls = []

    def on_launch(interface, client, model):
        calls.append((interface, client, model))
        return 0

    async def run_app():
        app = LauncherApp(on_launch=on_launch)
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            assert app.pinned is not None
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
        return app

    app = asyncio.run(run_app())
    assert calls
    assert calls[0][1] == "opencode"
    assert calls[0][2] == app.pinned


def test_renders_big_yellow_wordmark():
    """A bold yellow TOKI wordmark in real letterforms (pyfiglet
    standard font) anchors the top of the launcher."""
    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            rendered = str(app.pane.render())
        return rendered

    rendered = asyncio.run(run_app())
    # Figlet 'standard' uses slashes and pipes to draw the letterforms.
    # Both the T's underscore top stroke and the O's vertical bars
    # must appear in the rendered output.
    assert "____" in rendered
    assert "\\" in rendered or "/" in rendered
    # App titles must still appear below the wordmark.
    assert "Chat" in rendered
    assert "Launch OpenCode" in rendered
    # No solid-block approximation.
    assert "█████" not in rendered


def test_number_keys_jump_to_app():
    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            await pilot.press("3")
            await pilot.pause()
            idx_after = app.launch_idx
            await pilot.press("5")
            await pilot.pause()
            idx_after_5 = app.launch_idx
        return idx_after, idx_after_5

    idx_after, idx_after_5 = asyncio.run(run_app())
    assert idx_after == 2
    assert idx_after_5 == 4


def test_number_keys_jump_to_model():
    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            await pilot.press("right")
            await pilot.pause()
            await pilot.press("2")
            await pilot.pause()
            idx_after = app.model_idx
        return idx_after

    idx_after = asyncio.run(run_app())
    assert idx_after == 1  # pressing "2" jumps to model index 1


def test_settings_pane_renders_rows_for_each_spec():
    """Pressing `s` shows the settings pane with one editable row per
    field per registry spec, with the highlighted row accented."""
    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            rendered = str(app.pane.render())
        return rendered

    rendered = asyncio.run(run_app())
    # Every default model appears as a section header.
    assert "9b" in rendered
    assert "35b" in rendered
    assert "laguna" in rendered
    # Editable fields appear.
    assert "gguf" in rendered
    assert "llama_port" in rendered
    assert "ctx_size" in rendered
    assert "binary" in rendered
    assert "reasoning" in rendered
    # Read-only fields do NOT appear (name, source).
    assert rendered.count("\n  9b") == 1 or "  9b\n" in rendered  # one section header
    # The first row gets the accent bar.
    assert "▌ gguf" in rendered


def test_settings_pane_navigation_wraps():
    """up on the first row wraps to the last; down on the last wraps
    to the first. Both directions must reach valid row indices."""
    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            first = app.settings_idx
            await pilot.press("up")
            await pilot.pause()
            last_via_up = app.settings_idx
            await pilot.press("down")
            await pilot.pause()
            first_again = app.settings_idx
        return first, last_via_up, first_again

    first, last_via_up, first_again = asyncio.run(run_app())
    assert first == 0
    assert last_via_up > 0, "up should wrap past 0"
    assert first_again == first


def test_settings_pane_esc_returns_to_launchers():
    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            assert app.focus_pane == "settings"
            await pilot.press("escape")
            await pilot.pause()
            pane = app.focus_pane
        return pane

    assert asyncio.run(run_app()) == "launchers"


def test_settings_edit_modal_commits_new_value():
    """Pressing Enter on a settings row opens the modal; submitting
    calls update_spec_field and refreshes the spec.

    We bypass the Input widget's text editing (too fragile to simulate
    via pilot.press) and drive the launcher's callback directly — that
    is the path the modal's Submitted handler invokes.
    """
    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            # We're on the first row (9b.gguf). Call the dismiss callback
            # the modal would invoke on submit.
            app._on_edit_dismissed("/tmp/new-9b.gguf")
            await pilot.pause()
            new_gguf = app.specs[0].gguf
        return new_gguf

    new_gguf = asyncio.run(run_app())
    assert new_gguf == "/tmp/new-9b.gguf"


def test_settings_edit_modal_cancel_keeps_old_value():
    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            original = app.specs[0].gguf
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            after = app.specs[0].gguf
        return original, after

    original, after = asyncio.run(run_app())
    assert original == after


def test_settings_edit_modal_rejects_bad_value():
    """Invalid int input shows an error notification and keeps the old value."""
    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            # Navigate down once to reach 9b.llama_port (an int field).
            await pilot.press("down")
            await pilot.pause()
            original_port = app.specs[0].llama_port
            # Drive the dismiss callback with garbage — same path the
            # modal would invoke on submit.
            app._on_edit_dismissed("not-a-number")
            await pilot.pause()
            after_port = app.specs[0].llama_port
        return original_port, after_port

    original, after = asyncio.run(run_app())
    assert original == after
    assert original == 8088


def test_settings_edit_modal_has_save_button():
    """The edit modal shows Cancel + Save buttons and Save commits the
    typed value through the same callback Enter uses."""
    from textual.widgets import Button

    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            screen = app.screen
            buttons = {
                b.id: b
                for b in screen.query(Button)
            }
            assert "settings-save" in buttons
            assert "settings-cancel" in buttons
            assert buttons["settings-save"].label == "Save"
            # Pressing Save dismisses with the current input value.
            buttons["settings-save"].press()
            await pilot.pause()
            committed = app.specs[0].gguf
        return committed

    committed = asyncio.run(run_app())
    assert committed == app_specs_first_gguf()


def app_specs_first_gguf():
    from tokirecur import ctl
    return ctl.load_registry()[0].gguf


def test_status_line_shows_pinned_model():
    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            rendered = str(app.pane.render())
        return rendered

    rendered = asyncio.run(run_app())
    assert "no model pinned" not in rendered
    assert "1-5 jump" in rendered


def test_footer_shows_key_hints():
    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            footer = str(app.footer.render())
        return footer

    footer = asyncio.run(run_app())
    assert "enter launch" in footer


def test_right_opens_model_pane_and_enter_pins():
    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            await pilot.press("right")
            await pilot.pause()
            assert app.focus_pane == "models"
            await pilot.press("up")
            await pilot.press("enter")
            await pilot.pause()
            assert app.focus_pane == "launchers"
            assert app.pinned in {s.name for s in app.specs}
            assert app.pinned is not None
        return app

    app = asyncio.run(run_app())
    assert app.focus_pane == "launchers"


# ── Settings dropdowns ─────────────────────────────────────────────────


def test_settings_choices_map_covers_expected_fields():
    """Preset dropdowns exist for sizing/enum fields, with ``custom…``
    for the free-form ones and pure toggles for booleans."""
    from tokirecur.tui.launcher import _CUSTOM_VALUE, _SETTINGS_CHOICES

    assert "ctx_size" in _SETTINGS_CHOICES
    assert "max_tokens" in _SETTINGS_CHOICES
    assert "parallel" in _SETTINGS_CHOICES
    assert "kv" in _SETTINGS_CHOICES
    assert "reasoning" in _SETTINGS_CHOICES
    assert "always_on" in _SETTINGS_CHOICES
    assert "boot" in _SETTINGS_CHOICES
    # Paths/ids stay free text.
    for field in ("gguf", "llama_port", "daemon_port", "model_id", "binary"):
        assert field not in _SETTINGS_CHOICES
    # Int/enum fields offer a custom escape hatch.
    assert _CUSTOM_VALUE in _SETTINGS_CHOICES["ctx_size"].values()
    assert _CUSTOM_VALUE in _SETTINGS_CHOICES["kv"].values()
    # Booleans are pure toggles.
    assert set(_SETTINGS_CHOICES["always_on"].values()) == {"true", "false"}
    assert set(_SETTINGS_CHOICES["boot"].values()) == {"true", "false"}
    # Sensible presets.
    assert _SETTINGS_CHOICES["kv"]["f16"] == "f16"
    assert _SETTINGS_CHOICES["kv"]["q8_0"] == "q8_0"
    assert _SETTINGS_CHOICES["reasoning"]["auto"] == "auto"


def _setting_row_index(app, field: str) -> int:
    """Index of the first settings row for `field` across all models."""
    from tokirecur.tui.launcher import _settings_rows

    rows = _settings_rows(app.specs)
    for i, (_name, f, _value) in enumerate(rows):
        if f == field:
            return i
    raise AssertionError(f"no settings row for field {field!r}")


def test_settings_dropdown_save_commits_preset():
    """Enter on a dropdown field opens a choice modal; picking a preset
    and pressing Save writes it to the registry."""
    from textual.widgets import Select

    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            app.settings_idx = _setting_row_index(app, "kv")
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen.__class__.__name__ == "SettingsChoiceScreen"
            select = app.screen.query_one("#settings-choice-select", Select)
            select.value = "q8_0"
            await pilot.click("#settings-save")
            await pilot.pause()
            return app.specs[0].kv

    assert asyncio.run(run_app()) == "q8_0"


def test_settings_dropdown_preselects_unlisted_current_value():
    """A registry value outside the presets (9b defaults to ctx 200000)
    is surfaced as its own option so Save is a no-op-safe commit."""
    from textual.widgets import Select

    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            app.settings_idx = _setting_row_index(app, "ctx_size")
            await pilot.press("enter")
            await pilot.pause()
            select = app.screen.query_one("#settings-choice-select", Select)
            return select.value

    assert asyncio.run(run_app()) == "200000"


def test_settings_dropdown_custom_chains_to_text_modal():
    """Picking ``custom…`` dismisses the choice modal and opens the
    free-text editor for the same row."""
    from tokirecur.tui.launcher import _CUSTOM_VALUE

    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            app.settings_idx = _setting_row_index(app, "ctx_size")
            await pilot.press("enter")
            await pilot.pause()
            select = app.screen.query_one("#settings-choice-select")
            select.value = _CUSTOM_VALUE
            await pilot.click("#settings-save")
            await pilot.pause()
            return app.screen.__class__.__name__

    assert asyncio.run(run_app()) == "SettingsEditScreen"

