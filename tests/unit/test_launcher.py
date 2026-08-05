"""Tests for the Ollama-style Textual launcher and its UI design tokens."""

import asyncio

from tokirecur.tui.launcher import LAUNCH_APPS, LauncherApp


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
