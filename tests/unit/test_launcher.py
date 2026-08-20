"""Tests for the Ollama-style Textual launcher and its UI design tokens."""

import asyncio

import pytest

from tokirecur import ctl
from tokirecur.tui.launcher import LAUNCH_APPS, LauncherApp


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Point every launcher test at a throwaway registry + active-marker
    so running the suite never mutates the user's real config. Seeds
    the reference DEFAULT_MODELS so the launcher has rows to render."""
    import yaml

    reg = tmp_path / "models.yaml"
    marker = tmp_path / "active-model"
    reg.write_text(yaml.safe_dump({"models": ctl.DEFAULT_MODELS}, sort_keys=False))
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", marker)
    monkeypatch.setenv("TOKI_MODEL_REGISTRY", str(reg))
    monkeypatch.setenv("TOKI_ACTIVE_MODEL", str(marker))
    return reg


def test_apps_bind_to_pinned_model():
    keys = [a.key for a in LAUNCH_APPS]
    assert keys == ["toki", "opencode", "hermes", "openclaw", "codex", "web"]
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

    asyncio.run(run_app())
    assert calls


def test_launcher_opens_empty_registry():
    """A fresh install (zero models) still opens; nothing is pinned
    and no model row is highlighted."""
    import yaml

    ctl.DEFAULT_REGISTRY.write_text(yaml.safe_dump({"models": []}, sort_keys=False))

    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            assert app.specs == []
            assert app.pinned is None
        return app

    app = asyncio.run(run_app())
    assert app.specs == []
    assert app.pinned is None


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

    gguf is a model-swap field (restart-required + harder-stop), so the
    commit goes through ConfirmRestartScreen. Drive the modal flow:
    push the edit dismiss → assert the confirm modal opens → press
    Proceed → registry writes. The restart itself doesn't run because
    the daemon isn't reachable in tests, so the spec still gets the
    new gguf value before the runtime error fires (caught and surfaced
    as a warning notification, registry still updated).
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
            # The confirm modal opened; press Proceed.
            assert app.screen.__class__.__name__ == "ConfirmRestartScreen"
            await pilot.click("#confirm-ok")
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


def test_settings_choices_ctx_includes_50000_100000_200000():
    """The three ctx presets that match the registry's actual default
    values must be reachable from the dropdown without typing them in
    through custom…."""
    from tokirecur.tui.launcher import _SETTINGS_CHOICES

    ctx = _SETTINGS_CHOICES["ctx_size"]
    for v in ("50000", "100000", "200000"):
        assert ctx.get(v) == v, f"missing ctx_size preset {v}"


def test_settings_choices_ctx_ordering_ascending():
    """Numeric presets appear in ascending order so the dropdown is
    scannable; ``custom…`` stays last."""
    from tokirecur.tui.launcher import _CUSTOM_VALUE, _SETTINGS_CHOICES

    values = [
        int(v) for v in _SETTINGS_CHOICES["ctx_size"].values()
        if v != _CUSTOM_VALUE
    ]
    assert values == sorted(values), (
        f"ctx_size presets out of order: {values}"
    )
    # And custom is the last entry.
    keys = list(_SETTINGS_CHOICES["ctx_size"].keys())
    assert keys[-1] == "custom…"


def test_settings_choice_screen_cancel_returns_none():
    """Escape on the choice modal dismisses with None (no commit)."""
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
            await pilot.press("escape")
            await pilot.pause()
            return app.specs[0].kv

    # 9b defaults to f16; cancelling must leave it untouched.
    assert asyncio.run(run_app()) == "f16"


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
    auto-commits (no Save-button dance) and writes it to the registry.

    kv is a restart-required field, so the commit goes through the
    ConfirmRestartScreen after the choice modal dismisses. Press
    Proceed on the confirm modal; the restart itself fails (no live
    daemon in tests) but the registry write still happens before the
    RuntimeError fires — caught and surfaced as a warning.
    """
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
            # on_select_changed auto-commits: setting the value dismisses
            # the choice modal and the restart confirmation appears.
            app.screen.query_one("#settings-choice-select", Select).value = "q8_0"
            await pilot.pause()
            assert app.screen.__class__.__name__ == "ConfirmRestartScreen"
            await pilot.click("#confirm-ok")
            await pilot.pause()
            return app.specs[0].kv

    assert asyncio.run(run_app()) == "q8_0"


def test_settings_dropdown_preselects_unlisted_current_value():
    """A registry value outside the presets (9b defaults to ctx 200000)
    is surfaced as its own option so the dropdown opens pre-selected."""
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
    """Picking ``custom…`` auto-commits and dismisses the choice modal,
    which chains into the free-text editor for the same row."""
    from textual.widgets import Select

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
            # Auto-commit on change: setting to _CUSTOM_VALUE dismisses
            # the choice modal and the launcher chains into the text
            # editor.
            app.screen.query_one("#settings-choice-select", Select).value = _CUSTOM_VALUE
            await pilot.pause()
            return app.screen.__class__.__name__

    assert asyncio.run(run_app()) == "SettingsEditScreen"



# ── Settings panel v2: live values + confirmation modal + dynamic choices ──


def test_settings_renders_two_columns_when_registry_and_live_differ():
    """When the live slot context disagrees with the registry value
    (e.g. the user just edited 131072 but RAM-clamped to 64000), the
    settings row shows both numbers so the user can see what's running."""
    # 9b defaults to ctx_size 200000. Inject a fake status poll so the
    # "live" value disagrees with the registry.
    from tokirecur import ctl
    from tokirecur.tui.launcher import _settings_rows

    class FakeApp:
        _statuses = {"9b": {"ctx": 64000}}

    rows = _settings_rows(ctl.load_registry(), app=FakeApp())
    ctx_rows = [(n, f, v) for n, f, v in rows if f == "ctx_size"]
    assert len(ctx_rows) >= 1
    # At least one ctx_size row shows the two-column format.
    assert any("reg:" in v and "live:" in v for _, _, v in ctx_rows), (
        f"expected reg:/live: in some ctx row, got {ctx_rows!r}"
    )


def test_settings_renders_single_value_when_registry_and_live_match():
    """When the live slot context matches the registry, show the plain
    value (no reg: / live: prefix)."""
    from tokirecur import ctl
    from tokirecur.tui.launcher import _settings_rows

    class FakeApp:
        _statuses = {"9b": {"ctx": 200000}}

    rows = _settings_rows(ctl.load_registry(), app=FakeApp())
    ctx_rows = [(n, f, v) for n, f, v in rows if f == "ctx_size"]
    # 9b's ctx_size is 200000 (default), live is 200000 — no prefix.
    matching = [v for _, _, v in ctx_rows if v == "200000"]
    assert matching


def test_commit_edit_marks_restarting_field(monkeypatch):
    """A restart-required edit (ctx_size, kv, ...) registers an
    in-flight marker; sync-only fields do not."""
    import time as _time

    async def run_app():
        # Patch update_spec_field so the test never touches the daemon
        # but still records that _commit_edit was called.
        from tokirecur import ctl
        from tokirecur.tui.launcher import _RESTART_REQUIRED_FIELDS, _SYNC_ONLY_FIELDS

        async with LauncherApp().run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            app = pilot.app

            called: dict[str, str] = {}

            def fake_update(name, field, value, apply=False):
                called["field"] = field
                return None

            monkeypatch.setattr(ctl, "update_spec_field", fake_update)

            # Restart-required field.
            app._commit_edit("9b", "ctx_size", "4096", apply=True)
            assert ("9b", "ctx_size") in app._restarting
            assert _RESTART_REQUIRED_FIELDS and "ctx_size" in _RESTART_REQUIRED_FIELDS

            # Sync-only field must NOT register a marker.
            app._restarting.clear()
            app._commit_edit("9b", "max_tokens", "256", apply=False)
            assert not app._restarting
            assert _SYNC_ONLY_FIELDS and "max_tokens" in _SYNC_ONLY_FIELDS

    asyncio.run(run_app())
    _time.monotonic()  # ensure import stays


def test_poll_statuses_clears_restarting_when_live_matches_reg():
    """The in-flight marker drops the moment the live value agrees
    with the registry on the next poll."""
    import time as _time

    async def run_app():
        async with LauncherApp().run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            app = pilot.app

            # Inject a "just-committed" marker for ctx_size on 9b.
            app._restarting[("9b", "ctx_size")] = _time.monotonic()

            # And tell the launcher the live slot already agrees with
            # the registry (the restart finished between commit and
            # this poll).
            spec_9b = next(s for s in app.specs if s.name == "9b")
            app._statuses["9b"] = {"ctx": spec_9b.ctx_size, "llama": True,
                                    "daemon": True}

            app._poll_statuses()
            assert ("9b", "ctx_size") not in app._restarting

    asyncio.run(run_app())


def test_poll_statuses_clears_restarting_after_timeout(monkeypatch):
    """An 8 s ceiling drops the marker even if the live value never
    catches up (poll stalled, daemon wedged, etc.)."""
    import time as _time

    async def run_app():
        async with LauncherApp().run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            app = pilot.app

            # Pretend the marker was set 9 seconds ago.
            app._restarting[("9b", "ctx_size")] = _time.monotonic() - 9.0

            # Live value deliberately stays "wrong" — the cap is
            # what should save us.
            app._statuses["9b"] = {"ctx": 1, "llama": True, "daemon": True}

            app._poll_statuses()
            assert ("9b", "ctx_size") not in app._restarting

    asyncio.run(run_app())


def test_render_settings_shows_restarting_suffix():
    """A row whose (name, field) is in ``_restarting`` renders the
    yellow "⟳ restarting" suffix until cleared."""
    from rich.text import Text

    async def run_app():
        from tokirecur.tui.launcher import invalidate_settings_rows

        async with LauncherApp().run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            app = pilot.app
            app._restarting[("9b", "ctx_size")] = 0.0
            invalidate_settings_rows()
            rendered = app._render_settings(100)
            blob = Text().join(rendered).plain
            assert "⟳ restarting" in blob, (
                f"expected restarting suffix in rendered settings; "
                f"got: {blob!r}"
            )

    asyncio.run(run_app())


def test_settings_choices_for_ctx_filters_over_model_max(monkeypatch):
    """The dynamic ctx_size dropdown excludes values above the model's
    GGUF max context. A 9B with general.context_length=40960 shouldn't
    offer 131072."""
    from tokirecur.tui.launcher import _choices_for_field

    class FakeSpec:
        ctx_size = 200000  # registry value
        gguf = "/tmp/fake.gguf"

    # Stub _model_max_ctx to return 40960 (a small-model cap).
    from tokirecur import ctl
    monkeypatch.setattr(ctl, "_model_max_ctx", lambda spec: 40960)
    # RAM plan says we can fit 200000 — that should NOT widen the cap.
    monkeypatch.setattr(ctl, "_fit_plan", lambda spec, **kw: (200000, "f16", 1))

    choices = _choices_for_field(FakeSpec(), "ctx_size")
    assert choices is not None
    # All preset integers above 40960 should be filtered out.
    for label, value in choices.items():
        if value in ("__custom__",):
            continue
        assert int(value) <= 40960, (
            f"choice {label}={value} exceeds model max 40960"
        )
    # custom… option is always preserved.
    assert "__custom__" in choices.values()


def test_settings_choices_for_non_ctx_fields_are_static(monkeypatch):
    """Non-ctx_size fields return the same dict for every spec."""
    from tokirecur.tui.launcher import _SETTINGS_CHOICES, _choices_for_field

    class FakeSpec:
        pass

    assert _choices_for_field(FakeSpec(), "kv") == _SETTINGS_CHOICES["kv"]
    assert _choices_for_field(FakeSpec(), "parallel") == _SETTINGS_CHOICES["parallel"]
    assert _choices_for_field(FakeSpec(), "reasoning") == _SETTINGS_CHOICES["reasoning"]


def test_settings_sync_only_field_does_not_show_confirm():
    """Editing max_tokens (sync-only) commits directly without a
    ConfirmRestartScreen."""
    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            # Navigate to max_tokens (sync-only). The first row is gguf;
            # max_tokens is several rows down in the default ordering.
            app.settings_idx = _setting_row_index(app, "max_tokens")
            app._on_edit_dismissed("4096")
            await pilot.pause()
            # No confirm modal opened; the registry was updated.
            assert app.screen.__class__.__name__ != "ConfirmRestartScreen"
            return app.specs[0].max_tokens

    assert asyncio.run(run_app()) == 4096


def test_settings_noop_field_does_not_show_confirm():
    """Editing always_on (no-op field) commits directly."""
    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            app.settings_idx = _setting_row_index(app, "always_on")
            app._on_edit_dismissed("true")
            await pilot.pause()
            assert app.screen.__class__.__name__ != "ConfirmRestartScreen"
            return app.specs[0].always_on

    assert asyncio.run(run_app()) is True


def test_settings_restart_required_field_opens_confirm_modal():
    """Editing ctx_size (restart-required) pushes the ConfirmRestartScreen
    and the registry is NOT updated until the user presses Proceed."""
    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            original_ctx = app.specs[0].ctx_size
            app._on_edit_dismissed("8192")
            await pilot.pause()
            # Confirm modal should be on screen.
            assert app.screen.__class__.__name__ == "ConfirmRestartScreen"
            # Registry NOT updated yet (modal gates the apply).
            assert app.specs[0].ctx_size == original_ctx
            # Press Cancel: registry still untouched, modal dismisses.
            await pilot.click("#confirm-cancel")
            await pilot.pause()
            assert app.specs[0].ctx_size == original_ctx
        return None

    asyncio.run(run_app())


def test_settings_restart_required_confirm_proceed_triggers_apply(monkeypatch):
    """Pressing Proceed on the confirm modal drives the apply path:
    update_spec_field(apply=True) is called, which calls use_model."""
    from tokirecur import ctl as ctl_mod

    use_model_calls: list[str] = []
    sync_calls: list[str] = []

    def fake_use_model(name, **kwargs):
        use_model_calls.append(name)
        return ctl_mod.get_model(name)

    def fake_sync(spec, **kwargs):
        sync_calls.append(spec.name)

    monkeypatch.setattr(ctl_mod, "use_model", fake_use_model)
    monkeypatch.setattr(ctl_mod, "sync_cli_configs", fake_sync)

    async def run_app():
        app = LauncherApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            # Navigate to ctx_size (restart-required, not model-swap —
            # so the test doesn't have to navigate the harder-stop modal).
            app.settings_idx = _setting_row_index(app, "ctx_size")
            app._on_edit_dismissed("8192")
            await pilot.pause()
            assert app.screen.__class__.__name__ == "ConfirmRestartScreen"
            await pilot.click("#confirm-ok")
            await pilot.pause()
            return app.specs[0].ctx_size

    new_ctx = asyncio.run(run_app())
    assert new_ctx == 8192
    assert use_model_calls == ["9b"]
    # sync_cli_configs was NOT called separately (use_model calls it
    # internally); the TUI relies on use_model's existing sync.
    assert sync_calls == []
