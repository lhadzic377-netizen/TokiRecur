"""A dropdown has to be usable.

The launcher's arrow keys and Enter were priority bindings, and a priority
binding is checked before the focused widget. So while a settings modal
was open, the launcher took "down" and "enter" for itself: the option list
never moved, no value was ever picked, and nothing was ever committed. The
settings pane could be opened and looked at, and not edited.

These keys are not priority bindings now. Inside a modal the focused
widget gets them, and outside one there is nothing that consumes them, so
the main pane still navigates and launches.
"""

import pytest
from textual.widgets import Select

from tokirecur import ctl
from tokirecur.tui.launcher import LauncherApp, _settings_rows


@pytest.fixture
def one_model(monkeypatch, tmp_path):
    """A registry with one model, so the settings pane is one screen long."""
    import yaml

    reg = tmp_path / "models.yaml"
    marker = tmp_path / "active"
    weights = tmp_path / "m.gguf"
    weights.write_bytes(b"x")
    reg.write_text(yaml.safe_dump({"models": [{
        "name": "m", "label": "M", "gguf": str(weights),
        "llama_port": 9100, "daemon_port": 9101,
        "ctx_size": 200000, "max_tokens": 100000, "model_id": "m",
        "parallel": 1, "kv": "f16",
    }]}))
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", marker)
    monkeypatch.setenv("TOKI_MODEL_REGISTRY", str(reg))
    monkeypatch.setenv("TOKI_ACTIVE_MODEL", str(marker))
    return reg


async def _goto(pilot, app, field: str, name: str = "m") -> None:
    """Move the settings highlight to a row, by name.

    The pane opens on the machine-wide recorder row, so counting the
    downs to reach a model's field is a trap: a row added at the top
    shifts every model's field by one.
    """
    from tokirecur.tui.launcher import _settings_rows

    rows = _settings_rows(app.specs, app=app)
    idx = next(
        i for i, (row_name, row_field, _) in enumerate(rows)
        if row_name == name and row_field == field
    )
    for _ in range(idx):
        await pilot.press("down")
    await pilot.pause()


@pytest.mark.asyncio
async def test_the_dropdown_opens_and_the_option_list_moves(one_model):
    app = LauncherApp()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        # Down to ctx_size.
        await _goto(pilot, app, "ctx_size")
        await pilot.press("enter")
        await pilot.pause()
        assert type(app.screen).__name__ == "SettingsChoiceScreen"
        select = app.screen.query_one("#settings-choice-select", Select)
        assert not select.expanded, "the option list did not open"

        await pilot.press("down")
        await pilot.pause()
        assert select.expanded, "the option list did not open on down"
        assert app.focused.__class__.__name__ == "SelectOverlay", (
            "the launcher took the arrow key for itself"
        )


@pytest.mark.asyncio
async def test_a_picked_value_is_carried_out_of_the_modal(one_model, monkeypatch):
    """The pick has to reach the launcher, or the edit is a no-op."""
    app = LauncherApp()
    committed: list = []
    monkeypatch.setattr(
        "tokirecur.ctl.update_spec_field",
        lambda name, field, value, apply=False: committed.append((field, value)),
    )
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        await _goto(pilot, app, "ctx_size")
        await pilot.press("enter")
        await pilot.pause()
        select = app.screen.query_one("#settings-choice-select", Select)
        await pilot.press("down")     # open the list
        await pilot.pause()
        await pilot.press("down")     # move to another option
        await pilot.pause()
        await pilot.press("enter")    # pick it
        await pilot.pause()
        # ctx_size needs a restart, so a confirmation comes next.
        assert type(app.screen).__name__ == "ConfirmRestartScreen", (
            f"the pick did not reach the launcher: {type(app.screen).__name__}"
        )
        assert select.value != "200000", "the pick did not change the value"


@pytest.mark.asyncio
async def test_the_recorder_row_is_in_the_settings_pane(one_model, monkeypatch, tmp_path):
    """The recorder is a setting, so it belongs in the settings pane.

    It leads the list: it is the one setting that is not a model's
    business, and the one that writes what the user says to disk.
    """
    flag = tmp_path / "tap-flag"
    monkeypatch.setenv("TOKI_TAP_FLAG", str(flag))
    monkeypatch.setenv("TOKI_TAP_DIR", str(tmp_path / "tap"))
    monkeypatch.delenv("TOKI_TAP", raising=False)

    app = LauncherApp()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()

        rows = _settings_rows(app.specs, app=app)
        assert rows[0][0] == "machine", f"the recorder is not first: {rows[0]}"
        assert rows[0][1] == "record turns"
        assert rows[0][2] == "off", f"the recorder should read off: {rows[0][2]}"

        # Enter opens the on/off choice.
        await pilot.press("enter")
        await pilot.pause()
        assert type(app.screen).__name__ == "SettingsChoiceScreen"
        app.screen.query_one("#settings-choice-select", Select)
        await pilot.press("down")     # open the list
        await pilot.pause()
        await pilot.press("up")       # back to "on"
        await pilot.pause()
        await pilot.press("enter")    # pick it
        await pilot.pause()

        assert flag.is_file(), "the pick did not reach the switch"
        from tokirecur import tap

        assert tap.is_enabled() is True
        # The row follows the switch.
        rows = _settings_rows(app.specs, app=app)
        assert rows[0][2].startswith("on"), f"the row still reads off: {rows[0][2]}"


@pytest.mark.asyncio
async def test_the_main_pane_still_navigates_and_launches(one_model):
    """The keys the modals need are not priority bindings any more, so the
    main pane has to keep working without them."""
    app = LauncherApp()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        assert app.focus_pane == "launchers"
        before = app.launch_idx
        await pilot.press("down")
        await pilot.pause()
        assert app.launch_idx == before + 1, "the down key did not move"
        await pilot.press("up")
        await pilot.pause()
        assert app.launch_idx == before, "the up key did not move back"


@pytest.mark.asyncio
async def test_the_text_editor_takes_typing_and_enter(one_model, monkeypatch):
    """The same keys, in the other modal.

    An Input consumes printable keys, so the typing was never the problem —
    but Enter behind the modal was starting a client instead of saving,
    because the launcher's Enter was a priority binding.
    """
    app = LauncherApp()
    committed: list = []
    monkeypatch.setattr(
        "tokirecur.ctl.update_spec_field",
        lambda name, field, value, apply=False: committed.append((field, value)),
    )
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        await _goto(pilot, app, "gguf")     # a text field
        await pilot.press("enter")
        await pilot.pause()
        assert type(app.screen).__name__ == "SettingsEditScreen"
        await pilot.press("ctrl+u")          # clear the Input
        await pilot.pause()
        await pilot.press(*"hello")
        await pilot.pause()
        assert app.focused.__class__.__name__ == "Input"
        await pilot.press("enter")
        await pilot.pause()
        # gguf is a model-swap field, so a confirmation comes next.
        assert type(app.screen).__name__ == "ConfirmRestartScreen", (
            f"Enter did not reach the text field: {type(app.screen).__name__}"
        )
