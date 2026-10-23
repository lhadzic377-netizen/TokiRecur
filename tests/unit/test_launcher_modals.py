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

from tokirecur import ctl
from tokirecur.tui.launcher import LauncherApp
from textual.widgets import Select


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


@pytest.mark.asyncio
async def test_the_dropdown_opens_and_the_option_list_moves(one_model):
    app = LauncherApp()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        # Down to ctx_size (gguf, llama_port, daemon_port, ctx_size).
        await pilot.press("down", "down", "down")
        await pilot.pause()
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
        await pilot.press("down", "down", "down")
        await pilot.pause()
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
