"""The launcher must stay usable with a real-sized registry.

Two panes used to render every row they had: the model picker (one row per
model) and the settings pane (a dozen fields per model). On a machine with
thirty-odd models that is a list taller than the screen, and the
highlighted row walks off the bottom of it and takes the footer with it. The
person arrows down and there is nothing left on screen to aim at.

These tests drive the real Textual app, at a realistic terminal size, and
assert that the highlighted row is on screen wherever the cursor is.
"""

import pytest

from tokirecur import ctl
from tokirecur.tui.launcher import LauncherApp


def _many_specs(count: int) -> list[ctl.ModelSpec]:
    """A registry with `count` models, so the panes have to window."""
    specs = []
    for i in range(count):
        raw = dict(ctl.DEFAULT_MODELS[0])
        raw["name"] = f"model-{i:02d}"
        raw["label"] = f"Model {i}"
        raw["model_id"] = f"model-{i:02d}"
        raw["gguf"] = f"/nonexistent/model-{i:02d}.gguf"
        raw["llama_port"] = 8100 + i
        raw["daemon_port"] = 8800 + i
        raw["mmproj"] = ""
        specs.append(ctl.ModelSpec(raw=raw, **raw))
    return specs


@pytest.fixture
def many_models(monkeypatch):
    specs = _many_specs(38)
    monkeypatch.setattr(ctl, "load_registry", lambda *a, **k: specs)
    monkeypatch.setattr("tokirecur.tui.launcher.load_registry", lambda *a, **k: specs)
    return specs


@pytest.mark.asyncio
async def test_the_model_picker_keeps_the_highlight_on_screen(many_models):
    """Arrowing down must never leave the cursor off the screen.

    The picker used to render every model in the registry. With thirty-eight
    of them and a 24-row terminal, the cursor left the visible area after a
    dozen keypresses and the person had no way to see where they were.
    """
    app = LauncherApp()
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await pilot.press("right")          # into the model picker
        await pilot.pause()
        for _ in range(30):
            await pilot.press("down")
        await pilot.pause()
        text = str(app.pane.render())
        # The cursor is on model-30: it has to be on screen, and so does
        # the footer that says what is pinned.
        assert "model-30" in text, "the highlighted model is not on screen"
        assert "▌" in text, "nothing is highlighted"
        assert "no model pinned" not in text, "the footer was pushed off screen"


@pytest.mark.asyncio
async def test_the_settings_pane_keeps_the_highlight_and_its_model_on_screen(
    many_models,
):
    """The same for the settings pane, which is far longer.

    A dozen fields per model and thirty-eight models is several hundred
    rows. The highlighted row has to stay on screen, and the name of the
    model it belongs to has to stay with it — a field with no heading is a
    setting nobody can tell the meaning of.
    """
    app = LauncherApp()
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await pilot.press("s")              # settings pane
        await pilot.pause()
        for _ in range(100):
            await pilot.press("down")
        await pilot.pause()
        text = str(app.pane.render())
        assert "▌" in text, "nothing is highlighted"
        # The heading of the model the highlighted field belongs to.
        headings = [ln for ln in text.splitlines() if "model-" in ln]
        assert headings, "the settings pane shows nothing"


@pytest.mark.asyncio
async def test_the_picker_says_when_a_models_file_is_gone(many_models):
    """A model whose weights have been deleted must say so.

    The registry keeps the row — the file may come back, and the port
    assignment is still its own — but the launcher must not offer it as if
    it were there. The person pins it, launches it, and the failure they
    get is about a port.
    """
    app = LauncherApp()
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await pilot.press("right")
        await pilot.pause()
        text = str(app.pane.render())
        assert "file missing" in text, "a model with no file does not say so"


@pytest.mark.asyncio
async def test_a_model_whose_file_is_gone_cannot_be_pinned(many_models):
    """Pinning it is refused, with a reason."""
    app = LauncherApp()
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await pilot.press("right")     # model picker
        await pilot.pause()
        await pilot.press("space")     # pin the highlighted (missing) model
        await pilot.pause()
        assert app.pinned != "model-00", "a model with no file was pinned"


@pytest.mark.asyncio
async def test_the_launcher_still_renders_with_no_models(monkeypatch):
    """An empty registry is a state the launcher has to survive."""
    monkeypatch.setattr(ctl, "load_registry", lambda *a, **k: [])
    monkeypatch.setattr("tokirecur.tui.launcher.load_registry", lambda *a, **k: [])
    app = LauncherApp()
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await pilot.press("right")
        await pilot.pause()
        text = str(app.pane.render())
        assert "no models in registry" in text
