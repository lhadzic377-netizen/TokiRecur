"""The launcher's public-server screen.

"p" has to reach a screen that starts nothing until Enter, and the
credentials have to be readable and copyable once the tunnel is live.
"""

import pytest
from textual.widgets import Static

from tokirecur import ctl
from tokirecur.public_server import PublicServer
from tokirecur.tui.launcher import LauncherApp, PublicServerScreen

URL = "https://griffin-designing-sold-polls.trycloudflare.com"


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    import yaml

    reg = tmp_path / "models.yaml"
    marker = tmp_path / "active-model"
    reg.write_text(yaml.safe_dump({"models": [{
        "name": "m", "label": "M", "gguf": str(tmp_path / "m.gguf"),
        "llama_port": 9100, "daemon_port": 9101, "ctx_size": 200000,
        "max_tokens": 100000, "model_id": "m", "parallel": 1, "kv": "f16",
    }]}))
    (tmp_path / "m.gguf").write_bytes(b"x")
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", marker)
    monkeypatch.setenv("TOKI_MODEL_REGISTRY", str(reg))
    monkeypatch.setenv("TOKI_ACTIVE_MODEL", str(marker))
    monkeypatch.setattr(ctl, "daemon_reachable", lambda spec: True)
    return reg


def _spec():
    return ctl.load_registry()[0]


class TrackedServer(PublicServer):
    """A PublicServer that records whether it was stopped."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stopped = False

    def stop(self, wait=4.0):
        self.stopped = True


def _fake_starter(seen: dict, factory=TrackedServer):
    def starter(*, port, api_key, model):
        seen.update(port=port, api_key=api_key, model=model)
        return factory(url=URL, api_key=api_key, model=model, port=port)
    return starter


async def test_p_opens_the_public_server_screen():
    app = LauncherApp()
    async with app.run_test() as pilot:
        await pilot.press("p")
        await pilot.pause()
        assert isinstance(app.screen, PublicServerScreen)


async def test_enter_starts_the_tunnel_and_shows_the_credentials():
    app = LauncherApp()
    seen: dict = {}
    async with app.run_test() as pilot:
        await app.push_screen(
            PublicServerScreen("m", _spec(), starter=_fake_starter(seen))
        )
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause(0.05)
            body = app.screen.query_one("#public-body", Static).render()
            if URL in str(body):
                break
        body = str(app.screen.query_one("#public-body", Static).render())

    assert seen["port"] == 9101
    assert seen["api_key"]
    assert URL in body
    assert f"{URL}/v1" in body
    assert f"{URL}/v1beta" in body
    assert "API key" in body


async def test_escape_closes_and_stops_the_tunnel():
    app = LauncherApp()
    async with app.run_test() as pilot:
        screen = PublicServerScreen("m", _spec(), starter=_fake_starter({}))
        await app.push_screen(screen)
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause(0.05)
            if screen._server is not None:
                break
        server = screen._server
        assert isinstance(server, TrackedServer)
        await pilot.press("escape")
        for _ in range(40):
            await pilot.pause(0.05)
            if server.stopped:
                break
        assert server.stopped
        assert not isinstance(app.screen, PublicServerScreen)
