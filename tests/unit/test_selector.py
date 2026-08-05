"""Tests for the interactive launcher selector and systemd service mgmt."""

from dataclasses import dataclass

from tokirecur.selector import default_index, launch_cmd
from tokirecur.service import unit_text


@dataclass
class FakeSpec:
    name: str
    model_id: str
    llama_port: int
    daemon_port: int
    ctx_size: int = 131072

    @property
    def llama_url(self) -> str:
        return f"http://127.0.0.1:{self.llama_port}"

    @property
    def daemon_url(self) -> str:
        return f"http://127.0.0.1:{self.daemon_port}"


def spec(name: str = "laguna", model_id: str = "laguna-s-2.1") -> FakeSpec:
    return FakeSpec(name=name, model_id=model_id, llama_port=8090, daemon_port=8789)


def test_chat_hermes_interactive():
    plan = launch_cmd("chat", "hermes", spec())
    assert plan["argv"] == ["hermes", "chat", "--tui", "--provider", "tokirecur-laguna",
                            "-m", "laguna-s-2.1"]
    assert not plan["detach"]


def test_chat_opencode_opens_tui_with_model():
    plan = launch_cmd("chat", "opencode", spec())
    assert plan["argv"] == ["opencode", "-m", "tokirecur-laguna/laguna-s-2.1"]
    assert not plan["detach"]


def test_chat_openclaw_opens_gateway_tui():
    plan = launch_cmd("chat", "openclaw", spec())
    assert plan["argv"] == ["openclaw", "tui"]
    assert not plan["detach"]


def test_chat_toki_uses_model_urls():
    plan = launch_cmd("chat", "toki", spec())
    # The launcher's Chat entry opens the TUI chat client, not the
    # plain-text REPL.
    assert plan["argv"] == [
        "toki", "tui", "--raw-url", "http://127.0.0.1:8090",
        "--daemon-url", "http://127.0.0.1:8789",
    ]


def test_tui_opencode_preselects_model():
    plan = launch_cmd("tui", "opencode", spec())
    assert plan["argv"] == ["opencode", "-m", "tokirecur-laguna/laguna-s-2.1"]
    assert not plan["detach"]


def test_tui_hermes_uses_provider():
    plan = launch_cmd("tui", "hermes", spec())
    assert plan["argv"] == ["hermes", "--tui", "chat", "--provider", "tokirecur-laguna",
                            "-m", "laguna-s-2.1"]
    assert not plan["detach"]


def test_tui_toki_includes_auth_token(monkeypatch):
    monkeypatch.setenv("TOKI_AUTH_TOKEN", "demo-token")
    plan = launch_cmd("tui", "toki", spec())
    assert "--auth-token" in plan["argv"]


def test_web_toki_detaches_and_opens_url():
    plan = launch_cmd("web", "toki", spec())
    assert plan["detach"]
    assert plan["argv"] == ["toki", "web", "--detach",
                            "--raw-url", "http://127.0.0.1:8090",
                            "--daemon-url", "http://127.0.0.1:8789"]
    assert plan["url"] == "http://127.0.0.1:7777"


def test_web_openclaw_just_opens_browser():
    plan = launch_cmd("web", "openclaw", spec())
    assert plan["argv"] == []
    assert plan["url"] == "http://127.0.0.1:18789"


def test_web_hermes_dashboard():
    plan = launch_cmd("web", "hermes", spec())
    assert plan["argv"] == ["hermes", "dashboard", "--port", "9119"]
    assert plan["url"] == "http://127.0.0.1:9119"


def test_unknown_combo_raises():
    import pytest
    with pytest.raises(ValueError):
        launch_cmd("web", "nope", spec())
    with pytest.raises(ValueError):
        launch_cmd("desktop", "opencode", spec())


def test_default_index():
    options = [("9b", ""), ("laguna", ""), ("35b", "")]
    assert default_index(options, "laguna") == 2
    assert default_index(options, None) == 1
    assert default_index(options, "missing") == 1


def test_unit_template_contains_execstart_and_token():
    text = unit_text()
    assert "%h/.local/bin/toki model use %i --keep-others" in text
    assert "WantedBy=default.target" in text
    assert "TOKI_AUTH_TOKEN=" in text
    assert "TimeoutStartSec=1800" in text
