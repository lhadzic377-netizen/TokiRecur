"""The public server, from the command line.

`toki public-server` is the launcher's public-server row without the
screen: the same quick tunnel, the same credentials, printed where
they can be copied from.
"""

import time

import pytest
from click.testing import CliRunner

from tokirecur import ctl
from tokirecur.cli import cli
from tokirecur.public_server import PublicServer

URL = "https://griffin-designing-sold-polls.trycloudflare.com"


@pytest.fixture(autouse=True)
def one_model(tmp_path, monkeypatch):
    """A registry with one model, and a daemon that answers."""
    import yaml

    registry = tmp_path / "models.yaml"
    registry.write_text(yaml.safe_dump({"models": [{
        "name": "m", "label": "M", "gguf": str(tmp_path / "m.gguf"),
        "llama_port": 9100, "daemon_port": 9101, "ctx_size": 200000,
        "max_tokens": 100000, "model_id": "qwen3.8-flash", "parallel": 1,
        "kv": "f16",
    }]}))
    (tmp_path / "m.gguf").write_bytes(b"x")
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", registry)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", tmp_path / "active")
    monkeypatch.setenv("TOKI_MODEL_REGISTRY", str(registry))
    (tmp_path / "active").write_text("m\n")
    monkeypatch.setenv("TOKI_ACTIVE_MODEL", str(tmp_path / "active"))
    monkeypatch.setattr(ctl, "daemon_reachable", lambda spec: True)
    return registry


def _fake_starter(seen: dict):
    def starter(port, api_key, model, **kwargs):
        seen.update(port=port, api_key=api_key, model=model)
        return PublicServer(url=URL, api_key=api_key, model=model, port=port)
    return starter


def test_the_command_prints_the_credentials_and_the_endpoints(monkeypatch):
    from tokirecur import public_server

    seen: dict = {}
    monkeypatch.setattr(public_server, "start_public_server", _fake_starter(seen))
    # The command holds the tunnel open until Ctrl-C; the test sends it.
    def _interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(time, "sleep", _interrupt)

    result = CliRunner().invoke(cli, ["public-server"])

    assert result.exit_code == 0, result.output
    assert seen["port"] == 9101
    assert seen["model"] == "qwen3.8-flash"
    assert URL in result.output
    assert "Base URL (OpenAI)" in result.output
    assert f"{URL}/v1" in result.output
    assert f"{URL}/v1beta" in result.output
    assert "Anthropic" in result.output
    assert "generateContent" in result.output


def test_it_refuses_when_the_model_is_not_serving(monkeypatch):
    from tokirecur import public_server

    monkeypatch.setattr(ctl, "daemon_reachable", lambda spec: False)
    monkeypatch.setattr(
        public_server, "start_public_server",
        lambda *a, **k: pytest.fail("the tunnel must not start"),
    )

    result = CliRunner().invoke(cli, ["public-server"])

    assert result.exit_code != 0
    assert "not serving" in result.output


def test_the_command_is_registered():
    result = CliRunner().invoke(cli, ["public-server", "--help"])
    assert result.exit_code == 0
    assert "public HTTPS" in result.output
