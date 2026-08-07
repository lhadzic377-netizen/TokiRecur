"""Unit tests for the Windows CLI chat (ollama-style REPL)."""

import asyncio

import pytest
import yaml

from tokirecur import clichat, ctl


@pytest.fixture
def env(tmp_path, monkeypatch):
    reg = tmp_path / "models.yaml"
    state = tmp_path / "state"
    marker = tmp_path / "active-model"
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    monkeypatch.setattr(ctl, "STATE_ROOT", state)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", marker)
    return tmp_path


def _write_registry(env, specs):
    ctl.DEFAULT_REGISTRY.write_text(yaml.safe_dump({"models": specs}))


def _spec(name, **kw):
    raw = {
        "name": name, "label": name, "gguf": f"/tmp/{name}.gguf",
        "llama_port": 8088, "daemon_port": 8787, "ctx_size": 4096,
        "parallel": 2, "kv": "f16", "max_tokens": 2048,
        "model_id": f"acme/{name}-GGUF",
    }
    raw.update(kw)
    return raw


def test_resolve_model_explicit_name(env):
    _write_registry(env, [_spec("a"), _spec("b")])
    spec = clichat._resolve_model("b")
    assert spec.name == "b"


def test_resolve_model_active(env):
    _write_registry(env, [_spec("a"), _spec("b")])
    ctl.set_active("a")
    spec = clichat._resolve_model(None)
    assert spec.name == "a"


def test_resolve_model_single(env):
    _write_registry(env, [_spec("solo")])
    spec = clichat._resolve_model(None)
    assert spec.name == "solo"


def test_resolve_model_empty_registry(env, capsys):
    _write_registry(env, [])
    with pytest.raises(SystemExit) as exc:
        clichat._resolve_model(None)
    assert exc.value.code == 1
    assert "no models installed" in capsys.readouterr().out


def test_resolve_model_picker(env, monkeypatch):
    _write_registry(env, [_spec("first"), _spec("second")])
    monkeypatch.setattr("builtins.input", lambda _: "2")
    spec = clichat._resolve_model(None)
    assert spec.name == "second"


def test_resolve_model_unknown_name(env):
    _write_registry(env, [_spec("a")])
    with pytest.raises(KeyError):
        clichat._resolve_model("nope")


def test_max_tokens_env(monkeypatch):
    monkeypatch.delenv("TOKI_MAX_TOKENS", raising=False)
    assert clichat._max_tokens() == 2048
    monkeypatch.setenv("TOKI_MAX_TOKENS", "4096")
    assert clichat._max_tokens() == 4096
    monkeypatch.setenv("TOKI_MAX_TOKENS", "junk")
    assert clichat._max_tokens() == 2048


def test_read_prompt_plain(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "hello world")
    assert asyncio.run(clichat._read_prompt(">>> ")) == "hello world"


def test_read_prompt_eof(monkeypatch, capsys):
    def boom(_):
        raise EOFError

    monkeypatch.setattr("builtins.input", boom)
    assert asyncio.run(clichat._read_prompt(">>> ")) is None


def test_read_prompt_multiline(monkeypatch):
    answers = iter(['"""', "line one", "line two", '"""'])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    result = asyncio.run(clichat._read_prompt(">>> "))
    assert result == "line one\nline two"


def test_read_prompt_multiline_inline(monkeypatch):
    answers = iter(['"""start here', "more text", 'end"""'])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    result = asyncio.run(clichat._read_prompt(">>> "))
    assert result == "start here\nmore text\nend"


def test_help_lists_commands():
    assert "/bye" in clichat._HELP
    assert "/clear" in clichat._HELP
    assert "/help" in clichat._HELP
