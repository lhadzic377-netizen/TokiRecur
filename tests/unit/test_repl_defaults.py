"""The interactive shell talks to the model that is actually running.

Three faults, all found by driving `toki interactive` in a terminal:

- Its defaults were a hard-coded path to a starter download and the
  ports 8088/8787, so every command that talked to the model failed with
  "connection refused" while the daemon was plainly running — on 8811.
- It sent no auth token, so the daemon's 401 came back as "connection
  refused" in the message and the hint blamed the daemon for not being up.
- `run` asked for a single JSON object and then parsed it as
  server-sent events, so the text never arrived and the whole envelope
  was printed instead.
"""


import pytest

from tokirecur import ctl, repl


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A registry with one model on its own ports."""
    import yaml

    reg = tmp_path / "models.yaml"
    marker = tmp_path / "active"
    weights = tmp_path / "m.gguf"
    weights.write_bytes(b"x")
    reg.write_text(yaml.safe_dump({"models": [{
        "name": "m", "label": "M", "gguf": str(weights),
        "llama_port": 9100, "daemon_port": 9101,
        "ctx_size": 4096, "max_tokens": 1000, "model_id": "m",
        "parallel": 1, "kv": "f16",
    }]}))
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", marker)
    marker.write_text("m")
    monkeypatch.setenv("TOKI_MODEL_REGISTRY", str(reg))
    monkeypatch.setenv("TOKI_ACTIVE_MODEL", str(marker))
    monkeypatch.delenv("TOKI_MODEL", raising=False)
    return reg


def test_the_shell_uses_the_active_models_ports(registry):
    cfg = repl.ReplConfig()
    assert cfg.llama_port == 9100
    assert cfg.daemon_port == 9101
    assert cfg.ctx_size == 4096
    assert cfg.backend_url == "http://127.0.0.1:9100"


def test_an_explicit_model_still_wins(registry, monkeypatch, tmp_path):
    monkeypatch.setenv("TOKI_MODEL", str(tmp_path / "other.gguf"))
    assert repl.ReplConfig().model == tmp_path / "other.gguf"


def test_the_daemon_token_is_sent(monkeypatch):
    monkeypatch.setenv("TOKI_AUTH_TOKEN", "s3cret")
    headers = repl._auth_headers()
    assert headers.get("Authorization") == "Bearer s3cret"


def test_a_refused_token_is_named_in_the_hint(registry):
    """401 is not the same as "not running", and the hint said it was."""
    hint = repl._why_it_failed(
        RuntimeError("401 Unauthorized"), 9101
    )
    assert "token" in hint
    assert "running" not in hint
    assert "9101" in hint


def test_a_refused_connection_is_named_in_the_hint(registry):
    hint = repl._why_it_failed(
        RuntimeError("[Errno 111] Connection refused"), 9101
    )
    assert "running" in hint
    assert "9101" in hint


def test_run_asks_for_the_stream_it_reads(registry, monkeypatch):
    """It used to ask for one JSON object and parse it as events."""
    sent = {}

    class _Stream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield 'data: {"choices": [{"delta": {"content": "OK"}}]}'
            yield "data: [DONE]"

    def fake_stream(method, url, json=None, timeout=None, headers=None):
        sent["payload"] = json
        sent["headers"] = headers
        return _Stream()

    monkeypatch.setattr("httpx.stream", fake_stream)
    printed: list[str] = []
    monkeypatch.setattr(
        repl.console, "print",
        lambda *a, **k: printed.append("".join(str(x) for x in a)),
    )
    repl.cmd_run(["hello"], repl.ReplConfig())
    assert sent["payload"]["stream"] is True, sent["payload"]
    assert "OK" in "".join(printed)
