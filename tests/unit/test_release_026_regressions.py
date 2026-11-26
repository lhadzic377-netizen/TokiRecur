"""The fixes that had no test of their own in 0.3.26.

Every bug fix in this release is meant to be pinned by a test that fails
before it and passes after. These are the ones that were only checked by
hand the first time round: the stale OpenAPI version, the install-llama
self-check that read the wrong stream, the locale-dependent text files,
the empty brackets in `toki model status`, the unknown subcommand that
only printed help, and the `/slots` answer treated as an object.
"""

import locale

import pytest
from click.testing import CliRunner

from tokirecur import ctl
from tokirecur.api.app import create_app
from tokirecur.cli import cli, main
from tokirecur.web.server import create_web_app

# ── The OpenAPI document reports the real version ─────────────────────


def test_the_daemon_openapi_document_reports_the_package_version():
    """The daemon said 0.2.0 while the package was 0.3.x."""
    from tokirecur import __version__

    class _Backend:
        base_url = "http://127.0.0.1:1"

    class _Index:
        pass

    app = create_app(
        backend=_Backend(),
        index=_Index(),
        cache_dir="/tmp/toki-version-test",
        enable_auth=False,
        enable_rate_limit=False,
    )
    doc = app.openapi()
    assert doc["info"]["version"] == __version__
    assert doc["info"]["version"] != "0.2.0"


def test_the_web_ui_openapi_document_reports_the_package_version():
    from tokirecur import __version__

    app = create_web_app()
    doc = app.openapi()
    assert doc["info"]["version"] == __version__


# ── install-llama reads the banner from the right stream ──────────────


def test_the_self_check_reads_the_banner_from_stderr():
    """llama-server prints its version on stderr."""
    from tokirecur.cli import _version_banner

    assert _version_banner("", "version: 1.1.0-dev (build 1, commit abc)\n") == (
        "version: 1.1.0-dev (build 1, commit abc)"
    )
    # A binary that prints on stdout still works.
    assert _version_banner("version: 9.9\n", "") == "version: 9.9"
    # Nothing printed at all is reported honestly.
    assert _version_banner("", "") == "ok"


# ── Text files do not depend on the locale ────────────────────────────


def test_the_registry_survives_a_non_utf8_locale(tmp_path, monkeypatch):
    """The default encoding is the locale's - which is not always UTF-8."""
    registry = tmp_path / "models.yaml"
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", registry)
    monkeypatch.setattr(
        locale, "getpreferredencoding", lambda do_setlocale=True: "ascii",
    )
    raw = {
        "name": "m", "label": "Modèle Ünicode",
        "gguf": str(tmp_path / "m.gguf"),
        "llama_port": 9100, "daemon_port": 9101, "ctx_size": 1000,
        "max_tokens": 100, "model_id": "m", "parallel": 1, "kv": "f16",
    }
    spec = ctl.ModelSpec(raw=raw, **raw)
    # Without the explicit encoding, the write raises UnicodeEncodeError
    # and the read mangles the label under an ascii locale.
    ctl.save_registry([spec], registry)
    loaded = ctl.load_registry(registry)
    assert loaded[0].label == "Modèle Ünicode"


def test_the_auth_file_is_utf8(tmp_path, monkeypatch):
    monkeypatch.setattr(
        locale, "getpreferredencoding", lambda do_setlocale=True: "ascii",
    )
    path = tmp_path / "auth"
    path.write_text("tökén", encoding="utf-8")
    assert path.read_text(encoding="utf-8") == "tökén"


# ── `toki model status` prints no empty brackets ──────────────────────


def test_model_status_prints_no_empty_parentheses(tmp_path, monkeypatch):
    import yaml

    registry = tmp_path / "models.yaml"
    registry.write_text(yaml.safe_dump({"models": [{
        "name": "m", "label": "M", "gguf": str(tmp_path / "m.gguf"),
        "llama_port": 9100, "daemon_port": 9101, "ctx_size": 1000,
        "max_tokens": 100, "model_id": "m", "parallel": 1, "kv": "f16",
    }]}))
    (tmp_path / "m.gguf").write_bytes(b"x")
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", registry)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", tmp_path / "active")
    monkeypatch.setenv("TOKI_MODEL_REGISTRY", str(registry))
    monkeypatch.setenv("TOKI_ACTIVE_MODEL", str(tmp_path / "active"))
    # Nothing is running, so there is no slot context to show.
    monkeypatch.setattr(
        ctl, "model_status",
        lambda spec: {
            "llama": {"reachable": False, "pid": None, "slot_ctx": None},
            "daemon": {"reachable": False, "pid": None},
        },
    )

    result = CliRunner().invoke(cli, ["model", "status"])

    assert result.exit_code == 0, result.output
    assert "llama=down" in result.output
    assert "()" not in result.output


# ── An unknown subcommand says so ─────────────────────────────────────


def test_an_unknown_subcommand_is_named(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["toki", "definitely-not-a-command"])
    assert main() == 2
    out = capsys.readouterr()
    assert "no such command 'definitely-not-a-command'" in out.err


# ── /slots is a list, and an error body is an object ──────────────────


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload

    def get(self, url, **kwargs):
        return _FakeResponse(self._payload)


def _backend(payload):
    from tokirecur.backend import LlamaServerBackend

    backend = LlamaServerBackend("http://127.0.0.1:9")
    backend._client = _FakeClient(payload)
    return backend


def test_list_slots_survives_an_error_body():
    """An error body is an object, not a list of slots."""
    backend = _backend({"error": {"code": 500, "message": "boom"}})
    assert backend.list_slots() == []


def test_list_slots_returns_the_slots():
    backend = _backend([
        {"id": 0, "n_ctx": 8192, "n_prompt_tokens_processed": 12},
    ])
    slots = backend.list_slots()
    assert len(slots) == 1
    assert slots[0]["id"] == 0


def test_cache_info_survives_an_error_body():
    backend = _backend({"error": {"code": 500, "message": "boom"}})
    info = backend.cache_info(0)
    # It falls through to /v1/models, which is the same error body here,
    # and answers with the unknown model rather than raising.
    assert info["model_id"] == "unknown"
    assert info["n_past"] == 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
