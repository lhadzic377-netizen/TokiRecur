"""Every command that talks to the model has to find the model.

The install defaults — llama-server on 8088, the daemon on 8787 — are for
a fresh install. Every model in the registry gets its own pair of ports,
so a command that used the defaults talked to a port nothing was on:

- `toki up` failed on a hard-coded path to a starter download.
- `toki interactive` answered "connection refused" while the daemon was
  plainly running on 8811.
- `toki web` served a page pointed at a port that was not there.
- `toki diff` compared a model that was not running against one that was.

They all take the active model's ports now.
"""

import pytest

from tokirecur import ctl
from tokirecur.cli import _active_urls


@pytest.fixture
def registry(tmp_path, monkeypatch):
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
    marker.write_text("m")
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", marker)
    monkeypatch.setenv("TOKI_MODEL_REGISTRY", str(reg))
    monkeypatch.setenv("TOKI_ACTIVE_MODEL", str(marker))
    return reg


def test_the_active_model_urls_are_used(registry):
    raw, daemon = _active_urls()
    assert raw == "http://127.0.0.1:9100"
    assert daemon == "http://127.0.0.1:9101"


def test_without_an_active_model_the_install_defaults_stand(registry, monkeypatch):
    monkeypatch.setattr(ctl, "read_active", lambda: "")
    raw, daemon = _active_urls()
    assert raw == "http://127.0.0.1:8088"
    assert daemon == "http://127.0.0.1:8787"


def test_the_commands_do_not_hard_code_a_port():
    """The options themselves must not carry a port."""
    from tokirecur.cli import diff, up, web

    for command in (diff, up, web):
        defaults = {p.name: p.default for p in command.params}
        for name, value in defaults.items():
            if isinstance(value, str) and "127.0.0.1" in value:
                assert "8088" not in value and "8787" not in value, (
                    f"{command.name} --{name} still defaults to {value}"
                )
