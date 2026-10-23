"""Where a model's weights actually are, and whether they are there at all.

The registry names a path, but that path is not always where the file is:
a portable entry names a bare filename, and a model that has been moved is
found by name under ``~/models/<name>/``. So the question of whether a
model is present has to be asked of the resolver rather than of the string
in the registry — and a model that has merely moved is not missing.
"""

import struct
from pathlib import Path

import pytest

from tokirecur import ctl
from tokirecur.ctl import ModelSpec


def _spec(name: str, gguf: str) -> ModelSpec:
    raw = {
        "name": name,
        "label": name,
        "gguf": gguf,
        "llama_port": 9000,
        "daemon_port": 9001,
        "ctx_size": 4096,
        "max_tokens": 4096,
        "model_id": name,
        "parallel": 1,
        "kv": "f16",
    }
    return ModelSpec(raw=raw, **raw)


def test_a_model_whose_file_is_gone_is_missing(tmp_path):
    spec = _spec("gone", str(tmp_path / "nope.gguf"))
    assert ctl.model_file_missing(spec) is True


def test_a_model_whose_file_is_there_is_not_missing(tmp_path):
    weights = tmp_path / "here.gguf"
    weights.write_bytes(b"x")
    spec = _spec("here", str(weights))
    assert ctl.model_file_missing(spec) is False


def test_a_model_that_has_moved_is_not_missing(tmp_path, monkeypatch):
    """The registry's path is stale, but the weights are on disk."""
    home = tmp_path / "home"
    model_dir = home / "models" / "moved"
    model_dir.mkdir(parents=True)
    weights = model_dir / "moved-Q4_K_M.gguf"
    weights.write_bytes(b"x")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))

    spec = _spec("moved", str(tmp_path / "old" / "moved-Q4_K_M.gguf"))
    assert ctl.model_file_missing(spec) is False
    assert ctl.resolve_gguf_path(spec) == weights


def test_the_resolver_never_hands_back_a_projector(tmp_path, monkeypatch):
    """A projector is a GGUF, and it lives next to the model it belongs to.

    When the model file has been deleted, the "first gguf in the
    directory" fallback used to find the projector and return it as the
    weights — a model that loads a vision tower and calls it a language
    model.
    """
    home = tmp_path / "home"
    model_dir = home / "models" / "only-a-projector"
    model_dir.mkdir(parents=True)
    # A minimal GGUF whose architecture is "clip": that is a projector.
    header = bytearray(b"GGUF")
    header += struct.pack("<I", 3)
    header += struct.pack("<Q", 0)
    header += struct.pack("<Q", 1)
    key = b"general.architecture"
    header += struct.pack("<Q", len(key)) + key
    header += struct.pack("<I", 8)          # string
    value = b"clip"
    header += struct.pack("<Q", len(value)) + value
    (model_dir / "mmproj-Model-f16.gguf").write_bytes(bytes(header))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))

    spec = _spec("only-a-projector", str(tmp_path / "gone" / "Model-Q4_K_M.gguf"))
    with pytest.raises(FileNotFoundError):
        ctl.resolve_gguf_path(spec)
    assert ctl.model_file_missing(spec) is True


def test_the_status_says_a_model_has_no_weights(tmp_path, monkeypatch):
    """A model that cannot be started is not "stopped".

    The registry keeps the row — the file may come back, and the port
    assignment is still its own — but a person reading the list would
    otherwise think they could switch to it.
    """
    from tokirecur.cli import status
    from click.testing import CliRunner

    import yaml

    reg = tmp_path / "models.yaml"
    marker = tmp_path / "active"
    here = tmp_path / "here.gguf"
    here.write_bytes(b"x")
    gone = tmp_path / "gone.gguf"
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "here", "label": "Here", "gguf": str(here),
         "llama_port": 9100, "daemon_port": 9101, "ctx_size": 4096,
         "max_tokens": 1000, "model_id": "here", "parallel": 1, "kv": "f16"},
        {"name": "gone", "label": "Gone", "gguf": str(gone),
         "llama_port": 9102, "daemon_port": 9103, "ctx_size": 4096,
         "max_tokens": 1000, "model_id": "gone", "parallel": 1, "kv": "f16"},
    ]}))
    marker.write_text("here")
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", marker)
    monkeypatch.setenv("TOKI_MODEL_REGISTRY", str(reg))
    monkeypatch.setenv("TOKI_ACTIVE_MODEL", str(marker))

    result = CliRunner().invoke(status, [])
    out = result.output
    assert "no weights on disk" in out
    assert "gone.gguf" in out
    assert "1 model(s) have no weights on disk" in out
