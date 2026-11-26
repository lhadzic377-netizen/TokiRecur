"""Tests for the plug-and-play fixes in 0.3.23.

- the ROCm install must not set a CUDA knob (it silently corrupts
  inference on some AMD GPUs)
- the registry repair must not throw away a pinned binary
- a model whose architecture the installed build does not know must be
  refused with an actionable message, not a server-side crash
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from tokirecur import arch, cli, ctl

# ── ROCm env ──────────────────────────────────────────────────────────────


def test_rocm_env_has_no_cuda_knobs():
    """GGML_CUDA_ENABLE_UNIFIED_MEMORY is read by the HIP backend too.

    With it on, an AMD box answers with repeated garbage and an empty
    message; the ROCm build already uses unified memory.
    """
    env = cli._rocm_env(None)
    assert "GGML_CUDA_ENABLE_UNIFIED_MEMORY" not in env
    assert env == {}


def test_rocm_env_keeps_the_gfx_override():
    env = cli._rocm_env("11.5.1")
    assert env["HSA_OVERRIDE_GFX_VERSION"] == "11.5.1"
    assert "GGML_CUDA_ENABLE_UNIFIED_MEMORY" not in env


# ── registry repair keeps a pinned binary ─────────────────────────────────


def test_backfill_keeps_a_pinned_binary(tmp_path, monkeypatch):
    """An entry that pins its own build keeps it when only env is empty.

    Pinning is how a model that needs a different llama.cpp build is
    served; the repair used to overwrite it with the default build.
    """
    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    registry = tmp_path / "models.yaml"
    state = tmp_path / "state"
    marker = tmp_path / "active-model"
    pinned = tmp_path / "pinned" / "llama-server"
    pinned.parent.mkdir(parents=True)
    pinned.write_bytes(b"")
    default = tmp_path / "bin" / "llama-server"
    default.parent.mkdir(parents=True)
    default.write_bytes(b"")

    registry.write_text(
        "models:\n"
        "- name: laguna\n"
        "  label: Laguna\n"
        "  gguf: /tmp/laguna.gguf\n"
        "  llama_port: 8112\n"
        "  daemon_port: 8812\n"
        "  ctx_size: 8192\n"
        "  parallel: 2\n"
        "  kv: f16\n"
        "  max_tokens: 4096\n"
        "  model_id: laguna\n"
        f"  binary: {default}\n"
        "  env: {}\n"
        "- name: pinned-model\n"
        "  label: Pinned\n"
        "  gguf: /tmp/none.gguf\n"
        "  llama_port: 8149\n"
        "  daemon_port: 8849\n"
        "  ctx_size: 8192\n"
        "  parallel: 2\n"
        "  kv: f16\n"
        "  max_tokens: 4096\n"
        "  model_id: pinned-model\n"
        f"  binary: {pinned}\n"
        "  env: {}\n"
        "  source: pulled\n"
    )
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", registry)
    monkeypatch.setattr(ctl, "STATE_ROOT", state)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", marker)
    ctl.invalidate_registry_cache()

    ctl.backfill_binary_env()
    ctl.invalidate_registry_cache()
    assert ctl.get_model("pinned-model").binary == str(pinned)


# ── the architecture guard ────────────────────────────────────────────────


def _write_gguf(path: Path, architecture: str) -> Path:
    """A tiny GGUF header carrying only ``general.architecture``."""
    def write_string(fh, text: str) -> None:
        raw = text.encode("utf-8")
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"GGUF")
        fh.write(struct.pack("<I", 3))      # version
        fh.write(struct.pack("<Q", 0))      # tensor_count
        fh.write(struct.pack("<Q", 1))      # kv_count
        write_string(fh, "general.architecture")
        fh.write(struct.pack("<I", 8))      # string
        write_string(fh, architecture)
    return path


def _fake_build(folder: Path, archs: list[str]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    binary = folder / "llama-server"
    binary.write_bytes(b"#!/bin/sh\n")
    lib = folder / "libllama.so"
    lib.write_bytes(b"\0".join(a.encode() for a in archs))
    return binary


def test_arch_support_is_read_from_the_build(tmp_path):
    binary = _fake_build(tmp_path / "b", ["qwen4exp", "inkling2_loop"])
    assert arch.binary_supports_arch(str(binary), "inkling2_loop") is True
    assert arch.binary_supports_arch(str(binary), "qwen4exp") is True


def test_arch_support_is_false_for_an_unknown_arch(tmp_path):
    binary = _fake_build(tmp_path / "b", ["qwen4exp", "inkling2"])
    assert arch.binary_supports_arch(str(binary), "inkling2_loop") is False


def test_arch_support_has_no_opinion_without_libraries(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"#!/bin/sh\n")
    assert arch.binary_supports_arch(str(binary), "inkling2_loop") is None


def test_arch_support_ignores_a_missing_binary(tmp_path):
    assert arch.binary_supports_arch(
        str(tmp_path / "nope"), "inkling2_loop") is None


def test_start_model_refuses_a_model_the_build_cannot_load(tmp_path, monkeypatch):
    """The refusal names the architecture and the way out.

    Without this the server dies with "unknown model architecture"
    inside llama.log, which reads like a broken download.
    """
    # The architecture check, not the RAM gate, is what this test is for.
    monkeypatch.setattr(ctl, "_available_gb", lambda: 1000.0)
    gguf = _write_gguf(tmp_path / "m.gguf", "inkling2_loop")
    binary = _fake_build(tmp_path / "bin", ["qwen4exp", "inkling2"])
    spec = ctl.ModelSpec(
        raw={}, name="inkling-2-mini-gguf", label="Inkling",
        gguf=str(gguf), llama_port=8149, daemon_port=8849,
        ctx_size=16384, parallel=2, kv="f16", max_tokens=8192,
        model_id="inkling", always_on=False, boot=False,
        binary=str(binary), env={}, reasoning="auto", source="pulled",
    )
    with pytest.raises(RuntimeError) as err:
        ctl.start_model(spec)
    message = str(err.value)
    assert "inkling2_loop" in message
    assert "install-llama" in message


def test_arch_hint_mentions_the_upstream_merge():
    assert "#32694" in arch.arch_support_hint("inkling2_loop")


# ── the ROCm env repair ───────────────────────────────────────────────────


def test_repair_drops_the_cuda_knob_from_a_rocm_model(tmp_path, monkeypatch):
    """An AMD box that installed toki before the fix carries the CUDA
    knob in its registry; the repair takes it out again."""
    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    registry = tmp_path / "models.yaml"
    state = tmp_path / "state"
    marker = tmp_path / "active-model"
    binary = tmp_path / "bin" / "llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"")

    registry.write_text(
        "models:\n"
        "- name: rocm-model\n"
        "  label: ROCm Model\n"
        "  gguf: /tmp/rocm.gguf\n"
        "  llama_port: 8112\n"
        "  daemon_port: 8812\n"
        "  ctx_size: 8192\n"
        "  parallel: 2\n"
        "  kv: f16\n"
        "  max_tokens: 4096\n"
        "  model_id: rocm-model\n"
        f"  binary: {binary}\n"
        "  env:\n"
        "    HSA_OVERRIDE_GFX_VERSION: '11.5.1'\n"
        "    GGML_CUDA_ENABLE_UNIFIED_MEMORY: '1'\n"
    )
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", registry)
    monkeypatch.setattr(ctl, "STATE_ROOT", state)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", marker)
    monkeypatch.setattr(ctl, "_binary_backend", lambda _b: "ROCm")
    ctl.invalidate_registry_cache()

    healed = ctl.backfill_binary_env()
    ctl.invalidate_registry_cache()
    assert "rocm-model" in healed
    env = ctl.get_model("rocm-model").env
    assert "GGML_CUDA_ENABLE_UNIFIED_MEMORY" not in env
    assert env["HSA_OVERRIDE_GFX_VERSION"] == "11.5.1"
