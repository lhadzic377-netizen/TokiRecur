"""The doctor has to point at the thing that is actually wrong.

Two false alarms it was giving:

- It probed the install defaults (8080 / 8787) for the stack, so on any
  machine whose models had their own ports it reported a stack that was
  up as down.
- When a model came up without the GPU it said "the log may just be
  terse, or -ngl was forced to 0" — and the log, which it had just
  read, said exactly why.
"""

import pytest

from tokirecur import ctl, diagnose


@pytest.fixture
def iso_env(tmp_path, monkeypatch):
    monkeypatch.setattr(ctl, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", tmp_path / "models.yaml")
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", tmp_path / "active")
    return tmp_path


def _spec(tmp_path, name="qwen", llama_port=8112, daemon_port=8811):
    gguf = tmp_path / f"{name}.gguf"
    gguf.write_bytes(b"dummy")
    return ctl.ModelSpec(
        name=name, label=name, gguf=str(gguf),
        llama_port=llama_port, daemon_port=daemon_port,
        ctx_size=100, parallel=2, kv="f16", max_tokens=90,
        model_id=f"{name}-GGUF",
    )


def test_the_stack_is_probed_on_the_active_models_ports(iso_env, monkeypatch):
    spec = _spec(iso_env, llama_port=8112, daemon_port=8811)
    monkeypatch.setattr(diagnose, "_active_spec", lambda: spec)
    assert diagnose.active_stack_ports() == (8112, 8811)


def test_without_an_active_model_the_install_defaults_are_used(iso_env, monkeypatch):
    monkeypatch.setattr(diagnose, "_active_spec", lambda: None)
    assert diagnose.active_stack_ports() == (8080, 8787)


def test_the_log_says_why_the_gpu_was_not_used(iso_env):
    spec = _spec(iso_env)
    d = spec.state_dir(root=iso_env)
    d.mkdir(parents=True, exist_ok=True)
    (d / "llama.log").write_text(
        "0.00.844 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 19902.90 MiB "
        "on device 0: cudaMalloc failed: out of memory\n"
        "0.00.902 E srv load_model: failed to load model\n"
    )
    reason = diagnose.gpu_fallback_reason(spec)
    assert reason is not None
    assert "out of memory" in reason


def test_a_pinned_layer_count_is_named_too(iso_env):
    spec = _spec(iso_env)
    d = spec.state_dir(root=iso_env)
    d.mkdir(parents=True, exist_ok=True)
    (d / "llama.log").write_text(
        "0.00.516 W common_fit_params: failed to fit params to free device memory: "
        "n_gpu_layers already set by user to 99, abort\n"
    )
    reason = diagnose.gpu_fallback_reason(spec)
    assert reason is not None
    assert "pinned" in reason


def test_a_healthy_log_gives_no_reason(iso_env):
    """A model that came up on the GPU has nothing to explain."""
    spec = _spec(iso_env)
    d = spec.state_dir(root=iso_env)
    d.mkdir(parents=True, exist_ok=True)
    (d / "llama.log").write_text(
        "0.06.000 I load_tensors: offloaded 40/40 layers to GPU\n"
    )
    assert diagnose.gpu_fallback_reason(spec) is None
