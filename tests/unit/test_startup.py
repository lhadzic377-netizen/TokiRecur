"""Tests for RAM-aware context clamping, fail-fast startup, and the
effective-context single source of truth."""


import pytest

from tokirecur import ctl
from tokirecur.ggufmeta import GgufMeta


@pytest.fixture
def env(tmp_path, monkeypatch):
    reg = tmp_path / "models.yaml"
    state = tmp_path / "state"
    marker = tmp_path / "active-model"
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    monkeypatch.setattr(ctl, "STATE_ROOT", state)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", marker)
    return tmp_path


def _spec(tmp_path, name="9b", ctx=200000, parallel=2):
    gguf = tmp_path / f"{name}.gguf"
    gguf.write_bytes(b"x" * 100)
    raw = {
        "name": name, "label": name, "gguf": str(gguf),
        "llama_port": 8088, "daemon_port": 8787,
        "ctx_size": ctx, "max_tokens": 100000, "parallel": parallel,
        "kv": "f16", "model_id": "qwen3.5-9b", "source": "pulled",
    }
    return ctl.ModelSpec(raw=raw, **raw)


# ── _kv_bytes_per_token / _clamp_ctx ────────────────────────────────


def test_kv_bytes_per_token_uses_gqa_dims(env, monkeypatch):
    spec = _spec(env)
    monkeypatch.setattr(ctl, "read_gguf_meta", lambda p: GgufMeta(
        n_embd=4096, n_layer=32, n_head=16, n_head_kv=4))
    # head_dim 256 * 4 kv heads * 2 (k+v) * 2.0 bytes * 32 layers * 2 slots
    assert ctl._kv_bytes_per_token(spec) == 256 * 4 * 2 * 2.0 * 32 * 2


def test_clamp_ctx_reduces_on_small_ram(env, monkeypatch):
    spec = _spec(env)
    monkeypatch.setattr(ctl, "read_gguf_meta", lambda p: GgufMeta(
        n_embd=4096, n_layer=32, n_head=16, n_head_kv=4))
    # Budget = 16 GB avail - 0 weights - 4 GB headroom = 12 GB.
    clamped = ctl._clamp_ctx(spec, avail_gb=16.0, weight_gb=0.0)
    assert clamped < spec.ctx_size
    # 12e9 / (256*4*2*2.0*32*2 = 262144 B/token) = 45776
    assert clamped == 45776


def test_clamp_ctx_noop_when_meta_unreadable(env, monkeypatch):
    spec = _spec(env)
    monkeypatch.setattr(ctl, "read_gguf_meta", lambda p: GgufMeta())
    assert ctl._clamp_ctx(spec, avail_gb=8.0, weight_gb=0.0) == spec.ctx_size


def test_clamp_ctx_noop_when_ram_unknown(env, monkeypatch):
    spec = _spec(env)
    monkeypatch.setattr(ctl, "read_gguf_meta", lambda p: GgufMeta(
        n_embd=4096, n_layer=32, n_head=16, n_head_kv=4))
    assert ctl._clamp_ctx(spec, avail_gb=float("inf"), weight_gb=0.0) == spec.ctx_size


def test_clamp_ctx_keeps_floor(env, monkeypatch):
    spec = _spec(env)
    monkeypatch.setattr(ctl, "read_gguf_meta", lambda p: GgufMeta(
        n_embd=4096, n_layer=32, n_head=16, n_head_kv=4))
    # Budget 0.1 GB -> max_fit ~381 tokens, floored up to 512.
    assert ctl._clamp_ctx(spec, avail_gb=4.1, weight_gb=0.0) == ctl._CTX_FLOOR


# ── effective ctx plumbing ──────────────────────────────────────────


def test_effective_ctx_falls_back_to_configured(env):
    spec = _spec(env)
    assert ctl._effective_ctx(spec) == spec.ctx_size


def test_effective_ctx_reads_state_file(env):
    spec = _spec(env)
    spec.state_dir().mkdir(parents=True)
    (spec.state_dir() / "effective_ctx.txt").write_text("8192")
    assert ctl._effective_ctx(spec) == 8192


def test_provider_entry_advertises_effective_ctx(env):
    spec = _spec(env)
    spec.state_dir().mkdir(parents=True)
    (spec.state_dir() / "effective_ctx.txt").write_text("4096")
    entry = ctl._provider_entry(spec)
    model = entry["models"][spec.model_id]
    assert model["contextWindow"] == 4096
    assert model["limit"]["context"] == 4096
    assert model["limit"]["output"] == 2048


def test_llama_args_honors_ctx_override(env):
    spec = _spec(env)
    args = ctl._llama_args(spec, ctx_override=4096)
    assert args[args.index("-c") + 1] == "4096"


# ── fail-fast startup ───────────────────────────────────────────────


def test_start_model_fails_fast_with_log_tail(env, monkeypatch):
    spec = _spec(env)
    spec.state_dir().mkdir(parents=True)
    (spec.state_dir() / "llama.log").write_text(
        "loading model...\nllama_model_load: mmap failed\nout of memory"
    )
    monkeypatch.setattr(ctl, "llama_reachable", lambda s: False)
    monkeypatch.setattr(ctl, "daemon_reachable", lambda s: True)

    class DeadProc:
        pid = 4242
        returncode = 1

        def poll(self):
            return 1

    monkeypatch.setattr(ctl.subprocess, "Popen", lambda *a, **k: DeadProc())
    with pytest.raises(RuntimeError) as exc:
        ctl.start_model(spec)
    msg = str(exc.value)
    assert "exited during startup" in msg
    assert "out of memory" in msg


def test_start_model_writes_effective_ctx(env, monkeypatch):
    spec = _spec(env)
    monkeypatch.setattr(ctl, "llama_reachable", lambda s: False)
    monkeypatch.setattr(ctl, "daemon_reachable", lambda s: True)
    monkeypatch.setattr(ctl, "read_gguf_meta", lambda p: GgufMeta(
        n_embd=4096, n_layer=32, n_head=16, n_head_kv=4))

    class FakeProc:
        pid = 4242

        def poll(self):
            return None

    monkeypatch.setattr(ctl.subprocess, "Popen", lambda *a, **k: FakeProc())
    monkeypatch.setattr(ctl, "_real_slot_ctx", lambda url: 16384)

    class Probe:
        status_code = 200

    monkeypatch.setattr(ctl.httpx, "get", lambda url, timeout=2.0: Probe())
    ctl.start_model(spec)
    effective = int((spec.state_dir() / "effective_ctx.txt").read_text())
    assert effective == 16384
