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
    # head_dim 256 * 4 kv heads * 2 (k+v) * 2.0 bytes * 32 layers.
    # Unified KV (-kvu) shares one n_ctx pool across slots, so the
    # slot count does not multiply the per-token estimate.
    assert ctl._kv_bytes_per_token(spec) == 256 * 4 * 2 * 2.0 * 32


def test_clamp_ctx_reduces_on_small_ram(env, monkeypatch):
    spec = _spec(env)
    monkeypatch.setattr(ctl, "read_gguf_meta", lambda p: GgufMeta(
        n_embd=4096, n_layer=32, n_head=16, n_head_kv=4))
    # Budget = 16 GB avail - 0 weights - 4 GB headroom = 12 GB.
    clamped = ctl._clamp_ctx(spec, avail_gb=16.0, weight_gb=0.0)
    assert clamped < spec.ctx_size
    # 12e9 / (256*4*2*2.0*32 = 131072 B/token) = 91552
    assert clamped == 91552


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
    # Budget 0.06 GB -> max_fit ~457 tokens, floored up to 512.
    assert ctl._clamp_ctx(spec, avail_gb=4.06, weight_gb=0.0) == ctl._CTX_FLOOR


def test_clamp_ctx_zero_budget_keeps_floor(env, monkeypatch):
    spec = _spec(env)
    monkeypatch.setattr(ctl, "read_gguf_meta", lambda p: GgufMeta(
        n_embd=4096, n_layer=32, n_head=16, n_head_kv=4))
    # Weights + headroom alone (16.8 GB) exceed the machine: the old code
    # returned the raw 200k cap and let llama.cpp silently shrink the
    # window — every client was told a context that never existed. The
    # smallest honest value wins instead; `_check_ram` raises before
    # launch anyway.
    assert ctl._clamp_ctx(spec, avail_gb=16.0, weight_gb=14.0) == ctl._CTX_FLOOR


# ── fit planner (ctx / kv / parallel that best fits RAM) ─────────────


def test_fit_plan_picks_biggest_window(env, monkeypatch):
    spec = _spec(env, ctx=200000, parallel=2)
    monkeypatch.setattr(ctl, "read_gguf_meta", lambda p: GgufMeta(
        n_embd=4096, n_layer=32, n_head=16, n_head_kv=4))
    # 12 GB budget: f16 reaches 91552, q4_1 reaches the full 200k.
    # The planner must trade KV precision for the window — on the
    # 16 GB class that's the difference between "not enough
    # context" and a usable model. Unified KV keeps the spec's
    # slots (parallel no longer costs KV memory).
    assert ctl._fit_plan(spec, avail_gb=16.0, weight_gb=0.0) == (
        200000, "q4_1", 2)


def test_fit_plan_prefers_spec_kv_on_tie(env, monkeypatch):
    spec = _spec(env, ctx=200000, parallel=2)
    spec.kv = "q4_0"
    monkeypatch.setattr(ctl, "read_gguf_meta", lambda p: GgufMeta(
        n_embd=4096, n_layer=32, n_head=16, n_head_kv=4))
    # q4_1 and q4_0 both reach the 200k cap; the spec's own kv wins the
    # tie (no quality surprise on machines with plenty of RAM).
    assert ctl._fit_plan(spec, avail_gb=16.0, weight_gb=0.0) == (
        200000, "q4_0", 2)


def test_fit_plan_noop_unknown_ram(env, monkeypatch):
    spec = _spec(env)
    monkeypatch.setattr(ctl, "read_gguf_meta", lambda p: GgufMeta(
        n_embd=4096, n_layer=32, n_head=16, n_head_kv=4))
    # Unbounded/unknown RAM: spec's own config, untouched.
    assert ctl._fit_plan(spec, avail_gb=float("inf"), weight_gb=0.0) == (
        spec.ctx_size, spec.kv, spec.parallel)


def test_fit_plan_noop_zero_budget(env, monkeypatch):
    spec = _spec(env)
    monkeypatch.setattr(ctl, "read_gguf_meta", lambda p: GgufMeta(
        n_embd=4096, n_layer=32, n_head=16, n_head_kv=4))
    # Weights alone don't fit — `_check_ram` refuses the launch, so the
    # planner leaves the spec config alone (never a lying cap).
    assert ctl._fit_plan(spec, avail_gb=16.0, weight_gb=14.0) == (
        spec.ctx_size, spec.kv, spec.parallel)


def test_effective_ctx_falls_back_to_plan(env, monkeypatch):
    spec = _spec(env, ctx=200000)
    monkeypatch.setattr(ctl, "read_gguf_meta", lambda p: GgufMeta(
        n_embd=4096, n_layer=32, n_head=16, n_head_kv=4))
    monkeypatch.setattr(ctl, "_available_gb", lambda: 8.0)
    # 8 GB budget, weights ignored: f16 reaches 30517, but the
    # planner launches the biggest window that fits — q4_0 reaches
    # 108506. The effective context must match what llama-server
    # actually runs with, not the spec's own worst-case
    # kv/slots (the "select 65k, still says 31k" bug).
    assert ctl._effective_ctx(spec) == 108506


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


# ── strict daemon reachability (self-heal) ───────────────────────────


class _FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_daemon_reachable_requires_matching_live_backend(env, monkeypatch):
    """A daemon is only reachable when its upstream is live AND wired to
    THIS model's llama_url — a stale daemon proxying a dead or different
    backend must count as down so start_model takes over."""
    spec = _spec(env)
    monkeypatch.delenv("TOKI_AUTH_TOKEN", raising=False)
    calls = []

    def fake_get(url, timeout=2.0, headers=None):
        calls.append((url, headers))
        if url.endswith("/healthz"):
            return _FakeResp(200)
        if url.endswith("/v1/health/llama"):
            return _FakeResp(200, {"status": "ok", "backend": spec.llama_url})
        return _FakeResp(404)

    monkeypatch.setattr(ctl.httpx, "get", fake_get)
    assert ctl.daemon_reachable(spec) is True

    # Wrong backend (stale daemon pointing at an old llama) → not reachable.
    monkeypatch.setattr(
        ctl.httpx, "get",
        lambda url, timeout=2.0, headers=None: _FakeResp(
            200, {"status": "ok", "backend": "http://127.0.0.1:9999"}
        ),
    )
    assert ctl.daemon_reachable(spec) is False

    # Dead backend (status unreachable) → not reachable.
    monkeypatch.setattr(
        ctl.httpx, "get",
        lambda url, timeout=2.0, headers=None: _FakeResp(
            200, {"status": "unreachable", "backend": spec.llama_url}
        ),
    )
    assert ctl.daemon_reachable(spec) is False

    # Old-code daemon that cannot state its backend → not reachable.
    monkeypatch.setattr(
        ctl.httpx, "get",
        lambda url, timeout=2.0, headers=None: _FakeResp(200, {"status": "ok"}),
    )
    assert ctl.daemon_reachable(spec) is False

    # healthz down → not reachable.
    monkeypatch.setattr(
        ctl.httpx, "get",
        lambda url, timeout=2.0, headers=None: _FakeResp(500),
    )
    assert ctl.daemon_reachable(spec) is False


def test_daemon_reachable_sends_auth_token_when_configured(env, monkeypatch):
    spec = _spec(env)
    monkeypatch.setenv("TOKI_AUTH_TOKEN", "sekrit")
    seen = {}

    def fake_get(url, timeout=2.0, headers=None):
        seen["headers"] = headers
        return _FakeResp(200, {"status": "ok", "backend": spec.llama_url})

    monkeypatch.setattr(ctl.httpx, "get", fake_get)
    ctl.daemon_reachable(spec)
    assert seen["headers"] == {"Authorization": "Bearer sekrit"}


def test_takeover_stale_daemon_replaces_own_stale_code_daemon(env, monkeypatch):
    """Our own managed daemon running old code (no backend report) must be
    stopped so the fresh spawn can bind — a daemon that fails the strict
    check is dead weight."""
    spec = _spec(env)
    state = spec.state_dir()
    state.mkdir(parents=True, exist_ok=True)
    (state / "daemon.pid").write_text("4242")

    monkeypatch.setattr(ctl, "_pid_on_port", lambda port: 4242)
    monkeypatch.setattr(ctl, "_is_alive", lambda pid: True)
    stopped = []

    def fake_stop(path, grace=8.0):
        stopped.append(path)
        return True

    monkeypatch.setattr(ctl, "_stop_pidfile", fake_stop)
    ctl._takeover_stale_daemon(spec)
    assert stopped == [state / "daemon.pid"]


def test_takeover_stale_daemon_ignores_non_daemon_listener(env, monkeypatch):
    """An unrelated process on the daemon port must never be killed."""
    spec = _spec(env)

    monkeypatch.setattr(ctl, "_pid_on_port", lambda port: 7777)
    monkeypatch.setattr(ctl, "_is_alive", lambda pid: True)
    monkeypatch.setattr(ctl, "daemon_pid", lambda spec: None)
    monkeypatch.setattr(
        ctl.httpx, "get",
        lambda url, timeout=2.0: _FakeResp(404),  # not a daemon healthz
    )
    killed = []

    def fake_signal(pid, sig):
        killed.append(pid)

    monkeypatch.setattr(ctl, "_signal_proc", fake_signal)
    monkeypatch.setattr(ctl, "_hard_kill", lambda pid: killed.append(pid))
    ctl._takeover_stale_daemon(spec)
    assert killed == []


def test_takeover_stale_daemon_stops_foreign_stale_daemon(env, monkeypatch):
    """A stale daemon bound to the model's port with a dead backend must be
    stopped so the correct daemon can bind."""
    spec = _spec(env)

    monkeypatch.setattr(ctl, "_pid_on_port", lambda port: 9999)
    monkeypatch.setattr(ctl, "_is_alive", lambda pid: True)
    monkeypatch.setattr(ctl, "daemon_pid", lambda spec: None)
    monkeypatch.setattr(
        ctl.httpx, "get",
        lambda url, timeout=2.0: _FakeResp(200, {"status": "ok"}),  # daemon
    )
    signalled = []

    def fake_signal(pid, sig):
        signalled.append(pid)
        monkeypatch.setattr(ctl, "_is_alive", lambda pid: False)

    monkeypatch.setattr(ctl, "_signal_proc", fake_signal)
    ctl._takeover_stale_daemon(spec)
    assert signalled == [9999]


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
