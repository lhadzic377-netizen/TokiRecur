"""Unit tests for the multi-model plug-and-play manager.

Covers: registry loading (with default file creation), active-model
marker, and the opencode / openclaw / hermes config writers
(merge-safe, backup-first). No real processes are spawned.
"""

import json
import os
from pathlib import Path

import pytest
import yaml

from tokirecur import ctl


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolate registry/state/marker paths into a temp dir."""
    reg = tmp_path / "models.yaml"
    state = tmp_path / "state"
    marker = tmp_path / "active-model"
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    monkeypatch.setattr(ctl, "STATE_ROOT", state)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", marker)
    return tmp_path


def test_registry_defaults_created(env):
    specs = ctl.load_registry()
    assert [s.name for s in specs] == ["9b", "35b", "laguna"]
    by_name = {s.name: s for s in specs}
    assert by_name["9b"].llama_port == 8088 and not by_name["9b"].always_on
    assert by_name["35b"].model_id == "qwen3.6-35b-a3b"
    assert by_name["laguna"].ctx_size == 131072
    assert by_name["laguna"].kv == "q8_0"
    assert by_name["laguna"].max_tokens == 64000
    assert ctl.DEFAULT_REGISTRY.exists()


def test_registry_custom(env):
    reg = env / "models.yaml"
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "x", "gguf": "/tmp/x.gguf", "llama_port": 1, "daemon_port": 2,
         "ctx_size": 100, "max_tokens": 90, "model_id": "x-model"},
    ]}))
    specs = ctl.load_registry(reg)
    assert len(specs) == 1
    assert specs[0].label == "x"  # defaults filled


def test_get_model_unknown(env):
    with pytest.raises(KeyError):
        ctl.get_model("nope")


def test_active_marker_roundtrip(env):
    assert ctl.read_active() is None  # no preloaded/always-on models
    ctl.set_active("laguna")
    assert ctl.read_active() == "laguna"
    (ctl.ACTIVE_MARKER).write_text("gone")
    assert ctl.read_active() is None  # stale marker, no always-on default


def _specs(env):
    return {s.name: s for s in ctl.load_registry()}


@pytest.fixture
def with_pulled_model(env, hf):
    """Register an extra pulled model alongside the 3 defaults.

    The pulled model has no `binary` or `env` (the broken state users
    hit before the repair command existed) and a port pair that's
    distinct from the defaults so port-collision isn't tested here.
    """
    pulled = ctl.pull_model("acme/Pulled-GGUF", "pulled")
    return {**_specs(env), pulled.name: pulled}


def test_opencode_config_writer(env, with_pulled_model):
    spec = with_pulled_model["35b"]
    cfg_path = env / "opencode.json"
    cfg_path.write_text(json.dumps({"small_model": "tokirecur/qwen3.5-9b",
                                    "provider": {"existing": {"name": "keep me"}}}))
    out = ctl.write_opencode_config(spec, path=cfg_path)
    cfg = json.loads(out.read_text())
    assert cfg["provider"]["existing"]["name"] == "keep me"
    # All registry specs get a provider entry — defaults AND pulled.
    for s in ctl.load_registry():
        p = cfg["provider"][f"tokirecur-{s.name}"]
        assert p["options"]["baseURL"] == f"{s.daemon_url}/v1"
        assert p["npm"] == "@ai-sdk/openai-compatible"
    # Active spec is selected.
    assert cfg["model"] == "tokirecur-35b/qwen3.6-35b-a3b"
    # small_model points at the always-on spec (9b in the default registry).
    assert cfg["small_model"] == "tokirecur/9b"
    # Pulled model has its own provider entry.
    pulled = with_pulled_model["pulled"]
    assert cfg["provider"][f"tokirecur-{pulled.name}"]["options"]["baseURL"] == \
        f"{pulled.daemon_url}/v1"
    assert cfg_path.with_suffix(".json.bak-model").exists()


def test_openclaw_config_writer(env, with_pulled_model):
    spec = with_pulled_model["laguna"]
    cfg_path = env / "openclaw.json"
    cfg_path.write_text(json.dumps({"agents": {"defaults": {"model": {"primary": "old"}}}}))
    out = ctl.write_openclaw_config(spec, path=cfg_path)
    cfg = json.loads(out.read_text())
    prov = cfg["models"]["providers"]["tokirecur-laguna"]
    assert prov["api"] == "openai-completions"
    assert prov["timeoutSeconds"] == 3600
    assert prov["baseUrl"] == "http://127.0.0.1:8789/v1"
    assert prov["models"][0]["id"] == "laguna-s-2.1"
    assert prov["models"][0]["contextWindow"] == 64000
    assert cfg["agents"]["defaults"]["model"]["primary"] == "tokirecur-laguna/laguna-s-2.1"
    assert cfg["agents"]["defaults"]["timeoutSeconds"] == 3600
    # Pulled model also has a provider entry.
    pulled = with_pulled_model["pulled"]
    pulled_prov = cfg["models"]["providers"][f"tokirecur-{pulled.name}"]
    assert pulled_prov["baseUrl"] == f"{pulled.daemon_url}/v1"
    assert pulled_prov["models"][0]["id"] == pulled.model_id


def test_hermes_config_writer(env, with_pulled_model):
    spec = with_pulled_model["9b"]
    cfg_path = env / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "memory": {"memory_enabled": True},
        "providers": {"ollama-launch": {"api": "http://127.0.0.1:11434/v1"}},
    }))
    out = ctl.write_hermes_config(spec, path=cfg_path)
    cfg = yaml.safe_load(out.read_text())
    assert cfg["memory"]["memory_enabled"] is True  # untouched
    assert cfg["providers"]["ollama-launch"]["api"] == "http://127.0.0.1:11434/v1"
    assert cfg["providers"]["tokirecur-9b"]["api"] == "http://127.0.0.1:8787/v1"
    # Hermes lists every spec's model_id under every provider (legacy).
    expected_ids = {s.model_id for s in ctl.load_registry()}
    assert set(cfg["providers"]["tokirecur-9b"]["models"]) == expected_ids
    assert cfg["model"] == {
        "api_key": "demo-token",
        "base_url": "http://127.0.0.1:8787/v1",
        "default": "qwen3.5-9b",
        "provider": "tokirecur-9b",
    }
    # Pulled model also wired.
    pulled = with_pulled_model["pulled"]
    pulled_prov = cfg["providers"][f"tokirecur-{pulled.name}"]
    assert pulled_prov["api"] == f"{pulled.daemon_url}/v1"
    assert pulled_prov["default_model"] == pulled.model_id
    assert cfg_path.with_suffix(".yaml.bak-model").exists()


def test_sync_cli_configs_writes_all(env, with_pulled_model):
    spec = with_pulled_model["laguna"]
    paths = ctl.sync_cli_configs(spec)
    assert len(paths) == 3
    for p in paths.values():
        assert Path(p).exists()
    # And every CLI config picked up the pulled provider.
    pulled_name = with_pulled_model["pulled"].name
    oc = json.loads(paths["opencode"].read_text())
    assert f"tokirecur-{pulled_name}" in oc["provider"]
    cla = json.loads(paths["openclaw"].read_text())
    assert f"tokirecur-{pulled_name}" in cla["models"]["providers"]
    h = yaml.safe_load(paths["hermes"].read_text())
    assert f"tokirecur-{pulled_name}" in h["providers"]


def test_model_status_down(env):
    reg = env / "models.yaml"
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "t", "gguf": "/tmp/t.gguf", "llama_port": 29901, "daemon_port": 29902,
         "ctx_size": 100, "max_tokens": 90, "model_id": "t-model"},
    ]}))
    spec = ctl.load_registry(reg)[0]
    s = ctl.model_status(spec)  # ports closed -> down, no crash
    assert s["llama"]["reachable"] is False
    assert s["daemon"]["reachable"] is False


def test_use_model_tolerates_broken_always_on(env, monkeypatch):
    """A broken always-on spec (e.g. missing GGUF -> FileNotFoundError)
    must not abort the model switch."""
    ctl.update_spec_field("9b", "always_on", "true")

    def fake_start_model(spec, wait=True):
        if spec.name == "9b":
            raise FileNotFoundError(f"model file not found: {spec.gguf}")
        return spec

    monkeypatch.setattr(ctl, "start_model", fake_start_model)
    monkeypatch.setattr(ctl, "sync_cli_configs", lambda spec: {})
    monkeypatch.setattr(ctl, "backfill_binary_env", lambda specs=None: [])

    spec = ctl.use_model("35b", stop_others=False)
    assert spec.name == "35b"
    assert ctl.read_active() == "35b"


def test_use_model_tolerates_broken_target(env, monkeypatch):
    """A missing-GGUF target surfaces the FileNotFoundError so the user
    sees a clean error instead of a silent no-op."""
    def fake_start_model(spec, wait=True):
        if spec.name == "35b":
            raise FileNotFoundError(f"model file not found: {spec.gguf}")
        return spec

    monkeypatch.setattr(ctl, "start_model", fake_start_model)
    monkeypatch.setattr(ctl, "sync_cli_configs", lambda spec: {})
    monkeypatch.setattr(ctl, "backfill_binary_env", lambda specs=None: [])

    with pytest.raises(FileNotFoundError):
        ctl.use_model("35b", stop_others=False)


# ── Registry save / pull ──────────────────────────────────────────────


def test_next_free_ports(env):
    assert ctl.next_free_ports(ctl.load_registry()) == (8091, 8790)


def test_save_registry_roundtrip_preserves_raw(env):
    specs = ctl.load_registry()
    assert ctl.save_registry(specs) == ctl.DEFAULT_REGISTRY
    again = ctl.load_registry()
    assert [s.name for s in again] == ["9b", "35b", "laguna"]
    # untouched entries keep their extra keys (binary, env, extra_args...)
    assert again[2].binary == specs[2].binary
    assert again[2].env == specs[2].env
    assert again[2].raw == specs[2].raw


@pytest.fixture
def hf(env, monkeypatch):
    """Fake huggingface_hub: no network, downloads land under tmp_path."""
    from types import SimpleNamespace

    import huggingface_hub

    calls = {"files": [], "downloads": [], "sizes": {}}

    def fake_list(self, repo_id, **kw):
        calls["files"].append(repo_id)
        return list(calls["repo_files"])

    def fake_download(repo_id, *, allow_patterns=None, local_dir=None,
                      resume_download=True):
        calls["downloads"].append(
            (repo_id, allow_patterns, str(local_dir), os.environ.get("HF_ENDPOINT")))
        return local_dir

    def fake_sizes(self, repo_id, *, paths=None, **kw):
        if getattr(self, "sizes_fail", False):
            raise OSError("sizes unavailable")
        return [SimpleNamespace(path=p, size=calls["sizes"].get(p))
                for p in paths]

    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", fake_list)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    monkeypatch.setattr(huggingface_hub.HfApi, "get_paths_info", fake_sizes)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: env))
    calls["repo_files"] = ["README.md", "model-Q4_K_M.gguf"]
    return calls


def test_pull_model_registers_entry(env, hf):
    spec = ctl.pull_model("acme/My-Model-GGUF", "mymodel", ctx_size=8192, kv="q8_0")
    assert spec.name == "mymodel"
    assert spec.label == "model-Q4_K_M"
    assert spec.llama_port == 8091 and spec.daemon_port == 8790
    assert spec.ctx_size == 8192 and spec.kv == "q8_0"
    assert spec.max_tokens == 4096
    assert spec.model_id == "acme/My-Model-GGUF"
    assert str(env / "models" / "mymodel" / "model-Q4_K_M.gguf") == spec.gguf
    # persisted and visible to the launcher/list
    assert ctl.get_model("mymodel").name == "mymodel"
    assert hf["downloads"][0][1] == ["*model-Q4_K_M.gguf"]


def test_pull_model_endpoint_sets_hf_env(env, hf):
    ctl.pull_model("acme/My-Model-GGUF", "mirrored",
                   endpoint="http://localhost:8090")
    assert hf["downloads"][0][3] == "http://localhost:8090"
    assert os.environ.get("HF_ENDPOINT") is None  # restored afterwards


def test_pull_model_file_choice(env, hf):
    hf["repo_files"] = ["a-Q4_K_M.gguf", "b-Q8_0.gguf"]
    with pytest.raises(RuntimeError, match="several GGUF"):
        ctl.pull_model("acme/Repo", "amb")
    spec = ctl.pull_model("acme/Repo", "chosen", file="b-Q8_0.gguf")
    assert spec.label == "b-Q8_0"


def test_pull_model_choose_file_callback(env, hf):
    hf["repo_files"] = ["a-Q4_K_M.gguf", "b-Q8_0.gguf", "c-BF16.gguf"]
    hf["sizes"] = {"a-Q4_K_M.gguf": 100, "b-Q8_0.gguf": 200, "c-BF16.gguf": 300}
    picked = []

    def pick(files):
        picked.append(list(files))
        return files[1][0]

    spec = ctl.pull_model("acme/Repo", "viacb", choose_file=pick)
    assert spec.label == "b-Q8_0"
    assert picked == [[
        ("a-Q4_K_M.gguf", 100), ("b-Q8_0.gguf", 200), ("c-BF16.gguf", 300),
    ]]


def test_pull_model_choose_file_sizes_best_effort(env, hf, monkeypatch):
    import huggingface_hub

    hf["repo_files"] = ["a-Q4_K_M.gguf", "b-Q8_0.gguf"]
    monkeypatch.setattr(huggingface_hub.HfApi, "get_paths_info", _boom_sizes)
    picked = []

    def pick(files):
        picked.append(list(files))
        return files[0][0]

    ctl.pull_model("acme/Repo", "nosizes", choose_file=pick)
    assert picked == [[("a-Q4_K_M.gguf", None), ("b-Q8_0.gguf", None)]]


def _boom_sizes(self, repo_id, *, paths=None, **kw):
    raise OSError("sizes unavailable")


def test_pull_model_choose_file_bad_result(env, hf):
    hf["repo_files"] = ["a-Q4_K_M.gguf", "b-Q8_0.gguf"]
    with pytest.raises(RuntimeError, match="unknown file"):
        ctl.pull_model("acme/Repo", "x", choose_file=lambda files: "nope.gguf")


def test_recommend_quant_prefers_q4_k_m():
    files = ["x-Q4_0.gguf", "y-Q4_K_M.gguf", "z-BF16.gguf"]
    assert ctl.recommend_quant(files) == "y-Q4_K_M.gguf"
    assert ctl.recommend_quant(["a-BF16.gguf"]) == "a-BF16.gguf"

def test_cli_picker_interactive_and_eof_fallback():
    import click
    from click.testing import CliRunner

    from tokirecur.cli import _pick_gguf_interactive

    files = [
        ("a-Q4_0.gguf", 500_000_000),
        ("b-Q4_K_M.gguf", 400_000_000),
        ("c-BF16.gguf", 1_000_000_000),
    ]

    @click.command()
    def pick():
        print(_pick_gguf_interactive(files))

    # typed choice
    res = CliRunner().invoke(pick, input="1\n")
    assert res.exit_code == 0
    assert "a-Q4_0.gguf" in res.output
    assert "476.8 MB" in res.output  # real size rendered
    assert "b-Q4_K_M.gguf  (381.5 MB)  (recommended)" in res.output
    # scripted (EOF) -> recommended quant, no crash
    res2 = CliRunner().invoke(pick, input="")
    assert res2.exit_code == 0
    assert "Using recommended: b-Q4_K_M.gguf" in res2.output


def test_pull_model_no_gguf(env, hf):
    hf["repo_files"] = ["model.safetensors", "config.json"]
    with pytest.raises(RuntimeError, match="no GGUF"):
        ctl.pull_model("acme/Safetensors-Only", "stub")


def test_pull_model_file_not_in_repo(env, hf):
    with pytest.raises(RuntimeError, match="candidates"):
        ctl.pull_model("acme/My-Model-GGUF", "stub", file="nope.gguf")


def test_pull_model_name_derived_from_repo(env, hf):
    spec = ctl.pull_model("Nanbeige/Nanbeige4.2-3B")
    assert spec.name == "nanbeige4.2-3b"
    assert ctl.get_model("nanbeige4.2-3b").name == "nanbeige4.2-3b"


def test_pull_model_name_slugified(env, hf):
    spec = ctl.pull_model("Acme/My Model_GGUF", "Bad Name!")
    assert spec.name == "bad-name"


def test_pull_model_name_uppercase_ok(env, hf):
    spec = ctl.pull_model("Acme/My-Model-GGUF", "Nanbeige4.2-3B")
    assert spec.name == "nanbeige4.2-3b"


def test_pull_model_underivable_name(env, hf):
    with pytest.raises(ValueError, match="cannot derive"):
        ctl.pull_model("!!!")


def test_pull_model_name_clash(env, hf):
    with pytest.raises(ValueError, match="already exists"):
        ctl.pull_model("acme/My-Model-GGUF", "9b")


def test_pull_model_explicit_name_case_insensitive_clash(env, hf):
    with pytest.raises(ValueError, match="already exists"):
        ctl.pull_model("acme/My-Model-GGUF", "9B")


def test_pull_model_network_error(env, hf, monkeypatch):
    import huggingface_hub

    def boom(self, repo_id, **kw):
        raise OSError("connection refused")

    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", boom)
    with pytest.raises(RuntimeError, match="cannot reach"):
        ctl.pull_model("acme/Down", "downed")


# ── pull defaults (binary / env inheritance) ────────────────────────


def test_pull_defaults_inherit_reference_spec_binary(env, monkeypatch):
    """The in-code default registry points laguna at the tuned llama.cpp
    build; pulled models should pick that up instead of the PATH default."""
    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    expected_binary = ctl.DEFAULT_MODELS[2]["binary"]
    binary, env_vars = ctl.pull_defaults(ctl.load_registry())
    assert binary == expected_binary
    # Default registry has no ROCm env block (users add their own); a
    # blank dict is the correct inherited value.
    assert env_vars == {}


def test_pull_defaults_env_override_takes_precedence(env, monkeypatch):
    monkeypatch.setenv("TOKI_LLAMA_SERVER", "/custom/llama-server")
    binary, env_vars = ctl.pull_defaults(ctl.load_registry())
    assert binary == "/custom/llama-server"
    assert env_vars == {}


def test_pull_defaults_env_inherited_when_reference_has_env(env, monkeypatch):
    """A reference spec that defines ROCm env (e.g. the user's real
    registry) propagates those vars to the new spec."""
    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    reg = env / "models.yaml"
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "laguna", "gguf": "/tmp/lag.gguf", "llama_port": 1,
         "daemon_port": 2, "ctx_size": 100, "max_tokens": 90,
         "model_id": "laguna-s-2.1",
         "binary": "/usr/local/bin/llama-server",
         "env": {"HSA_OVERRIDE_GFX_VERSION": "11.5.1",
                 "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"}},
    ]}))
    binary, env_vars = ctl.pull_defaults(ctl.load_registry(reg))
    assert binary == "/usr/local/bin/llama-server"
    assert env_vars.get("HSA_OVERRIDE_GFX_VERSION") == "11.5.1"
    assert env_vars.get("GGML_CUDA_ENABLE_UNIFIED_MEMORY") == "1"


def test_pull_defaults_fallback_when_no_reference(env, monkeypatch):
    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    reg = env / "models.yaml"
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "x", "gguf": "/tmp/x.gguf", "llama_port": 1, "daemon_port": 2,
         "ctx_size": 100, "max_tokens": 90, "model_id": "x-model"},
    ]}))
    binary, env_vars = ctl.pull_defaults(ctl.load_registry(reg))
    assert binary == "llama-server"
    assert env_vars == {}


def test_pull_model_inherits_binary(env, hf):
    spec = ctl.pull_model("acme/My-Model-GGUF", "inherits")
    assert spec.binary == ctl.DEFAULT_MODELS[2]["binary"]
    # No reference env → empty.
    assert spec.env == {}


def test_pull_model_reasoning_field_persisted(env, hf):
    spec_off = ctl.pull_model("acme/My-Model-GGUF", "reasonoff", reasoning="off")
    assert spec_off.reasoning == "off"
    spec_auto = ctl.pull_model("acme/My-Model-GGUF", "reasonauto")
    assert spec_auto.reasoning == "auto"
    # Survives registry round-trip.
    assert ctl.get_model("reasonoff").reasoning == "off"


def test_llama_args_appends_reasoning_off_for_off_spec(env):
    # The default 9b entry points at a dev-machine path; point it at an
    # existing temp file so _llama_args' existence check passes on CI.
    gguf = env / "9b.gguf"
    gguf.write_bytes(b"dummy")
    raw = dict(ctl.load_registry()[0].raw)
    raw["gguf"] = str(gguf)
    spec = ctl.ModelSpec(raw=raw, **raw)
    assert "--reasoning" not in ctl._llama_args(spec)
    spec_off = spec
    spec_off.reasoning = "off"
    assert "--reasoning" in ctl._llama_args(spec_off)
    assert "off" in ctl._llama_args(spec_off)


def test_llama_args_appends_reasoning_on_for_on_spec(env):
    gguf = env / "9b.gguf"
    gguf.write_bytes(b"dummy")
    raw = dict(ctl.load_registry()[0].raw)
    raw["gguf"] = str(gguf)
    spec = ctl.ModelSpec(raw=raw, **raw)
    spec.reasoning = "on"
    args = ctl._llama_args(spec)
    assert "--reasoning" in args
    assert args[args.index("--reasoning") + 1] == "on"


# ── Registry repair (legacy pulled entries) ────────────────────────


def test_backfill_binary_env_inherits_reference(env, hf, monkeypatch):
    """A pulled entry missing binary/env inherits the reference spec's
    values; the repair returns the names it updated."""
    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    pulled = ctl.pull_model("acme/Pulled-GGUF", "pulled")
    assert pulled.source == "pulled"
    assert pulled.binary == ctl.DEFAULT_MODELS[2]["binary"]

    # Simulate an older pulled entry written before `pull_defaults` —
    # clear its binary/env so it has only the PATH default.
    reg = ctl.DEFAULT_REGISTRY
    data = yaml.safe_load(reg.read_text())
    for entry in data["models"]:
        if entry["name"] == "pulled":
            entry["binary"] = "llama-server"
            entry["env"] = {}
    reg.write_text(yaml.safe_dump(data, sort_keys=False))

    healed = ctl.backfill_binary_env()
    assert healed == ["pulled"]
    refreshed = ctl.get_model("pulled")
    assert refreshed.binary == ctl.DEFAULT_MODELS[2]["binary"]
    assert refreshed.source == "pulled"


def test_backfill_binary_env_idempotent(env, hf, monkeypatch):
    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    # No entries need repair → empty list.
    assert ctl.backfill_binary_env() == []
    # Pulled entry already has a custom binary → no-op.
    ctl.pull_model("acme/Pulled-GGUF", "pulled")
    assert ctl.backfill_binary_env() == []


def test_backfill_binary_env_skips_configured_defaults(env, monkeypatch):
    """Configured defaults (source="default") are NEVER rewritten even
    when they look bare — they intentionally use the PATH llama-server."""
    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    # 9b / 35b have no custom binary. They should be left alone.
    healed = ctl.backfill_binary_env()
    assert healed == []
    assert ctl.get_model("9b").source == "default"
    assert ctl.get_model("35b").source == "default"


def test_backfill_binary_env_no_reference_noop(env, hf, monkeypatch):
    """If no spec has a custom binary/env, there's nothing to inherit."""
    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    # Override registry to a single spec with default binary/env.
    reg = env / "models.yaml"
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "x", "gguf": "/tmp/x.gguf", "llama_port": 1, "daemon_port": 2,
         "ctx_size": 100, "max_tokens": 90, "model_id": "x-model",
         "source": "pulled"},
    ]}))
    assert ctl.backfill_binary_env(ctl.load_registry(reg)) == []


def test_repair_cli_noop_when_clean(env, hf, monkeypatch):
    from click.testing import CliRunner

    from tokirecur.cli import cli

    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    res = CliRunner().invoke(cli, ["model", "repair"])
    assert res.exit_code == 0, res.output
    assert "nothing to repair" in res.output


def test_repair_cli_heals_legacy_entry(env, hf, monkeypatch):
    from click.testing import CliRunner

    from tokirecur.cli import cli

    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    ctl.pull_model("acme/Pulled-GGUF", "pulled")
    # Simulate a legacy entry: blow away binary/env but keep source.
    reg = ctl.DEFAULT_REGISTRY
    data = yaml.safe_load(reg.read_text())
    for entry in data["models"]:
        if entry["name"] == "pulled":
            entry["binary"] = "llama-server"
            entry["env"] = {}
    reg.write_text(yaml.safe_dump(data, sort_keys=False))

    res = CliRunner().invoke(cli, ["model", "repair"])
    assert res.exit_code == 0, res.output
    assert "repaired: pulled" in res.output
    assert ctl.get_model("pulled").binary == ctl.DEFAULT_MODELS[2]["binary"]


# ── Settings: update_spec_field ───────────────────────────────────


def test_update_spec_field_changes_value_and_persists(env):
    spec = ctl.update_spec_field("9b", "ctx_size", "4096")
    assert spec.ctx_size == 4096
    assert ctl.get_model("9b").ctx_size == 4096
    # Survives reload.
    assert ctl.load_registry()[0].ctx_size == 4096


def test_update_spec_field_int_validators(env):
    ctl.update_spec_field("9b", "llama_port", "9999")
    assert ctl.get_model("9b").llama_port == 9999
    with pytest.raises(ValueError, match="bad value for 'llama_port'"):
        ctl.update_spec_field("9b", "llama_port", "not-a-number")


def test_update_spec_field_bool_truthy(env):
    ctl.update_spec_field("35b", "always_on", "true")
    assert ctl.get_model("35b").always_on is True
    ctl.update_spec_field("35b", "always_on", "off")
    assert ctl.get_model("35b").always_on is False
    with pytest.raises(ValueError, match="expected true/false"):
        ctl.update_spec_field("35b", "always_on", "maybe")


def test_update_spec_field_unknown_field(env):
    with pytest.raises(ValueError, match="not editable"):
        ctl.update_spec_field("9b", "name", "newname")


def test_update_spec_field_readonly_source(env):
    with pytest.raises(ValueError, match="not editable"):
        ctl.update_spec_field("9b", "source", "pulled")


def test_update_spec_field_unknown_model(env):
    with pytest.raises(KeyError, match="unknown model 'nope'"):
        ctl.update_spec_field("nope", "ctx_size", "100")


def test_update_spec_field_preserves_other_fields(env):
    """Editing one field doesn't blow away the spec's other keys
    (raw dict merge — only the edited field changes)."""
    # Register a spec with a non-default env so we can prove it survives.
    reg = env / "models.yaml"
    # Trigger _ensure_registry so the file exists, then mutate.
    ctl.load_registry()
    ctl.invalidate_registry_cache()
    data = yaml.safe_load(reg.read_text())
    for entry in data["models"]:
        if entry["name"] == "laguna":
            entry["env"] = {"HSA_OVERRIDE_GFX_VERSION": "11.5.1",
                            "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"}
            entry["extra_args"] = ["--reasoning", "off"]
    reg.write_text(yaml.safe_dump(data, sort_keys=False))
    ctl.invalidate_registry_cache()

    spec = ctl.update_spec_field("laguna", "ctx_size", "99999")
    # Laguna's binary + env + extra_args must survive.
    assert spec.ctx_size == 99999
    assert spec.env.get("HSA_OVERRIDE_GFX_VERSION") == "11.5.1"
    assert spec.env.get("GGML_CUDA_ENABLE_UNIFIED_MEMORY") == "1"
    assert spec.extra_args == ["--reasoning", "off"]
    assert spec.binary == str(ctl.Path.home() / "llama.cpp/build/bin/llama-server")


def test_repair_heals_legacy_entry_without_source_field(env, monkeypatch):
    """The first wave of pulled entries were written before the
    ``source`` field existed — they have no ``source`` in the raw
    dict. The repair should still heal them and tag them as pulled."""
    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    reg = env / "models.yaml"
    reg.write_text(yaml.safe_dump({"models": [
        # Configured defaults (keep `source: default` explicit).
        {"name": "9b", "gguf": "/tmp/9b.gguf", "llama_port": 8088,
         "daemon_port": 8787, "ctx_size": 200000, "max_tokens": 100000,
         "model_id": "qwen3.5-9b", "always_on": True, "boot": True,
         "source": "default"},
        {"name": "35b", "gguf": "/tmp/35b.gguf", "llama_port": 8089,
         "daemon_port": 8788, "ctx_size": 200000, "max_tokens": 100000,
         "model_id": "qwen3.6-35b-a3b", "source": "default"},
        # Reference: a custom-build spec.
        {"name": "laguna", "gguf": "/tmp/lag.gguf", "llama_port": 8090,
         "daemon_port": 8789, "ctx_size": 131072, "max_tokens": 64000,
         "model_id": "laguna-s-2.1", "boot": True,
         "binary": "/home/lukehadzic/llama.cpp/build/bin/llama-server",
         "source": "default"},
        # Legacy pulled entry: no `source`, no `binary`, no `env`.
        {"name": "oldpull", "gguf": "/tmp/oldpull.gguf", "llama_port": 8091,
         "daemon_port": 8790, "ctx_size": 200000, "max_tokens": 100000,
         "model_id": "unsloth/oldpull"},
    ]}))
    healed = ctl.backfill_binary_env()
    assert healed == ["oldpull"]
    refreshed = ctl.get_model("oldpull")
    assert refreshed.binary == "/home/lukehadzic/llama.cpp/build/bin/llama-server"
    assert refreshed.source == "pulled"
    # Re-run is a no-op.
    assert ctl.backfill_binary_env() == []


def test_use_model_syncs_cli_configs(env, hf, monkeypatch):
    """`toki model pull --use` and `toki model use <name>` must end
    with the opencode/openclaw/hermes configs rewritten so every
    registry spec (defaults + pulled) has a provider entry — otherwise
    the clients can't see the pulled stack."""
    import json as _json

    import yaml as _yaml

    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)

    # Point the CLI config writers at temp files.
    oc_path = env / "opencode.json"
    cla_path = env / "openclaw.json"
    her_path = env / "config.yaml"
    monkeypatch.setenv("OPENCODE_CONFIG", str(oc_path))
    monkeypatch.setenv("OPENCLAW_CONFIG", str(cla_path))
    monkeypatch.setenv("HERMES_CONFIG", str(her_path))

    # Avoid actually starting llama-server + daemon.
    monkeypatch.setattr(ctl, "start_model", lambda spec: None)
    monkeypatch.setattr(ctl, "set_active", lambda name, registry=None: None)
    monkeypatch.setattr(ctl, "stop_model", lambda spec, force=False: None)
    monkeypatch.setattr(ctl, "llama_reachable", lambda spec: False)
    monkeypatch.setattr(ctl, "daemon_reachable", lambda spec: False)

    pulled = ctl.pull_model("acme/Pulled-GGUF", "pulled")
    ctl.use_model("pulled")

    # Every CLI config must now know about the pulled spec.
    oc = _json.loads(oc_path.read_text())
    cla = _json.loads(cla_path.read_text())
    h = _yaml.safe_load(her_path.read_text())
    assert f"tokirecur-{pulled.name}" in oc["provider"]
    assert f"tokirecur-{pulled.name}" in cla["models"]["providers"]
    assert f"tokirecur-{pulled.name}" in h["providers"]
    # And the active model line points at the pulled spec.
    assert oc["model"] == f"tokirecur-{pulled.name}/{pulled.model_id}"
    assert cla["agents"]["defaults"]["model"]["primary"] == \
        f"tokirecur-{pulled.name}/{pulled.model_id}"
    assert h["model"]["provider"] == f"tokirecur-{pulled.name}"


# ── Daemon reasoning headroom ────────────────────────────────────────


def test_resolve_max_tokens_floors_small_requests():
    from tokirecur.api.app import (
        MAX_TOKENS_FLOOR,
        THINKING_TOKENS_FLOOR,
        _resolve_max_tokens,
    )

    # Thinking is enabled by default (auto) → the higher floor so
    # verbose reasoning leaves room for content.
    assert _resolve_max_tokens(8) == THINKING_TOKENS_FLOOR
    assert _resolve_max_tokens(0) == THINKING_TOKENS_FLOOR
    # reasoning=off keeps the small floor.
    assert _resolve_max_tokens(8, reasoning="off") == MAX_TOKENS_FLOOR
    # Larger requests are respected (never capped down).
    assert _resolve_max_tokens(2048) == 2048
    assert _resolve_max_tokens(MAX_TOKENS_FLOOR, reasoning="off") == MAX_TOKENS_FLOOR


def test_chat_completion_includes_reasoning_content(tmp_path, monkeypatch):
    """End-to-end: a reasoning response from the engine surfaces as
    ``reasoning_content`` in the daemon's non-stream JSON answer, and
    the request's small ``max_tokens`` is floored to the daemon's floor."""
    from fastapi.testclient import TestClient

    from tokirecur.api.app import THINKING_TOKENS_FLOOR, create_app
    from tokirecur.backend import LlamaServerBackend
    from tokirecur.core.recycling import RecyclingEngine, RecyclingResult
    from tokirecur.index import Index

    backend = LlamaServerBackend(base_url="http://llama.invalid")
    index = Index()
    app = create_app(
        backend=backend, index=index, cache_dir=tmp_path,
        enable_auth=False, enable_rate_limit=False,
    )

    observed_max_tokens: dict[str, int] = {}

    def fake_serve_chat(self, messages, *, max_tokens, **kw):
        observed_max_tokens["v"] = max_tokens
        return RecyclingResult(
            hit=False, soft_hit=False, cache_id=None, reuse_depth=0,
            embedding_similarity=0.0, prefilled_tokens=4, skipped_tokens=0,
            generated_tokens=12, ttft_ms=15.0, total_ms=80.0,
            prefill_tokens_per_sec=100.0, generated_tokens_per_sec=50.0,
            output_text="Hello!", reasoning="thinking happened…",
            slot_id=0, spilled=False, tool_calls=None, finish_reason="stop",
        )

    monkeypatch.setattr(RecyclingEngine, "serve_chat", fake_serve_chat)

    client = TestClient(app)
    res = client.post("/v1/chat/completions", json={
        "model": "test", "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,  # way below the floor
    })
    assert res.status_code == 200, res.text
    body = res.json()
    msg = body["choices"][0]["message"]
    assert msg["content"] == "Hello!"
    assert msg["reasoning_content"] == "thinking happened…"
    # The engine saw the floored budget, not the client's 8.
    assert observed_max_tokens["v"] == THINKING_TOKENS_FLOOR


def test_chat_completion_stream_emits_reasoning_sse(tmp_path, monkeypatch):
    """The streaming path emits ``reasoning_content`` deltas as their own
    SSE chunks alongside the visible ``content`` deltas."""
    from fastapi.testclient import TestClient

    from tokirecur.api.app import create_app
    from tokirecur.backend import LlamaServerBackend
    from tokirecur.index import Index

    backend = LlamaServerBackend(base_url="http://llama.invalid")
    index = Index()
    app = create_app(
        backend=backend, index=index, cache_dir=tmp_path,
        enable_auth=False, enable_rate_limit=False,
    )

    async def fake_serve_chat_stream(self, messages, *, max_tokens, **kw):
        yield {"kind": "reasoning", "delta": "step 1 "}
        yield {"kind": "reasoning", "delta": "step 2"}
        yield {"kind": "delta", "delta": "Answer"}

        from tokirecur.core.recycling import RecyclingResult
        yield {
            "kind": "result",
            "result": RecyclingResult(
                hit=False, soft_hit=False, cache_id=None, reuse_depth=0,
                embedding_similarity=0.0, prefilled_tokens=4,
                skipped_tokens=0, generated_tokens=2,
                ttft_ms=15.0, total_ms=80.0,
                prefill_tokens_per_sec=100.0,
                generated_tokens_per_sec=50.0,
                output_text="Answer", reasoning="step 1 step 2",
                slot_id=0, spilled=False,
                tool_calls=None, finish_reason="stop",
            ),
        }

    import tokirecur.core.recycling as recycling_mod
    monkeypatch.setattr(
        recycling_mod.RecyclingEngine, "serve_chat_stream",
        fake_serve_chat_stream,
    )

    client = TestClient(app)
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "test", "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8, "stream": True,
    }) as res:
        assert res.status_code == 200
        body = "".join(res.iter_text())

    # Reasoning is delivered as its own SSE delta field, not folded into content.
    assert '"reasoning_content": "step 1 "' in body
    assert '"reasoning_content": "step 2"' in body
    assert '"content": "Answer"' in body
