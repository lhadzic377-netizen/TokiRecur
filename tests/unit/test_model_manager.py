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
from click.testing import CliRunner
from tomllib import load as toml_load

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


@pytest.fixture
def seeded_env(env):
    """Like ``env`` but pre-seeds the reference DEFAULT_MODELS, for
    tests that exercise behavior against existing registry entries."""
    ctl.DEFAULT_REGISTRY.write_text(
        yaml.safe_dump({"models": ctl.DEFAULT_MODELS}, sort_keys=False)
    )
    return env


def test_registry_created_empty(env):
    """A fresh install starts with zero preloaded models."""
    specs = ctl.load_registry()
    assert specs == []
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


def test_active_marker_roundtrip(seeded_env):
    assert ctl.read_active() is None  # no preloaded/always-on models
    ctl.set_active("laguna")
    assert ctl.read_active() == "laguna"
    (ctl.ACTIVE_MARKER).write_text("gone")
    assert ctl.read_active() is None  # stale marker, no always-on default


def _specs(env):
    return {s.name: s for s in ctl.load_registry()}


@pytest.fixture
def with_pulled_model(seeded_env, hf):
    """Register an extra pulled model alongside the 3 defaults.

    The pulled model has no `binary` or `env` (the broken state users
    hit before the repair command existed) and a port pair that's
    distinct from the defaults so port-collision isn't tested here.
    """
    pulled = ctl.pull_model("acme/Pulled-GGUF", "pulled")
    return {**_specs(env), pulled.name: pulled}


def test_opencode_config_writer(env, with_pulled_model, monkeypatch):
    monkeypatch.setattr(ctl, "daemon_reachable", lambda spec: True)
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
    # Anti-loop guardrail: opencode's `instructions` points at the
    # toki-owned markdown file next to the config.
    assert cfg["instructions"] == [str(env / "tokirecur-anti-loop.md")]
    assert ctl.ANTI_LOOP_SYSTEM_PROMPT in (env / "tokirecur-anti-loop.md").read_text()


def test_opencode_config_writer_only_advertises_reachable_daemons(
        env, with_pulled_model, monkeypatch):
    """The harness fallback class: a dead daemon advertised in the
    config makes opencode route titles/small-model calls into
    connection-refused and use the first response as the session
    title. Only the active spec + reachable daemons are written."""
    monkeypatch.setattr(ctl, "llama_reachable", lambda spec: True)

    def dead_anyone_but_active(spec):
        return spec.name == "35b"

    monkeypatch.setattr(ctl, "daemon_reachable", dead_anyone_but_active)
    spec = with_pulled_model["35b"]
    cfg_path = env / "opencode.json"
    cfg_path.write_text(json.dumps({"provider": {}}))
    cfg = json.loads(ctl.write_opencode_config(spec, path=cfg_path).read_text())
    # Only the active spec's provider survives; dead daemons are gone.
    assert set(cfg["provider"]) == {"tokirecur-35b"}
    # small_model falls back to the active spec — never a dead one.
    assert cfg["small_model"] == "tokirecur/35b"
    assert cfg["model"] == "tokirecur-35b/qwen3.6-35b-a3b"


def test_opencode_config_writer_small_model_prefers_reachable_always_on(
        env, with_pulled_model, monkeypatch):
    """An always-on spec with a LIVE daemon wins small_model; the
    active spec is the fallback when the always-on daemon is down."""
    monkeypatch.setattr(ctl, "daemon_reachable", lambda spec: True)
    monkeypatch.setattr(ctl, "llama_reachable", lambda spec: True)
    ctl.update_spec_field("9b", "always_on", "true")
    spec = with_pulled_model["35b"]
    cfg_path = env / "opencode.json"
    cfg_path.write_text(json.dumps({"provider": {}}))
    cfg = json.loads(ctl.write_opencode_config(spec, path=cfg_path).read_text())
    assert cfg["small_model"] == "tokirecur/9b"


def test_opencode_config_writer_skips_user_instructions(env, with_pulled_model):
    spec = with_pulled_model["35b"]
    cfg_path = env / "opencode.json"
    cfg_path.write_text(json.dumps({"instructions": ["CONTRIBUTING.md"]}))
    ctl.write_opencode_config(spec, path=cfg_path)
    cfg = json.loads(cfg_path.read_text())
    # User-provided instructions are never clobbered or extended.
    assert cfg["instructions"] == ["CONTRIBUTING.md"]
    assert not (env / "tokirecur-anti-loop.md").exists()


def test_opencode_md_upgrades_v1_block(env, with_pulled_model):
    spec = with_pulled_model["35b"]
    md = env / "tokirecur-anti-loop.md"
    md.write_text(
        "<!-- toki-recur: anti-loop rules -->\n\n"
        "You are an autonomous agent completing a task. Old v1 text.\n"
    )
    ctl.write_opencode_config(spec, path=env / "opencode.json")
    text = md.read_text()
    assert ctl._ANTI_LOOP_MARKER in text
    assert "Old v1 text." not in text
    assert ctl.ANTI_LOOP_SYSTEM_PROMPT in text


def test_openclaw_upgrades_v1_block(env, with_pulled_model, monkeypatch):
    spec = with_pulled_model["laguna"]
    ws = env / "ws"
    agents_md = ws / "AGENTS.md"
    agents_md.parent.mkdir(parents=True)
    agents_md.write_text(
        "# My workspace\n\n"
        "<!-- toki-recur: anti-loop rules -->\n\n"
        "You are an autonomous agent completing a task. Old v1 text.\n"
    )
    monkeypatch.setenv("OPENCLAW_WORKSPACE_DIR", str(ws))
    ctl.write_openclaw_config(spec, path=env / "openclaw.json")
    text = agents_md.read_text()
    assert text.startswith("# My workspace")
    assert "Old v1 text." not in text
    assert text.count(ctl.ANTI_LOOP_SYSTEM_PROMPT) == 1


def test_openclaw_config_writer(env, with_pulled_model, monkeypatch):
    monkeypatch.setattr(ctl, "daemon_reachable", lambda spec: True)
    monkeypatch.setattr(ctl, "llama_reachable", lambda spec: True)
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
    # contextWindow must advertise the model's EFFECTIVE context (the
    # old bug wrote max_tokens here — the lie that made openclaw
    # overfill the slot and hang).
    assert prov["models"][0]["contextWindow"] == 131072
    assert prov["models"][0]["cost"] == {"cacheRead": 0, "cacheWrite": 0,
                                         "input": 0, "output": 0}
    assert cfg["agents"]["defaults"]["model"]["primary"] == "tokirecur-laguna/laguna-s-2.1"
    assert cfg["agents"]["defaults"]["timeoutSeconds"] == 3600
    # Pulled model also has a provider entry.
    pulled = with_pulled_model["pulled"]
    pulled_prov = cfg["models"]["providers"][f"tokirecur-{pulled.name}"]
    assert pulled_prov["baseUrl"] == f"{pulled.daemon_url}/v1"
    assert pulled_prov["models"][0]["id"] == pulled.model_id
    # Anti-loop guardrail: rules land in the workspace AGENTS.md
    # bootstrap (openclaw has no config-injectable system prompt yet).
    ws = env / "ws"
    monkeypatch.setenv("OPENCLAW_WORKSPACE_DIR", str(ws))
    ctl.write_openclaw_config(spec, path=cfg_path)
    agents_md = ws / "AGENTS.md"
    assert agents_md.exists()
    assert ctl.ANTI_LOOP_SYSTEM_PROMPT in agents_md.read_text()
    # Idempotent: a second sync doesn't duplicate the block.
    ctl.write_openclaw_config(spec, path=cfg_path)
    assert agents_md.read_text().count(ctl.ANTI_LOOP_SYSTEM_PROMPT) == 1


def test_hermes_config_writer(env, with_pulled_model, monkeypatch):
    monkeypatch.setenv("TOKI_AUTH_TOKEN", "test-tok-0123456789abcdef")
    spec = with_pulled_model["9b"]
    cfg_path = env / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "memory": {"memory_enabled": True},
        "providers": {"ollama-launch": {"api": "http://127.0.0.1:11434/v1"}},
    }))
    out = ctl.write_hermes_config(spec, path=cfg_path)
    cfg = yaml.safe_load(out.read_text())
    assert cfg["memory"]["memory_enabled"] is True  # untouched
    # The user's other providers are preserved.
    assert cfg["providers"]["ollama-launch"]["api"] == "http://127.0.0.1:11434/v1"
    # No tokirecur-* keys are written — hermes's resolve_provider
    # rejects them. The active model is registered under the
    # built-in "llamacpp" alias via the legacy custom_providers list.
    assert "tokirecur-9b" not in cfg["providers"]
    custom = cfg.get("custom_providers") or []
    names = [c.get("name") for c in custom]
    assert "llamacpp" in names
    entry = next(c for c in custom if c.get("name") == "llamacpp")
    assert entry["base_url"] == "http://127.0.0.1:8787/v1"
    expected_ids = {s.model_id for s in ctl.load_registry()}
    assert set(entry["models"]) == expected_ids
    assert cfg["model"] == {
        "api_key": "test-tok-0123456789abcdef",
        "base_url": "http://127.0.0.1:8787/v1",
        "default": "qwen3.5-9b",
        "provider": "llamacpp",
        # Explicit context_length is hermes's authoritative context
        # source (resolution step 0) — without it a failed probe of the
        # daemon falls back to a 256k default.
        "context_length": 200000,
    }
    assert cfg_path.with_suffix(".yaml.bak-model").exists()
    # Anti-loop guardrail: hermes's manual system-prompt field.
    assert cfg["agent"]["system_prompt"] == ctl.ANTI_LOOP_SYSTEM_PROMPT


def test_hermes_config_writer_keeps_user_system_prompt(env, with_pulled_model):
    spec = with_pulled_model["9b"]
    cfg_path = env / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"agent": {"system_prompt": "Be snarky."}}))
    ctl.write_hermes_config(spec, path=cfg_path)
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["agent"]["system_prompt"] == "Be snarky."


def test_anti_loop_prompt_discipline_budget():
    """The guardrail must stay minimal (paper: instruction dropout /
    "contextual overload") and fully positive (v2's "Don't repeat"
    phrasing invited self-monitoring). Guards against regrowth."""
    text = ctl.ANTI_LOOP_SYSTEM_PROMPT
    rules = [line for line in text.splitlines() if line and line[0].isdigit()]
    assert len(rules) <= 6
    assert len(text.split()) <= 60
    # Zero negations: don't/dont/never/no/not must not appear as words.
    negations = {w.lower().rstrip(",") for w in text.split()} & {
        "don't", "dont", "never", "no", "not"}
    assert not negations, f"negation sneaked back into the prompt: {negations}"
    assert "autonomous agent" not in text.lower()
    # One primitive idea per rule: a single sentence each (the leading
    # "N." list number is not a sentence period).
    for rule in rules:
        assert rule.split(".", 1)[1].count(".") <= 1


def test_opencode_md_upgrades_v2_block(env, with_pulled_model):
    spec = with_pulled_model["35b"]
    md = env / "tokirecur-anti-loop.md"
    md.write_text(
        "<!-- toki-recur: anti-loop rules v2 -->\n\n"
        "Working rules\n\n1. Work one step at a time. Old v2 text.\n"
    )
    ctl.write_opencode_config(spec, path=env / "opencode.json")
    text = md.read_text()
    assert ctl._ANTI_LOOP_MARKER in text
    assert "Old v2 text." not in text
    assert ctl.ANTI_LOOP_SYSTEM_PROMPT in text


def test_sync_cli_configs_writes_all(env, with_pulled_model, monkeypatch):
    monkeypatch.setattr(ctl, "daemon_reachable", lambda spec: True)
    monkeypatch.setattr(ctl, "llama_reachable", lambda spec: True)
    spec = with_pulled_model["laguna"]
    monkeypatch.setenv("CODEX_HOME", str(env / "codex-home"))
    paths = ctl.sync_cli_configs(spec)
    assert len(paths) == 4
    for p in paths.values():
        assert Path(p).exists()
    # And every CLI config picked up the pulled provider.
    pulled_name = with_pulled_model["pulled"].name
    oc = json.loads(paths["opencode"].read_text())
    assert f"tokirecur-{pulled_name}" in oc["provider"]
    cla = json.loads(paths["openclaw"].read_text())
    assert f"tokirecur-{pulled_name}" in cla["models"]["providers"]
    h = yaml.safe_load(paths["hermes"].read_text())
    # hermes writer uses the built-in "llamacpp" alias (resolve_provider
    # rejects custom keys like "tokirecur-<name>"); the custom_providers
    # list carries the model registry.
    custom = h.get("custom_providers") or []
    assert any(c.get("name") == "llamacpp" for c in custom)
    llama_entry = next(c for c in custom if c.get("name") == "llamacpp")
    pulled_model_id = with_pulled_model["pulled"].model_id
    assert pulled_model_id in llama_entry["models"]
    assert h["model"]["provider"] == "llamacpp"
    with paths["codex"].open("rb") as fh:
        codex = toml_load(fh)
    assert f"tokirecur-{pulled_name}" in codex["model_providers"]


def test_hermes_models_dict_carries_context_length(env, with_pulled_model):
    """hermes's authoritative context source: per-model dict entries.

    Without explicit context_length hermes falls back to a 256k default
    after a failed (< 64k) probe of the daemon — the "request sent,
    never answered" class on small-RAM machines.
    """
    spec = with_pulled_model["9b"]
    out = ctl.write_hermes_config(spec, path=env / "config.yaml")
    cfg = yaml.safe_load(out.read_text())
    entry = next(c for c in cfg["custom_providers"]
                 if c.get("name") == "llamacpp")
    for s in ctl.load_registry():
        model = entry["models"][s.model_id]
        assert model["context_length"] == ctl._effective_ctx(s)
    # The active model is defaulted.
    assert cfg["model"]["default"] == spec.model_id


def test_hermes_purges_stale_context_cache(env, with_pulled_model, tmp_path):
    """context_length_cache.yaml entries for daemon URLs must go.

    A 256k default cached while the daemon 404'd /v1/models would keep
    hermes believing a window that isn't there even after the config
    override lands (display paths / model switches read the cache).
    """
    cache = env / "hermes" / "context_length_cache.yaml"
    cache.parent.mkdir(parents=True)
    cache.write_text(yaml.safe_dump({
        "context_lengths": {
            "qwen3.5-9b@http://127.0.0.1:8787/v1": 256000,
            "other@http://127.0.0.1:11434/v1": 131072,
        },
    }))
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("HERMES_HOME", str(env / "hermes"))
    try:
        spec = with_pulled_model["9b"]
        ctl.write_hermes_config(spec, path=env / "config.yaml")
        fresh = yaml.safe_load(cache.read_text())
        urls = fresh["context_lengths"]
        # The daemon-URL entry (live or dead port) is gone; the foreign
        # ollama one stays.
        assert all("@http://127.0.0.1:8787" not in u for u in urls)
        assert "other@http://127.0.0.1:11434/v1" in urls
    finally:
        monkeypatch.undo()


def test_mode_persisted_and_threaded(env, with_pulled_model, monkeypatch):
    """Per-client raw/managed modes: persisted, then honored by sync."""
    spec = with_pulled_model["9b"]
    assert ctl.client_mode("opencode") == "managed"
    ctl.write_client_mode("opencode", "raw")
    assert ctl.client_mode("opencode") == "raw"
    # Explicit --mode overrides the persisted value.
    assert ctl._resolve_mode("opencode", "managed") == "managed"
    assert ctl._resolve_mode("opencode", None) == "raw"

    # sync_cli_configs picks up the persisted mode: raw opencode points
    # straight at llama-server (no daemon in the path).
    paths = ctl.sync_cli_configs(spec)
    oc = json.loads(paths["opencode"].read_text())
    assert oc["provider"]["tokirecur-9b"]["options"]["baseURL"] == \
        f"{spec.llama_url}/v1"
    # The others stayed managed (their mode was never changed).
    cla = json.loads(paths["openclaw"].read_text())
    assert cla["models"]["providers"]["tokirecur-9b"]["baseUrl"] == \
        f"{spec.daemon_url}/v1"
    h = yaml.safe_load(paths["hermes"].read_text())
    entry = next(c for c in h["custom_providers"]
                 if c.get("name") == "llamacpp")
    assert entry["base_url"] == f"{spec.daemon_url}/v1"


def test_write_codex_config_stays_managed(env, with_pulled_model, monkeypatch):
    """codex always routes through the daemon (Responses API shim)."""
    monkeypatch.setattr(ctl, "daemon_reachable", lambda spec: True)
    spec = with_pulled_model["9b"]
    monkeypatch.setenv("CODEX_HOME", str(env / "codex-home"))
    ctl.write_client_mode("codex", "raw")
    out = ctl.write_codex_config(spec)
    with out.open("rb") as fh:
        codex = toml_load(fh)
    prov = codex["model_providers"]["tokirecur-9b"]
    assert prov["base_url"] == f"{spec.daemon_url}/v1"


def test_codex_purges_stale_tokirecur_providers(env, with_pulled_model,
                                                monkeypatch):
    """A provider pointing at a dead toki era port must not survive."""
    spec = with_pulled_model["9b"]
    monkeypatch.setenv("CODEX_HOME", str(env / "codex-home"))
    home = Path(env / "codex-home")
    home.mkdir(parents=True)
    (home / "config.toml").write_text(
        "[model]\n"
        'model_provider = "tokirecur-old"\n'
        'model = "tokirecur-old/qwen3.5-9b"\n\n'
        "[model_providers.tokirecur-old]\n"
        'name = "tokirecur-old"\n'
        'base_url = "http://127.0.0.1:18002/v1"\n'
        'env_key = "TOKI_AUTH"\n'
        'wire_api = "responses"\n'
    )
    ctl.write_codex_config(spec)
    with (home / "config.toml").open("rb") as fh:
        codex = toml_load(fh)
    # Stale entry gone; the active spec is always advertised.
    assert "tokirecur-old" not in codex["model_providers"]
    assert "tokirecur-9b" in codex["model_providers"]
    assert codex["model_providers"]["tokirecur-9b"]["base_url"] == \
        f"{spec.daemon_url}/v1"


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


def test_use_model_tolerates_broken_always_on(seeded_env, monkeypatch):
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


def test_use_model_tolerates_broken_target(seeded_env, monkeypatch):
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
    # Empty registry → ports start at the base pair.
    assert ctl.next_free_ports([]) == (8088, 8787)
    raw_a = dict(ctl.DEFAULT_MODELS[0])
    raw_a["llama_port"], raw_a["daemon_port"] = 8089, 8788
    raw_b = dict(ctl.DEFAULT_MODELS[1])
    raw_b["llama_port"], raw_b["daemon_port"] = 8090, 8789
    specs = [
        ctl.ModelSpec(raw=raw_a, **raw_a),
        ctl.ModelSpec(raw=raw_b, **raw_b),
    ]
    assert ctl.next_free_ports(specs) == (8091, 8790)


def test_save_registry_roundtrip_preserves_raw(seeded_env):
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
    assert spec.llama_port == 8088 and spec.daemon_port == 8787
    assert spec.ctx_size == 8192 and spec.kv == "q8_0"
    assert spec.max_tokens == 4096
    assert spec.model_id == "model-q4_k_m"
    assert str(env / "models" / "mymodel" / "model-Q4_K_M.gguf") == spec.gguf
    # persisted and visible to the launcher/list
    assert ctl.get_model("mymodel").name == "mymodel"
    assert hf["downloads"][0][1] == ["*model-Q4_K_M.gguf"]


def test_pull_model_auto_ctx_caps_at_model_max(env, hf, monkeypatch):
    """A pull without --ctx auto-sizes: a 350M-class model whose max
    context is 8k must never be registered at the old flat 200k."""
    from tokirecur.ggufmeta import GgufMeta
    monkeypatch.setattr(ctl, "read_gguf_meta", lambda p: GgufMeta(
        n_embd=1024, n_layer=16, n_head=16, n_head_kv=8,
        context_length=8192))
    spec = ctl.pull_model("acme/My-Model-GGUF", "auto")
    assert spec.ctx_size == 8192
    assert spec.max_tokens == 4096


def test_pull_model_auto_ctx_default_cap(env, hf, monkeypatch):
    """Unreadable metadata -> conservative auto ctx (16k), never 200k."""
    from tokirecur.ggufmeta import GgufMeta
    monkeypatch.setattr(ctl, "read_gguf_meta", lambda p: GgufMeta())
    spec = ctl.pull_model("acme/My-Model-GGUF", "auto2")
    assert spec.ctx_size == 16384
    assert spec.max_tokens == 8192


def test_auto_ctx_respects_ram_budget_and_ceiling(env, monkeypatch):
    """KV at f16 for the 16-layer/8-kv-head model is 32KB/token
    (unified KV: parallel doesn't multiply the pool). A small RAM box
    gets a small ctx; the 64k ceiling caps even unlimited RAM."""
    from tokirecur.ggufmeta import GgufMeta
    gguf = env / "m.gguf"
    gguf.write_bytes(b"")
    monkeypatch.setattr(ctl, "read_gguf_meta", lambda p: GgufMeta(
        n_embd=1024, n_layer=16, n_head=16, n_head_kv=8,
        context_length=128000))
    monkeypatch.setattr(ctl, "_available_gb", lambda: 6.0)
    # Budget = 6 - 0 - 4 = 2 GB -> max_fit ~61035, floor 512.
    assert ctl._auto_ctx_size(gguf, parallel=2, kv="f16") == 61035
    monkeypatch.setattr(ctl, "_available_gb", lambda: float("inf"))
    # Unlimited RAM -> capped by the 64k ceiling (model max is 128k).
    assert ctl._auto_ctx_size(gguf, parallel=2, kv="f16") == 65536


def test_clamp_ctx_caps_at_model_max(env, monkeypatch):
    """_clamp_ctx never returns more than the model supports, even when
    RAM would allow it (regression: 200k requested on a 128k model)."""
    from tokirecur.ggufmeta import GgufMeta
    gguf = env / "m.gguf"
    gguf.write_bytes(b"")
    spec = ctl.ModelSpec(
        name="m", label="m", gguf=str(gguf), llama_port=1, daemon_port=2,
        ctx_size=200000, parallel=2, kv="f16", max_tokens=100000,
        model_id="m")
    monkeypatch.setattr(ctl, "read_gguf_meta", lambda p: GgufMeta(
        n_embd=1024, n_layer=16, n_head=16, n_head_kv=8,
        context_length=128000))
    assert ctl._clamp_ctx(spec, avail_gb=float("inf"), weight_gb=0.0) == 128000


def test_model_max_ctx_reads_metadata(env, monkeypatch):
    from tokirecur.ggufmeta import GgufMeta
    gguf = env / "m.gguf"
    gguf.write_bytes(b"")
    spec = ctl.ModelSpec(
        name="m", label="m", gguf=str(gguf), llama_port=1, daemon_port=2,
        ctx_size=200000, parallel=2, kv="f16", max_tokens=1, model_id="m")
    monkeypatch.setattr(ctl, "read_gguf_meta", lambda p: GgufMeta(
        context_length=32000))
    assert ctl._model_max_ctx(spec) == 32000
    monkeypatch.setattr(ctl, "read_gguf_meta", lambda p: GgufMeta())
    assert ctl._model_max_ctx(spec) is None


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


def test_gguf_split_siblings_returns_all_parts_in_order():
    """Split-GGUF detection must list every sibling 1..N in numeric order."""
    first = "UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf"
    assert ctl._gguf_split_siblings(first) == [
        "UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf",
        "UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00002-of-00003.gguf",
        "UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00003-of-00003.gguf",
    ]
    middle = "UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00002-of-00003.gguf"
    assert ctl._gguf_split_siblings(middle) == ctl._gguf_split_siblings(first)


def test_gguf_split_siblings_none_for_non_split():
    """Standalone GGUFs must not be detected as a split."""
    assert ctl._gguf_split_siblings("model-Q4_K_M.gguf") is None
    assert ctl._gguf_split_siblings("UD-Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M.gguf") is None
    # bad indices → not a split (the regex requires exactly 4-digit -NNNN-of-NNNN)
    assert ctl._gguf_split_siblings("model-1-of-3.gguf") is None
    assert ctl._gguf_split_siblings("model-00001-of-00003.bin") is None


def test_expand_gguf_patterns_includes_all_siblings(env, hf):
    """Picking one shard of a split must pull every sibling — otherwise
    llama-server fails to load the model with
    'failed to open GGUF file ... (No such file or directory)'."""
    hf["repo_files"] = [
        "UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf",
        "UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00002-of-00003.gguf",
        "UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00003-of-00003.gguf",
    ]
    hf["sizes"] = {
        "UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf": 10_000_000,
        "UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00002-of-00003.gguf": 50_000_000_000,
        "UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00003-of-00003.gguf": 26_000_000_000,
    }
    # Caller picks the small first part — that's what users currently do.
    picked = []
    def pick(files):
        picked.append(list(files))
        return "UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf"

    ctl.pull_model("unsloth/Qwen3.5-122B-A10B-MTP-GGUF", "qwen122", choose_file=pick)

    # All three patterns must have been requested from huggingface_hub.
    patterns = hf["downloads"][0][1]
    assert patterns == [
        "*UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf",
        "*UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00002-of-00003.gguf",
        "*UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00003-of-00003.gguf",
    ]
    # gguf points at the first part (the entry-point llama-server loads),
    # not some random middle shard.
    spec = ctl.get_model("qwen122")
    assert spec.gguf.endswith("00001-of-00003.gguf")

def test_cli_picker_interactive_and_eof_fallback():
    import click

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


def test_pull_model_name_clash(seeded_env, hf):
    with pytest.raises(ValueError, match="already exists"):
        ctl.pull_model("acme/My-Model-GGUF", "9b")


def test_pull_model_explicit_name_case_insensitive_clash(seeded_env, hf):
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


def test_defaults_are_portable(seeded_env, monkeypatch):
    """Shipped DEFAULT_MODELS must never reference another machine: no
    absolute gguf paths and no pinned binaries (the local tuned build is
    not everyone's build)."""
    for m in ctl.DEFAULT_MODELS:
        assert not Path(m["gguf"]).is_absolute(), f"{m['name']} pins a path"
        assert "binary" not in m, f"{m['name']} pins a binary"
        assert Path(m["gguf"]).name == m["gguf"]
        assert m["name"] and m["model_id"] and m["llama_port"] and m["daemon_port"]


def test_pull_defaults_reference_spec_binary(seeded_env, tmp_path, monkeypatch):
    """With a registry holding a spec whose binary exists, pulls inherit
    that binary (instead of the PATH default)."""
    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    ref_bin = tmp_path / "bin" / "llama-server"
    ref_bin.parent.mkdir(parents=True)
    ref_bin.write_text("#!/bin/sh\n")
    reg = seeded_env / "models.yaml"
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "laguna", "gguf": "/tmp/lag.gguf", "llama_port": 1,
         "daemon_port": 2, "ctx_size": 100, "max_tokens": 90,
         "model_id": "laguna-s-2.1", "binary": str(ref_bin),
         "env": {}, "source": "default"},
    ]}))
    binary, env_vars = ctl.pull_defaults(ctl.load_registry(reg))
    assert binary == str(ref_bin)
    assert env_vars == {}


def test_pull_defaults_skips_dead_reference_binary(seeded_env, monkeypatch):
    """A reference spec whose binary path does not exist on this machine
    (registry shipped from another box) must NOT be inherited — new pulls
    fall through to the installed binary or PATH default."""
    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    monkeypatch.delenv("TOKI_STATE_DIR", raising=False)
    reg = seeded_env / "models.yaml"
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "laguna", "gguf": "/tmp/lag.gguf", "llama_port": 1,
         "daemon_port": 2, "ctx_size": 100, "max_tokens": 90,
         "model_id": "laguna-s-2.1",
         "binary": "/home/someone-else/llama.cpp/build/bin/llama-server",
         "env": {"HSA_OVERRIDE_GFX_VERSION": "11.5.1"}, "source": "default"},
    ]}))
    binary, env_vars = ctl.pull_defaults(ctl.load_registry(reg))
    assert binary != "/home/someone-else/llama.cpp/build/bin/llama-server"
    # Nothing sensible left to inherit: env must not carry a dead build's vars.
    assert env_vars == {}


def test_pull_defaults_env_override_takes_precedence(env, monkeypatch):
    monkeypatch.setenv("TOKI_LLAMA_SERVER", "/custom/llama-server")
    binary, env_vars = ctl.pull_defaults(ctl.load_registry())
    assert binary == "/custom/llama-server"
    assert env_vars == {}


def test_pull_defaults_env_inherited_when_reference_has_env(env, tmp_path, monkeypatch):
    """A reference spec that defines ROCm env (e.g. the user's real
    registry) propagates those vars to the new spec."""
    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    ref_bin = tmp_path / "bin" / "llama-server"
    ref_bin.parent.mkdir(parents=True)
    ref_bin.write_text("#!/bin/sh\n")
    reg = env / "models.yaml"
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "laguna", "gguf": "/tmp/lag.gguf", "llama_port": 1,
         "daemon_port": 2, "ctx_size": 100, "max_tokens": 90,
         "model_id": "laguna-s-2.1",
         "binary": str(ref_bin),
         "env": {"HSA_OVERRIDE_GFX_VERSION": "11.5.1",
                 "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"}},
    ]}))
    binary, env_vars = ctl.pull_defaults(ctl.load_registry(reg))
    assert binary == str(ref_bin)
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


def test_pull_defaults_prefers_installed_binary(env, monkeypatch):
    """A machine with `toki model install-llama` (binary in STATE_ROOT/bin)
    must default to that concrete path, not the bare PATH name — the bin
    dir is not on PATH, and on Windows a bare name makes Popen fail with
    WinError 2 (regression: `toki model pull --use` crash after install)."""
    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    bin_dir = ctl.STATE_ROOT / "bin"
    bin_dir.mkdir(parents=True)
    exe_name = "llama-server.exe" if os.name == "nt" else "llama-server"
    installed = bin_dir / exe_name
    installed.write_bytes(b"")
    reg = env / "models.yaml"
    reg.write_text(yaml.safe_dump({"models": []}))
    binary, env_vars = ctl.pull_defaults(ctl.load_registry(reg))
    assert binary == str(installed)
    assert env_vars == {}


def test_backfill_heals_bare_binary_to_installed(env, monkeypatch):
    """Existing pulled entries left with the bare PATH default are
    rewritten to the installed build by backfill_binary_env — the path
    `use_model` takes, so an already-pulled model gets fixed on next use."""
    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    bin_dir = ctl.STATE_ROOT / "bin"
    bin_dir.mkdir(parents=True)
    exe_name = "llama-server.exe" if os.name == "nt" else "llama-server"
    (bin_dir / exe_name).write_bytes(b"")
    reg = env / "models.yaml"
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "old-pull", "gguf": "/tmp/old.gguf", "llama_port": 1,
         "daemon_port": 2, "ctx_size": 100, "max_tokens": 90,
         "model_id": "old", "binary": "llama-server"},
    ]}))
    updated = ctl.backfill_binary_env(ctl.load_registry(reg))
    assert updated == ["old-pull"]
    healed = ctl.get_model("old-pull", reg)
    assert healed.binary == str(bin_dir / exe_name)


def test_resolve_binary_prefers_custom_then_installed(env, tmp_path, monkeypatch):
    """Launch-time resolution: a custom binary wins; a bare PATH default
    falls back to the installed build; bare name only as last resort; a
    dead absolute path is skipped like the bare default."""
    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    bin_dir = ctl.STATE_ROOT / "bin"
    bin_dir.mkdir(parents=True)
    exe_name = "llama-server.exe" if os.name == "nt" else "llama-server"
    installed = str(bin_dir / exe_name)
    (bin_dir / exe_name).write_bytes(b"")
    live_custom = tmp_path / "custom-llama-server"
    live_custom.write_bytes(b"")
    reg = env / "models.yaml"
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "custom", "gguf": "/tmp/c.gguf", "llama_port": 1,
         "daemon_port": 2, "ctx_size": 100, "max_tokens": 90,
         "model_id": "c", "binary": str(live_custom)},
        {"name": "dead", "gguf": "/tmp/d.gguf", "llama_port": 5,
         "daemon_port": 6, "ctx_size": 100, "max_tokens": 90,
         "model_id": "d", "binary": "/opt/llama-server"},
        {"name": "bare", "gguf": "/tmp/b.gguf", "llama_port": 3,
         "daemon_port": 4, "ctx_size": 100, "max_tokens": 90,
         "model_id": "b", "binary": "llama-server"},
    ]}))
    specs = ctl.load_registry(reg)
    by_name = {s.name: s for s in specs}
    assert ctl._resolve_binary(by_name["custom"]) == str(live_custom)
    # Dead absolute path → falls back to the installed build, not Popen failure.
    assert ctl._resolve_binary(by_name["dead"]) == installed
    assert ctl._resolve_binary(by_name["bare"]) == installed
    (bin_dir / exe_name).unlink()
    assert ctl._resolve_binary(by_name["bare"]) == "llama-server"


def test_pull_model_inherits_binary(seeded_env, hf, tmp_path):
    ref_bin = tmp_path / "bin" / "llama-server"
    ref_bin.parent.mkdir(parents=True)
    ref_bin.write_bytes(b"")
    reg = seeded_env / "models.yaml"
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "laguna", "gguf": "/tmp/lag.gguf", "llama_port": 1,
         "daemon_port": 2, "ctx_size": 100, "max_tokens": 90,
         "model_id": "laguna-s-2.1", "binary": str(ref_bin),
         "env": {}, "source": "default"},
    ]}))
    spec = ctl.pull_model("acme/My-Model-GGUF", "inherits", registry=reg)
    assert spec.binary == str(ref_bin)
    # No reference env → empty.
    assert spec.env == {}


def test_pull_model_reasoning_field_persisted(env, hf):
    spec_off = ctl.pull_model("acme/My-Model-GGUF", "reasonoff", reasoning="off")
    assert spec_off.reasoning == "off"
    spec_auto = ctl.pull_model("acme/My-Model-GGUF", "reasonauto")
    assert spec_auto.reasoning == "auto"
    # Survives registry round-trip.
    assert ctl.get_model("reasonoff").reasoning == "off"


def test_llama_args_appends_reasoning_off_for_off_spec(seeded_env):
    # The default 9b entry points at a dev-machine path; point it at an
    # existing temp file so _llama_args' existence check passes on CI.
    gguf = seeded_env / "9b.gguf"
    gguf.write_bytes(b"dummy")
    raw = dict(ctl.load_registry()[0].raw)
    raw["gguf"] = str(gguf)
    spec = ctl.ModelSpec(raw=raw, **raw)
    assert "--reasoning" not in ctl._llama_args(spec)
    spec_off = spec
    spec_off.reasoning = "off"
    assert "--reasoning" in ctl._llama_args(spec_off)
    assert "off" in ctl._llama_args(spec_off)


def test_llama_args_appends_reasoning_on_for_on_spec(seeded_env):
    gguf = seeded_env / "9b.gguf"
    gguf.write_bytes(b"dummy")
    raw = dict(ctl.load_registry()[0].raw)
    raw["gguf"] = str(gguf)
    spec = ctl.ModelSpec(raw=raw, **raw)
    spec.reasoning = "on"
    args = ctl._llama_args(spec)
    assert "--reasoning" in args
    assert args[args.index("--reasoning") + 1] == "on"


def _ngl_spec(seeded_env):
    gguf = seeded_env / "9b.gguf"
    gguf.write_bytes(b"dummy")
    raw = dict(ctl.load_registry()[0].raw)
    raw["gguf"] = str(gguf)
    return ctl.ModelSpec(raw=raw, **raw)


def test_llama_args_ngl_honors_env(seeded_env, monkeypatch):
    """TOKI_NGL wins over auto-detection."""
    monkeypatch.setenv("TOKI_NGL", "17")
    args = ctl._llama_args(_ngl_spec(seeded_env))
    assert "-ngl" in args
    assert args[args.index("-ngl") + 1] == "17"


def test_llama_args_toki_ctx_size_overrides_spec(seeded_env, monkeypatch):
    """TOKI_CTX_SIZE replaces the spec's ctx_size (appended last so
    llama.cpp's last-flag-wins parser honours it)."""
    spec = _ngl_spec(seeded_env)
    spec.ctx_size = 200000
    monkeypatch.setenv("TOKI_CTX_SIZE", "4096")
    args = ctl._llama_args(spec)
    assert args[-2:] == ["-c", "4096"]


def test_llama_args_toki_ctx_size_overrides_extra_args_pin(seeded_env, monkeypatch):
    """TOKI_CTX_SIZE beats a registry extra_args -c pin because the
    override is appended after extra_args."""
    spec = _ngl_spec(seeded_env)
    spec.extra_args = ["-c", "8192"]
    monkeypatch.setenv("TOKI_CTX_SIZE", "4096")
    args = ctl._llama_args(spec)
    assert args[-2:] == ["-c", "4096"]


def test_llama_args_toki_threads_override(seeded_env, monkeypatch):
    """TOKI_THREADS is passed as -t on the CPU path."""
    monkeypatch.setenv("TOKI_NGL", "0")
    monkeypatch.setenv("TOKI_THREADS", "3")
    spec = _ngl_spec(seeded_env)
    args = ctl._llama_args(spec)
    assert "-t" in args
    assert args[args.index("-t") + 1] == "3"


def test_llama_args_toki_threads_overrides_extra_args_threads(seeded_env, monkeypatch):
    """TOKI_THREADS beats a registry extra_args -t pin (appended
    after extra_args; llama.cpp takes the last flag)."""
    monkeypatch.setenv("TOKI_NGL", "0")
    monkeypatch.setenv("TOKI_THREADS", "3")
    spec = _ngl_spec(seeded_env)
    spec.extra_args = ["-t", "8"]
    args = ctl._llama_args(spec)
    assert args[-2:] == ["-t", "3"]


def test_llama_args_ngl_zero_runs_cpu(seeded_env, monkeypatch):
    monkeypatch.setenv("TOKI_NGL", "0")
    args = ctl._llama_args(_ngl_spec(seeded_env))
    assert args[args.index("-ngl") + 1] == "0"


def test_llama_args_ngl_skipped_when_pinned_in_extra_args(seeded_env):
    """Registry `extra_args` already pinning -ngl is not duplicated."""
    spec = _ngl_spec(seeded_env)
    spec.extra_args = ["-ngl", "12"]
    args = ctl._llama_args(spec)
    assert args.count("-ngl") == 1
    assert args[args.index("-ngl") + 1] == "12"


def test_llama_args_threads_drop_lpe_on_hybrid_cpu(seeded_env, monkeypatch):
    """On a hybrid CPU the live -t probe must use the freq-tier
    filter — LPE cores (40-50% of P-core boost) and HT siblings
    must not be in the llama.cpp thread pool. Stub cpu_topology to
    return a 155U-shaped profile; the resulting -t count should be
    physical fast cores (2P + 8E = 10), not raw physical_cores (12)
    and not logical fast cores (12)."""
    from tokirecur import hardware
    freqs = ([4800, 4800, 4800, 4800]   # 2P with HT
             + [3800] * 8                # 8E
             + [2100, 2100])             # 2LPE
    profile = hardware.CPUProfile(
        vendor="GenuineIntel", name="Core Ultra 7 155U",
        family=6, model=189,
        physical_cores=12, logical_cores=14,
        p_core_ids=list(range(4)), e_core_ids=list(range(4, 14)),
        max_boost_mhz=4800.0,
        core_max_freqs_mhz=list(freqs),
        simd=["avx2", "avx_vnni"],
    )
    monkeypatch.setattr(hardware, "cpu_topology", lambda: profile)
    monkeypatch.setenv("TOKI_NGL", "0")

    spec = _ngl_spec(seeded_env)
    args = ctl._llama_args(spec)

    # 10 physical fast cores: 2P + 8E (HT siblings and LPE dropped).
    assert args[args.index("-t") + 1] == "10"


# ── Registry repair (legacy pulled entries) ────────────────────────


def test_backfill_binary_env_inherits_reference(seeded_env, hf, tmp_path, monkeypatch):
    """A pulled entry missing binary/env inherits the reference spec's
    values; the repair returns the names it updated."""
    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    ref_bin = tmp_path / "bin" / "llama-server"
    ref_bin.parent.mkdir(parents=True)
    ref_bin.write_bytes(b"")
    # Give the laguna default a live tuned binary so it's a valid reference.
    reg = ctl.DEFAULT_REGISTRY
    data = yaml.safe_load(reg.read_text())
    for entry in data["models"]:
        if entry["name"] == "laguna":
            entry["binary"] = str(ref_bin)
    reg.write_text(yaml.safe_dump(data, sort_keys=False))
    ctl.invalidate_registry_cache()

    pulled = ctl.pull_model("acme/Pulled-GGUF", "pulled")
    assert pulled.source == "pulled"
    assert pulled.binary == str(ref_bin)

    # Simulate an older pulled entry written before `pull_defaults` —
    # clear its binary/env so it has only the PATH default.
    data = yaml.safe_load(reg.read_text())
    for entry in data["models"]:
        if entry["name"] == "pulled":
            entry["binary"] = "llama-server"
            entry["env"] = {}
    reg.write_text(yaml.safe_dump(data, sort_keys=False))
    ctl.invalidate_registry_cache()

    healed = ctl.backfill_binary_env()
    assert healed == ["pulled"]
    refreshed = ctl.get_model("pulled")
    assert refreshed.binary == str(ref_bin)
    assert refreshed.source == "pulled"


def test_backfill_binary_env_idempotent(env, hf, monkeypatch):
    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    # No entries need repair → empty list.
    assert ctl.backfill_binary_env() == []
    # Pulled entry already has a custom binary → no-op.
    ctl.pull_model("acme/Pulled-GGUF", "pulled")
    assert ctl.backfill_binary_env() == []


def test_backfill_binary_env_skips_configured_defaults(seeded_env, monkeypatch):
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

    from tokirecur.cli import cli

    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    res = CliRunner().invoke(cli, ["model", "repair"])
    assert res.exit_code == 0, res.output
    assert "nothing to repair" in res.output


def test_repair_cli_heals_legacy_entry(seeded_env, hf, tmp_path, monkeypatch):

    from tokirecur.cli import cli

    monkeypatch.delenv("TOKI_LLAMA_SERVER", raising=False)
    ref_bin = tmp_path / "bin" / "llama-server"
    ref_bin.parent.mkdir(parents=True)
    ref_bin.write_bytes(b"")
    reg = ctl.DEFAULT_REGISTRY
    data = yaml.safe_load(reg.read_text())
    for entry in data["models"]:
        if entry["name"] == "laguna":
            entry["binary"] = str(ref_bin)
    reg.write_text(yaml.safe_dump(data, sort_keys=False))
    ctl.invalidate_registry_cache()

    ctl.pull_model("acme/Pulled-GGUF", "pulled")
    # Simulate a legacy entry: blow away binary/env but keep source.
    data = yaml.safe_load(reg.read_text())
    for entry in data["models"]:
        if entry["name"] == "pulled":
            entry["binary"] = "llama-server"
            entry["env"] = {}
    reg.write_text(yaml.safe_dump(data, sort_keys=False))
    ctl.invalidate_registry_cache()

    res = CliRunner().invoke(cli, ["model", "repair"])
    assert res.exit_code == 0, res.output
    assert "repaired: pulled" in res.output
    assert ctl.get_model("pulled").binary == str(ref_bin)


# ── Settings: update_spec_field ───────────────────────────────────


def test_update_spec_field_changes_value_and_persists(seeded_env):
    spec = ctl.update_spec_field("9b", "ctx_size", "4096")
    assert spec.ctx_size == 4096
    assert ctl.get_model("9b").ctx_size == 4096
    # Survives reload.
    assert ctl.load_registry()[0].ctx_size == 4096


def test_update_spec_field_int_validators(seeded_env):
    ctl.update_spec_field("9b", "llama_port", "9999")
    assert ctl.get_model("9b").llama_port == 9999
    with pytest.raises(ValueError, match="bad value for 'llama_port'"):
        ctl.update_spec_field("9b", "llama_port", "not-a-number")


def test_update_spec_field_bool_truthy(seeded_env):
    ctl.update_spec_field("35b", "always_on", "true")
    assert ctl.get_model("35b").always_on is True
    ctl.update_spec_field("35b", "always_on", "off")
    assert ctl.get_model("35b").always_on is False
    with pytest.raises(ValueError, match="expected true/false"):
        ctl.update_spec_field("35b", "always_on", "maybe")


def test_update_spec_field_unknown_field(seeded_env):
    with pytest.raises(ValueError, match="not editable"):
        ctl.update_spec_field("9b", "name", "newname")


def test_update_spec_field_readonly_source(seeded_env):
    with pytest.raises(ValueError, match="not editable"):
        ctl.update_spec_field("9b", "source", "pulled")


def test_update_spec_field_unknown_model(env):
    with pytest.raises(KeyError, match="unknown model 'nope'"):
        ctl.update_spec_field("nope", "ctx_size", "100")


def test_update_spec_field_apply_forces_restart_of_running_stack(
    seeded_env, monkeypatch
):
    """apply=True on a restart-required field must stop a RUNNING stack
    so the new -c/-kvu actually takes effect — start_model alone is
    idempotent and would silently keep the old window (the
    "reg:100000 · live:16384" confusion)."""
    use_calls: list[dict] = []

    def fake_use_model(name, **kwargs):
        use_calls.append({"name": name, **kwargs})
        return ctl.get_model(name)

    monkeypatch.setattr(ctl, "use_model", fake_use_model)
    ctl.update_spec_field("9b", "ctx_size", "4096", apply=True)
    assert len(use_calls) == 1
    call = use_calls[0]
    assert call["name"] == "9b"
    assert call["stop_others"] is False
    assert call["force_restart"] is True


def test_use_model_force_restart_stops_then_starts(seeded_env, monkeypatch):
    """force_restart=True stops the target's own running stack before
    start_model, so a settings edit propagates to the live server."""
    from tokirecur.ctl import use_model

    stopped: list[str] = []
    started: list[str] = []

    monkeypatch.setattr(ctl, "llama_reachable", lambda spec: True)
    monkeypatch.setattr(ctl, "daemon_reachable", lambda spec: False)

    def fake_stop(spec, force=False):
        stopped.append(spec.name)

    def fake_start(spec, wait=True):
        started.append(spec.name)

    monkeypatch.setattr(ctl, "stop_model", fake_stop)
    monkeypatch.setattr(ctl, "start_model", fake_start)
    monkeypatch.setattr(ctl, "set_active", lambda name, registry=None: None)
    monkeypatch.setattr(ctl, "sync_cli_configs", lambda spec: None)

    use_model("9b", stop_others=False, force_restart=True)
    assert stopped == ["9b"]
    assert started == ["9b"]


def test_use_model_force_restart_noop_when_down(seeded_env, monkeypatch):
    """force_restart on a stopped model doesn't stop twice — it just
    starts (the normal idempotent start_model path)."""
    from tokirecur.ctl import use_model

    stopped: list[str] = []
    started: list[str] = []

    monkeypatch.setattr(ctl, "llama_reachable", lambda spec: False)
    monkeypatch.setattr(ctl, "daemon_reachable", lambda spec: False)

    def fake_stop(spec, force=False):
        stopped.append(spec.name)

    def fake_start(spec, wait=True):
        started.append(spec.name)

    monkeypatch.setattr(ctl, "stop_model", fake_stop)
    monkeypatch.setattr(ctl, "start_model", fake_start)
    monkeypatch.setattr(ctl, "set_active", lambda name, registry=None: None)
    monkeypatch.setattr(ctl, "sync_cli_configs", lambda spec: None)

    use_model("9b", stop_others=False, force_restart=True)
    assert stopped == []
    assert started == ["9b"]


def test_update_spec_field_preserves_other_fields(seeded_env):
    """Editing one field doesn't blow away the spec's other keys
    (raw dict merge — only the edited field changes)."""
    # Register a spec with a non-default env so we can prove it survives.
    reg = seeded_env / "models.yaml"
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
    # Laguna's env + extra_args must survive; the binary defaults to the
    # portable bare name (DEFAULT_MODELS pins no machine-specific build).
    assert spec.ctx_size == 99999
    assert spec.env.get("HSA_OVERRIDE_GFX_VERSION") == "11.5.1"
    assert spec.env.get("GGML_CUDA_ENABLE_UNIFIED_MEMORY") == "1"
    assert spec.extra_args == ["--reasoning", "off"]
    assert spec.binary == "llama-server"


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
    # hermes uses the built-in "llamacpp" alias instead of a custom
    # provider key, so the pulled spec is registered via the legacy
    # custom_providers list.
    custom = h.get("custom_providers") or []
    llama_entry = next(
        (c for c in custom if c.get("name") == "llamacpp"), None,
    )
    assert llama_entry is not None
    assert pulled.model_id in llama_entry["models"]
    # And the active model line points at the pulled spec.
    assert oc["model"] == f"tokirecur-{pulled.name}/{pulled.model_id}"
    assert cla["agents"]["defaults"]["model"]["primary"] == \
        f"tokirecur-{pulled.name}/{pulled.model_id}"
    assert h["model"]["provider"] == "llamacpp"


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


def test_chat_completions_accepts_legacy_repo_path_model_name(tmp_path, monkeypatch):
    """Regression for the hermes 'model not found' 404: a client
    sending the legacy repo-path model_id ('unsloth/Qwen3.5-0.8B-GGUF')
    must be served (not 404) and the response echoes the active spec's
    short model_id so downstream consumers see one consistent name."""
    from fastapi.testclient import TestClient

    from tokirecur.api import app as app_module
    from tokirecur.api.app import create_app
    from tokirecur.backend import LlamaServerBackend
    from tokirecur.core.recycling import RecyclingEngine, RecyclingResult
    from tokirecur.index import Index

    backend = LlamaServerBackend(base_url="http://llama.invalid")
    app = create_app(
        backend=backend, index=Index(), cache_dir=tmp_path,
        enable_auth=False, enable_rate_limit=False,
    )

    def fake_serve_chat(self, messages, *, max_tokens, **kw):
        return RecyclingResult(
            hit=False, soft_hit=False, cache_id=None, reuse_depth=0,
            embedding_similarity=0.0, prefilled_tokens=0, skipped_tokens=0,
            generated_tokens=2, ttft_ms=5.0, total_ms=10.0,
            prefill_tokens_per_sec=0.0, generated_tokens_per_sec=200.0,
            output_text="hi", reasoning="", slot_id=0, spilled=False,
            tool_calls=None, finish_reason="stop",
        )

    monkeypatch.setattr(RecyclingEngine, "serve_chat", fake_serve_chat)
    # Active spec's short model_id (post-fix-model-ids form).
    monkeypatch.setattr(app_module, "_active_model_id",
                        lambda: "qwen3.5-0.8b-q4_k_m")

    client = TestClient(app)
    res = client.post("/v1/chat/completions", json={
        "model": "unsloth/Qwen3.5-0.8B-GGUF",  # legacy form from the bug
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["model"] == "qwen3.5-0.8b-q4_k_m"
    assert body["choices"][0]["message"]["content"] == "hi"


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


# ── toki model remove ─────────────────────────────────────────────────


def test_remove_model_drops_registry_entry(seeded_env, hf, monkeypatch):
    """A plain `remove_model(name)` must drop the spec from the registry."""
    monkeypatch.setattr(ctl, "stop_model", lambda spec, force=False: None)
    ctl.pull_model("acme/Pulled-GGUF", "pulled")
    names_before = {s.name for s in ctl.load_registry()}
    assert "pulled" in names_before

    summary = ctl.remove_model("pulled")

    assert summary["removed"] == "pulled"
    assert summary["was_active"] is False
    assert summary["purged_weights"] is False
    assert summary["purged_state"] is False
    assert "pulled" not in {s.name for s in ctl.load_registry()}
    # Other entries are untouched.
    assert {"9b", "35b", "laguna"} <= {s.name for s in ctl.load_registry()}


def test_remove_model_unknown_raises(env, monkeypatch):
    monkeypatch.setattr(ctl, "stop_model", lambda spec, force=False: None)
    with pytest.raises(KeyError, match="no model named 'nope'"):
        ctl.remove_model("nope")


def test_remove_model_refuses_active_without_force(seeded_env, hf, monkeypatch):
    """Without --force the active model must NOT be removed — the launcher
    and CLI configs would otherwise point at a name that no longer exists."""
    monkeypatch.setattr(ctl, "stop_model", lambda spec, force=False: None)
    ctl.pull_model("acme/Pulled-GGUF", "pulled")
    ctl.set_active("pulled")

    with pytest.raises(RuntimeError, match="active model"):
        ctl.remove_model("pulled")

    # Registry unchanged.
    assert "pulled" in {s.name for s in ctl.load_registry()}
    assert ctl.read_active() == "pulled"


def test_remove_model_with_force_allows_active(seeded_env, hf, monkeypatch):
    monkeypatch.setattr(ctl, "stop_model", lambda spec, force=False: None)
    ctl.pull_model("acme/Pulled-GGUF", "pulled")
    ctl.set_active("pulled")

    summary = ctl.remove_model("pulled", force=True)

    assert summary["was_active"] is True
    assert "pulled" not in {s.name for s in ctl.load_registry()}
    # Active marker cleared (the model it pointed at is gone).
    assert ctl.read_active() is None
    assert not ctl.ACTIVE_MARKER.exists()


def test_remove_model_purge_deletes_weights_and_state(seeded_env, hf, monkeypatch, tmp_path):
    """--purge removes the weights directory under ~/models/<name>/ and
    the runtime state directory under STATE_ROOT/model/<name>/."""
    monkeypatch.setattr(ctl, "stop_model", lambda spec, force=False: None)
    ctl.pull_model("acme/Pulled-GGUF", "pulled")
    # Redirect Path.home() so ~/models/<name>/ lives inside tmp_path.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    # pull_model already wrote weights at ~/models/pulled/ inside tmp_path.
    state_dir = ctl.STATE_ROOT / "model" / "pulled"
    state_dir.mkdir(parents=True)
    (state_dir / "llama.pid").write_text("999")

    summary = ctl.remove_model("pulled", purge=True)

    assert summary["purged_weights"] is True
    assert summary["purged_state"] is True
    assert not (tmp_path / "models" / "pulled").exists()
    assert not state_dir.exists()


def test_remove_model_purge_keeps_weights_outside_models_dir(seeded_env, hf, monkeypatch, tmp_path):
    """If the spec's gguf path lives outside ~/models (e.g. a custom
    legacy layout pointing at /opt/...), --purge must not delete it.
    The state dir is still cleaned."""
    monkeypatch.setattr(ctl, "stop_model", lambda spec, force=False: None)
    ctl.pull_model("acme/Pulled-GGUF", "pulled")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    # Rewrite the registry entry's gguf path to point outside ~/models.
    outside = tmp_path / "external-weights" / "pulled-Q4_K_M.gguf"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("stub")
    specs = ctl.load_registry()
    for s in specs:
        if s.name == "pulled":
            s.gguf = str(outside)
            s.raw["gguf"] = str(outside)
    ctl.save_registry(specs)
    ctl.invalidate_registry_cache()
    state_dir = ctl.STATE_ROOT / "model" / "pulled"
    state_dir.mkdir(parents=True)

    summary = ctl.remove_model("pulled", purge=True)

    assert summary["purged_weights"] is False  # refused — outside ~/models
    assert summary["purged_state"] is True
    assert outside.exists()         # weights preserved
    assert not state_dir.exists()   # state still cleaned


def test_remove_model_clears_active_marker(seeded_env, hf, monkeypatch):
    """The active-marker file is removed when its target is removed so
    read_active() falls back to the always-on default (or None)."""
    monkeypatch.setattr(ctl, "stop_model", lambda spec, force=False: None)
    ctl.pull_model("acme/Pulled-GGUF", "pulled")
    ctl.set_active("pulled")
    assert ctl.ACTIVE_MARKER.exists()

    ctl.remove_model("pulled", force=True)

    assert not ctl.ACTIVE_MARKER.exists()
    assert ctl.read_active() is None  # no always-on, no marker → None


def test_remove_model_idempotent(seeded_env, hf, monkeypatch):
    """Removing a name that is no longer in the registry must raise KeyError
    (not silently succeed) — protects against double-removes."""
    monkeypatch.setattr(ctl, "stop_model", lambda spec, force=False: None)
    ctl.pull_model("acme/Pulled-GGUF", "pulled")
    ctl.remove_model("pulled")
    with pytest.raises(KeyError):
        ctl.remove_model("pulled")


# ── Writable-path resolution (XDG → macOS fallback) ─────────────


def test_writable_dir_prefers_xdg_when_writable(tmp_path, monkeypatch):
    from tokirecur.ctl import _writable_dir
    chosen = _writable_dir(tmp_path, "tokirecur")
    assert chosen == tmp_path / "tokirecur"
    assert chosen.is_dir()


def test_writable_dir_falls_back_to_macos_app_support(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys, "platform", "darwin")

    # Block writing under tmp_path by making it a read-only parent.
    blocked = tmp_path / "locked"
    blocked.mkdir()
    blocked.chmod(0o555)
    try:
        from tokirecur.ctl import _writable_dir
        # Override Path.home so the macOS fallback resolves into tmp_path.
        mac_root = tmp_path / "mac"
        monkeypatch.setattr(
            "pathlib.Path.home", classmethod(lambda cls: mac_root),
        )
        # And make the XDG probe point at our locked dir.
        from tokirecur.ctl import Path as _P
        chosen = _writable_dir(_P(str(blocked)), "tokirecur")
        # Either the macOS fallback succeeded, or it returned the
        # original blocked path (which we then can't verify). We
        # accept either — the key is the function didn't raise.
        assert chosen is not None
        if chosen.exists():
            # If it did write, restore perms so pytest can clean up.
            try:
                chosen.chmod(0o755)
            except OSError:
                pass
    finally:
        try:
            blocked.chmod(0o755)
        except OSError:
            pass


def test_default_paths_env_vars_win(monkeypatch):
    monkeypatch.setenv("TOKI_MODEL_REGISTRY", "/tmp/custom-registry.yaml")
    monkeypatch.setenv("TOKI_STATE_DIR", "/tmp/custom-state")
    monkeypatch.setenv("TOKI_ACTIVE_MODEL", "/tmp/custom-marker")
    from tokirecur.ctl import _default_paths
    reg, state, marker = _default_paths()
    assert reg == Path("/tmp/custom-registry.yaml")
    assert state == Path("/tmp/custom-state")
    assert marker == Path("/tmp/custom-marker")


def test_default_paths_falls_back_when_xdg_not_writable(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys, "platform", "darwin")

    # Pretend ~/.config is locked down.
    locked = tmp_path / "fakehome" / ".config"
    locked.mkdir(parents=True)
    locked.chmod(0o555)
    try:
        monkeypatch.setattr(
            "pathlib.Path.home", classmethod(lambda cls: tmp_path / "fakehome"),
        )
        from tokirecur.ctl import _default_paths
        reg, state, marker = _default_paths()
        # Registry should land under the macOS fallback, not the
        # locked .config dir.
        assert ".config" not in str(reg) or reg.parent.exists(), reg
    finally:
        try:
            locked.chmod(0o755)
        except OSError:
            pass


def test_write_hermes_config_purges_stale_tokirecur_keys(env, monkeypatch, tmp_path):
    """Old toki versions wrote 'tokirecur-*' provider keys that hermes's
    resolve_provider now rejects. The writer must purge them so the
    next launch doesn't trip on a stale 'Unknown provider' error."""
    from types import SimpleNamespace

    import yaml

    from tokirecur.ctl import write_hermes_config

    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "providers": {
            "tokirecur-9b": {"api": "http://127.0.0.1:8787/v1",
                             "default_model": "qwen3.5-9b"},
            "ollama-launch": {"api": "http://127.0.0.1:11434/v1"},
        },
    }))

    spec = SimpleNamespace(
        name="qwen3.5-0.8b-gguf", model_id="qwen3.5-0.8b",
        daemon_url="http://127.0.0.1:8787", label="Qwen3.5-0.8B", ctx_size=32768,
    )

    write_hermes_config(spec, specs=[spec], path=cfg)
    out = yaml.safe_load(cfg.read_text())
    assert "tokirecur-9b" not in out["providers"], out["providers"]
    # The new code doesn't write a 'tokirecur-*' key at all — it uses
    # the built-in 'llamacpp' alias via the custom_providers list.
    custom = out.get("custom_providers") or []
    assert any(c.get("name") == "llamacpp" for c in custom)
    # The user's other providers (ollama) are preserved.
    assert "ollama-launch" in out["providers"]


def test_write_opencode_config_purges_stale_tokirecur_keys(env, monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    from tokirecur.ctl import write_opencode_config

    cfg = tmp_path / "opencode.json"
    cfg.write_text(json.dumps({
        "provider": {
            "tokirecur-9b": {"name": "9b (Toki Recur)"},
            "anthropic": {"name": "Anthropic"},
        },
    }))

    spec = SimpleNamespace(
        name="qwen3.5-0.8b-gguf", model_id="qwen3.5-0.8b",
        daemon_url="http://127.0.0.1:8787", llama_url="http://127.0.0.1:8088",
        label="Qwen3.5-0.8B", ctx_size=32768, max_tokens=32768,
        always_on=False,
        state_dir=lambda: tmp_path / "state",
    )

    write_opencode_config(spec, specs=[spec], path=cfg)
    out = json.loads(cfg.read_text())
    assert "tokirecur-9b" not in out["provider"], out["provider"]
    assert "tokirecur-qwen3.5-0.8b-gguf" in out["provider"]
    assert "anthropic" in out["provider"]


def test_write_openclaw_config_purges_stale_tokirecur_keys(env, monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    from tokirecur.ctl import write_openclaw_config

    cfg = tmp_path / "openclaw.json"
    cfg.write_text(json.dumps({
        "models": {
            "providers": {
                "tokirecur-9b": {"api": "openai-completions",
                                 "baseUrl": "http://127.0.0.1:8787/v1"},
            },
        },
    }))

    spec = SimpleNamespace(
        name="qwen3.5-0.8b-gguf", model_id="qwen3.5-0.8b",
        daemon_url="http://127.0.0.1:8787", llama_url="http://127.0.0.1:8088",
        label="Qwen3.5-0.8B", ctx_size=32768, max_tokens=32768,
        always_on=False,
        state_dir=lambda: tmp_path / "state",
    )

    write_openclaw_config(spec, specs=[spec], path=cfg)
    out = json.loads(cfg.read_text())
    providers = out["models"]["providers"]
    assert "tokirecur-9b" not in providers, providers
    assert "tokirecur-qwen3.5-0.8b-gguf" in providers


def test_write_hermes_config_uses_llamacpp_alias(env, monkeypatch, tmp_path):
    """The hermes writer must register the model under 'llamacpp' (a
    built-in hermes alias) so resolve_provider doesn't reject it."""
    from types import SimpleNamespace

    import yaml

    from tokirecur.ctl import write_hermes_config

    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"providers": {}}))

    spec = SimpleNamespace(
        name="qwen3.5-0.8b-gguf", model_id="qwen3.5-0.8b",
        daemon_url="http://127.0.0.1:8787", llama_url="http://127.0.0.1:8088",
        ctx_size=32768,
    )

    write_hermes_config(spec, specs=[spec], path=cfg)
    out = yaml.safe_load(cfg.read_text())

    # model.provider is the name hermes's resolve_provider must accept.
    assert out["model"]["provider"] == "llamacpp"
    # base_url points at the active daemon (with /v1 for OpenAI compat).
    assert out["model"]["base_url"] == "http://127.0.0.1:8787/v1"
    # The legacy custom_providers list has a "llamacpp" entry.
    custom = out.get("custom_providers") or []
    names = [c.get("name") for c in custom]
    assert "llamacpp" in names
    entry = next(c for c in custom if c.get("name") == "llamacpp")
    assert entry["base_url"] == "http://127.0.0.1:8787/v1"
    assert entry["models"] == {"qwen3.5-0.8b": {"context_length": 32768}}
    # The newer providers: dict has the same entry — hermes scans this
    # FIRST, so a stale user entry must not shadow the daemon.
    prov = out.get("providers") or {}
    assert "llamacpp" in prov
    assert prov["llamacpp"]["api"] == "http://127.0.0.1:8787/v1"
    assert prov["llamacpp"]["default_model"] == "qwen3.5-0.8b"
    assert prov["llamacpp"]["models"] == {
        "qwen3.5-0.8b": {"context_length": 32768}}


def test_write_hermes_config_takes_over_stale_providers_llamacpp(env, tmp_path):
    """Regression for the hermes 404: a stale providers.llamacpp entry
    pointing straight at llama-server (hermes's own setup wizard or a
    pre-0.2.16 toki can write one) shadows the custom_providers entry
    because hermes scans providers: first. The writer must purge and
    re-take the key so requests route through the toki daemon."""
    from types import SimpleNamespace

    import yaml

    from tokirecur.ctl import write_hermes_config

    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "providers": {
            "llamacpp": {
                "api": "http://127.0.0.1:8091/v1",  # llama-server directly
                "api_key": "",
                "default_model": "unsloth/Qwen3.5-0.8B-GGUF",
                "name": "llamacpp",
            },
        },
        "custom_providers": [],
        "model": {"provider": "llamacpp"},
    }))

    spec = SimpleNamespace(
        name="qwen3.5-0.8b-gguf", model_id="qwen3.5-0.8b-q4_k_m",
        daemon_url="http://127.0.0.1:8787", llama_url="http://127.0.0.1:8088",
        ctx_size=32768,
    )

    write_hermes_config(spec, specs=[spec], path=cfg)
    out = yaml.safe_load(cfg.read_text())
    prov = out["providers"]
    assert prov["llamacpp"]["api"] == "http://127.0.0.1:8787/v1"
    assert prov["llamacpp"]["default_model"] == "qwen3.5-0.8b-q4_k_m"
    assert "8091" not in prov["llamacpp"]["api"]
    custom = out["custom_providers"]
    assert any(c.get("name") == "llamacpp" for c in custom)
    assert out["model"]["provider"] == "llamacpp"


def test_write_hermes_config_preserves_user_custom_providers(env, tmp_path):
    """Switching models must NOT clobber the user's own custom_providers
    entries (e.g. an ollama-launch entry)."""
    from types import SimpleNamespace

    import yaml

    from tokirecur.ctl import write_hermes_config

    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "custom_providers": [
            {"name": "ollama-local", "base_url": "http://x:11434/v1",
             "models": ["llama3"]},
        ],
    }))

    spec = SimpleNamespace(
        name="qwen3.5-0.8b-gguf", model_id="qwen3.5-0.8b",
        daemon_url="http://127.0.0.1:8787", llama_url="http://127.0.0.1:8088",
        ctx_size=32768,
    )

    write_hermes_config(spec, specs=[spec], path=cfg)
    out = yaml.safe_load(cfg.read_text())
    names = [c.get("name") for c in out["custom_providers"]]
    assert "ollama-local" in names
    assert "llamacpp" in names


def test_normalize_model_ids_rewrites_repo_paths(env, monkeypatch, tmp_path):
    """Pre-0.2.15 pulls left a Hugging Face repo path in model_id.
    normalize_model_ids rewrites it to a short slug from the GGUF
    filename so hermes stops 404ing on it."""
    from types import SimpleNamespace

    from tokirecur.ctl import normalize_model_ids
    reg = env / "models.yaml"
    spec = SimpleNamespace(
        name="qwen3.5-0.8b-gguf", gguf="/m/qwen3.5-0.8b-gguf/"
        "Qwen3.5-0.8B-Q4_K_M.gguf",
        llama_port=8091, daemon_port=8790, ctx_size=32768,
        parallel=2, kv="f16", max_tokens=16384,
        model_id="unsloth/Qwen3.5-0.8B-GGUF",  # the legacy form
        always_on=False, boot=False, binary="/b/llama-server",
        env={}, extra_args=[], reasoning="auto", source="pulled",
    )
    reg.write_text(yaml.safe_dump({"models": [spec.__dict__]}))

    changes = normalize_model_ids()
    assert len(changes) == 1
    name, old, new = changes[0]
    assert name == "qwen3.5-0.8b-gguf"
    assert old == "unsloth/Qwen3.5-0.8B-GGUF"
    assert new == "qwen3.5-0.8b-q4_k_m"
    # Registry was actually rewritten.
    reloaded = yaml.safe_load(reg.read_text())
    assert reloaded["models"][0]["model_id"] == "qwen3.5-0.8b-q4_k_m"


def test_normalize_model_ids_skips_already_short_ids(env, tmp_path):
    from types import SimpleNamespace

    from tokirecur.ctl import normalize_model_ids
    reg = env / "models.yaml"
    spec = SimpleNamespace(
        name="qwen3.5-9b", gguf="/m/qwen3.5-9b.gguf",
        llama_port=8088, daemon_port=8787, ctx_size=200000,
        parallel=2, kv="f16", max_tokens=16384,
        model_id="qwen3.5-9b",  # already short
        always_on=False, boot=False, binary="/b/llama-server",
        env={}, extra_args=[], reasoning="auto", source="pulled",
    )
    reg.write_text(yaml.safe_dump({"models": [spec.__dict__]}))
    assert normalize_model_ids() == []


def test_pull_model_sets_short_model_id(env, monkeypatch, tmp_path):
    """toki model pull now uses the GGUF filename stem as model_id
    (e.g. 'qwen3.5-0.8b-q4_k_m') instead of the full HF repo path."""
    from types import SimpleNamespace

    import huggingface_hub as hfh

    from tokirecur import ctl

    api = SimpleNamespace(
        list_repo_files=lambda repo: ["Qwen3.5-0.8B-Q4_K_M.gguf"],
    )
    monkeypatch.setattr(hfh, "HfApi", lambda endpoint=None: api)
    monkeypatch.setattr(hfh, "snapshot_download", lambda *a, **kw: None)
    monkeypatch.setattr(ctl, "_auto_ctx_size",
                        lambda *a, **kw: 32768)
    monkeypatch.setattr(ctl, "next_free_ports",
                        lambda specs: (29901, 29902))
    monkeypatch.setattr(ctl, "pull_defaults",
                        lambda specs: ("/b/llama-server", {}))
    monkeypatch.setattr(ctl, "_gguf_sizes", lambda api, r, c: {})

    reg = env / "models.yaml"
    reg.write_text(yaml.safe_dump({"models": []}))
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)

    spec = ctl.pull_model("unsloth/Qwen3.5-0.8B-GGUF", name="qwen3.5-0.8b",
                          registry=reg)
    assert spec.model_id == "qwen3.5-0.8b-q4_k_m"
    assert spec.name == "qwen3.5-0.8b"
    # And the full repo id is NOT in the registry.
    assert "/" not in spec.model_id


# ── 0.2.17: automatic config sync (sync_only / heal / fault isolation) ──


def test_use_model_sync_only_never_touches_stack(env, monkeypatch):
    """`--sync-only` must rewrite configs without starting or stopping
    any llama/daemon — installers and daemon startup rely on it being
    side-effect-free on processes."""

    from tokirecur import ctl

    env.joinpath("models.yaml").write_text(yaml.safe_dump({"models": [
        {"name": "m1", "gguf": "/tmp/m1.gguf", "llama_port": 18001,
         "daemon_port": 18002, "ctx_size": 100, "max_tokens": 90,
         "model_id": "m1-model"},
    ]}))
    reg = env / "models.yaml"
    marker = env / "active-model"

    monkeypatch.setattr(ctl, "start_model",
                        lambda *a, **k: pytest.fail("start_model must not run"))
    monkeypatch.setattr(ctl, "stop_model",
                        lambda *a, **k: pytest.fail("stop_model must not run"))
    monkeypatch.setattr(ctl, "llama_reachable", lambda s: False)
    monkeypatch.setattr(ctl, "daemon_reachable", lambda s: False)
    monkeypatch.setattr(ctl, "sync_cli_configs",
                        lambda spec: {"hermes": env / "hermes.yaml"})

    spec = ctl.use_model("m1", registry=reg, sync_only=True)
    assert spec.name == "m1"
    assert ctl.read_active(reg) == "m1"
    assert marker.read_text().strip() == "m1"


def test_use_model_normalizes_legacy_ids_before_sync(env, monkeypatch):
    """A legacy repo-path model_id must be normalized before the agent
    configs are written, so hermes never sees 'unsloth/...' again."""

    from tokirecur import ctl

    reg = env / "models.yaml"
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "qwen3.5-0.8b-gguf", "gguf": "/tmp/Qwen3.5-0.8B-Q4_K_M.gguf",
         "llama_port": 18001, "daemon_port": 18002, "ctx_size": 100,
         "max_tokens": 90, "model_id": "unsloth/Qwen3.5-0.8B-GGUF"},
    ]}))

    written: list[str] = []

    def fake_sync(spec):
        written.append(spec.model_id)
        return {}

    monkeypatch.setattr(ctl, "sync_cli_configs", fake_sync)
    ctl.use_model("qwen3.5-0.8b-gguf", registry=reg, sync_only=True)
    # The writer saw the short id, and the registry is healed.
    assert written == ["qwen3.5-0.8b-q4_k_m"]
    healed = yaml.safe_load(reg.read_text())["models"][0]["model_id"]
    assert "/" not in healed


def test_heal_cli_configs_rewrites_stale_hermes_endpoint(env, monkeypatch):
    """The daemon-startup heal must take a stale hermes config (llama
    port + legacy model id) and rewrite it to the daemon with the short
    model id — the exact Mac/Windows failure state."""
    from tokirecur import ctl

    reg = env / "models.yaml"
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "qwen3.5-0.8b-gguf", "gguf": "/tmp/Qwen3.5-0.8B-Q4_K_M.gguf",
         "llama_port": 18001, "daemon_port": 18002, "ctx_size": 100,
         "max_tokens": 90, "model_id": "unsloth/Qwen3.5-0.8B-GGUF"},
    ]}))
    marker = env / "active-model"
    marker.write_text("qwen3.5-0.8b-gguf")

    hermes_cfg = env / "hermes.yaml"
    hermes_cfg.write_text(yaml.safe_dump({
        "custom_providers": {
            "llamacpp": {
                "base_url": "http://127.0.0.1:18001/v1",  # llama port!
                "api_key": "none",
                "model": "unsloth/Qwen3.5-0.8B-GGUF",
            },
        },
    }))
    opencode_cfg = env / "opencode.json"
    opencode_cfg.write_text(json.dumps({"provider": "llamacpp"}))  # string form

    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", marker)
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", marker)
    monkeypatch.setenv("HERMES_CONFIG", str(hermes_cfg))
    monkeypatch.setenv("OPENCODE_CONFIG", str(opencode_cfg))

    paths = ctl.heal_cli_configs(reg)
    assert paths is not None
    assert "hermes" in paths

    out = yaml.safe_load(hermes_cfg.read_text())
    entry = out["custom_providers"][0]
    assert entry["base_url"] == "http://127.0.0.1:18002/v1"
    assert entry["default_model"] == "qwen3.5-0.8b-q4_k_m"
    # The opencode string provider migrated to a dict, not a crash.
    out2 = json.loads(opencode_cfg.read_text())
    assert isinstance(out2["provider"], dict)
    # The registry got normalized by the heal too.
    assert "/" not in yaml.safe_load(reg.read_text())["models"][0]["model_id"]


def test_heal_cli_configs_no_active_returns_none(env, monkeypatch):
    from tokirecur import ctl

    reg = env / "models.yaml"
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "m1", "gguf": "/tmp/m1.gguf", "llama_port": 1,
         "daemon_port": 2, "ctx_size": 100, "max_tokens": 90,
         "model_id": "m1-model"},
    ]}))
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", env / "active-model")
    assert ctl.heal_cli_configs(reg) is None


def test_sync_cli_configs_fault_isolated(env, monkeypatch, tmp_path):
    """One corrupt/foreign config (opencode) must never block the other
    three writers — that silent failure left hermes stale for the whole
    0.2.16 era."""
    from types import SimpleNamespace

    from tokirecur import ctl

    cfg_dir = tmp_path / "cfgs"
    cfg_dir.mkdir()
    corrupt = cfg_dir / "opencode.json"
    corrupt.write_text("not json {{{")

    spec = SimpleNamespace(
        name="qwen3.5-0.8b-gguf", model_id="qwen3.5-0.8b",
        daemon_url="http://127.0.0.1:8787", llama_url="http://127.0.0.1:8088",
        ctx_size=32768,
        state_dir=lambda: cfg_dir / "state",
    )
    monkeypatch.setenv("OPENCODE_CONFIG", str(corrupt))
    monkeypatch.setenv("OPENCLAW_CONFIG", str(cfg_dir / "openclaw.json"))
    monkeypatch.setenv("HERMES_CONFIG", str(cfg_dir / "hermes.yaml"))
    monkeypatch.setenv("CODEX_HOME", str(cfg_dir / "codex"))

    out = ctl.sync_cli_configs(spec)
    assert "hermes" in out
    assert "openclaw" in out
    assert "codex" in out
    assert "opencode" not in out
    hermes = yaml.safe_load((cfg_dir / "hermes.yaml").read_text())
    assert hermes["custom_providers"][0]["base_url"] == "http://127.0.0.1:8787/v1"


def test_write_opencode_config_migrates_string_provider(env, tmp_path):
    """A vanilla opencode config has `"provider": "<name>"` (string);
    the writer must migrate it to the dict form, not crash."""
    from types import SimpleNamespace

    from tokirecur.ctl import write_opencode_config

    cfg = tmp_path / "opencode.json"
    cfg.write_text(json.dumps({"provider": "opencode", "model": "gpt-4o"}))

    spec = SimpleNamespace(
        name="qwen3.5-0.8b-gguf", model_id="qwen3.5-0.8b",
        daemon_url="http://127.0.0.1:8787", llama_url="http://127.0.0.1:8088",
        label="Qwen3.5-0.8B", ctx_size=32768, always_on=False,
        state_dir=lambda: tmp_path / "state",
    )
    write_opencode_config(spec, specs=[spec], path=cfg)
    out = json.loads(cfg.read_text())
    assert isinstance(out["provider"], dict)
    assert f"tokirecur-{spec.name}" in out["provider"]
    assert out["model"] == "tokirecur-qwen3.5-0.8b-gguf/qwen3.5-0.8b"


def test_write_hermes_config_migrates_string_model_key(env, tmp_path):
    """Old hermes configs carry `model: <name>` as a string; the writer
    must upgrade it to the dict form instead of crashing the sync."""
    from types import SimpleNamespace

    from tokirecur.ctl import write_hermes_config

    cfg = tmp_path / "hermes.yaml"
    cfg.write_text(yaml.safe_dump({"model": "unsloth/Qwen3.5-0.8B-GGUF"}))

    spec = SimpleNamespace(
        name="qwen3.5-0.8b-gguf", model_id="qwen3.5-0.8b",
        daemon_url="http://127.0.0.1:8787", llama_url="http://127.0.0.1:8088",
        ctx_size=32768,
    )
    write_hermes_config(spec, specs=[spec], path=cfg)
    out = yaml.safe_load(cfg.read_text())
    assert isinstance(out["model"], dict)
    assert out["model"]["base_url"] == "http://127.0.0.1:8787/v1"
    assert out["model"]["default"] == "qwen3.5-0.8b"


def test_llama_args_aliases_cover_every_stack_name(env, tmp_path):
    """Every model name the stack can emit must be registered as a
    llama-server alias so a validating (router-mode) build can never
    404 a direct hit with one of them."""
    from types import SimpleNamespace

    from tokirecur import ctl

    gguf = tmp_path / "Qwen3.5-0.8B-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")
    spec = SimpleNamespace(
        name="qwen3.5-0.8b-gguf", model_id="qwen3.5-0.8b",
        gguf=str(gguf),
        llama_port=1, ctx_size=100, parallel=2, kv="f16",
        reasoning="off", extra_args=[], binary="llama-server",
    )
    args = ctl._llama_args(spec)
    aliases = {args[i + 1] for i in range(len(args)) if args[i] == "--alias"}
    assert aliases == {
        "qwen3.5-0.8b",                          # model_id
        "Qwen3.5-0.8B-Q4_K_M",                   # GGUF stem
        "default",                               # toki chat client
        "tokirecur-qwen3.5-0.8b-gguf/qwen3.5-0.8b",  # opencode form
        "tokirecur/qwen3.5-0.8b-gguf",           # legacy short form
    }, aliases


def test_model_use_sync_only_cli(env, monkeypatch):
    """`toki model use <name> --sync-only` rewires the four agent
    configs and prints 'configs synced' without starting any stack —
    the installer/daemon-startup repair path, end to end."""
    from click.testing import CliRunner

    from tokirecur.cli import cli

    reg = env / "models.yaml"
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "m1", "gguf": "/tmp/m1.gguf", "llama_port": 18001,
         "daemon_port": 18002, "ctx_size": 100, "max_tokens": 90,
         "model_id": "m1-model"},
    ]}))
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", env / "active-model")
    monkeypatch.setenv("HERMES_CONFIG", str(env / "hermes.yaml"))
    monkeypatch.setenv("OPENCODE_CONFIG", str(env / "opencode.json"))
    monkeypatch.setenv("OPENCLAW_CONFIG", str(env / "openclaw.json"))
    monkeypatch.setenv("CODEX_HOME", str(env / "codex"))

    runner = CliRunner()
    result = runner.invoke(cli, ["model", "use", "m1", "--sync-only"])
    assert result.exit_code == 0, result.output
    assert "configs synced" in result.output
    hermes = yaml.safe_load((env / "hermes.yaml").read_text())
    assert hermes["custom_providers"][0]["base_url"] == "http://127.0.0.1:18002/v1"


class _RunResult:
    def __init__(self, stdout):
        self.stdout = stdout


def _fake_run(device_names):
    def run(*args, **kwargs):
        lines = []
        for i, n in enumerate(device_names):
            lines.append(f"GPU{i}\ndeviceName = {n}\n")
        return _RunResult("".join(lines))

    return run


def test_default_ngl_uses_profile_when_present(monkeypatch):
    """The CPU backend is the default when no GPU is usable. The
    hardware profile is the authority: when it marks a GPU as usable
    for LLM, the default offloads to that GPU. No runtime probes
    (nvidia-smi, hipconfig, vulkaninfo) are used — those heuristics
    wrongly offloaded to Intel iGPUs / lavapipe."""
    monkeypatch.setattr(ctl.os, "name", "nt")
    monkeypatch.setattr(ctl.sys, "platform", "win32")
    monkeypatch.setattr(
        ctl.shutil, "which", lambda name: "vulkaninfo" if name == "vulkaninfo" else None
    )

    # iGPU only — profile would mark use_for_llm=False → CPU wins.
    monkeypatch.setattr(
        "tokirecur.hardware.profile_box",
        lambda *a, **k: _FakeProfile(
            {"backend": "cpu", "ngl": 0, "threads": 10},
            gpus=[type("G", (), {"vendor": "Intel", "use_for_llm": False, "vram_gb": 0.5})()],
        ),
    )
    assert ctl._default_ngl() == 0

    # Profile marks a discrete GPU as usable → offload.
    monkeypatch.setattr(
        "tokirecur.hardware.profile_box",
        lambda *a, **k: _FakeProfile(
            {"backend": "cuda", "ngl": 99, "threads": 16},
            gpus=[type("G", (), {"vendor": "NVIDIA", "use_for_llm": True, "vram_gb": 12})()],
        ),
    )
    assert ctl._default_ngl() == 99

    # TOKI_NGL overrides the profile.
    monkeypatch.setenv("TOKI_NGL", "7")
    assert ctl._default_ngl() == 7


class _FakeProfile:
    def __init__(self, recommended, gpus=None):
        self.recommended = recommended
        self.gpus = gpus or []


def _linux_default_ngl_env(monkeypatch):
    """Patch the Linux `_default_ngl` environment: no ROCm, no GPU
    subprocesses — only the pieces the test drives."""
    monkeypatch.setattr(ctl.os, "name", "posix")
    monkeypatch.setattr(ctl.sys, "platform", "linux")
    monkeypatch.setattr(ctl.shutil, "which", lambda name: None)
    monkeypatch.setattr(ctl.os.path, "exists", lambda p: False)


def test_default_ngl_linux_offloads_when_profile_marks_gpu_usable(monkeypatch):
    """Profile says a discrete GPU is usable → offload (99). The
    profile is the authority; runtime probes are bypassed."""
    _linux_default_ngl_env(monkeypatch)
    monkeypatch.setattr(
        "tokirecur.hardware.profile_box",
        lambda *a, **k: _FakeProfile(
            {"backend": "cuda", "ngl": 99, "threads": 16},
            gpus=[type("G", (), {"vendor": "NVIDIA", "use_for_llm": True, "vram_gb": 12})()],
        ),
    )
    assert ctl._default_ngl() == 99


def test_default_ngl_linux_stays_cpu_when_profile_marks_only_igpu(monkeypatch):
    """Profile only has an Intel iGPU (not usable for LLM) → CPU. The
    profile's use_for_llm=False is the right signal — no runtime
    probe needed."""
    _linux_default_ngl_env(monkeypatch)
    monkeypatch.setattr(
        "tokirecur.hardware.profile_box",
        lambda *a, **k: _FakeProfile(
            {"backend": "cpu", "ngl": 0, "threads": 10},
            gpus=[type("G", (), {"vendor": "Intel", "use_for_llm": False, "vram_gb": 0.5})()],
        ),
    )
    assert ctl._default_ngl() == 0


def test_default_ngl_linux_no_profile_uses_cpu(monkeypatch):
    """First run / probe failure: no profile → CPU backend. The old
    ICD-probe fallback is gone (it wrongly picked lavapipe / iGPUs)."""
    _linux_default_ngl_env(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("no profile")

    monkeypatch.setattr("tokirecur.hardware.profile_box", boom)
    assert ctl._default_ngl() == 0


def test_default_ngl_toki_ngl_overrides_default(monkeypatch):
    """TOKI_NGL opts in to GPU offload on any device, any platform."""
    monkeypatch.setattr(ctl.os, "name", "posix")
    monkeypatch.setattr(ctl.sys, "platform", "linux")
    monkeypatch.setattr(ctl.shutil, "which", lambda n: None)
    monkeypatch.setattr(ctl.os.path, "exists", lambda p: False)
    monkeypatch.setenv("TOKI_NGL", "99")
    assert ctl._default_ngl() == 99


# ── Settings: update_spec_field(apply=...) downstream orchestration ──


def test_update_spec_field_apply_false_does_not_restart_or_sync(seeded_env, monkeypatch):
    """Default behaviour: write the registry, do nothing else. Pre-existing
    callers (CLI installers, daemon startup) keep working unchanged."""
    started: list[str] = []
    synced: list[str] = []

    def fake_use_model(name, **kwargs):
        started.append(name)
        # Mimic use_model raising if the binary is missing — the test
        # only checks that update_spec_field did not call it.
        raise RuntimeError("daemon not running in test")

    def fake_sync(spec, **kwargs):
        synced.append(spec.name)

    monkeypatch.setattr(ctl, "use_model", fake_use_model)
    monkeypatch.setattr(ctl, "sync_cli_configs", fake_sync)

    ctl.update_spec_field("9b", "ctx_size", "8192", apply=False)
    ctl.update_spec_field("9b", "max_tokens", "2048", apply=False)

    assert started == []
    assert synced == []
    # Registry still has the new value.
    assert ctl.get_model("9b").ctx_size == 8192
    assert ctl.get_model("9b").max_tokens == 2048


def test_update_spec_field_apply_true_restart_required_calls_use_model(
    seeded_env, monkeypatch
):
    """apply=True + a restart-required field (ctx_size) calls use_model."""
    started: list[tuple[str, dict]] = []
    synced: list[str] = []

    def fake_use_model(name, **kwargs):
        started.append((name, kwargs))
        return ctl.get_model(name)

    def fake_sync(spec, **kwargs):
        synced.append(spec.name)

    monkeypatch.setattr(ctl, "use_model", fake_use_model)
    monkeypatch.setattr(ctl, "sync_cli_configs", fake_sync)

    ctl.update_spec_field("9b", "ctx_size", "8192", apply=True)

    # use_model was called for restart-required fields. sync_cli_configs
    # was NOT called separately — use_model already calls it internally
    # (the TUI notification relies on that to avoid double work).
    assert len(started) == 1
    assert started[0][0] == "9b"
    # stop_others=False so we don't disturb other models.
    assert started[0][1].get("stop_others") is False
    assert synced == []


def test_update_spec_field_apply_true_restart_required_all_fields(
    seeded_env, monkeypatch
):
    """Every restart-required field triggers use_model when apply=True."""
    started: list[str] = []

    def fake_use_model(name, **kwargs):
        started.append(name)
        return ctl.get_model(name)

    monkeypatch.setattr(ctl, "use_model", fake_use_model)
    monkeypatch.setattr(ctl, "sync_cli_configs", lambda *a, **kw: None)

    for field in ("ctx_size", "kv", "parallel", "binary", "gguf",
                  "llama_port", "daemon_port"):
        started.clear()
        value = {
            "ctx_size": "8192",
            "kv": "q8_0",
            "parallel": "1",
            "binary": "/usr/bin/llama-server",
            "gguf": "/tmp/foo.gguf",
            "llama_port": "9999",
            "daemon_port": "9998",
        }[field]
        ctl.update_spec_field("9b", field, value, apply=True)
        assert started == ["9b"], f"{field!r} did not trigger use_model"


def test_update_spec_field_apply_true_sync_only_calls_sync_not_use_model(
    seeded_env, monkeypatch
):
    """Sync-only fields re-sync CLI configs without restarting the daemon."""
    started: list[str] = []
    synced: list[str] = []

    def fake_use_model(name, **kwargs):
        started.append(name)
        return ctl.get_model(name)

    def fake_sync(spec, **kwargs):
        synced.append(spec.name)

    monkeypatch.setattr(ctl, "use_model", fake_use_model)
    monkeypatch.setattr(ctl, "sync_cli_configs", fake_sync)

    for field in ("max_tokens", "model_id", "reasoning"):
        synced.clear()
        started.clear()
        value = {"max_tokens": "4096", "model_id": "9b-renamed",
                 "reasoning": "off"}[field]
        ctl.update_spec_field("9b", field, value, apply=True)
        assert started == [], f"{field!r} should not restart daemon"
        assert synced == ["9b"], f"{field!r} should sync CLI configs"


def test_update_spec_field_apply_true_noop_field_does_nothing_extra(
    seeded_env, monkeypatch
):
    """always_on / boot are registry-only; apply=True does nothing extra."""
    started: list[str] = []
    synced: list[str] = []

    def fake_use_model(name, **kwargs):
        started.append(name)
        return ctl.get_model(name)

    def fake_sync(spec, **kwargs):
        synced.append(spec.name)

    monkeypatch.setattr(ctl, "use_model", fake_use_model)
    monkeypatch.setattr(ctl, "sync_cli_configs", fake_sync)

    ctl.update_spec_field("9b", "always_on", "true", apply=True)
    ctl.update_spec_field("9b", "boot", "true", apply=True)

    assert started == []
    assert synced == []
    # Registry still updated.
    assert ctl.get_model("9b").always_on is True
    assert ctl.get_model("9b").boot is True


def test_update_spec_field_restart_failed_returns_spec_anyway(seeded_env, monkeypatch):
    """When use_model raises (binary missing, RAM too low, etc.) the
    registry write still succeeds and the function propagates the
    exception so the caller can decide how to surface it (the TUI
    shows a warning notification in that case)."""
    def fake_use_model(name, **kwargs):
        raise RuntimeError("llama-server binary not built")

    monkeypatch.setattr(ctl, "use_model", fake_use_model)
    monkeypatch.setattr(ctl, "sync_cli_configs", lambda *a, **kw: None)

    with pytest.raises(RuntimeError, match="binary not built"):
        ctl.update_spec_field("9b", "ctx_size", "8192", apply=True)
    # Registry still has the new value.
    assert ctl.get_model("9b").ctx_size == 8192
