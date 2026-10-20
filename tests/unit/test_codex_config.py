"""Tests for the OpenAI Codex CLI config writer.

Codex parity with opencode/mimo/openclaw/hermes: `toki model use` writes
~/.codex/config.toml (or $CODEX_HOME/config.toml) with a provider
entry per registry spec and the active model selected, and the launcher
menu offers codex like the other clients.
"""

from pathlib import Path

import pytest
import tomllib

from tokirecur import ctl
from tokirecur.selector import CLIENTS, launch_cmd


@pytest.fixture
def env(tmp_path, monkeypatch):
    reg = tmp_path / "models.yaml"
    state = tmp_path / "state"
    marker = tmp_path / "active-model"
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    monkeypatch.setattr(ctl, "STATE_ROOT", state)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", marker)
    monkeypatch.setenv("MIMOCODE_CONFIG", str(tmp_path / "mimocode.json"))
    monkeypatch.setenv("QWEN_CONFIG", str(tmp_path / "qwen-settings.json"))
    monkeypatch.setenv("FACTORY_CONFIG", str(tmp_path / "factory-settings.json"))
    monkeypatch.setenv("CLINE_TOKIRECUR_CONFIG", str(tmp_path / "cline-settings"))
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "dsh"))
    monkeypatch.setenv("GROK_HOME", str(tmp_path / "grok"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("KILO_CONFIG", str(tmp_path / "kilo.jsonc"))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi-agent"))
    monkeypatch.setenv("KIMI_CODE_HOME", str(tmp_path / "kimi"))
    monkeypatch.setenv("OMP_CONFIG_DIR", str(tmp_path / "omp-agent"))
    return tmp_path


def _spec(env, name="9b", model_id="qwen3.5-9b", daemon_port=8787):
    return ctl.ModelSpec(
        raw={
            "name": name, "label": name, "gguf": "/tmp/x.gguf",
            "llama_port": 8088, "daemon_port": daemon_port,
            "ctx_size": 200000, "parallel": 4, "kv": "q4_k_m",
            "max_tokens": 100000, "model_id": model_id,
            "source": "pulled",
        },
        name=name, label=name, gguf="/tmp/x.gguf",
        llama_port=8088, daemon_port=daemon_port,
        ctx_size=200000, parallel=4, kv="q4_k_m",
        max_tokens=100000, model_id=model_id, source="pulled",
    )


def _load(path: Path) -> dict:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def test_codex_config_writer(env, monkeypatch):
    monkeypatch.setenv("TOKI_AUTH_TOKEN", "test-tok-0123456789abcdef")
    spec = _spec(env)
    cfg_path = env / "codex" / "config.toml"
    out = ctl.write_codex_config(spec, [spec], path=cfg_path)
    assert out == cfg_path
    cfg = _load(cfg_path)
    assert cfg["model_provider"] == "tokirecur-9b"
    assert cfg["model"] == "qwen3.5-9b"
    prov = cfg["model_providers"]["tokirecur-9b"]
    assert prov["base_url"] == "http://127.0.0.1:8787/v1"
    assert prov["wire_api"] == "responses"
    assert prov["env_key"] == "TOKI_AUTH_TOKEN"
    assert "env" not in prov
    ctl.write_codex_config(spec, [spec], path=cfg_path)
    assert Path(str(cfg_path) + ".bak-model").exists()
    # Anti-loop guardrail: codex has NO system_prompt config key (v1's
    # top-level scalar was silently ignored). The rules land in the
    # global instructions file ~/.codex/AGENTS.md, which codex reads as
    # user context instructions.
    assert "system_prompt" not in _load(cfg_path)
    agents_md = cfg_path.parent / "AGENTS.md"
    assert agents_md.exists()
    text = agents_md.read_text()
    assert ctl.ANTI_LOOP_SYSTEM_PROMPT in text
    assert ctl._ANTI_LOOP_MARKER in text
    # Idempotent: a second sync doesn't duplicate the block.
    ctl.write_codex_config(spec, [spec], path=cfg_path)
    assert agents_md.read_text().count(ctl.ANTI_LOOP_SYSTEM_PROMPT) == 1


def test_codex_config_preserves_user_agents_md(env):
    spec = _spec(env)
    cfg_path = env / "codex" / "config.toml"
    agents_md = cfg_path.parent / "AGENTS.md"
    agents_md.parent.mkdir(parents=True)
    agents_md.write_text("# My rules\n\nBe concise.\n")
    ctl.write_codex_config(spec, [spec], path=cfg_path)
    text = agents_md.read_text()
    # User content survives; the block is appended after it.
    assert text.startswith("# My rules\n\nBe concise.")
    assert ctl.ANTI_LOOP_SYSTEM_PROMPT in text


def test_codex_config_upgrades_v1_block(env):
    spec = _spec(env)
    cfg_path = env / "codex" / "config.toml"
    agents_md = cfg_path.parent / "AGENTS.md"
    agents_md.parent.mkdir(parents=True)
    v1 = (
        "# My rules\n\n"
        "<!-- toki-recur: anti-loop rules -->\n\n"
        "You are an autonomous agent completing a task. Old text.\n"
    )
    agents_md.write_text(v1)
    ctl.write_codex_config(spec, [spec], path=cfg_path)
    text = agents_md.read_text()
    # The stale v1 block is replaced by the current versioned one.
    assert "Old text." not in text
    assert text.count(ctl.ANTI_LOOP_SYSTEM_PROMPT) == 1
    assert text.startswith("# My rules")
    # The dead v1 system_prompt scalar is dropped from the TOML.
    assert "system_prompt" not in _load(cfg_path)


def test_codex_config_keeps_user_config_scalar(env):
    spec = _spec(env)
    cfg_path = env / "codex" / "config.toml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text('system_prompt = "My custom prompt"\n')
    ctl.write_codex_config(spec, [spec], path=cfg_path)
    cfg = _load(cfg_path)
    # A user's own (non-toki) scalar is preserved as-is.
    assert cfg["system_prompt"] == "My custom prompt"


def test_codex_config_writes_all_specs(env, monkeypatch):
    monkeypatch.setattr(ctl, "daemon_reachable", lambda spec: True)
    spec = _spec(env)
    pulled = _spec(env, name="pulled", model_id="pulled-model", daemon_port=8788)
    cfg_path = env / "codex" / "config.toml"
    ctl.write_codex_config(spec, [spec, pulled], path=cfg_path)
    cfg = _load(cfg_path)
    assert "tokirecur-pulled" in cfg["model_providers"]
    assert cfg["model_providers"]["tokirecur-pulled"]["base_url"].endswith(":8788/v1")


def test_codex_config_preserves_existing_keys(env):
    spec = _spec(env)
    cfg_path = env / "codex" / "config.toml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(
        "model = \"gpt-5\"\n"
        "sandbox_mode = \"workspace-write\"\n"
        "approval_policy = \"on-request\"\n"
        "\n"
        "[model_providers.myother]\n"
        "name = \"Other\"\n"
        "base_url = \"https://example.com/v1\"\n"
        "wire_api = \"chat\"\n"
    )
    ctl.write_codex_config(spec, [spec], path=cfg_path)
    cfg = _load(cfg_path)
    assert cfg["sandbox_mode"] == "workspace-write"
    assert cfg["approval_policy"] == "on-request"
    assert cfg["model_providers"]["myother"]["base_url"] == "https://example.com/v1"
    assert cfg["model_providers"]["tokirecur-9b"]["wire_api"] == "responses"
    assert cfg["model"] == "qwen3.5-9b"


def test_codex_config_honors_cod_home(env, monkeypatch):
    spec = _spec(env)
    home = env / "fake-codex-home"
    monkeypatch.setenv("CODEX_HOME", str(home))
    out = ctl.write_codex_config(spec, [spec])
    assert out == home / "config.toml"
    assert out.exists()


def test_codex_config_idempotent(env):
    spec = _spec(env)
    cfg_path = env / "codex" / "config.toml"
    ctl.write_codex_config(spec, [spec], path=cfg_path)
    first = cfg_path.read_bytes()
    ctl.write_codex_config(spec, [spec], path=cfg_path)
    assert cfg_path.read_bytes() == first


def test_sync_cli_configs_writes_codex(env, monkeypatch):
    spec = _spec(env)
    monkeypatch.setattr(ctl, "load_registry", lambda: [spec])
    monkeypatch.setenv("CODEX_HOME", str(env / "codex"))
    paths = ctl.sync_cli_configs(spec)
    assert "codex" in paths
    cfg = _load(paths["codex"])
    assert cfg["model_provider"] == "tokirecur-9b"


def _fake_spec():
    return ctl.ModelSpec(
        raw={}, name="9b", label="9b", gguf="/tmp/x.gguf",
        llama_port=8088, daemon_port=8787, ctx_size=200000,
        parallel=4, kv="q4_k_m", max_tokens=100000,
        model_id="qwen3.5-9b", source="pulled",
    )


def test_selector_lists_codex():
    assert "codex" in CLIENTS


def test_chat_codex_opens_interactive():
    plan = launch_cmd("chat", "codex", _fake_spec())
    assert plan["argv"] == ["codex"]
    assert not plan["detach"]
    assert plan["url"] is None


def test_tui_codex_opens_interactive():
    plan = launch_cmd("tui", "codex", _fake_spec())
    assert plan["argv"] == ["codex"]
    assert not plan["detach"]


def test_web_codex_app_server():
    plan = launch_cmd("web", "codex", _fake_spec())
    assert plan["argv"] == ["codex", "app-server"]
    assert plan["detach"]
    assert plan["url"] is None
