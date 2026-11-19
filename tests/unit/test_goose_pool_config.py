"""goose and pool: two harnesses that reach a provider their own way.

Neither is an opencode clone, so neither takes a provider block in a
JSON file. goose reads a provider from ``custom_providers/*.json`` and
picks it with two keys in ``config.yaml``; pool only speaks to an
OpenAI-compatible endpoint in standalone mode, and only the environment
selects that mode. Both have to end up on the local model, and both
have to leave the user's own settings alone.
"""

import json
from pathlib import Path

import pytest
import yaml

from tokirecur import ctl
from tokirecur.selector import launch_cmd


@pytest.fixture
def env(tmp_path, monkeypatch):
    """The registry, the state and the marker, all inside the test."""
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", tmp_path / "models.yaml")
    monkeypatch.setattr(ctl, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", tmp_path / "active-model")
    monkeypatch.setenv("TOKI_AUTH_TOKEN", "test-tok-0123456789abcdef")
    ctl.DEFAULT_REGISTRY.write_text(
        yaml.safe_dump({"models": ctl.DEFAULT_MODELS}, sort_keys=False)
    )
    return tmp_path


@pytest.fixture
def model(env, monkeypatch):
    monkeypatch.setattr(ctl, "daemon_reachable", lambda spec: True)
    return ctl.get_model("35b")


def test_the_goose_provider_file_points_at_the_local_model(env, model):
    """goose's provider file is the whole contract.

    The base_url is the FULL request path: goose appends nothing to it,
    so a URL that stops at ``/v1`` 404s.
    """
    cfg_dir = env / "goose"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(
        "GOOSE_TELEMETRY_ENABLED: false\n"
        "extensions:\n"
        "  developer:\n"
        "    enabled: true\n"
    )

    out = ctl.write_goose_config(model, path=cfg_dir / "config.yaml")
    # The provider file is the one that names the endpoint and the
    # models, so it is the path toki reports.
    assert out == cfg_dir / "custom_providers" / "tokirecur.json"

    provider = json.loads(
        (cfg_dir / "custom_providers" / "tokirecur.json").read_text()
    )
    assert provider["name"] == "tokirecur"
    assert provider["engine"] == "openai"
    assert provider["base_url"] == f"{model.daemon_url}/v1/chat/completions"
    # The key is read from the environment, never written to disk.
    assert provider["api_key_env"] == "TOKI_AUTH_TOKEN"
    assert "api_key" not in provider
    names = {m["name"] for m in provider["models"]}
    assert model.model_id in names
    assert all(m["context_limit"] > 0 for m in provider["models"])

    cfg = yaml.safe_load((cfg_dir / "config.yaml").read_text())
    assert cfg["GOOSE_PROVIDER"] == "tokirecur"
    assert cfg["GOOSE_MODEL"] == model.model_id
    # The user's own goose settings are still there.
    assert cfg["GOOSE_TELEMETRY_ENABLED"] is False
    assert cfg["extensions"]["developer"]["enabled"] is True


def test_the_goose_provider_file_is_owner_only(env, model):
    cfg_dir = env / "goose"
    cfg_dir.mkdir()
    ctl.write_goose_config(model, path=cfg_dir / "config.yaml")
    provider = cfg_dir / "custom_providers" / "tokirecur.json"
    assert provider.stat().st_mode & 0o077 == 0


def test_the_pool_settings_keep_the_users_own_things(env, model):
    """pool's settings file is the user's: their agent servers stay."""
    cfg_path = env / "poolside" / "settings.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(
        "agent_servers:\n"
        "    Poolside:\n"
        "        command: pool\n"
        "        args:\n"
        "            - acp\n"
    )

    out = ctl.write_pool_config(model, path=cfg_path)
    assert out == cfg_path
    cfg = yaml.safe_load(out.read_text())
    assert cfg["pool"]["api_url"] == f"{model.daemon_url}/v1"
    assert cfg["agent_servers"]["Poolside"]["command"] == "pool"


def test_pool_is_launched_in_standalone_mode(monkeypatch):
    """Standalone mode is the environment's job - no setting does it."""
    monkeypatch.setattr("tokirecur.selector.client_mode", lambda _: "managed")
    monkeypatch.setattr("tokirecur.selector._refresh_client_config",
                        lambda *a, **k: None)
    spec = type("S", (), {
        "name": "ornith",
        "model_id": "ornith-1.5-35b-q4_k_m",
        "llama_port": 8112,
        "daemon_port": 8811,
        "llama_url": "http://127.0.0.1:8112",
        "daemon_url": "http://127.0.0.1:8811",
        "ctx_size": 262144,
    })()
    plan = launch_cmd("chat", "pool", spec)
    assert plan["argv"] == ["pool"]
    assert plan["env"]["POOLSIDE_STANDALONE_BASE_URL"] == (
        "http://127.0.0.1:8811/v1"
    )
    assert plan["env"]["POOLSIDE_STANDALONE_MODEL"] == spec.model_id
    assert plan["env"]["POOLSIDE_API_KEY"]
    assert int(plan["env"]["POOLSIDE_STANDALONE_CONTEXT_LENGTH"]) > 0


def test_goose_is_launched_with_the_provider_named(monkeypatch):
    monkeypatch.setattr("tokirecur.selector.client_mode", lambda _: "managed")
    monkeypatch.setattr("tokirecur.selector._refresh_client_config",
                        lambda *a, **k: None)
    spec = type("S", (), {
        "name": "ornith",
        "model_id": "ornith-1.5-35b-q4_k_m",
        "llama_port": 8112,
        "daemon_port": 8811,
        "llama_url": "http://127.0.0.1:8112",
        "daemon_url": "http://127.0.0.1:8811",
        "ctx_size": 262144,
    })()
    plan = launch_cmd("chat", "goose", spec)
    assert plan["argv"] == ["goose"]
    assert plan["env"]["GOOSE_PROVIDER"] == "tokirecur"
    assert plan["env"]["GOOSE_MODEL"] == spec.model_id
    assert plan["env"]["TOKI_AUTH_TOKEN"]


def test_both_clients_are_synced_with_the_rest(env, model, monkeypatch):
    """A model switch rewrites their configs like every other client's."""
    monkeypatch.setattr(ctl, "daemon_reachable", lambda spec: True)
    monkeypatch.setenv("TOKI_GOOSE_CONFIG", str(env / "goose"))
    monkeypatch.setenv("TOKI_POOL_CONFIG", str(env / "poolside" / "settings.yaml"))
    written = ctl.sync_cli_configs(model)
    assert "goose" in written
    assert "pool" in written
    assert Path(written["goose"]).exists()
    assert Path(written["pool"]).exists()
