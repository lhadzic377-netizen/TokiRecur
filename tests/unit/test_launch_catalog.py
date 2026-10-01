import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tokirecur.launch_catalog import (
    LAUNCH_APPS,
    configured_provider_model,
    get_launch_app,
    unavailable_reason,
)
from tokirecur.selector import launch_cmd

# Meta Code, Cursor Agent and Devin were removed from the catalog: all three
# authenticate against their own account rather than a local stack, so none of
# them could ever use a model bound here.
REMOVED_ACCOUNT_CLIENTS = ("muse", "cursor-agent", "devin")
NEW_CLIENTS = (
    "qwen", "droid", "cline", "dsh", "grok", "kilo", "pi", "copilot",
    "kimi", "omp",
)


def _spec():
    return SimpleNamespace(
        name="ornith",
        model_id="ornith-1.5-35b-q4_k_m",
        llama_port=8112,
        daemon_port=8811,
        llama_url="http://127.0.0.1:8112",
        daemon_url="http://127.0.0.1:8811",
        ctx_size=262144,
    )


def test_catalog_contains_all_requested_clients():
    assert [app.key for app in LAUNCH_APPS] == [
        "toki", "ornithcode", "opencode", "hermes", "openclaw", "codex", "mimo",
        "qwen", "droid", "cline", "dsh", "grok", "kilo", "pi",
        "copilot", "claude", "kimi", "omp", "web",
    ]
    assert all(get_launch_app(key) is not None for key in NEW_CLIENTS)
    assert all(get_launch_app(key) is None for key in REMOVED_ACCOUNT_CLIENTS)
    with pytest.raises(ValueError):
        launch_cmd("chat", "muse", None)


def test_provider_protocols_keep_openai_and_anthropic_routes_separate():
    assert get_launch_app("claude").protocol == "anthropic"
    assert all(
        app.protocol == "openai"
        for app in LAUNCH_APPS
        if app.model_binding and app.client != "claude"
    )
    assert all(
        get_launch_app(key) is None
        for key in REMOVED_ACCOUNT_CLIENTS
    )
    assert not any(app.protocol == "account" for app in LAUNCH_APPS)


def test_claude_configured_anthropic_model_is_read_without_credentials(monkeypatch, tmp_path):
    settings = tmp_path / ".claude"
    settings.mkdir()
    (settings / "settings.json").write_text(
        '{"env": {"ANTHROPIC_MODEL": "MiniMax-M3", "ANTHROPIC_AUTH_TOKEN": "secret"}}'
    )
    for name in (
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("tokirecur.launch_catalog.Path.home", lambda: tmp_path)
    assert configured_provider_model("claude") == "MiniMax-M3"
    assert configured_provider_model("opencode") is None


def test_claude_route_uses_local_anthropic_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("TOKI_CLAUDE_SETTINGS", str(tmp_path / "claude.json"))
    monkeypatch.setenv("TOKI_AUTH_TOKEN", "test-claude-token")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "stale-user-token")
    plan = launch_cmd("chat", "claude", _spec())
    assert plan["argv"][:2] == ["claude", "--settings"]
    assert "test-claude-token" not in " ".join(plan["argv"])
    settings = json.loads((tmp_path / "claude.json").read_text())
    assert settings["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8811/anthropic"
    assert settings["env"]["ANTHROPIC_MODEL"] == "ornith-1.5-35b-q4_k_m"
    assert "OPENAI_BASE_URL" not in plan["env"]
    assert "ANTHROPIC_AUTH_TOKEN" not in plan["env"]
    assert plan["env"]["ANTHROPIC_API_KEY"] == "test-claude-token"


def test_claude_lean_flags_are_opt_in(monkeypatch, tmp_path):
    monkeypatch.setenv("TOKI_CLAUDE_SETTINGS", str(tmp_path / "claude.json"))
    monkeypatch.setenv("TOKI_AUTH_TOKEN", "test-claude-token")
    for name in ("TOKI_CLAUDE_BARE", "TOKI_CLAUDE_MCP", "TOKI_CLAUDE_EFFORT"):
        monkeypatch.delenv(name, raising=False)
    default_argv = launch_cmd("chat", "claude", _spec())["argv"]
    assert "--bare" not in default_argv
    assert "--strict-mcp-config" not in default_argv
    assert "--effort" not in default_argv

    monkeypatch.setenv("TOKI_CLAUDE_BARE", "1")
    monkeypatch.setenv("TOKI_CLAUDE_MCP", "none")
    monkeypatch.setenv("TOKI_CLAUDE_EFFORT", "low")
    lean = launch_cmd("chat", "claude", _spec())["argv"]
    assert "--bare" in lean
    assert "--strict-mcp-config" in lean
    assert lean[lean.index("--effort") + 1] == "low"
    assert lean[:2] == ["claude", "--settings"]


def test_dsh_availability_honors_dsh_home(monkeypatch, tmp_path):
    home = tmp_path / "dsh-home"
    (home / "profiles" / "tui").mkdir(parents=True)
    monkeypatch.setenv("DSH_HOME", str(home))
    monkeypatch.setattr("tokirecur.launch_catalog.shutil.which", lambda _: "/usr/bin/dsh")
    assert unavailable_reason(get_launch_app("dsh")) is None


@pytest.mark.parametrize("key", NEW_CLIENTS)
def test_model_bound_new_clients_require_a_model(key):

    with pytest.raises(ValueError, match="requires an active model"):
        launch_cmd("chat", key, None)


def test_new_client_launch_plans_pin_the_selected_model(monkeypatch):
    monkeypatch.setattr("tokirecur.selector.client_mode", lambda _: "managed")
    spec = _spec()
    expected = {
        "qwen": ["qwen", "--model", spec.model_id,
                 "--openai-base-url", "http://127.0.0.1:8811/v1",
                 "--auth-type", "openai"],
        "droid": ["droid"],
        "cline": ["cline", "--config",
                  str(Path.home() / ".cline" / "tokirecur"),
                  "--data-dir",
                  str(Path.home() / ".cline" / "tokirecur" / "data"),
                  "--tui", "--provider", "openai-compatible", "--model", spec.model_id],
        "dsh": ["dsh", "--profile", "tui"],
        "grok": ["grok", "--model", "tokirecur-ornith"],
        "kilo": ["kilo", "--model", f"openai-compatible/{spec.model_id}"],
        "pi": ["pi", "--provider", "tokirecur", "--model", spec.model_id],
        "copilot": ["copilot", "--model", spec.model_id],
        "kimi": ["kimi", "--model", f"tokirecur/{spec.model_id}"],
        "omp": ["omp", "--provider", "tokirecur", "--model", spec.model_id],
    }
    for key, argv in expected.items():
        plan = launch_cmd("chat", key, spec)
        assert plan["argv"] == argv
        assert plan["env"]["TOKI_AUTH_TOKEN"]
        assert plan["env"].get("OPENAI_BASE_URL") in {
            None, "http://127.0.0.1:8811/v1",
        }
        if key == "cline":
            assert plan["env"]["CLINE_DATA_DIR"] == str(
                Path.home() / ".cline" / "tokirecur" / "data"
            )


def test_toki_launch_does_not_put_auth_token_in_argv(monkeypatch):
    monkeypatch.setenv("TOKI_AUTH_TOKEN", "test-token")
    monkeypatch.setattr("tokirecur.selector.client_mode", lambda _: "managed")
    plan = launch_cmd("chat", "toki", _spec())
    assert "--auth-token" not in plan["argv"]
    assert plan["env"]["TOKI_AUTH_TOKEN"] == "test-token"


def test_copilot_launch_uses_effective_limits(monkeypatch):
    monkeypatch.setattr("tokirecur.selector.client_mode", lambda _: "managed")
    monkeypatch.setattr("tokirecur.selector._effective_ctx", lambda _: 4096)
    spec = _spec()
    spec.max_output_tokens = 1024
    plan = launch_cmd("chat", "copilot", spec)
    assert plan["env"]["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"] == "4096"
    assert plan["env"]["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"] == "1024"


def test_new_client_raw_mode_uses_llama_endpoint(monkeypatch):
    monkeypatch.setattr("tokirecur.selector.client_mode", lambda _: "raw")
    plan = launch_cmd("chat", "qwen", _spec())
    assert plan["env"]["OPENAI_BASE_URL"] == "http://127.0.0.1:8112/v1"


def test_model_bound_client_still_requires_a_model():
    with pytest.raises(ValueError, match="requires an active model"):
        launch_cmd("chat", "opencode", None)


def test_web_client_still_requires_a_model():
    with pytest.raises(ValueError, match="requires an active model"):
        launch_cmd("web", "toki", None)


def test_removed_account_clients_are_not_offered():
    """The catalog is the single source of what the TUI offers."""
    for key in REMOVED_ACCOUNT_CLIENTS:
        assert get_launch_app(key) is None
        assert key not in [app.key for app in LAUNCH_APPS]




def test_dsh_profile_patch_binds_the_profile_to_the_bound_model(monkeypatch, tmp_path):
    """settings.yaml is not the user override layer for a named profile.

    `dsh --profile tui --dump-config` and `--dump-default-config` are
    identical, so a profile never reads ~/.dsh/settings.yaml and falls through
    to deepseek-official/deepseek-flash. Without this, the TUI boots and then
    asks for a DEEPSEEK_API_KEY this machine does not have.
    """
    import yaml

    from tokirecur import ctl

    dsh = tmp_path / ".dsh"
    profile = dsh / "profiles" / "tui"
    profile.mkdir(parents=True)
    monkeypatch.setenv("DSH_HOME", str(dsh))

    spec = ctl.get_model(ctl.load_registry()[0].name)
    out = ctl._write_dsh_profile_patch(spec)
    assert out == profile / "cordis.patch.yml"

    patch = yaml.safe_load(out.read_text())
    by_id = {entry["id"]: entry for entry in patch}
    assert by_id["agent-default-model"]["config"]["provider"] == "tokirecur"
    assert by_id["agent-default-model"]["config"]["model"] == spec.model_id
    provider = by_id["llm-pi-ai"]["config"]["providers"]["tokirecur"]
    assert provider["apiKeyEnv"] == "TOKI_AUTH_TOKEN"
    assert provider["api"] == "openai-completions"
    assert provider["baseURL"].startswith("http://127.0.0.1:")
    assert [m["id"] for m in provider["models"]] == [spec.model_id]


def test_dsh_profile_patch_skips_a_profile_that_was_never_created(monkeypatch, tmp_path):
    """Inventing a profile would install a plugin tree nobody asked for."""
    from tokirecur import ctl

    dsh = tmp_path / ".dsh"
    dsh.mkdir()
    monkeypatch.setenv("DSH_HOME", str(dsh))
    spec = ctl.get_model(ctl.load_registry()[0].name)
    assert ctl._write_dsh_profile_patch(spec) is None


def test_dsh_config_write_also_updates_the_profile_layer(monkeypatch, tmp_path):
    """One call keeps both layers in step; they cannot drift apart."""
    import yaml

    from tokirecur import ctl

    dsh = tmp_path / ".dsh"
    profile = dsh / "profiles" / "tui"
    profile.mkdir(parents=True)
    monkeypatch.setenv("DSH_HOME", str(dsh))
    spec = ctl.get_model(ctl.load_registry()[0].name)

    ctl.write_dsh_config(spec)

    settings = yaml.safe_load((dsh / "settings.yaml").read_text())
    patch = yaml.safe_load((profile / "cordis.patch.yml").read_text())
    by_id = {entry["id"]: entry for entry in patch}
    # Same model in both, or one of them is pointing somewhere stale.
    assert settings["agent-default-model"]["model"] == by_id["agent-default-model"]["config"]["model"]
    assert (settings["llm-pi-ai"]["providers"]["tokirecur"]["baseURL"]
            == by_id["llm-pi-ai"]["config"]["providers"]["tokirecur"]["baseURL"])
