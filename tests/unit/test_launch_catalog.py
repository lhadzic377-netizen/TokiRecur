import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tokirecur import ctl
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
        "toki", "tokirecur", "opencode", "hermes", "openclaw", "codex", "mimo",
        "qwen", "droid", "cline", "dsh", "grok", "kilo", "pi",
        "copilot", "claude", "kimi", "omp", "web", "dsh-web",
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
    # The stale user token must not reach the daemon: Claude Code sends
    # ANTHROPIC_AUTH_TOKEN as-is, so it has to be ours. An inherited
    # ANTHROPIC_API_KEY would be preferred over it, so it goes.
    assert plan["env"]["ANTHROPIC_AUTH_TOKEN"] == "test-claude-token"
    assert "stale-user-token" not in plan["env"].values()
    assert "ANTHROPIC_API_KEY" not in plan["env"]


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




def test_dsh_profile_patch_binds_the_profile_to_the_bound_model(monkeypatch, tmp_path, isolated_registry):
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
    # A profile is only bound if it has a user patch layer to write into.
    # Creating one where the user has none would be inventing structure.
    (profile / "cordis.patch.yml").write_text("[]\n")
    monkeypatch.setenv("DSH_HOME", str(dsh))

    spec = ctl.get_model(ctl.load_registry()[0].name)
    out = ctl._write_dsh_profile_patch(spec, "tui")
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


def test_dsh_profile_patch_skips_a_profile_that_was_never_created(monkeypatch, tmp_path, isolated_registry):
    """Inventing a profile would install a plugin tree nobody asked for."""
    from tokirecur import ctl

    dsh = tmp_path / ".dsh"
    dsh.mkdir()
    monkeypatch.setenv("DSH_HOME", str(dsh))
    spec = ctl.get_model(ctl.load_registry()[0].name)
    assert ctl._write_dsh_profile_patch(spec, "tui") is None


def test_dsh_config_write_also_updates_the_profile_layer(monkeypatch, tmp_path, isolated_registry):
    """One call keeps both layers in step; they cannot drift apart."""
    import yaml

    from tokirecur import ctl

    dsh = tmp_path / ".dsh"
    profile = dsh / "profiles" / "tui"
    profile.mkdir(parents=True)
    (profile / "cordis.patch.yml").write_text("[]\n")
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


def test_dsh_binds_every_profile_not_just_tui(monkeypatch, tmp_path, isolated_registry):
    """An unbound `web` profile serves its UI and fails the first message.

    That reads as a broken web UI rather than a missing binding, which is
    exactly how this was reported: the TUI worked and the browser did not.
    """
    from tokirecur import ctl

    dsh = tmp_path / ".dsh"
    for name in ("tui", "web", "headless"):
        (dsh / "profiles" / name).mkdir(parents=True)
    (dsh / "profiles" / "tui" / "cordis.patch.yml").write_text("[]\n")
    (dsh / "profiles" / "web" / "cordis.patch.yml").write_text("[]\n")
    # headless has no patch layer of its own; settings.yaml is its config.
    monkeypatch.setenv("DSH_HOME", str(dsh))

    spec = ctl.get_model(ctl.load_registry()[0].name)
    assert ctl._existing_dsh_profiles() == ["tui", "web"]

    written = ctl._write_all_dsh_profile_patches(spec)
    assert [p.parent.name for p in written] == ["tui", "web"]
    for path in written:
        assert "agent-default-model" in path.read_text()


def test_dsh_profile_list_ignores_directories_without_a_patch_layer(monkeypatch, tmp_path):
    from tokirecur import ctl

    dsh = tmp_path / ".dsh"
    (dsh / "profiles" / "web").mkdir(parents=True)  # no cordis.patch.yml
    (dsh / "profiles" / "node_modules").mkdir(parents=True)
    monkeypatch.setenv("DSH_HOME", str(dsh))
    assert ctl._existing_dsh_profiles() == []


def test_dsh_web_is_offered_as_its_own_entry():
    """It was unreachable: the only web row was a hardcoded toki entry."""
    from tokirecur.launch_catalog import LAUNCH_APPS

    web_rows = {a.client for a in LAUNCH_APPS if a.interface == "web"}
    assert "dsh" in web_rows
    dsh_web = next(a for a in LAUNCH_APPS if a.key == "dsh-web")
    # Not model-bound: dsh web reads the same settings file as the other
    # profiles, it is not a tokirecur model client.
    assert dsh_web.model_binding is False


def test_dsh_web_plan_detaches_and_lets_us_open_the_browser(isolated_registry):
    from tokirecur.selector import launch_cmd

    spec = ctl.get_model(ctl.load_registry()[0].name)
    plan = launch_cmd("web", "dsh", spec)
    assert plan["argv"] == ["dsh", "web", "--no-open"]
    assert plan["detach"] is True
    # No url: dsh prints a tokenised one at boot that we read from its log.
    # A hardcoded http://127.0.0.1:3080/ gets a 401.
    assert plan["url"] is None


def test_dsh_web_does_not_shadow_the_chat_tui_launch(isolated_registry):
    """A branch in the chat block was unreachable, shadowed by the real one."""
    from tokirecur.selector import launch_cmd

    spec = ctl.get_model(ctl.load_registry()[0].name)
    chat = launch_cmd("chat", "dsh", spec)
    assert chat["argv"] == ["dsh", "--profile", "tui"]
    assert chat["detach"] is False


def test_dsh_url_is_read_from_its_output(tmp_path):
    """The token lives in dsh's own boot line, nowhere else."""
    from tokirecur.selector import _DSH_URL_RE

    log = tmp_path / "dsh-web.log"
    log.write_text(
        "dsh web: http://127.0.0.1:3080/?token=abc_DEF-123=\n"
    )
    found = _DSH_URL_RE.search(log.read_text())
    assert found is not None
    assert found.group(0) == "http://127.0.0.1:3080/?token=abc_DEF-123="


def test_dsh_url_regex_ignores_a_bare_port():
    from tokirecur.selector import _DSH_URL_RE

    # The bare port 401s, so it must never be handed to the browser.
    assert _DSH_URL_RE.search("dsh web: http://127.0.0.1:3080/\n") is None
    assert _DSH_URL_RE.search("dsh: some unrelated line\n") is None


def test_dsh_web_gives_up_when_dsh_never_prints_a_url(tmp_path):
    """Should say so rather than appear to hang for the full timeout."""
    from tokirecur.selector import _await_dsh_web_url

    log = tmp_path / "dsh-web.log"
    log.write_text("dsh: some unrelated line\n")
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert _await_dsh_web_url(proc, str(log), timeout=5.0) is None


def test_dsh_web_ignores_urls_left_by_a_previous_run(tmp_path):
    """The log is appended, so scanning from 0 hands back a dead token.

    Without a start offset the previous run's URL matches first, and the
    browser opens a session that no longer exists.
    """
    from tokirecur.selector import _await_dsh_web_url, _log_size

    log = tmp_path / "dsh-web.log"
    log.write_text("dsh web: http://127.0.0.1:3080/?token=OLD_TOKEN\n")
    start = _log_size(str(log))

    dead = subprocess.Popen([sys.executable, "-c", "raise SystemExit(1)"])
    dead.wait()
    assert _await_dsh_web_url(dead, str(log), timeout=2.0, start=start) is None

    # And with fresh bytes it does find the new one.
    alive = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        with log.open("a") as fh:
            fh.write("dsh web: http://127.0.0.1:3080/?token=NEW_TOKEN\n")
        found = _await_dsh_web_url(alive, str(log), timeout=3.0, start=start)
        assert found is not None
        assert found.endswith("token=NEW_TOKEN")
    finally:
        alive.kill()
        alive.wait()


def test_dsh_web_reuses_a_running_instance(tmp_path):
    """A second `dsh web` dies with EADDRINUSE and prints no URL.

    The first one is serving fine, so report that rather than a failure the
    user cannot act on.
    """
    from tokirecur.selector import _existing_dsh_web_url

    log = tmp_path / "dsh-web.log"
    log.write_text(
        "dsh web: http://127.0.0.1:3080/?token=first\n"
        "dsh: startup failed: EADDRINUSE\n"
        "dsh web: http://127.0.0.1:3080/?token=second\n"
    )
    # The most recent one wins.
    assert _existing_dsh_web_url(str(log)).endswith("token=second")

    log.write_text("dsh: startup failed: EADDRINUSE\n")
    assert _existing_dsh_web_url(str(log)) is None
    assert _existing_dsh_web_url(str(tmp_path / "missing.log")) is None


def test_dsh_chat_plan_brings_the_web_ui_alongside_the_tui(isolated_registry):
    """One selection, both surfaces - the TUI in the terminal, web in a tab."""
    from tokirecur.selector import launch_cmd

    spec = ctl.get_model(ctl.load_registry()[0].name)
    plan = launch_cmd("chat", "dsh", spec)
    assert plan["detach"] is False
    assert plan["companion"] == {
        "client": "dsh",
        "argv": ["dsh", "web", "--no-open"],
    }


def test_only_dsh_declares_a_companion(isolated_registry):
    from tokirecur.selector import launch_cmd

    spec = ctl.get_model(ctl.load_registry()[0].name)
    for client in ("opencode", "codex", "claude", "toki"):
        assert launch_cmd("chat", client, spec).get("companion") is None


def test_companion_starts_before_the_foreground_exec():
    """A foreground launch replaces the process; later children never exist."""
    from tokirecur.tui.launcher import _execute_launch

    order = []
    import tokirecur.selector as sel

    original = sel.start_companion
    original_fg = sel._launch_foreground
    try:
        sel.start_companion = lambda c, e: order.append("companion")
        sel._launch_foreground = lambda argv, env: order.append("foreground") or 0
        _execute_launch({
            "interface": "chat", "client": "dsh", "argv": ["dsh"], "env": {},
            "detach": False, "url": None,
            "companion": {"client": "dsh", "argv": ["dsh", "web", "--no-open"]},
        })
    finally:
        sel.start_companion = original
        sel._launch_foreground = original_fg
    assert order == ["companion", "foreground"]


def test_companion_failure_does_not_block_the_primary_launch(monkeypatch):
    """A browser that will not open must not cost you the TUI."""
    import tokirecur.selector as sel
    from tokirecur.tui.launcher import _execute_launch

    def boom(companion, env):
        raise RuntimeError("no browser here")

    original = sel.start_companion
    original_fg = sel._launch_foreground
    try:
        sel.start_companion = boom
        sel._launch_foreground = lambda argv, env: 0
        rc = _execute_launch({
            "interface": "chat", "client": "dsh", "argv": ["dsh"], "env": {},
            "detach": False, "url": None,
            "companion": {"client": "dsh", "argv": ["dsh", "web", "--no-open"]},
        })
    finally:
        sel.start_companion = original
        sel._launch_foreground = original_fg
    assert rc == 0
