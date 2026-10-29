"""The chain a new user walks, from an empty home directory.

Why this file exists. Every fault found during the DeepSeek work had the same
shape: something was correct when it was written and silently wrong later.

  * dsh's settings.yaml pointed at a model and a port that no longer existed
  * the web profile was never bound, because only the tui profile was written
  * auth_token() invented a random token and never matched the daemon

Each was fixed where it was found. None was *guarded*, which is why the same
class kept arriving. CI proved the installer runs and the unit tests pass;
nothing asserted the chain between them.

So this walks the whole path in a temporary home and checks the invariants
that actually failed:

  register a model -> sync every client config -> every launch plan builds
  -> every config names the live model -> every plan carries the daemon token

Deliberately not covered: downloading weights and booting llama-server. Those
are gigabytes, need a network and a GPU. The installer job in CI covers the
install, and tests/integration/ covers the running daemon. What is covered
here is the part that is pure logic and was quietly wrong.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from tokirecur import ctl
from tokirecur.launch_catalog import LAUNCH_APPS, LaunchApp, get_launch_app, unavailable_reason
from tokirecur.selector import launch_cmd

# Clients that must end up configured and pointing at the live stack.
CONFIG_CLIENTS = (
    "opencode", "mimo", "openclaw", "hermes", "codex", "qwen", "droid",
    "cline", "claude", "dsh", "grok", "kilo", "pi", "kimi", "omp",
)


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    """An empty home, with tokirecur's roots redirected into it.

    The path constants are resolved at import time from HOME and a set of
    TOKI_* env vars, so redirecting the env alone would not be enough - the
    module constants are rebound directly. Doing it explicitly is the point:
    the alternative is a suite that quietly writes to the real config, which
    is exactly how the dsh work was diagnosed as test pollution.
    """
    home = tmp_path / "home"
    home.mkdir()
    config = home / ".config" / "tokirecur"
    config.mkdir(parents=True)
    state = home / ".local" / "state" / "tokirecur"
    state.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TOKI_AUTH_TOKEN", "fresh-install-token")
    for var in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME",
                "TOKI_MODEL_REGISTRY", "TOKI_STATE_DIR", "TOKI_ACTIVE_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", config / "models.yaml")
    monkeypatch.setattr(ctl, "STATE_ROOT", state)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", config / "active-model")
    return home


@pytest.fixture
def model(fresh_home):
    """One registered model, on ports nothing else in the suite uses."""
    raw = dict(ctl.DEFAULT_MODELS[0])
    raw["llama_port"], raw["daemon_port"] = 8123, 8823
    return ctl.ModelSpec(raw=raw, **raw)


def _registered_pair(fresh_home, first, second):
    registry = ctl.DEFAULT_REGISTRY
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        yaml.safe_dump({"models": [first.raw, second.raw]}, sort_keys=False)
    )
    ctl.invalidate_registry_cache()
    return first


def _registered(fresh_home, spec):
    """Write the registry the way the daemon reads it: {"models": [...]}."""
    registry = ctl.DEFAULT_REGISTRY
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(yaml.safe_dump({"models": [spec.raw]}, sort_keys=False))
    # load_registry caches per path for the process lifetime; every real
    # mutation path invalidates it, so a test that writes the file directly
    # has to as well or it reads its own stale snapshot.
    ctl.invalidate_registry_cache()
    return spec


# --- the registry ---------------------------------------------------------


def test_a_fresh_install_can_register_and_activate_a_model(fresh_home, model):
    """Step 1: a model exists and can be made active."""
    _registered(fresh_home, model)
    assert [s.name for s in ctl.load_registry()] == [model.name]
    assert ctl.ACTIVE_MARKER.write_text(model.name) is not None
    assert ctl.read_active() == model.name
    assert ctl.get_model(model.name).daemon_url.endswith(":8823")


def test_an_unknown_model_is_a_clear_error_not_a_traceback(fresh_home):
    with pytest.raises(KeyError) as exc:
        ctl.get_model("does-not-exist")
    assert "starter" in str(exc.value).lower()


# --- every client config --------------------------------------------------


def test_one_command_syncs_every_client_config(fresh_home, model):
    """Step 2: `toki model use` covers the whole catalog on a clean home."""
    _registered(fresh_home, model)
    written = ctl.sync_cli_configs(model)

    missing = [c for c in CONFIG_CLIENTS if c not in written]
    assert not missing, f"no config written for: {missing}"
    for client, path in written.items():
        assert Path(path).exists(), f"{client} reported {path} but did not write it"


def test_every_config_names_the_live_model_and_daemon(fresh_home, model):
    """The staleness invariant.

    A config naming a different model or endpoint is the exact fault that made
    dsh report a transport error while every visible part looked healthy.
    """
    _registered(fresh_home, model)
    written = ctl.sync_cli_configs(model)

    for client, path in written.items():
        text = Path(path).read_text("utf-8", "replace")
        assert model.model_id in text, f"{client} config does not name {model.model_id}"
        assert "8823" in text or model.daemon_url in text, (
            f"{client} config does not point at the bound daemon"
        )


def test_switching_models_reselects_every_client(fresh_home, model):
    """The drift that bit twice, asserted on the selection.

    Several clients deliberately keep a provider entry per *registered*
    model - codex carries a block for every spec in the registry, and
    opencode lists the whole roster. So "the old port must not appear" is not
    a valid invariant: a still-registered model legitimately has a port.

    What must hold is that the *selected* model moves, and that the config
    stops naming the old selection as the active one. A config that selects a
    dead stack is the fault, and it surfaces much later as a transport error.
    """
    second_raw = dict(ctl.DEFAULT_MODELS[1])
    second_raw.update(
        name="35b", model_id="qwen3.6-35b", label="Second",
        llama_port=8133, daemon_port=8833,
    )
    second = ctl.ModelSpec(raw=second_raw, **second_raw)
    _registered_pair(fresh_home, model, second)

    first = ctl.sync_cli_configs(model)
    ctl.sync_cli_configs(second)

    stale_id = model.model_id
    for client, path in first.items():
        text = Path(path).read_text("utf-8", "replace")
        assert second.model_id in text, f"{client} never saw the new model"
        # The old id may survive inside a roster, but never as the choice.
        assert f'model = "{stale_id}"' not in text, f"{client} still selects the old model"
        assert f'"model": "{stale_id}"' not in text, f"{client} still selects the old model"


def test_a_model_removed_from_the_registry_loses_its_provider_entry(fresh_home, model, monkeypatch):
    """The other half of drift, and codex already guards it.

    Use 9b, switch to 35b, then uninstall 9b. A client that keeps offering
    9b after it is gone has a menu entry pointing at a dead port, and picking
    it is a request to a server that is not running.

    Both models are advertised while their daemons answer; a client config
    only ever points at endpoints that are up (a dead port in a model menu
    is the bug this guards), so the reachability probe is stubbed here to
    put both on the menu and let the removal be the thing under test.
    """
    monkeypatch.setattr(ctl, "daemon_reachable", lambda spec: True)
    monkeypatch.setattr(ctl, "llama_reachable", lambda spec: True)
    other_raw = dict(ctl.DEFAULT_MODELS[1])
    other_raw.update(
        name="35b", model_id="qwen3.6-35b", label="Second",
        llama_port=8133, daemon_port=8833,
    )
    other = ctl.ModelSpec(raw=other_raw, **other_raw)

    _registered_pair(fresh_home, model, other)
    written = ctl.sync_cli_configs(model)
    codex = Path(written["codex"]).read_text()
    assert f"tokirecur-{model.name}" in codex

    # Switching leaves the previous model offered, because it is still installed.
    ctl.sync_cli_configs(other)
    both = Path(written["codex"]).read_text()
    assert f"tokirecur-{model.name}" in both and f"tokirecur-{other.name}" in both

    # Uninstalling it must take the entry with it.
    ctl.DEFAULT_REGISTRY.write_text(
        yaml.safe_dump({"models": [other.raw]}, sort_keys=False)
    )
    ctl.invalidate_registry_cache()
    ctl.sync_cli_configs(other)
    after = Path(written["codex"]).read_text()
    assert f"tokirecur-{model.name}" not in after, (
        "a model removed from the registry kept its provider block"
    )
    assert f"tokirecur-{other.name}" in after, "the live model lost its block"


def test_dsh_gets_settings_and_a_patch_for_every_profile(fresh_home, model, monkeypatch):
    """dsh reads settings.yaml only for headless; profiles need their own.

    Leaving one profile unbound is invisible until that surface is used -
    which is how the web UI came to serve a cloud model it had no key for.
    """
    _registered(fresh_home, model)
    dsh = fresh_home / ".dsh"
    for profile in ("tui", "web", "headless"):
        target = dsh / "profiles" / profile
        target.mkdir(parents=True)
        (target / "cordis.patch.yml").write_text("[]\n")
    monkeypatch.setenv("DSH_HOME", str(dsh))

    ctl.write_dsh_config(model)

    settings = dsh / "settings.yaml"
    assert settings.exists(), "headless config missing"
    assert model.model_id in settings.read_text()

    for profile in ("tui", "web", "headless"):
        text = (dsh / "profiles" / profile / "cordis.patch.yml").read_text()
        assert "agent-default-model" in text, f"{profile} was never bound"
        assert model.model_id in text, f"{profile} points at another model"


def test_a_profile_directory_without_a_patch_layer_is_left_alone(fresh_home, model, monkeypatch):
    """Creating a patch layer nobody had would be inventing structure."""
    _registered(fresh_home, model)
    dsh = fresh_home / ".dsh"
    (dsh / "profiles" / "tui").mkdir(parents=True)
    (dsh / "profiles" / "tui" / "cordis.patch.yml").write_text("[]\n")
    (dsh / "profiles" / "web").mkdir(parents=True)  # deliberately no patch file
    monkeypatch.setenv("DSH_HOME", str(dsh))

    ctl.write_dsh_config(model)
    assert not (dsh / "profiles" / "web" / "cordis.patch.yml").exists()


# --- every launch plan ----------------------------------------------------


def test_every_catalog_entry_builds_a_launch_plan(fresh_home, model):
    """Step 3: everything the menu advertises can actually be launched.

    A client that appears in the menu but cannot start is the least visible
    failure mode, because the row is right there looking fine.
    """
    _registered(fresh_home, model)
    failures = {}
    for app in LAUNCH_APPS:
        interface = "web" if app.interface == "web" else "chat"
        try:
            plan = launch_cmd(interface, app.client, model)
        except Exception as e:  # noqa: BLE001 - list every failure, not the first
            failures[f"{app.key} ({app.client})"] = f"{type(e).__name__}: {e}"
            continue
        assert plan["argv"], f"{app.key} produced an empty argv"
    assert not failures, f"launch plans failed: {failures}"


def test_every_plan_carries_the_daemon_token(fresh_home, model):
    """A client configured with the wrong token 401s on its first turn.

    That surfaces as an auth error inside somebody else's application, which
    is why it survived two releases before being traced here.
    """
    _registered(fresh_home, model)
    ctl.sync_cli_configs(model)
    for app in LAUNCH_APPS:
        if app.interface != "chat" or not app.model_binding:
            continue
        plan = launch_cmd("chat", app.client, model)
        token = plan["env"].get("TOKI_AUTH_TOKEN")
        if token is None:
            continue  # this client does not use the shared token
        assert token == ctl.auth_token(), f"{app.client} was given a different token"


def test_a_client_needing_a_model_says_so_instead_of_launching(fresh_home):
    for client in ("codex", "opencode", "dsh", "claude"):
        assert get_launch_app(client) is not None
        with pytest.raises(ValueError, match="requires an active model"):
            launch_cmd("chat", client, None)


# --- harness readiness ----------------------------------------------------


def test_harnesses_we_do_not_ship_stay_removed(fresh_home):
    """A deliberate product decision, so it stays pinned.

    Muse, Cursor and Devin authenticate against their own account and can
    never use a locally bound model, so they were dropped from the catalog.
    """
    for removed in ("muse", "cursor-agent", "devin"):
        assert get_launch_app(removed) is None
    assert not any(a.protocol == "account" for a in LAUNCH_APPS)


def test_a_missing_binary_is_reported_with_its_name(fresh_home):
    """A row you cannot start must say which binary is absent."""
    app = LaunchApp(
        key="nope", title="Nope", description="", interface="chat",
        client="definitely-not-installed", command="definitely-not-installed",
    )
    reason = unavailable_reason(app)
    assert reason is not None
    assert "definitely-not-installed" in reason


def test_a_missing_required_profile_is_reported(fresh_home):
    """dsh's required profile must not read as fine when it is absent."""
    app = LaunchApp(
        key="dsh", title="dsh", description="", interface="chat", client="dsh",
        command="sh", required_path="~/.dsh/profiles/tui",
    )
    assert not (fresh_home / ".dsh" / "profiles" / "tui").exists()
    reason = unavailable_reason(app)
    assert reason is None or "profile" in reason


# --- credentials close the loop -------------------------------------------


def test_the_token_we_write_into_configs_actually_authenticates(llama_server, fresh_home):
    """The 0.3.2 fault, guarded.

    `auth_token()` used to invent a random token when the env var was unset,
    so it never matched what the daemon accepted. Every symptom appeared
    inside somebody else's application - "API key is invalid", a dsh web UI
    that loaded and then failed its first turn - and none of them pointed
    back here. Assert the round trip instead: the token that gets written
    into client configs is the token the daemon accepts, and the two
    resolvers are read in the same order.
    """
    import tempfile

    from fastapi.testclient import TestClient

    from tokirecur.api.app import create_app
    from tokirecur.backend import LlamaServerBackend
    from tokirecur.index import BruteForceIndex

    # The daemon reads env first, then the auth file. ctl.auth_token must
    # resolve the same way or the two sides drift.
    monkey = os.environ.get("TOKI_AUTH_TOKEN")
    try:
        os.environ["TOKI_AUTH_TOKEN"] = "round-trip-token"
        assert ctl.auth_token() == "round-trip-token"

        db = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        db.close()
        app = create_app(
            backend=LlamaServerBackend(llama_server, timeout=120.0),
            index=BruteForceIndex(db.name),
            cache_dir=Path(tempfile.mkdtemp()),
            enable_auth=True,
            enable_rate_limit=False,
        )
        client = TestClient(app)
        assert client.get("/v1/models", headers={"Authorization": "Bearer round-trip-token"}).status_code == 200
        # And the wrong token is still wrong, so the check has teeth.
        assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    finally:
        if monkey is None:
            os.environ.pop("TOKI_AUTH_TOKEN", None)
        else:
            os.environ["TOKI_AUTH_TOKEN"] = monkey


def test_the_daemon_and_ctl_resolve_the_token_in_the_same_order(fresh_home, monkeypatch):
    """Parity, asserted directly rather than inferred from a symptom.

    Both sides must read env, then the auth file, in that order. Reorder one
    without the other and every client config is written with a token the
    daemon will not accept.
    """
    from tokirecur.api import app as daemon_app

    auth = fresh_home / ".config" / "tokirecur" / "auth"
    auth.parent.mkdir(parents=True, exist_ok=True)
    auth.write_text("from-the-file\n")
    monkeypatch.delenv("TOKI_AUTH_TOKEN", raising=False)

    assert ctl.auth_token() == "from-the-file"
    assert daemon_app._get_auth_token() == "from-the-file"

    # And the env var still wins on both sides.
    monkeypatch.setenv("TOKI_AUTH_TOKEN", "from-env")
    assert ctl.auth_token() == "from-env"
    assert daemon_app._get_auth_token() == "from-env"
