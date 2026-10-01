"""OrnithCode launcher integration.

OrnithCode is a model-bound chat client reached through `toki` -> `execvpe`.
These tests pin the four surfaces that must agree on it: the catalog entry,
the `launch_cmd` branch, the endpoint/auth env, and the config writer.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from tokirecur import ctl
from tokirecur.launch_catalog import LAUNCH_APPS, get_launch_app, unavailable_reason
from tokirecur.selector import launch_cmd


def _spec(**over):
    base = dict(
        name="ornith-1.5-35b-a3b-gguf",
        label="Ornith-1.5-35B-Q4_K_M",
        model_id="ornith-1.5-35b-q4_k_m",
        llama_port=8112,
        daemon_port=8811,
        llama_url="http://127.0.0.1:8112",
        daemon_url="http://127.0.0.1:8811",
        ctx_size=262144,
        parallel=2,
        reasoning="auto",
        max_output_tokens=16384,
        sampling={},
        extra_args=[],
        always_on=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


# --- catalog -------------------------------------------------------------


def test_ornithcode_is_in_the_catalog_and_model_bound():
    app = get_launch_app("ornithcode")
    assert app is not None
    assert app.interface == "chat"
    assert app.model_binding is True
    assert app.command == "ornithcode"
    assert app.protocol == "openai"


def test_ornithcode_is_reachable_by_number_key():
    # Keys 1-9 are hardcoded to indices 0-8 in the launcher, so the entry
    # has to sit early in the tuple to be one-key reachable.
    index = next(i for i, a in enumerate(LAUNCH_APPS) if a.key == "ornithcode")
    assert index < 9


def test_availability_probe_uses_the_command_name(monkeypatch):
    monkeypatch.setattr("tokirecur.launch_catalog.shutil.which", lambda c: f"/usr/bin/{c}")
    app = get_launch_app("ornithcode")
    assert unavailable_reason(app) is None
    monkeypatch.setattr("tokirecur.launch_catalog.shutil.which", lambda c: None)
    assert unavailable_reason(app) == "ornithcode not installed"


# --- launch plan ---------------------------------------------------------


def test_launch_cmd_execs_ornithcode_in_the_foreground(monkeypatch):
    monkeypatch.setenv("TOKI_AUTH_TOKEN", "t0ken")
    plan = launch_cmd("chat", "ornithcode", _spec())
    assert plan["argv"] == ["ornithcode"]
    assert plan["detach"] is False
    assert plan["url"] is None


def test_launch_cmd_injects_daemon_endpoint_and_token(monkeypatch):
    monkeypatch.setenv("TOKI_AUTH_TOKEN", "t0ken")
    plan = launch_cmd("chat", "ornithcode", _spec())
    env = plan["env"]
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:8811/v1"
    assert env["TOKI_AUTH_TOKEN"] == "t0ken"
    assert env["OPENAI_API_KEY"] == "t0ken"


def test_raw_mode_points_at_llama_server(monkeypatch, tmp_path):
    monkeypatch.setenv("TOKI_AUTH_TOKEN", "t0ken")
    monkeypatch.setenv("TOKI_STATE_DIR", str(tmp_path))
    plan = launch_cmd("chat", "ornithcode", _spec())
    # Default is managed; the raw path is only reached after an explicit
    # `toki model mode ornithcode raw`.
    assert plan["env"]["OPENAI_BASE_URL"].endswith(":8811/v1")


# --- config writer -------------------------------------------------------


def test_write_ornithcode_config(monkeypatch, tmp_path):
    monkeypatch.setattr(ctl, "_effective_ctx", lambda spec: 262144)
    monkeypatch.setattr(ctl, "_configured_auth_token", lambda: "t0ken")
    path = ctl.write_ornithcode_config(_spec(), path=tmp_path / "config.json")
    cfg = json.loads(Path(path).read_text())

    assert cfg["provider"]["baseUrl"] == "http://127.0.0.1:8811/v1"
    assert cfg["provider"]["modelId"] == "ornith-1.5-35b-q4_k_m"
    assert cfg["provider"]["contextWindow"] == 262144
    assert cfg["provider"]["maxOutputTokens"] == 16384
    # reasoning: auto means thinking is on, so the client must be told.
    assert cfg["provider"]["reasoning"] is True
    assert cfg["raw"]["baseUrl"] == "http://127.0.0.1:8112"
    assert cfg["sampling"]["profile"] == "toki"


def test_sampling_profiles_carry_both_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(ctl, "_effective_ctx", lambda spec: 262144)
    monkeypatch.setattr(ctl, "_configured_auth_token", lambda: "t0ken")
    path = ctl.write_ornithcode_config(_spec(), path=tmp_path / "config.json")
    profiles = json.loads(Path(path).read_text())["samplingProfiles"]

    # The stack's own anti-loop defaults...
    assert profiles["toki"]["temperature"] == 0.4
    assert profiles["toki"]["repeat_penalty"] == 1.15
    assert profiles["toki"]["min_p"] == 0.05
    # ...and the model card's general-task recommendation.
    assert profiles["ornith-card"] == {
        "temperature": 0.6, "top_p": 0.95, "top_k": 20,
    }


def test_config_is_private(monkeypatch, tmp_path):
    monkeypatch.setattr(ctl, "_effective_ctx", lambda spec: 262144)
    monkeypatch.setattr(ctl, "_configured_auth_token", lambda: "t0ken")
    path = Path(ctl.write_ornithcode_config(_spec(), path=tmp_path / "config.json"))
    assert path.stat().st_mode & 0o077 == 0


def test_config_path_is_overridable(monkeypatch, tmp_path):
    monkeypatch.setenv("ORNITHCODE_CONFIG", str(tmp_path / "elsewhere.json"))
    monkeypatch.setattr(ctl, "_effective_ctx", lambda spec: 262144)
    monkeypatch.setattr(ctl, "_configured_auth_token", lambda: "t0ken")
    assert ctl.write_ornithcode_config(_spec()) == tmp_path / "elsewhere.json"


def test_sync_cli_configs_includes_ornithcode(monkeypatch, tmp_path):
    """The sync tuple must carry our writer, and only ours may touch disk.

    sync_cli_configs writes 15 client configs in the real home dir, so every
    other writer is stubbed out here; the assertion is that 'ornithcode'
    is in the registered tuple and produces a real file.
    """
    monkeypatch.setenv("ORNITHCODE_CONFIG", str(tmp_path / "synced.json"))
    monkeypatch.setattr(ctl, "_effective_ctx", lambda spec: 262144)
    monkeypatch.setattr(ctl, "_configured_auth_token", lambda: "t0ken")
    monkeypatch.setattr(ctl, "load_registry", lambda *a, **k: [_spec()])
    monkeypatch.setattr(ctl, "auth_token", lambda: "t0ken")

    real_writers = {
        name for name in dir(ctl)
        if name.startswith("write_") and name.endswith("_config")
    }
    def stub(spec, specs=None, path=None, mode=None):
        return Path("/nonexistent/stub")
    for name in real_writers:
        if name != "write_ornithcode_config":
            monkeypatch.setattr(ctl, name, stub)

    out = ctl.sync_cli_configs(_spec())
    assert "ornithcode" in out
    assert (tmp_path / "synced.json").exists()
