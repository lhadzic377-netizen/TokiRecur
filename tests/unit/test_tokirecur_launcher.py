"""TokiRecur desktop launcher integration.

The launcher's second entry starts the desktop app itself: `tokirecur` is a
GUI, not a model-bound terminal client, so the plan detaches it and passes
the environment through untouched. These tests pin the three surfaces that
must agree on it: the catalog entry, the `launch_cmd` plan, and the absence
of the retired tokicode client.
"""

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


def test_tokirecur_is_in_the_catalog_and_not_model_bound():
    app = get_launch_app("tokirecur")
    assert app is not None
    assert app.interface == "chat"
    assert app.model_binding is False
    assert app.command == "tokirecur"
    assert app.detach is True


def test_tokirecur_is_reachable_by_number_key():
    # Keys 1-9 are hardcoded to indices 0-8 in the launcher, so the entry
    # has to sit early in the tuple to be one-key reachable.
    index = next(i for i, a in enumerate(LAUNCH_APPS) if a.key == "tokirecur")
    assert index < 9


def test_availability_probe_uses_the_command_name(monkeypatch):
    monkeypatch.setattr("tokirecur.launch_catalog.shutil.which", lambda c: f"/usr/bin/{c}")
    app = get_launch_app("tokirecur")
    assert unavailable_reason(app) is None
    monkeypatch.setattr("tokirecur.launch_catalog.shutil.which", lambda c: None)
    assert unavailable_reason(app) == "tokirecur not installed"


# --- launch plan ---------------------------------------------------------


def test_launch_cmd_spawns_the_desktop_app_detached():
    plan = launch_cmd("chat", "tokirecur", None)
    assert plan["argv"] == ["tokirecur"]
    assert plan["detach"] is True
    assert plan["url"] is None


def test_launch_cmd_does_not_inject_a_model_endpoint(monkeypatch):
    # The app reads models.yaml and the daemon auth itself; the launcher
    # must not steer it at whichever model happens to be pinned here.
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("TOKI_AUTH_TOKEN", raising=False)
    plan = launch_cmd("chat", "tokirecur", None)
    assert "OPENAI_BASE_URL" not in plan["env"]
    assert "TOKI_AUTH_TOKEN" not in plan["env"]


# --- the retired client --------------------------------------------------


def test_the_tokicode_client_is_gone():
    # The bun terminal agent retired with its name: no catalog entry, no
    # config writer, nothing left pointing at a `tokicode` binary.
    assert get_launch_app("tokicode") is None
    assert not hasattr(ctl, "write_tokicode_config")
    assert not any("tokicode" in repr(app) for app in LAUNCH_APPS)
