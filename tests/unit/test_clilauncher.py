"""Unit tests for the Windows CLI launcher (Textual-launcher stand-in)."""

import os

import pytest

from tokirecur import clilauncher


@pytest.fixture
def quit_after(monkeypatch):
    """Menu inputs fed to input(); ends with '0'."""
    def _run(*picks):
        answers = iter(list(picks) + ["0"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))
        return clilauncher.run_cli_launcher()
    return _run


def test_menu_shows_apps(quit_after, capsys):
    assert quit_after("0") == 0
    out = capsys.readouterr().out
    assert "toki launcher" in out
    assert "1) Chat" in out
    assert "2) Switch model" in out
    assert "Launch OpenCode" in out
    assert "Launch MiMo Code" in out
    assert "Launch Qwen Code" in out
    assert "Launch Factory Droid" in out
    assert "Launch Cline CLI" in out
    assert "Launch DeepSeek Harness" in out
    assert "Launch Grok Build" in out
    assert "Launch Kilo Code" in out
    assert "Launch Pi" in out
    assert "Launch GitHub Copilot" in out
    assert "Launch Claude Code" in out
    assert "Launch Kimi Code" in out
    assert "Launch OMP" in out
    assert "Web Dashboards" in out
    # Removed in 0.3.0: account-bound providers, so none could use a bound model.
    for gone in ("Launch Muse Code", "Launch Cursor Agent", "Launch Devin"):
        assert gone not in out
    assert "Doctor" in out
    assert "0) Quit" in out


def test_quit_word(quit_after):
    assert quit_after("q") == 0


def test_chat_entry_runs_cli_chat(monkeypatch, quit_after, capsys):
    calls = []
    from tokirecur import clichat

    def fake_chat():
        calls.append(1)
        return 0

    monkeypatch.setattr(clichat, "run_cli_chat", fake_chat)
    assert quit_after("1") == 0
    assert calls == [1]
    assert "bye." in capsys.readouterr().out


def test_invalid_pick_loops(monkeypatch, quit_after, capsys):
    from tokirecur import clichat
    calls = []

    def fake_chat():
        calls.append(1)
        return 0

    monkeypatch.setattr(clichat, "run_cli_chat", fake_chat)
    assert quit_after("abc", "99") == 0
    out = capsys.readouterr().out
    assert "invalid pick" in out
    assert calls == []


def test_web_opens_browsers(monkeypatch, quit_after, capsys):
    opened = []
    monkeypatch.setattr(clilauncher.webbrowser, "open", lambda u: opened.append(u))
    monkeypatch.setattr(clilauncher, "_active_spec", lambda: None)
    monkeypatch.setattr(clilauncher, "_port_open", lambda port: True)
    assert quit_after(str(len(clilauncher.AGENTS) + 3)) == 0
    assert opened == list(clilauncher.WEB_URLS)
    assert "dashboards" in capsys.readouterr().out


def test_web_starts_ui_when_port_closed(monkeypatch, quit_after, capsys):
    """Web Dashboards must start the toki web UI (7777) when nothing
    listens on it, not just open tabs to a dead port."""
    opened = []
    spawned = []
    monkeypatch.setattr(clilauncher.webbrowser, "open", lambda u: opened.append(u))
    monkeypatch.setattr(
        clilauncher, "_active_spec", lambda: _stub_spec()
    )

    class FakeCtl:
        def daemon_reachable(self, spec):
            return True

        def use_model(self, name):
            pass

        def get_model(self, name):
            return _stub_spec()

        def model_status(self, spec):
            return {"llama": {"reachable": True}, "daemon": {"reachable": True}}

        def gpu_state(self, spec):
            return {"backend": "CPU", "layers": 0}

    monkeypatch.setattr(clilauncher, "_ctl", lambda: FakeCtl())
    monkeypatch.setattr(clilauncher, "_find_binary", lambda app: "toki.exe")
    monkeypatch.setattr(
        clilauncher.subprocess, "Popen",
        lambda argv, **kw: spawned.append(argv),
    )

    def fake_port_open(port):
        return False

    monkeypatch.setattr(clilauncher, "_port_open", fake_port_open)
    monkeypatch.setattr(clilauncher, "_wait_for_port", lambda port, timeout: True)
    assert quit_after(str(len(clilauncher.AGENTS) + 3)) == 0
    assert "toki" in spawned[0][0] and "web" in spawned[0]
    assert "http://127.0.0.1:7777" in opened


def test_windows_fallback_binary_counts_as_available(monkeypatch, tmp_path):
    appdata = tmp_path / "npm"
    appdata.mkdir()
    (appdata / "opencode.cmd").write_text("@echo off\n")
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(clilauncher.shutil, "which", lambda _: None)
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert clilauncher._agent_available("opencode") is True


def test_missing_agent_warns(monkeypatch, quit_after, capsys):
    monkeypatch.setattr("shutil.which", lambda _: None)
    # Agents are numbered from 3 in catalog order; opencode is the 2nd.
    opencode_pick = str(next(
        i for i, (app, _, _) in enumerate(clilauncher.AGENTS, 3) if app == "opencode"
    ))
    assert quit_after(opencode_pick) == 0
    assert "cannot launch opencode" in capsys.readouterr().out


def test_menu_marks_missing_agents(monkeypatch, quit_after, capsys):
    monkeypatch.setattr("shutil.which", lambda _: None)
    quit_after("0")
    out = capsys.readouterr().out
    assert "[not installed]" in out


def _stub_spec():
    """Fake ModelSpec-shaped object sufficient for selector.launch_cmd."""
    from types import SimpleNamespace
    return SimpleNamespace(
        name="qwen3",
        model_id="Qwen3-8B-GGUF",
        llama_url="http://127.0.0.1:8091",
        daemon_url="http://127.0.0.1:8790",
    )




def test_selector_launches_mimo_with_model_ref():
    from tokirecur.selector import launch_cmd

    plan = launch_cmd("chat", "mimo", _stub_spec())
    assert plan["argv"] == [
        "mimo", "-m", "tokirecur-qwen3/Qwen3-8B-GGUF",
    ]
    assert plan["detach"] is False


def test_launch_agent_uses_selector_launch_cmd(monkeypatch, capsys):
    """The CLI launcher must route through selector.launch_cmd so it
    can never drift from the Textual launcher."""
    from tokirecur import clilauncher as cl

    monkeypatch.setattr("tokirecur.ctl.daemon_reachable", lambda spec: True)
    monkeypatch.setattr(cl, "_active_spec", _stub_spec)
    monkeypatch.setattr(cl.shutil, "which", lambda _: r"C:\bin\opencode.exe")
    monkeypatch.setattr(os, "name", "nt")
    captured = {}

    class FakeProc:
        pid = 1

    monkeypatch.setattr(
        cl.subprocess, "Popen",
        lambda argv, **kw: (
            captured.update(argv=argv, env=kw.get("env")),
            FakeProc(),
        )[1],
    )

    cl._launch_agent("opencode")
    assert captured["argv"] == [
        r"C:\bin\opencode.exe", "-m", "tokirecur-qwen3/Qwen3-8B-GGUF",
    ]
    assert captured["env"] == os.environ


def test_launch_agent_openclaw_falls_back_to_chat_when_gateway_down(monkeypatch, capsys):
    """openclaw TUI needs the gateway; fall back to local chat when it's down."""
    from tokirecur import clilauncher as cl
    from tokirecur import selector

    monkeypatch.setattr("tokirecur.ctl.daemon_reachable", lambda spec: True)
    monkeypatch.setattr(cl, "_active_spec", _stub_spec)
    monkeypatch.setattr(cl.shutil, "which", lambda _: r"C:\bin\openclaw.exe")
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(selector, "_gateway_up", lambda: False)
    captured = {}

    class FakeProc:
        pid = 2

    monkeypatch.setattr(
        cl.subprocess, "Popen",
        lambda argv, **kw: (captured.__setitem__("argv", argv), FakeProc())[1],
    )

    cl._launch_agent("openclaw")
    assert captured["argv"] == [r"C:\bin\openclaw.exe", "chat"]
    assert "falling back" in capsys.readouterr().out


def test_launch_agent_openclaw_uses_tui_when_gateway_up(monkeypatch, capsys):
    from tokirecur import clilauncher as cl
    from tokirecur import selector
    monkeypatch.setattr("tokirecur.ctl.daemon_reachable", lambda spec: True)

    monkeypatch.setattr(cl, "_active_spec", _stub_spec)
    monkeypatch.setattr(cl.shutil, "which", lambda _: r"C:\bin\openclaw.exe")
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(selector, "_gateway_up", lambda: True)
    captured = {}

    class FakeProc:
        pid = 3

    monkeypatch.setattr(
        cl.subprocess, "Popen",
        lambda argv, **kw: (captured.__setitem__("argv", argv), FakeProc())[1],
    )

    cl._launch_agent("openclaw")
    assert captured["argv"] == [r"C:\bin\openclaw.exe", "tui"]


def test_launch_agent_no_active_spec(monkeypatch, capsys):
    from tokirecur import clilauncher as cl

    monkeypatch.setattr(cl, "_active_spec", lambda: None)
    monkeypatch.setattr(os, "name", "nt")
    called = {"popen": False}
    monkeypatch.setattr(
        cl.subprocess, "Popen",
        lambda *a, **kw: called.__setitem__("popen", True),
    )

    cl._launch_agent("opencode")
    assert called["popen"] is False
    assert "no active model" in capsys.readouterr().out


def test_find_binary_path(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: r"C:\tools\opencode.exe")
    assert clilauncher._find_binary("opencode") == r"C:\tools\opencode.exe"


def test_find_binary_windows_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr("shutil.which", lambda _: None)
    npm = tmp_path / "npm"
    npm.mkdir()
    (npm / "codex.cmd").write_text("")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    found = clilauncher._find_binary("codex")
    assert found and found.endswith("codex.cmd")


def test_agent_launch_cmd_wrapper(monkeypatch, capsys):
    """npm/pipx .cmd shims must launch via cmd.exe, not CreateProcess."""
    from tokirecur import clilauncher as cl
    monkeypatch.setattr("tokirecur.ctl.daemon_reachable", lambda spec: True)
    monkeypatch.setattr(cl.shutil, "which", lambda _: r"C:\npm\opencode.cmd")
    monkeypatch.setattr(cl, "_active_spec", _stub_spec)
    monkeypatch.setattr(os, "name", "nt")
    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(argv, **kw):
        captured["argv"] = argv
        captured["creationflags"] = kw.get("creationflags")
        return FakeProc()

    monkeypatch.setattr(cl.subprocess, "Popen", fake_popen)
    cl._launch_agent("opencode")
    assert captured["argv"] == ["cmd.exe", "/c", r"C:\npm\opencode.cmd", "-m",
                                "tokirecur-qwen3/Qwen3-8B-GGUF"]
    assert "pid 4242" in capsys.readouterr().out


def test_agent_launch_exe(monkeypatch, capsys):
    from tokirecur import clilauncher as cl
    monkeypatch.setattr("tokirecur.ctl.daemon_reachable", lambda spec: True)
    monkeypatch.setattr(cl.shutil, "which", lambda _: r"C:\bin\codex.exe")
    monkeypatch.setattr(cl, "_active_spec", _stub_spec)
    monkeypatch.setenv("TOKI_AUTH_TOKEN", "test-token")
    monkeypatch.setattr(os, "name", "nt")
    captured = {}

    class FakeProc:
        pid = 7

    def fake_popen(argv, **kw):
        captured["argv"] = argv
        return FakeProc()

    monkeypatch.setattr(cl.subprocess, "Popen", fake_popen)
    cl._launch_agent("codex")
    assert captured["argv"] == [r"C:\bin\codex.exe"]
    assert "pid 7" in capsys.readouterr().out


def test_switch_model_by_number(monkeypatch, tmp_path, capsys):
    from tokirecur import ctl
    reg = tmp_path / "models.yaml"
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    monkeypatch.setattr(ctl, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", tmp_path / "active")

    import yaml
    # The weights have to exist: a model whose file is gone cannot be
    # started, and the launcher says so rather than trying.
    (tmp_path / "a.gguf").write_bytes(b"x")
    (tmp_path / "b.gguf").write_bytes(b"x")
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "a", "gguf": str(tmp_path / "a.gguf"), "llama_port": 1, "daemon_port": 2,
         "ctx_size": 100, "max_tokens": 90, "model_id": "A-GGUF"},
        {"name": "b", "gguf": str(tmp_path / "b.gguf"), "llama_port": 3, "daemon_port": 4,
         "ctx_size": 100, "max_tokens": 90, "model_id": "B-GGUF"},
    ]}))
    used = []
    monkeypatch.setattr(ctl, "use_model", lambda name, **kw: used.append(name) or ctl.get_model(name))
    monkeypatch.setattr("builtins.input", lambda _: "2")
    clilauncher.switch_model()
    assert used == ["b"]
    assert "b" in capsys.readouterr().out


def test_switch_model_by_name(monkeypatch, tmp_path):
    from tokirecur import ctl
    reg = tmp_path / "models.yaml"
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    monkeypatch.setattr(ctl, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", tmp_path / "active")

    import yaml
    (tmp_path / "a.gguf").write_bytes(b"x")
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "alpha", "gguf": str(tmp_path / "a.gguf"), "llama_port": 1,
         "daemon_port": 2, "ctx_size": 100, "max_tokens": 90, "model_id": "A"},
    ]}))
    used = []
    monkeypatch.setattr(ctl, "use_model", lambda name, **kw: used.append(name) or ctl.get_model(name))
    monkeypatch.setattr("builtins.input", lambda _: "alpha")
    clilauncher.switch_model()
    assert used == ["alpha"]


def test_switch_model_blank_keeps_active(monkeypatch, tmp_path):
    from tokirecur import ctl
    reg = tmp_path / "models.yaml"
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    monkeypatch.setattr(ctl, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", tmp_path / "active")

    import yaml
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "alpha", "gguf": "/tmp/a.gguf", "llama_port": 1,
         "daemon_port": 2, "ctx_size": 100, "max_tokens": 90, "model_id": "A"},
    ]}))
    used = []
    monkeypatch.setattr(ctl, "use_model", lambda name, **kw: used.append(name) or ctl.get_model(name))
    monkeypatch.setattr("builtins.input", lambda _: "")
    clilauncher.switch_model()
    assert used == []


def test_windows_routing_in_selector(monkeypatch):
    """selector.run() on Windows must route to the CLI launcher."""
    from tokirecur import selector
    monkeypatch.setattr(os, "name", "nt")
    ran = {"called": False}

    def fake_launcher():
        ran["called"] = True
        return 0

    monkeypatch.setattr("tokirecur.clilauncher.run_cli_launcher", fake_launcher)
    assert selector.run() == 0
    assert ran["called"]


def test_launch_agent_boots_stack_when_daemon_down(monkeypatch, capsys):
    """0.2.17: launching an agent with the stack down must bring the
    stack up first (which also re-syncs the agent configs), then launch
    against the refreshed spec."""
    from tokirecur import clilauncher as cl
    from tokirecur import ctl

    calls = {"use_model": 0}
    monkeypatch.setattr(cl, "_active_spec", _stub_spec)
    monkeypatch.setattr(cl.shutil, "which", lambda _: r"C:\bin\opencode.exe")
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(ctl, "daemon_reachable", lambda spec: False)
    monkeypatch.setattr(ctl, "use_model",
                        lambda name, **kw: calls.__setitem__("use_model", calls["use_model"] + 1) or ctl.get_model(name))
    monkeypatch.setattr(ctl, "get_model", lambda name: _stub_spec())
    captured = {}

    class FakeProc:
        pid = 9

    monkeypatch.setattr(
        cl.subprocess, "Popen",
        lambda argv, **kw: (captured.__setitem__("argv", argv), FakeProc())[1],
    )

    cl._launch_agent("opencode")
    assert calls["use_model"] == 1
    assert captured["argv"][0] == r"C:\bin\opencode.exe"
    assert "starting 'qwen3' stack" in capsys.readouterr().out
