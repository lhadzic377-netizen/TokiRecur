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
    assert "Web Dashboards" in out
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
    assert quit_after("7") == 0
    assert opened == list(clilauncher.WEB_URLS)
    assert "dashboards" in capsys.readouterr().out


def test_missing_agent_warns(monkeypatch, quit_after, capsys):
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert quit_after("3") == 0
    assert "not found in PATH: opencode" in capsys.readouterr().out


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


def test_launch_agent_uses_selector_launch_cmd(monkeypatch, capsys):
    """The CLI launcher must route through selector.launch_cmd so it
    can never drift from the Textual launcher."""
    from tokirecur import clilauncher as cl

    monkeypatch.setattr(cl, "_active_spec", _stub_spec)
    monkeypatch.setattr(cl.shutil, "which", lambda _: r"C:\bin\opencode.exe")
    monkeypatch.setattr(os, "name", "nt")
    captured = {}

    class FakeProc:
        pid = 1

    monkeypatch.setattr(
        cl.subprocess, "Popen",
        lambda argv, **kw: (captured.__setitem__("argv", argv), FakeProc())[1],
    )

    cl._launch_agent("opencode")
    assert captured["argv"] == [
        r"C:\bin\opencode.exe", "-m", "tokirecur-qwen3/Qwen3-8B-GGUF",
    ]


def test_launch_agent_openclaw_falls_back_to_chat_when_gateway_down(monkeypatch, capsys):
    """openclaw TUI needs the gateway; fall back to local chat when it's down."""
    from tokirecur import clilauncher as cl
    from tokirecur import selector

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
    monkeypatch.setattr(cl.shutil, "which", lambda _: r"C:\bin\codex.exe")
    monkeypatch.setattr(cl, "_active_spec", _stub_spec)
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
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "a", "gguf": "/tmp/a.gguf", "llama_port": 1, "daemon_port": 2,
         "ctx_size": 100, "max_tokens": 90, "model_id": "A-GGUF"},
        {"name": "b", "gguf": "/tmp/b.gguf", "llama_port": 3, "daemon_port": 4,
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
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "alpha", "gguf": "/tmp/a.gguf", "llama_port": 1,
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
