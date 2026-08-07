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
    assert "Web Dashboards" in out
    assert "0) Quit" in out


def test_model_line_none(monkeypatch):
    monkeypatch.setattr("tokirecur.ctl.read_active", lambda: None)
    assert clilauncher._model_line() == "model: none set (chat will ask)"


def test_model_line_active(monkeypatch):
    monkeypatch.setattr("tokirecur.ctl.read_active", lambda: "my-model")
    assert clilauncher._model_line() == "model: my-model"


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
    assert quit_after("6") == 0
    assert opened == list(clilauncher.WEB_URLS)
    assert "dashboards" in capsys.readouterr().out


def test_missing_agent_warns(monkeypatch, quit_after, capsys):
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert quit_after("2") == 0
    assert "not found in PATH: opencode" in capsys.readouterr().out


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
