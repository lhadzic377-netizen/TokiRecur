"""Windows launch/process helpers.

These run on POSIX by patching ``winproc.IS_WINDOWS``: the point is the
decision table (what gets run, how it is detached), not a real Windows
kernel.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tokirecur import winproc


def test_pid_alive_true_for_self():
    assert winproc.pid_alive(os.getpid()) is True


def test_pid_alive_false_for_dead_pid():
    # PIDs are not reused that fast; 4 million is far above the Linux
    # pid_max default and Windows allocates from a much smaller pool.
    assert winproc.pid_alive(4_000_000) is False


@pytest.mark.parametrize("pid", [0, -1])
def test_pid_alive_rejects_nonpositive(pid):
    assert winproc.pid_alive(pid) is False


def test_resolve_argv_posix_is_identity(monkeypatch):
    monkeypatch.setattr(winproc, "IS_WINDOWS", False)
    argv = ["opencode", "--model", "x"]
    assert winproc.resolve_argv(argv) == argv


def test_resolve_argv_windows_self_launch(monkeypatch):
    """`toki` must become the running interpreter, not a PATH shim."""
    monkeypatch.setattr(winproc, "IS_WINDOWS", True)
    monkeypatch.setattr(sys, "executable", r"C:\venv\Scripts\python.exe")
    assert winproc.resolve_argv(["toki", "tui", "--raw-url", "http://x"]) == [
        r"C:\venv\Scripts\python.exe",
        "-m",
        "tokirecur.cli",
        "tui",
        "--raw-url",
        "http://x",
    ]


def test_resolve_argv_windows_wraps_cmd_shim(monkeypatch):
    """CreateProcess cannot run a .cmd; cmd.exe /c can."""
    monkeypatch.setattr(winproc, "IS_WINDOWS", True)
    monkeypatch.setenv("COMSPEC", r"C:\Windows\system32\cmd.exe")
    monkeypatch.setattr(
        winproc.shutil, "which", lambda name: r"C:\npm\opencode.cmd"
    )
    assert winproc.resolve_argv(["opencode", "--model", "x"]) == [
        r"C:\Windows\system32\cmd.exe",
        "/c",
        r"C:\npm\opencode.cmd",
        "--model",
        "x",
    ]


def test_resolve_argv_windows_plain_exe(monkeypatch):
    monkeypatch.setattr(winproc, "IS_WINDOWS", True)
    monkeypatch.setattr(winproc.shutil, "which", lambda name: r"C:\bin\codex.exe")
    assert winproc.resolve_argv(["codex"]) == [r"C:\bin\codex.exe"]


def test_resolve_argv_windows_unresolved_left_alone(monkeypatch):
    """Nothing found -> unchanged, so Popen raises the real error."""
    monkeypatch.setattr(winproc, "IS_WINDOWS", True)
    monkeypatch.setattr(winproc.shutil, "which", lambda name: None)
    assert winproc.resolve_argv(["nope", "arg"]) == ["nope", "arg"]


def test_detach_kwargs_windows_hides_console(monkeypatch):
    monkeypatch.setattr(winproc, "IS_WINDOWS", True)
    assert winproc.detach_kwargs() == {
        "creationflags": winproc.CREATE_NEW_PROCESS_GROUP | winproc.CREATE_NO_WINDOW
    }


def test_detach_kwargs_windows_can_keep_console(monkeypatch):
    monkeypatch.setattr(winproc, "IS_WINDOWS", True)
    assert winproc.detach_kwargs(no_console=False) == {
        "creationflags": winproc.CREATE_NEW_PROCESS_GROUP
    }


def test_detach_kwargs_posix_uses_start_new_session(monkeypatch):
    monkeypatch.setattr(winproc, "IS_WINDOWS", False)
    assert winproc.detach_kwargs() == {"start_new_session": True}
    assert winproc.detach_kwargs(start_new_session=False) == {
        "start_new_session": False
    }


def test_format_cmdline_matches_platform(monkeypatch):
    monkeypatch.setattr(winproc, "IS_WINDOWS", False)
    assert winproc.format_cmdline(["toki", "a b"]) == "toki 'a b'"
    monkeypatch.setattr(winproc, "IS_WINDOWS", True)
    assert winproc.format_cmdline(["toki", "a b"]) == subprocess.list2cmdline(
        ["toki", "a b"]
    )
