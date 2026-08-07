"""Cross-platform compatibility tests.

Covers the helpers that must behave correctly on Linux, macOS, and
Windows (process signalling, port->pid lookup, browser opening).
"""

from __future__ import annotations

import signal
import subprocess
from types import SimpleNamespace

from tokirecur import ctl, lifecycle, selector


def _clear_port_cache() -> None:
    ctl._PORT_PID_CACHE.clear()


def test_signal_proc_uses_killpg_when_available(monkeypatch):
    seen = {}

    def fake_killpg(pgid, sig):
        seen["killpg"] = (pgid, sig)
        raise ProcessLookupError("no such process")

    monkeypatch.setattr(ctl.os, "killpg", fake_killpg)
    monkeypatch.setattr(ctl.os, "getpgid", lambda pid: 999)
    ctl._signal_proc(123, signal.SIGTERM)
    assert seen["killpg"] == (999, signal.SIGTERM)


def test_signal_proc_falls_back_to_kill_without_killpg(monkeypatch):
    # Windows has no os.killpg/os.getpgid at all.
    monkeypatch.delattr(ctl.os, "killpg", raising=False)
    monkeypatch.delattr(ctl.os, "getpgid", raising=False)
    seen = {}
    monkeypatch.setattr(ctl.os, "kill", lambda pid, sig: seen.update(pid=pid, sig=sig))
    ctl._signal_proc(123, signal.SIGTERM)
    assert seen == {"pid": 123, "sig": signal.SIGTERM}


def test_detach_kwargs_windows_uses_new_process_group(monkeypatch):
    """start_new_session is a no-op on Windows, so spawned services must
    get CREATE_NEW_PROCESS_GROUP instead — otherwise Ctrl+C in the
    launching console kills llama-server/daemon with it (reported as
    "exited during startup")."""
    monkeypatch.setattr(ctl.os, "name", "nt")
    if not hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
    kw = ctl._detach_kwargs()
    assert kw == {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}


def test_detach_kwargs_posix_uses_start_new_session(monkeypatch):
    monkeypatch.setattr(ctl.os, "name", "posix")
    monkeypatch.setattr(ctl, "_DETACH", True)
    assert ctl._detach_kwargs() == {"start_new_session": True}
    monkeypatch.setattr(ctl, "_DETACH", False)
    assert ctl._detach_kwargs() == {"start_new_session": False}



def test_signal_proc_falls_back_to_kill_when_killpg_raises(monkeypatch):
    # POSIX fallback path: process group gone, signal the pid directly.
    monkeypatch.setattr(ctl.os, "killpg", lambda pgid, sig: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(ctl.os, "getpgid", lambda pid: 999)
    seen = {}
    monkeypatch.setattr(ctl.os, "kill", lambda pid, sig: seen.update(pid=pid, sig=sig))
    ctl._signal_proc(123, signal.SIGTERM)
    assert seen == {"pid": 123, "sig": signal.SIGTERM}


def test_stop_pidfile_no_killpg_windows_style(monkeypatch, tmp_path):
    # Simulate Windows (no killpg) end-to-end without touching a real pid.
    monkeypatch.delattr(ctl.os, "killpg", raising=False)
    monkeypatch.delattr(ctl.os, "getpgid", raising=False)
    monkeypatch.setattr(ctl, "_read_pid", lambda path: 12345)
    alive = iter([True, False])

    def fake_is_alive(pid):
        return next(alive)

    monkeypatch.setattr(ctl, "_is_alive", fake_is_alive)
    seen = []
    monkeypatch.setattr(ctl.os, "kill", lambda pid, sig: seen.append(sig))
    pidfile = tmp_path / "llama.pid"
    pidfile.write_text("12345")

    assert ctl._stop_pidfile(pidfile) is True
    assert seen == [signal.SIGTERM]
    assert not pidfile.exists()


def test_lifecycle_stop_pid_no_killpg(monkeypatch, tmp_path):
    monkeypatch.delattr(lifecycle.os, "killpg", raising=False)
    monkeypatch.delattr(lifecycle.os, "getpgid", raising=False)
    monkeypatch.setattr(lifecycle, "_read_pid", lambda path: 12345)
    alive = iter([True, False])

    def fake_is_alive(pid):
        return next(alive)

    monkeypatch.setattr(lifecycle, "_is_alive", fake_is_alive)
    seen = []
    monkeypatch.setattr(lifecycle.os, "kill", lambda pid, sig: seen.append(sig))
    pidfile = tmp_path / "tokid.pid"
    pidfile.write_text("12345")

    assert lifecycle.stop_pid(pidfile, "toki daemon") is True
    assert seen == [signal.SIGTERM]


def test_hard_kill_windows_uses_taskkill(monkeypatch):
    monkeypatch.setattr(ctl.os, "name", "nt")
    seen = []
    monkeypatch.setattr(
        ctl.subprocess, "run", lambda argv, **kw: seen.append(argv),
    )
    ctl._hard_kill(12345)
    assert seen == [["taskkill", "/pid", "12345", "/f"]]


def test_hard_kill_posix_uses_sigkill(monkeypatch):
    monkeypatch.setattr(ctl.os, "name", "posix")
    monkeypatch.setattr(
        ctl.os, "killpg",
        lambda pgid, sig: (_ for _ in ()).throw(ProcessLookupError()),
    )
    monkeypatch.setattr(ctl.os, "getpgid", lambda pid: 999)
    seen = {}
    monkeypatch.setattr(ctl.os, "kill", lambda pid, sig: seen.update(pid=pid, sig=sig))
    ctl._hard_kill(12345)
    assert seen == {"pid": 12345, "sig": signal.SIGKILL}


def test_stop_pidfile_hard_kill_windows_style(monkeypatch, tmp_path):
    # Graceful stop times out -> hard kill must use taskkill, never
    # signal.SIGKILL (which does not exist on Windows Python).
    monkeypatch.setattr(ctl.os, "name", "nt")
    monkeypatch.delattr(ctl.os, "killpg", raising=False)
    monkeypatch.delattr(ctl.os, "getpgid", raising=False)
    monkeypatch.setattr(ctl, "_read_pid", lambda path: 12345)
    monkeypatch.setattr(ctl, "_is_alive", lambda pid: True)
    seen = []
    monkeypatch.setattr(ctl.os, "kill", lambda pid, sig: seen.append(sig))
    tasks = []
    monkeypatch.setattr(
        ctl.subprocess, "run", lambda argv, **kw: tasks.append(argv),
    )
    pidfile = tmp_path / "llama.pid"
    pidfile.write_text("12345")

    assert ctl._stop_pidfile(pidfile, grace=0.01) is True
    assert seen == [signal.SIGTERM]
    assert tasks == [["taskkill", "/pid", "12345", "/f"]]
    assert not pidfile.exists()


def test_lifecycle_stop_pid_hard_kill_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(lifecycle.os, "name", "nt")
    monkeypatch.delattr(lifecycle.os, "killpg", raising=False)
    monkeypatch.delattr(lifecycle.os, "getpgid", raising=False)
    monkeypatch.setattr(lifecycle, "_read_pid", lambda path: 12345)
    monkeypatch.setattr(lifecycle, "_is_alive", lambda pid: True)
    seen = []
    monkeypatch.setattr(lifecycle.os, "kill", lambda pid, sig: seen.append(sig))
    tasks = []
    monkeypatch.setattr(
        lifecycle.subprocess, "run", lambda argv, **kw: tasks.append(argv),
    )
    pidfile = tmp_path / "tokid.pid"
    pidfile.write_text("12345")

    assert lifecycle.stop_pid(pidfile, "toki daemon", grace=0.01) is True
    assert seen == [signal.SIGTERM]
    assert tasks == [["taskkill", "/pid", "12345", "/f"]]
    assert not pidfile.exists()


def test_pid_on_port_linux_ss(monkeypatch):
    _clear_port_cache()
    monkeypatch.setattr(ctl.sys, "platform", "linux")
    monkeypatch.setattr(
        ctl.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout=(
                "LISTEN 0 4096 127.0.0.1:8787 0.0.0.0:* "
                'users:(("uvicorn",pid=4321,fd=6))\n'
            )
        ),
    )
    assert ctl._pid_on_port(8787) == 4321


def test_pid_on_port_macos_lsof(monkeypatch):
    _clear_port_cache()
    monkeypatch.setattr(ctl.sys, "platform", "darwin")
    monkeypatch.setattr(
        ctl.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout=(
                "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\n"
                "python3 5555 luke 12u IPv4 0x1234 0t0 TCP 127.0.0.1:8787 (LISTEN)\n"
            )
        ),
    )
    assert ctl._pid_on_port(8787) == 5555


def test_pid_on_port_windows_netstat(monkeypatch):
    _clear_port_cache()
    monkeypatch.setattr(ctl.sys, "platform", "win32")
    monkeypatch.setattr(
        ctl.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout=(
                "  TCP    127.0.0.1:8787    127.0.0.1:0    LISTENING       7777\n"
                "  TCP    127.0.0.1:8788    127.0.0.1:0    LISTENING       8888\n"
            )
        ),
    )
    assert ctl._pid_on_port(8787) == 7777


def test_pid_on_port_none_when_unparsable(monkeypatch):
    _clear_port_cache()
    monkeypatch.setattr(ctl.sys, "platform", "linux")
    monkeypatch.setattr(
        ctl.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="nothing here\n")
    )
    assert ctl._pid_on_port(8787) is None


def test_open_browser_macos_uses_open(monkeypatch):
    monkeypatch.setattr(selector.sys, "platform", "darwin")
    seen = []
    monkeypatch.setattr(
        selector.subprocess,
        "Popen",
        lambda *a, **k: seen.append((a[0], k.get("start_new_session"))),
    )
    selector._open_browser("http://127.0.0.1:7777")
    assert seen[0][0] == ["open", "http://127.0.0.1:7777"]
    assert seen[0][1] is True


def test_open_browser_linux_uses_xdg_open(monkeypatch):
    monkeypatch.setattr(selector.sys, "platform", "linux")
    seen = []
    monkeypatch.setattr(
        selector.subprocess,
        "Popen",
        lambda *a, **k: seen.append((a[0], k.get("start_new_session"))),
    )
    selector._open_browser("http://127.0.0.1:7777")
    assert seen[0][0] == ["xdg-open", "http://127.0.0.1:7777"]
