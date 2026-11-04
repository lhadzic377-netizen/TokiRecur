"""Cross-platform compatibility tests.

Covers the helpers that must behave correctly on Linux, macOS, and
Windows (process signalling, port->pid lookup, browser opening).
"""

from __future__ import annotations

import os
import signal
import sys
from types import SimpleNamespace

import pytest

from tokirecur import ctl, lifecycle, selector, winproc


def _clear_port_cache() -> None:
    ctl._PORT_PID_CACHE.clear()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups only")
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
    "exited during startup"). CREATE_NO_WINDOW also keeps a console app
    from flashing a window and from dying with the launching console."""
    monkeypatch.setattr(winproc, "IS_WINDOWS", True)
    kw = ctl._detach_kwargs()
    assert kw == {
        "creationflags": winproc.CREATE_NEW_PROCESS_GROUP | winproc.CREATE_NO_WINDOW
    }


def test_detach_kwargs_posix_uses_start_new_session(monkeypatch):
    monkeypatch.setattr(winproc, "IS_WINDOWS", False)
    monkeypatch.setattr(ctl, "_DETACH", True)
    assert ctl._detach_kwargs() == {"start_new_session": True}
    monkeypatch.setattr(ctl, "_DETACH", False)
    assert ctl._detach_kwargs() == {"start_new_session": False}



@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups only")
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


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals only")
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
    monkeypatch.setattr(winproc, "IS_WINDOWS", True)
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
    assert tasks == [["taskkill", "/pid", "12345", "/f", "/t"]]
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


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS opens open")
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


@pytest.mark.skipif(sys.platform != "linux", reason="Linux opens xdg-open")
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


def test_is_alive_zero_or_negative(monkeypatch):
    assert ctl._is_alive(0) is False
    assert ctl._is_alive(-1) is False


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals only")
def test_is_alive_posix_uses_kill(monkeypatch):
    monkeypatch.setattr(ctl.os, "name", "posix")
    seen = {}
    monkeypatch.setattr(ctl.os, "kill",
                        lambda pid, sig: seen.update(pid=pid, sig=sig))

    class FakePath:
        def __init__(self, p):
            pass

        def read_bytes(self):
            return b"/usr/bin/llama-server -m x.gguf\x00"

    # Patch the cmdline read at the module level instead — simpler and
    # deterministic regardless of what pid 12345 happens to be on this
    # machine (an unrelated live process must not flip the result).
    monkeypatch.setattr(
        "tokirecur.ctl.Path",
        lambda p: type("P", (), {
            "read_bytes": staticmethod(
                lambda: b"/usr/bin/llama-server -m x.gguf\x00"
            ),
        })(),
    )
    assert ctl._is_alive(12345) is True
    assert seen == {"pid": 12345, "sig": 0}


def test_is_alive_posix_rejects_recycled_pid(monkeypatch):
    """A live pid whose cmdline is NOT llama-server/uvicorn/tokirecur
    (e.g. chromium got the recycled pid) must report dead so a stale
    pid file never masks a crashed stack."""
    monkeypatch.setattr(ctl.os, "name", "posix")
    monkeypatch.setattr(ctl.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(
        "tokirecur.ctl.Path",
        lambda p: type("P", (), {
            "read_bytes": staticmethod(
                lambda: b"/usr/lib/chromium/chromium --flag\x00"
            ),
        })(),
    )
    assert ctl._is_alive(12345) is False


def test_is_alive_posix_returns_false_on_oserror(monkeypatch):
    monkeypatch.setattr(ctl.os, "name", "posix")

    def fake_kill(pid, sig):
        raise OSError("no such process")

    monkeypatch.setattr(ctl.os, "kill", fake_kill)
    assert ctl._is_alive(12345) is False


def _patch_fake_ctypes(monkeypatch, open_proc, get_exit_code, close_handle):
    """Inject a fake ctypes module whose kernel32 functions are the
    supplied callables. Each callable is wrapped so it has the
    argtypes/restype attributes `_is_alive` assigns.

    The real ctypes WinDLL exposes functions whose attributes can be
    set; on a plain Python class those become bound methods, which
    reject attribute assignment. We attach the functions as INSTANCE
    attributes to sidestep the descriptor protocol.
    """
    import sys as _sys

    def _wrap(fn, argtypes, restype):
        fn.argtypes = argtypes
        fn.restype = restype
        return fn

    open_proc = _wrap(open_proc, [int, bool, int], int)
    get_exit_code = _wrap(get_exit_code, [int, object], bool)
    close_handle = _wrap(close_handle, [int], bool)

    class FakeWintypes:
        DWORD = _Ref
        BOOL = bool
        HANDLE = int

    class FakeWinDLL:
        pass

    def _new_dll():
        instance = FakeWinDLL()
        instance.OpenProcess = open_proc
        instance.GetExitCodeProcess = get_exit_code
        instance.CloseHandle = close_handle
        return instance

    fake_ctypes = type("C", (), {
        "WinDLL": lambda name, use_last_error=False: _new_dll(),
        "wintypes": FakeWintypes(),
        "byref": lambda x: x,
        "POINTER": lambda t: t,
    })
    monkeypatch.setitem(_sys.modules, "ctypes", fake_ctypes)


class _Ref:
    """Minimal ctypes.DWORD stand-in: defaults to 0 and exposes a
    mutable .value attribute so byref/GetExitCodeProcess interact with
    it the way ctypes.c_uint would."""
    __slots__ = ("value",)

    def __init__(self, value: int = 0) -> None:
        self.value = value


def test_is_alive_windows_uses_open_process(monkeypatch):
    """On Windows the probe must use OpenProcess + GetExitCodeProcess,
    not os.kill(pid, 0), which terminates the process there."""
    monkeypatch.setattr(winproc, "IS_WINDOWS", True)

    calls = []

    def open_proc(access, inherit, pid):
        calls.append(("OpenProcess", pid))
        return 0xCAFE

    def get_exit_code(handle, code_ref):
        calls.append(("GetExitCodeProcess", handle))
        code_ref.value = 259  # STILL_ACTIVE
        return True

    def close_handle(handle):
        calls.append(("CloseHandle", handle))
        return True

    _patch_fake_ctypes(monkeypatch, open_proc, get_exit_code, close_handle)

    assert ctl._is_alive(12345) is True
    assert ("OpenProcess", 12345) in calls
    assert ("CloseHandle", 0xCAFE) in calls


def test_is_alive_windows_returns_false_when_open_process_fails(monkeypatch):
    """OpenProcess returns 0 when the pid is gone or we lack access."""
    monkeypatch.setattr(winproc, "IS_WINDOWS", True)
    _patch_fake_ctypes(monkeypatch,
                       lambda *a, **k: 0,
                       lambda *a, **k: True,
                       lambda *a, **k: True)
    assert ctl._is_alive(99999) is False


def test_is_alive_windows_returns_false_on_get_exit_code_failure(monkeypatch):
    monkeypatch.setattr(winproc, "IS_WINDOWS", True)
    _patch_fake_ctypes(monkeypatch,
                       lambda *a, **k: 0xCAFE,
                       lambda *a, **k: False,
                       lambda *a, **k: True)
    assert ctl._is_alive(12345) is False
