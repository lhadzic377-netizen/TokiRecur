"""Platform process and launch helpers.

Windows differs from POSIX in three ways that quietly break a launcher:

* ``os.kill(pid, 0)`` is the POSIX idiom for "is this alive?"; on
  Windows ``os.kill`` is ``TerminateProcess`` for any signal except the
  two console events, so asking the question would *kill* the process.
* ``os.execvpe`` cannot run a ``.cmd`` shim (npm/pip console scripts on
  Windows are ``.cmd`` files), and ``CreateProcess`` cannot either.
* ``start_new_session`` is a silent no-op, so a child keeps the
  launcher's console and dies with Ctrl+C or a closed window.

Everything that needs to know this lives here, so the launcher, the
model lifecycle and the CLI all agree.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence

#: True when running on Windows. A module-level flag (rather than a
#: fresh ``os.name`` check at every call site) so the POSIX test suite
#: can exercise the Windows branches by patching one attribute.
IS_WINDOWS = os.name == "nt"

# Windows creation flags. Spelled out because ``subprocess`` only
# exposes them on Windows and this module is imported on POSIX too.
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008

#: Commands that mean "this program". Launching them through a console
#: script shim is fragile (the shim may not be on PATH, or may be a
#: ``.cmd``); run the module with the running interpreter instead.
_SELF_COMMANDS = frozenset({"toki", "tokirecur"})


def pid_alive(pid: int) -> bool:
    """Is ``pid`` running? Never signals or kills it.

    On Windows the answer comes from ``OpenProcess`` +
    ``GetExitCodeProcess``; on POSIX from ``os.kill(pid, 0)``.
    """
    if pid <= 0:
        return False
    if IS_WINDOWS:
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            open_process = kernel32.OpenProcess
            open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            open_process.restype = wintypes.HANDLE
            get_exit_code = kernel32.GetExitCodeProcess
            get_exit_code.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)
            ]
            get_exit_code.restype = wintypes.BOOL
            close_handle = kernel32.CloseHandle
            close_handle.argtypes = [wintypes.HANDLE]
            close_handle.restype = wintypes.BOOL
            handle = open_process(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            try:
                code = wintypes.DWORD()
                if not get_exit_code(handle, ctypes.byref(code)):
                    return False
                return code.value == STILL_ACTIVE
            finally:
                close_handle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def detach_kwargs(
    *,
    no_console: bool = True,
    start_new_session: bool = True,
) -> dict:
    """``Popen`` kwargs that detach a spawned service from the caller.

    Windows: a new process group so Ctrl+C does not reach the child,
    plus ``CREATE_NO_WINDOW`` so a console app does not flash a window
    (and does not die when the launching console is closed). POSIX:
    ``start_new_session``, exactly as before.
    """
    if IS_WINDOWS:
        flags = CREATE_NEW_PROCESS_GROUP
        if no_console:
            flags |= CREATE_NO_WINDOW
        return {"creationflags": flags}
    return {"start_new_session": start_new_session}


def format_cmdline(argv: Sequence[str]) -> str:
    """Render ``argv`` the way the platform's shell would need it."""
    if IS_WINDOWS:
        return subprocess.list2cmdline(list(argv))
    return shlex.join(argv)


def resolve_argv(argv: Sequence[str]) -> list[str]:
    """Make ``argv`` directly runnable with ``subprocess`` here.

    POSIX: returned unchanged; ``execvpe`` resolves PATH itself.

    Windows:

    * ``toki`` / ``tokirecur`` become ``python -m tokirecur.cli`` — the
      running interpreter already has the package installed, so this
      works no matter where the console-script shim ended up.
    * the executable is resolved with ``shutil.which`` (which honours
      ``PATHEXT``, unlike ``CreateProcess``);
    * a ``.cmd`` / ``.bat`` shim is run through ``cmd.exe /c`` because
      batch files are not executables.
    """
    out = list(argv)
    if not out or not IS_WINDOWS:
        return out
    if out[0] in _SELF_COMMANDS:
        return [sys.executable, "-m", "tokirecur.cli", *out[1:]]
    found = shutil.which(out[0])
    if found:
        out[0] = found
    if out[0].lower().endswith((".cmd", ".bat")):
        comspec = os.environ.get("COMSPEC") or "cmd.exe"
        return [comspec, "/c", *out]
    return out
