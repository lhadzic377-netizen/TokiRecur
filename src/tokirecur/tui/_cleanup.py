"""Defensive terminal cleanup for toki TUIs.

Textual's ``driver.stop_application_mode()`` emits the right escape
sequences on exit (alt-screen leave, cursor on, mouse off, bracketed
paste off, kitty kbd protocol off, termios restore). But it only runs
when Textual's own ``finally`` block reaches that point. On a busy
box the 3-second ``_poll_statuses`` interval can block the event loop
while the ExitApp message sits in the queue; an exception during
shutdown, a SIGTERM from logout, or a SIGHUP on SSH disconnect all
skip Textual's cleanup. The result: the terminal is left in alt-screen
/ raw / hidden-cursor mode and the user's subsequent keystrokes
aren't echoed.

This module registers an atexit handler (always runs on normal exit
and Python-level exceptions) and POSIX signal handlers for SIGINT /
SIGTERM / SIGHUP (run on signal-driven exit). Together they cover
every exit path and guarantee a sane terminal state on the way out.
"""
from __future__ import annotations

import atexit
import os
import signal
import sys

# All escape sequences we write on cleanup. Order matters: cursor-on
# before SGR-reset so the user sees a visible caret; alt-screen-leave
# LAST so we don't redraw onto a screen that the terminal is about to
# discard. See textual/drivers/linux_driver.py for the upstream
# equivalents.
_RESET_SEQ = (
    "\x1b[?25h",        # DECTCEM — cursor visible
    "\x1b[?1000l",      # mouse click reporting off
    "\x1b[?1003l",      # mouse motion reporting off
    "\x1b[?1015l",      # urxvt mouse off
    "\x1b[?1006l",      # SGR mouse off
    "\x1b[?2004l",      # bracketed paste off
    "\x1b[<u",          # kitty keyboard protocol — push all keys
    "\x1b[?7h",         # line wrap on
    "\x1b[?2048l",      # in-band window resize off
    "\x1b[?1049l",      # alt screen leave
    "\x1b[?1004l",      # focus-in/out reporting off
    "\x1b[0m",          # SGR reset
)

_installed = False
_original_termios = None
_original_windows_mode: tuple[int, int] | None = None


def _stdout_fileno() -> int | None:
    """Best-effort stdout file descriptor. Returns None when stdout
    has been redirected (pytest capture, pipes) so cleanup becomes a
    no-op rather than crashing on a missing tty."""
    try:
        return sys.stdout.fileno()
    except (OSError, ValueError, AttributeError):
        return None


def _snapshot_terminal_state(fileno: int) -> None:
    """Capture whatever termios / Windows console mode we need to
    restore later. Failure is silent — the worst case is we don't
    restore termios, which means the terminal stays in whatever mode
    Textual left it in (still better than nothing)."""
    global _original_termios, _original_windows_mode
    if os.name == "posix":
        try:
            import termios
            _original_termios = termios.tcgetattr(fileno)
        except (OSError, termios.error):
            _original_termios = None
        return
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.windll.kernel32
            handle = k32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = wintypes.DWORD()
            if k32.GetConsoleMode(handle, ctypes.byref(mode)):
                _original_windows_mode = (handle.value, int(mode.value))
        except Exception:
            _original_windows_mode = None


def restore_now() -> None:
    """Restore terminal to a sane state. Idempotent — calling twice
    is harmless and writes the same escape sequences again. Safe to
    call when stdout isn't a tty (the escape codes go nowhere)."""
    if os.name == "posix" and _original_termios is not None:
        try:
            import termios
            fileno = _stdout_fileno()
            if fileno is not None:
                termios.tcsetattr(
                    fileno, termios.TCSANOW, _original_termios,
                )
        except (OSError, termios.error):
            pass

    if os.name == "nt" and _original_windows_mode is not None:
        try:
            import ctypes
            handle_val, mode = _original_windows_mode
            ctypes.windll.kernel32.SetConsoleMode(ctypes.c_void_p(handle_val), mode)
        except Exception:
            pass

    try:
        sys.stdout.write("".join(_RESET_SEQ))
        sys.stdout.flush()
    except (OSError, ValueError, AttributeError):
        pass


def _on_signal(signum: int, frame: object) -> None:
    """POSIX signal handler: clean up and re-raise the default."""
    restore_now()
    try:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    except (OSError, ValueError):
        os._exit(1)


def _install_signal_handlers() -> None:
    """SIGINT / SIGTERM / SIGHUP. Skips any signal that already has
    a non-default handler — pytest, IDE debuggers, and parent shells
    sometimes set SIG_IGN on SIGHUP and we'd be a bad citizen to
    override it."""
    if os.name != "posix":
        return
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            prev = signal.getsignal(sig)
        except (ValueError, OSError):
            # Not on main thread, or unsupported.
            continue
        if prev not in (signal.SIG_DFL, None):
            continue
        try:
            signal.signal(sig, _on_signal)
        except (OSError, ValueError):
            pass


def install() -> None:
    """Snapshot terminal state, register atexit + signal cleanup.
    Idempotent — safe to call from every TUI entry point."""
    global _installed
    if _installed:
        return

    fileno = _stdout_fileno()
    if fileno is None:
        # stdout is captured / redirected; cleanup would be a no-op
        # AND would crash on a missing tty. Bail before registering
        # anything.
        return

    _snapshot_terminal_state(fileno)
    atexit.register(restore_now)
    _install_signal_handlers()
    _installed = True
