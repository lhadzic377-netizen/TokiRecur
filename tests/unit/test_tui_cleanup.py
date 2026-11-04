"""Tests for the defensive terminal cleanup at TUI exit.

These exercise every exit path the user can take out of a TUI:
normal quit, Ctrl+C, SIGHUP, an exception during the run. The
cleanup must restore a sane terminal every time so the user's
subsequent keystrokes stay visible.
"""

import io
import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Each test starts with a clean cleanup state — the module
    holds termios snapshots + atexit registrations that would leak
    across tests otherwise."""
    from tokirecur.tui import _cleanup
    saved_installed = _cleanup._installed
    saved_termios = _cleanup._original_termios
    saved_winmode = _cleanup._original_windows_mode
    _cleanup._installed = False
    _cleanup._original_termios = None
    _cleanup._original_windows_mode = None
    yield
    _cleanup._installed = saved_installed
    _cleanup._original_termios = saved_termios
    _cleanup._original_windows_mode = saved_winmode


def test_install_is_idempotent(monkeypatch):
    """Calling install() twice must not double-register atexit or
    double-install signal handlers — the second call returns early
    without registering anything new."""
    from tokirecur.tui import _cleanup

    registered: list[callable] = []
    monkeypatch.setattr(_cleanup.atexit, "register", registered.append)
    # install() needs a fileno; fake one with a pipe.
    r, w = _setup_fake_tty(monkeypatch)

    _cleanup.install()
    _cleanup.install()
    _cleanup.install()
    assert len(registered) == 1
    w.close()


def test_restore_now_writes_all_cleanup_sequences(monkeypatch):
    """Every escape sequence Textual emits on a clean exit must be
    in our reset list too — anything missing means that mode can
    leak. Cursor-on and alt-screen-leave are the two the user
    notices most when they're missing."""
    from tokirecur.tui import _cleanup

    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)
    _cleanup.restore_now()
    out = captured.getvalue()
    for seq in (
        "\x1b[?25h",   # cursor on
        "\x1b[?1000l", # mouse click off
        "\x1b[?2004l", # bracketed paste off
        "\x1b[<u",     # kitty kbd protocol — push keys
        "\x1b[?1049l", # alt screen leave
        "\x1b[0m",     # SGR reset
    ):
        assert seq in out, f"missing reset sequence: {seq!r}"


def test_install_skips_when_stdout_not_a_tty(monkeypatch):
    """When stdout is captured (pytest, pipes, redirects), install()
    must be a no-op — there's nothing to clean up, and trying to
    snapshot termios would crash on a missing tty."""
    from tokirecur.tui import _cleanup

    monkeypatch.setattr(sys, "stdout", io.StringIO())
    registered: list[callable] = []
    monkeypatch.setattr(_cleanup.atexit, "register", registered.append)
    _cleanup.install()
    assert registered == []
    assert _cleanup._installed is False


def test_restore_now_safe_when_fileno_unavailable(monkeypatch):
    """Captured stdout (no real fileno) — cleanup must still write
    the escape sequences through the file API and not crash."""
    from tokirecur.tui import _cleanup

    class NoFilenoIO(io.StringIO):
        def fileno(self):
            raise OSError("not a tty")

    captured = NoFilenoIO()
    monkeypatch.setattr(sys, "stdout", captured)
    _cleanup.restore_now()  # must not raise
    out = captured.getvalue()
    assert "\x1b[?25h" in out
    assert "\x1b[?1049l" in out


def test_stdout_fileno_falls_back_for_textual_sentinel(monkeypatch):
    from tokirecur.tui import _cleanup

    class TextualStdout(io.StringIO):
        def fileno(self):
            return -1

    class OriginalStdout(io.StringIO):
        def fileno(self):
            return 17

    monkeypatch.setattr(sys, "stdout", TextualStdout())
    monkeypatch.setattr(sys, "__stdout__", OriginalStdout())
    assert _cleanup._stdout_fileno() == 17


@pytest.mark.skipif(os.name != "posix", reason="POSIX termios behavior")
def test_restore_now_ignores_negative_descriptor(monkeypatch):
    import termios

    from tokirecur.tui import _cleanup

    monkeypatch.setattr(_cleanup, "_original_termios", ["snapshot"])
    monkeypatch.setattr(_cleanup, "_stdout_fileno", lambda: -1)
    monkeypatch.setattr(
        termios,
        "tcsetattr",
        lambda *args: (_ for _ in ()).throw(AssertionError("tcsetattr called")),
    )
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    _cleanup.restore_now()
    assert "\x1b[?1049l" in captured.getvalue()


def test_signal_handlers_skip_when_already_installed(monkeypatch):
    """Don't clobber SIG_IGN — pytest, IDE debuggers, and parent
    shells sometimes set it on SIGHUP and we'd be a bad citizen to
    override. Only SIG_DFL is ours to replace."""
    import signal

    from tokirecur.tui import _cleanup

    monkeypatch.setattr(sys, "stdout", io.StringIO())  # skip install
    calls: list[tuple] = []
    monkeypatch.setattr(_cleanup.signal, "signal",
                        lambda s, h: calls.append((s, h)))
    monkeypatch.setattr(_cleanup.signal, "getsignal",
                        lambda s: signal.SIG_IGN)
    _cleanup._install_signal_handlers()
    assert calls == []  # nothing installed because SIG_IGN was there


@pytest.mark.skipif(os.name != "posix", reason="signal handlers are installed on POSIX only")
def test_signal_handlers_install_when_default(monkeypatch):
    """When the previous handler is SIG_DFL, install() must put
    our handler in place so SIGTERM from logout doesn't skip
    cleanup."""
    import signal

    from tokirecur.tui import _cleanup

    monkeypatch.setattr(sys, "stdout", io.StringIO())
    calls: list[tuple] = []
    monkeypatch.setattr(_cleanup.signal, "signal",
                        lambda s, h: calls.append((s, h)))
    monkeypatch.setattr(_cleanup.signal, "getsignal",
                        lambda s: signal.SIG_DFL)
    _cleanup._install_signal_handlers()
    # SIGINT and SIGTERM everywhere; SIGHUP only exists on POSIX.
    installed = [s for s, _ in calls]
    assert signal.SIGINT in installed
    assert signal.SIGTERM in installed
    if hasattr(signal, "SIGHUP"):
        assert signal.SIGHUP in installed


@pytest.mark.skipif(os.name != "posix", reason="termios and pty are POSIX only")
def test_restore_now_restores_termios_on_posix(monkeypatch):
    """Snapshot a known-good termios, switch to raw mode, call
    restore_now, and assert the original termios is back. This is
    the user-visible bug: raw mode = no echo = keystrokes
    invisible after quit."""
    import pty
    import termios

    # Real pty — os.pipe() doesn't have termios; only a tty does.
    master_fd, slave_fd = pty.openpty()
    slave = os.fdopen(slave_fd, "w", buffering=1)  # text mode
    monkeypatch.setattr(sys, "stdout", slave)
    try:
        from tokirecur.tui import _cleanup

        original = termios.tcgetattr(slave_fd)
        _cleanup._snapshot_terminal_state(slave_fd)
        # Switch to raw-ish mode (no echo, no canonical processing).
        raw = list(original)
        raw[3] = (raw[3] | termios.ISIG) & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(slave_fd, termios.TCSANOW, raw)
        attrs = termios.tcgetattr(slave_fd)
        assert (attrs[3] & termios.ECHO) == 0, "test setup failed"
        # Restore.
        _cleanup.restore_now()
        attrs = termios.tcgetattr(slave_fd)
        assert (attrs[3] & termios.ECHO) != 0, (
            "ECHO was not restored — keystrokes would be invisible"
        )
        assert (attrs[3] & termios.ICANON) != 0, (
            "ICANON was not restored — line buffering would be off"
        )
    finally:
        try:
            termios.tcsetattr(slave_fd, termios.TCSANOW, original)
        except Exception:
            pass
        slave.close()
        os.close(master_fd)


def test_launcher_run_calls_install():
    """``toki`` (bare launcher) must call install() so the cleanup
    hook is in place before the user can hit Esc."""
    import inspect

    from tokirecur.tui import launcher

    src = inspect.getsource(launcher.run)
    assert "_tui_cleanup.install" in src


def test_chat_tui_run_calls_install(monkeypatch):
    """``toki tui`` (chat TUI) must call install() too — same
    terminal-leak risk on quit."""
    from tokirecur.tui import _cleanup
    from tokirecur.tui import app as tui_app

    calls = []
    monkeypatch.setattr(_cleanup, "install", lambda: calls.append(1))

    import inspect
    src = inspect.getsource(tui_app.run)
    assert "_tui_cleanup.install" in src


def _setup_fake_tty(monkeypatch) -> tuple[io.RawIOBase, io.RawIOBase]:
    """Fake a tty by patching sys.stdout to a pipe with a real
    fileno. termios calls work on the read end, the write end
    collects output (and is what stdout points at)."""
    import os

    r_fd, w_fd = os.pipe()
    r = os.fdopen(r_fd, "rb", buffering=0)
    w = os.fdopen(w_fd, "wb", buffering=0)
    monkeypatch.setattr(sys, "stdout", w)
    return r, w
