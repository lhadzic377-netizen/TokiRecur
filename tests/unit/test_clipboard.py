"""Copying a link means the real desktop clipboard, not an OSC 52 toast.

Textual's copy writes OSC 52, which most terminals ignore. The helper
tries the desktop's own clipboard first and only reports "none" when
there is genuinely nothing it can use.
"""


from tokirecur import clipboard


def test_a_one_shot_command_takes_the_text(monkeypatch):
    """wl-copy is the first stop on a Wayland desktop."""
    seen: dict = {}

    class Result:
        returncode = 0

    def which(name):
        return "/usr/bin/" + name if name == "wl-copy" else None

    def run(argv, **kwargs):
        seen["argv"] = argv
        seen["text"] = kwargs.get("input")
        return Result()

    monkeypatch.setattr(clipboard.shutil, "which", which)
    monkeypatch.setattr(clipboard.subprocess, "run", run)

    assert clipboard.copy("https://example.com/v1") == "wl-copy"
    assert seen["argv"] == ["wl-copy"]
    assert seen["text"] == "https://example.com/v1"


def test_a_failed_command_falls_through_to_the_next(monkeypatch):
    class Failed:
        returncode = 1

    class Ok:
        returncode = 0

    results = [Failed(), Ok()]

    def which(name):
        return "/usr/bin/" + name if name in ("wl-copy", "xclip") else None

    def run(argv, **kwargs):
        return results.pop(0)

    monkeypatch.setattr(clipboard.shutil, "which", which)
    monkeypatch.setattr(clipboard.subprocess, "run", run)
    monkeypatch.setattr(clipboard, "_tk_owner", lambda text: "tkinter")

    assert clipboard.copy("text") == "xclip"


def test_tkinter_owns_the_selection_when_no_tool_exists(monkeypatch):
    """No wl-copy or xclip: a detached tkinter owner serves the paste."""
    seen: dict = {}

    class Popen:
        def __init__(self, argv, **kwargs):
            seen["argv"] = argv
            seen["kwargs"] = kwargs

    monkeypatch.setattr(clipboard.shutil, "which", lambda name: None)
    monkeypatch.setattr(clipboard.subprocess, "Popen", Popen)
    monkeypatch.setenv("DISPLAY", ":0")

    assert clipboard.copy("https://example.com/v1") == "tkinter"
    assert seen["argv"][-1] == "https://example.com/v1"
    assert seen["kwargs"]["start_new_session"] is True
    assert "-c" in seen["argv"]


def test_no_desktop_is_reported_honestly(monkeypatch):
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: None)
    monkeypatch.delenv("DISPLAY", raising=False)
    assert clipboard.copy("text") == "none"
    assert clipboard.copy("") == "none"


def test_os52_is_a_terminal_fallback(monkeypatch):
    """When the desktop copy returns 'none', Textual's OSC 52 still works
    in terminals that support it - the caller decides, not this module."""
    monkeypatch.setattr(clipboard, "_run_commands", lambda text: None)
    monkeypatch.setattr(clipboard, "_tk_owner", lambda text: None)
    assert clipboard.copy("text") == "none"


def test_tkinter_is_not_claimed_when_python_lacks_it(monkeypatch):
    """A Python without tkinter cannot own the selection."""
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: None)
    monkeypatch.setattr(clipboard.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setenv("DISPLAY", ":0")

    assert clipboard.copy("text") == "none"
