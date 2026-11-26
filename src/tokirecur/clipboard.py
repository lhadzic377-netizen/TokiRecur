"""Put text on the desktop clipboard, not just on the terminal's OSC 52.

Textual's ``App.copy_to_clipboard`` writes an OSC 52 escape sequence,
which GNOME Terminal (and most other terminals) ignore: the app says
"copied", but a paste into another app gets whatever was there before.
That is the wrong answer for the public-server screen, whose whole point
is a link you can paste into another app's custom-endpoint form.

The desktop's own clipboard is tried first:

* ``wl-copy`` on Wayland, ``xclip`` / ``xsel`` on X11, ``pbcopy`` on
  macOS, PowerShell on Windows - all one-shot commands that feed the
  text on stdin and let the desktop own it afterwards;
* otherwise, on Linux with a DISPLAY, a small detached tkinter process
  takes ownership of the X11 selection. GNOME's XWayland selection is
  what a Wayland browser reads, so this works on the Wayland desktops
  that have no wl-copy, and it needs no new dependency: tkinter ships
  with Python. The helper stays alive briefly because a clipboard is
  served, not stored - an owner that exits immediately serves nothing.

``copy`` returns the mechanism that took the text, or ``"none"`` when no
desktop clipboard is available; in that case the caller can still fall
back to OSC 52.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

#: Own the selection long enough for a person to paste it. A clipboard
#: owner that exits as soon as it has set the text serves nobody.
_OWNER_SECONDS = 300

# Set with tkinter, then serve the selection until the timer expires.
# The text is passed as an argument so the child needs no pipe kept open.
_TK_OWNER = """
import sys
import tkinter as tk

root = tk.Tk()
root.withdraw()
root.clipboard_clear()
root.clipboard_append(sys.argv[1])
root.update()
root.after({ms}, root.destroy)
root.mainloop()
"""


def _run_commands(text: str) -> str | None:
    """The one-shot clipboard commands, in desktop order."""
    commands = (
        ("wl-copy", ["wl-copy"]),
        ("xclip", ["xclip", "-selection", "clipboard"]),
        ("xsel", ["xsel", "--clipboard", "--input"]),
        ("pbcopy", ["pbcopy"]),
        ("powershell", ["powershell", "-NoProfile", "-Command",
                        "Set-Clipboard -Value $input"]),
    )
    for name, argv in commands:
        if not shutil.which(argv[0]):
            continue
        try:
            result = subprocess.run(
                argv, input=text, text=True, timeout=10, check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0:
            return name
    return None


def _tk_owner(text: str) -> str | None:
    """A detached tkinter process that owns the selection briefly."""
    if sys.platform.startswith("win") or not os.environ.get("DISPLAY"):
        return None
    try:
        subprocess.Popen(
            [sys.executable, "-c", _TK_OWNER.format(ms=_OWNER_SECONDS * 1000), text],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (OSError, ValueError):
        return None
    return "tkinter"


def copy(text: str) -> str:
    """Copy ``text`` to the desktop clipboard.

    Returns the name of the mechanism that took it, or ``"none"``. A
    caller that gets ``"none"`` can still write OSC 52 for terminals that
    support it.
    """
    if not text:
        return "none"
    taken = _run_commands(text)
    if taken is not None:
        return taken
    return _tk_owner(text) or "none"
