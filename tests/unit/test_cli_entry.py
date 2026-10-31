"""The `toki` entry points.

The Windows wrappers run ``python -m tokirecur.cli`` (not the console
script shim), so that module must exist and behave exactly like the
script. A missing ``cli/__main__.py`` is invisible on Linux where the
script is always installed — and fatal on the Windows wrapper.
"""

from __future__ import annotations

import subprocess
import sys


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tokirecur.cli", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_module_entry_point_prints_version():
    proc = _run("--version")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().startswith("toki, version ")


def test_module_entry_point_help_names_the_program():
    """`Usage:  [OPTIONS]` (empty name) is what a bare click context
    renders; the wrapper would show it to every user."""
    proc = _run("--help")
    assert proc.returncode == 0, proc.stderr
    assert "Usage: toki " in proc.stdout


def test_main_callable_matches_module_entry_point():
    """Both entry points are the same CLI."""
    script = subprocess.run(
        [sys.executable, "-c", "from tokirecur.cli import main; raise SystemExit(main())", "--version"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert script.returncode == 0, script.stderr
    assert script.stdout == _run("--version").stdout
