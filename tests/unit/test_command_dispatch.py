"""A command in the help has to run.

The dispatch used to check the first argument against a hand-written list
of the known subcommands, and the list had drifted from the commands that
were actually registered: `self-upgrade` was in the help and not in the
list, so it printed the help and exited; and `shell` was in the list and
not registered, so it was a "no such command" error.

The registered commands are the truth, and the dispatch asks them.
"""

import pathlib
import sys

import pytest

from tokirecur.cli import cli, main


@pytest.mark.parametrize("name", sorted(cli.commands))
def test_every_command_in_the_help_answers(name, monkeypatch):
    """Each registered command gets past the dispatch.

    `--help` on a command never runs the command's body, so this proves
    the dispatch reaches it and not that the command works.
    """
    monkeypatch.setattr(sys, "argv", ["toki", name, "--help"])
    try:
        code = main()
    except SystemExit as exit:
        code = exit.code
    assert code in (0, None), f"`toki {name} --help` did not answer"


def test_an_unknown_command_still_shows_the_help(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["toki", "not-a-command"])
    try:
        code = main()
    except SystemExit as exit:
        code = exit.code
    assert code == 2


def test_the_dispatch_asks_the_registered_commands(monkeypatch):
    """The list of known commands is the registration, not a copy of it."""
    from tokirecur import cli as cli_mod

    source = pathlib.Path(cli_mod.__file__).read_text()
    assert "_SUBCOMMANDS" not in source, (
        "a hand-written list of subcommands is how `self-upgrade` stopped "
        "working"
    )
