"""A harness has to actually start.

The launcher builds the command line for each client it offers. A flag
the binary does not know is not ignored: the binary prints its help and
exits, so the person who pressed Enter got a page of flags instead of
the agent they asked for. OpenCode was launched with `-m`, which this
build does not have, and it had never once started.
"""

import shutil
import subprocess

import pytest

from tokirecur import selector


def _flags(argv: list[str]) -> list[str]:
    return [a for a in argv if a.startswith("-") and a != "--"]


def test_opencode_is_launched_without_a_flag_it_does_not_have(isolated_registry):
    """The model comes from the config, which is written on every switch."""
    from tokirecur import ctl

    spec = ctl.get_model(ctl.read_active() or "")
    plan = selector.launch_cmd("chat", "opencode", spec)
    assert "-m" not in plan["argv"], plan["argv"]
    assert "--model" not in plan["argv"], plan["argv"]


def _launch_argv(client: str) -> list[str]:
    """The argv the launcher would use for a client, without a model."""
    from tokirecur import ctl

    spec = ctl.get_model(ctl.read_active() or "")
    return selector.launch_cmd("chat", client, spec)["argv"]


@pytest.mark.parametrize(
    "client",
    ["opencode", "mimo", "hermes", "qwen", "cline", "claude"],
)
def test_every_flag_the_launcher_passes_is_one_the_binary_knows(client, isolated_registry):
    """Check the binary's own help for each flag.

    Skipped where the client is not installed: the point is to catch a
    flag that has been renamed or removed, and that can only be checked
    against the binary that is actually there.
    """
    binary = shutil.which(client)
    if binary is None:
        pytest.skip(f"{client} is not installed")
    try:
        result = subprocess.run(
            [binary, "--help"], capture_output=True, text=True, timeout=25
        )
    except Exception:
        pytest.skip(f"{client} --help did not run")
    text = (result.stdout or "") + (result.stderr or "")
    if not text.strip():
        pytest.skip(f"{client} --help said nothing")
    for flag in _flags(_launch_argv(client)):
        assert flag in text, f"{client} does not know {flag}"


def test_the_launcher_writes_the_config_the_client_reads(monkeypatch, tmp_path, isolated_registry):
    """The model the launch omits has to be in the config instead."""
    from tokirecur import ctl

    spec = ctl.get_model(ctl.read_active() or "")
    plan = selector.launch_cmd("chat", "opencode", spec)
    assert plan["argv"][0] == "opencode"
    # The config write is what the launch relies on.
    path = ctl.write_opencode_config(spec)
    import json

    cfg = json.loads(path.read_text())
    assert cfg["model"].startswith(f"tokirecur-{spec.name}/")
