"""Shared test isolation.

Every unit test must not write into the developer's real home directory.
Client config writers resolve their path from ``Path.home()`` unless an env
var overrides it, so an autouse fixture points the whole client family at a
temp dir. Without this, any test that calls ``sync_cli_configs`` writes real
config files into ``~/.config`` — a failure mode that is invisible until a
later run silently reads a fixture's model or token.

The model registry gets the same treatment, for the same reason. A test that
does ``ctl.load_registry()[0]`` was reading whatever the developer happened
to have registered: green on a box with models, an IndexError on a clean CI
runner, and a different answer on every machine. The registry is state, and
state belongs in the test's tmp dir.
"""

from __future__ import annotations

import os

import pytest
import yaml

from tokirecur import ctl

#: The real Qwen Code version probe, kept so a test that checks the
#: launch flags against the installed binary can use it instead of the
#: stub below.
_real_qwen_major_version = ctl.qwen_major_version


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """A model registry with the default models, inside the test's tmp dir.

    Not autouse: a few tests assert on the *default* registry path and on a
    registry that does not exist yet. Those opt out by not asking for this
    fixture. The tests that just want "a model to exist" ask for it, so
    they behave the same on a developer box and on a clean CI runner.
    """
    from tokirecur import ctl

    reg = tmp_path / "models.yaml"
    reg.write_text(yaml.safe_dump({"models": ctl.DEFAULT_MODELS}, sort_keys=False))
    marker = tmp_path / "active-model"
    # The default models are not always_on, so without an active marker
    # `read_active()` returns None and every "launch the active model" test
    # fails on a machine that has never run `toki model use`.
    marker.write_text(str(ctl.DEFAULT_MODELS[0]["name"]) + "\n")
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    monkeypatch.setattr(ctl, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", marker)
    ctl.invalidate_registry_cache()
    yield reg
    ctl.invalidate_registry_cache()


# Env var -> path under the test's tmp dir. Keep in sync with the
# `*_config_path` helpers in tokirecur.ctl.
#
# Every writer's path is redirected here. A writer left out of this list
# writes into the developer's real home: `sync_cli_configs` runs all of
# them, so one missing name is enough to have the suite rewrite
# ~/.config/opencode/opencode.json with a fixture model over the
# endpoint the person's own model is served on.
_CLIENT_CONFIG_VARS = (
    "MIMOCODE_CONFIG",
    "QWEN_CONFIG",
    "FACTORY_CONFIG",
    "DSH_HOME",
    "GROK_HOME",
    "HERMES_HOME",
    "KILO_CONFIG",
    "PI_CODING_AGENT_DIR",
    "KIMI_CODE_HOME",
    "OMP_CONFIG_DIR",
    "TOKI_CLAUDE_SETTINGS",
    "TOKI_CLAUDE_CONFIG",
    "OPENCODE_CONFIG",
    "OPENCLAW_CONFIG",
    "OPENCLAW_WORKSPACE_DIR",
    "CLINE_TOKIRECUR_CONFIG",
    "CODEX_HOME",
    "MINIMAX_DATA_DIR",
    "MAVIS_DATA_DIR",
)


@pytest.fixture(autouse=True)
def _isolate_client_configs(tmp_path, monkeypatch):
    for name in _CLIENT_CONFIG_VARS:
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    # A real token in the ambient env would leak into config assertions.
    if "TOKI_AUTH_TOKEN" in os.environ:
        monkeypatch.setenv("TOKI_AUTH_TOKEN", "test-token")
    # The Qwen Code version decides which settings section the writer
    # fills and which --model form the launcher passes, so a developer's
    # installed qwen would change the output from one machine to the
    # next. Tests that care about the 1.0 shape set the version
    # themselves.
    monkeypatch.setattr("tokirecur.ctl.qwen_major_version", lambda: None)


@pytest.fixture
def real_qwen_version(monkeypatch):
    """The installed Qwen Code's version, for flag checks.

    The launcher picks its flags from the version of the binary that is
    actually there, so a check of those flags against that binary has
    to ask the real probe rather than the stub.
    """
    monkeypatch.setattr(
        "tokirecur.ctl.qwen_major_version", _real_qwen_major_version
    )
