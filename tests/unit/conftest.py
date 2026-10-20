"""Shared test isolation.

Every unit test must not write into the developer's real home directory.
Client config writers resolve their path from ``Path.home()`` unless an env
var overrides it, so an autouse fixture points the whole client family at a
temp dir. Without this, any test that calls ``sync_cli_configs`` writes real
config files into ``~/.config`` — a failure mode that is invisible until a
later run silently reads a fixture's model or token.
"""

from __future__ import annotations

import os

import pytest

# Env var -> path under the test's tmp dir. Keep in sync with the
# `*_config_path` helpers in tokirecur.ctl.
#
# CLINE_TOKIRECUR_CONFIG and CODEX_HOME are deliberately absent: the cline
# and codex launch-plan tests assert the *default* home-based path, so they
# need these unset. Their own fixtures redirect them when they write.
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
)


@pytest.fixture(autouse=True)
def _isolate_client_configs(tmp_path, monkeypatch):
    for name in _CLIENT_CONFIG_VARS:
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    # A real token in the ambient env would leak into config assertions.
    if "TOKI_AUTH_TOKEN" in os.environ:
        monkeypatch.setenv("TOKI_AUTH_TOKEN", "test-token")
