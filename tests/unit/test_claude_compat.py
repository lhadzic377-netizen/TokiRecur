"""Claude Code's API-key consent, and the 401 it causes when it is missing.

Claude Code only sends the ``ANTHROPIC_API_KEY`` it is given when that key
is on its approved list. When it is not, it falls back to another
credential and the local daemon answers 401 — a failure that looks like a
dead server. The local route therefore uses the auth token, which needs no
consent, and these tests cover that, plus the repair of a key Claude Code
declined earlier.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokirecur import claude_compat
from tokirecur.claude_compat import (
    api_key_fingerprint,
    api_key_status,
    approve_api_key,
)

TOKEN = "2IoCsm4xp409xnYSFArQZgMp7lIJ8JadjW-E1a35xYw"
FINGERPRINT = "p7lIJ8JadjW-E1a35xYw"


def _write_config(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def test_fingerprint_is_the_last_twenty_characters():
    assert api_key_fingerprint(TOKEN) == FINGERPRINT
    assert len(api_key_fingerprint(TOKEN)) == 20


def test_fingerprint_ignores_surrounding_whitespace():
    assert api_key_fingerprint(f"  {TOKEN}\n") == FINGERPRINT


def test_approve_adds_the_key_to_the_approved_list(tmp_path):
    cfg = tmp_path / ".claude.json"
    _write_config(cfg, {"customApiKeyResponses": {"approved": [], "rejected": []}})

    assert approve_api_key(TOKEN, cfg) is True

    data = json.loads(cfg.read_text())
    assert data["customApiKeyResponses"]["approved"] == [FINGERPRINT]
    assert data["customApiKeyResponses"]["rejected"] == []


def test_approve_undoes_a_previous_rejection(tmp_path):
    """The broken state: the key was declined once, so it is never sent."""
    cfg = tmp_path / ".claude.json"
    _write_config(cfg, {
        "customApiKeyResponses": {
            "approved": ["sk-no-auth"],
            "rejected": [FINGERPRINT],
        },
    })

    assert approve_api_key(TOKEN, cfg) is True

    data = json.loads(cfg.read_text())
    assert data["customApiKeyResponses"]["approved"] == ["sk-no-auth", FINGERPRINT]
    assert data["customApiKeyResponses"]["rejected"] == []


def test_approve_keeps_the_rest_of_the_config(tmp_path):
    cfg = tmp_path / ".claude.json"
    _write_config(cfg, {
        "numStartups": 60,
        "customApiKeyResponses": {"approved": [], "rejected": []},
        "projects": {"/home/lukehadzic": {"hasTrustDialogAccepted": True}},
    })

    approve_api_key(TOKEN, cfg)

    data = json.loads(cfg.read_text())
    assert data["numStartups"] == 60
    assert data["projects"] == {"/home/lukehadzic": {"hasTrustDialogAccepted": True}}


def test_approve_is_idempotent(tmp_path):
    cfg = tmp_path / ".claude.json"
    _write_config(cfg, {"customApiKeyResponses": {"approved": [], "rejected": []}})

    assert approve_api_key(TOKEN, cfg) is True
    assert approve_api_key(TOKEN, cfg) is False


def test_approve_leaves_a_missing_config_alone(tmp_path):
    cfg = tmp_path / ".claude.json"
    assert approve_api_key(TOKEN, cfg) is False
    assert not cfg.exists()


def test_approve_leaves_an_unparsable_config_alone(tmp_path):
    cfg = tmp_path / ".claude.json"
    cfg.write_text("{not json", encoding="utf-8")
    assert approve_api_key(TOKEN, cfg) is False
    assert cfg.read_text() == "{not json"


def test_status_names_the_state(tmp_path):
    cfg = tmp_path / ".claude.json"
    assert api_key_status(TOKEN, cfg) == "unknown"

    _write_config(cfg, {"customApiKeyResponses": {"approved": [], "rejected": []}})
    assert api_key_status(TOKEN, cfg) == "new"

    _write_config(cfg, {
        "customApiKeyResponses": {"approved": [], "rejected": [FINGERPRINT]},
    })
    assert api_key_status(TOKEN, cfg) == "rejected"

    approve_api_key(TOKEN, cfg)
    assert api_key_status(TOKEN, cfg) == "approved"


def test_config_path_honours_the_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TOKI_CLAUDE_CONFIG", str(tmp_path / "claude.json"))
    assert claude_compat.claude_config_path() == tmp_path / "claude.json"


def test_claude_settings_carry_our_token_as_the_fallback(monkeypatch, tmp_path):
    """Claude Code falls back to ANTHROPIC_AUTH_TOKEN when the API key is
    not approved. That fallback has to be ours, not another provider's."""
    from tokirecur import ctl

    monkeypatch.setenv("TOKI_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("TOKI_CLAUDE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("TOKI_CLAUDE_CONFIG", str(tmp_path / ".claude.json"))

    spec = ctl.ModelSpec(
        name="test-model",
        label="Test",
        gguf="/tmp/test.gguf",
        llama_port=9001,
        daemon_port=9002,
        ctx_size=8192,
        parallel=1,
        kv="f16",
        max_tokens=4096,
        model_id="test-model",
    )
    ctl.write_claude_config(spec)

    settings = json.loads((tmp_path / "settings.json").read_text())
    assert settings["env"]["ANTHROPIC_AUTH_TOKEN"] == TOKEN
    # No API key: Claude Code would ask about one it has never seen, and
    # the auth token is sent without a prompt.
    assert "ANTHROPIC_API_KEY" not in settings["env"]

    # And the key is registered with Claude Code, so it is sent at all.
    _write_config(tmp_path / ".claude.json", {
        "customApiKeyResponses": {"approved": [], "rejected": [FINGERPRINT]},
    })
    ctl.write_claude_config(spec)
    data = json.loads((tmp_path / ".claude.json").read_text())
    assert FINGERPRINT in data["customApiKeyResponses"]["approved"]
    assert FINGERPRINT not in data["customApiKeyResponses"]["rejected"]


def test_launch_plan_gives_claude_our_auth_token(monkeypatch, tmp_path):
    """An inherited ANTHROPIC_AUTH_TOKEN must not reach the daemon."""
    from tokirecur import ctl, selector

    monkeypatch.setenv("TOKI_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("TOKI_CLAUDE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("TOKI_CLAUDE_CONFIG", str(tmp_path / ".claude.json"))
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "somebody-elses-token")

    spec = ctl.ModelSpec(
        name="test-model",
        label="Test",
        gguf="/tmp/test.gguf",
        llama_port=9001,
        daemon_port=9002,
        ctx_size=8192,
        parallel=1,
        kv="f16",
        max_tokens=4096,
        model_id="test-model",
    )
    plan = selector.launch_cmd("chat", "claude", spec)

    assert plan["env"]["ANTHROPIC_AUTH_TOKEN"] == TOKEN
    assert "ANTHROPIC_API_KEY" not in plan["env"]


@pytest.mark.parametrize("status", ["approved", "rejected", "new", "unknown"])
def test_doctor_reports_a_claude_code_problem(tmp_path, monkeypatch, status):
    """A key Claude Code will not send is a 401 waiting to happen, and
    `toki doctor` has to say so."""
    from tokirecur import diagnose

    cfg = tmp_path / ".claude.json"
    responses = {"approved": [], "rejected": []}
    if status == "approved":
        responses["approved"] = [FINGERPRINT]
    elif status == "rejected":
        responses["rejected"] = [FINGERPRINT]
    if status != "unknown":
        _write_config(cfg, {"customApiKeyResponses": responses})
    monkeypatch.setenv("TOKI_CLAUDE_CONFIG", str(cfg))
    monkeypatch.setenv("TOKI_AUTH_TOKEN", TOKEN)

    report = diagnose.run()
    if status == "rejected":
        assert "claude code:" in report
    else:
        assert "claude code:" not in report
