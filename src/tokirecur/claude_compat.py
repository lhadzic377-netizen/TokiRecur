"""Claude Code's API-key consent, and the 401 it causes when it is missing.

Claude Code sends the ``ANTHROPIC_API_KEY`` from the environment only when
that key is on its approved list (``customApiKeyResponses.approved`` in
``~/.claude.json``). A key it has never seen gets a prompt — "Do you want
to use this API key?" — and a key the user declines is remembered as
rejected and is never sent again.

When the key is not sent, Claude Code falls back to the next credential it
can find. On a machine that also has a cloud provider configured in
``~/.claude/settings.json`` — an ``ANTHROPIC_AUTH_TOKEN`` belonging to
somebody else — that fallback is what goes on the wire, to our daemon,
which answers ``401 {"detail": "invalid token"}``. The symptom is Claude
Code printing "Please run /login" and retrying ten times against a local
server that is working perfectly, with no hint that the problem is the
credential rather than the server.

The local route does not use an API key at all: it uses the auth token,
which needs no consent and is always sent (see ``ctl.write_claude_config``).
What is left for this module is the repair: a machine where our token was
offered as an API key and declined keeps a rejected fingerprint, and that
state is worth clearing — and worth saying out loud, which is what
``toki doctor`` does with it.

Claude Code identifies a key by its last 20 characters, so that is the
fingerprint written into its config.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

#: Claude Code truncates the key to its last 20 characters when it
#: records a decision about it.
FINGERPRINT_LENGTH = 20


def api_key_fingerprint(token: str) -> str:
    """The way Claude Code identifies an API key: its last 20 characters."""
    return (token or "").strip()[-FINGERPRINT_LENGTH:]


def claude_config_path() -> Path:
    """Claude Code's global config file (``~/.claude.json``).

    ``TOKI_CLAUDE_CONFIG`` overrides the path, the same way the other
    client config writers are redirected in tests.
    """
    override = os.environ.get("TOKI_CLAUDE_CONFIG", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".claude.json"


def _read_config(path: Path) -> dict | None:
    """The config as a dict, or None when it is missing or unreadable."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def api_key_status(token: str, path: Path | None = None) -> str:
    """How Claude Code sees this token: approved, rejected, new or unknown.

    "unknown" means Claude Code has no config yet — it creates one itself
    and asks about the key the first time it sees it.
    """
    data = _read_config(path or claude_config_path())
    if data is None:
        return "unknown"
    responses = data.get("customApiKeyResponses")
    if not isinstance(responses, dict):
        return "new"
    fingerprint = api_key_fingerprint(token)
    approved = responses.get("approved") or []
    rejected = responses.get("rejected") or []
    if fingerprint in approved:
        return "approved"
    if fingerprint in rejected:
        return "rejected"
    return "new"


def approve_api_key(token: str, path: Path | None = None) -> bool:
    """Record the token as an approved custom API key for Claude Code.

    Returns True when the config changed. A missing or unparsable config is
    not an error: Claude Code writes its own, and a key it has never seen
    costs one prompt. The approval is also what un-does a previous
    rejection, which is how a machine ends up with a working daemon and a
    Claude Code that refuses to talk to it.
    """
    token = (token or "").strip()
    if not token:
        return False
    cfg_path = path or claude_config_path()
    data = _read_config(cfg_path)
    if data is None:
        return False
    responses = data.get("customApiKeyResponses")
    if not isinstance(responses, dict):
        responses = {}
    approved = [x for x in responses.get("approved") or [] if isinstance(x, str)]
    rejected = [x for x in responses.get("rejected") or [] if isinstance(x, str)]
    fingerprint = api_key_fingerprint(token)
    if fingerprint in approved and fingerprint not in rejected:
        return False
    if fingerprint not in approved:
        approved.append(fingerprint)
    rejected = [x for x in rejected if x != fingerprint]
    data["customApiKeyResponses"] = {"approved": approved, "rejected": rejected}
    try:
        _write_config(cfg_path, data)
    except OSError:
        return False
    return True


def _write_config(path: Path, data: dict) -> None:
    """Write the config back atomically, keeping it private."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.toki-tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)
