"""A placeholder is not a secret.

The README used to show ``"apiKey": "demo-token"`` in a copy-and-paste
opencode config, and the auth file on a machine that followed it held
"demo-token" — which is what the daemon accepted, and what every client
was told to send. The README's own documentation was the credential.

A well-known value in the auth file (or in TOKI_AUTH_TOKEN) is now
treated as no token at all and replaced with a real one, so the token
that protects the daemon is never one that is written down.
"""

import pytest

from tokirecur import ctl


@pytest.fixture
def auth_home(tmp_path, monkeypatch):
    """An auth file of our own, and no env var in the way."""
    home = tmp_path / "home"
    (home / ".config" / "tokirecur").mkdir(parents=True)
    monkeypatch.setattr(ctl.Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("TOKI_AUTH_TOKEN", raising=False)
    return home / ".config" / "tokirecur" / "auth"


def test_a_placeholder_in_the_file_is_replaced(auth_home):
    auth_home.write_text("demo-token\n")
    token = ctl.configured_auth_token()
    assert token and token != "demo-token"
    assert len(token) >= 32, "a real token, not a placeholder"
    # And the file now holds the real one, so the daemon and the clients
    # agree on it.
    assert auth_home.read_text().strip() == token


def test_a_placeholder_in_the_environment_is_ignored(auth_home):
    auth_home.write_text("a-real-looking-token-value-here\n")
    monkey = pytest.MonkeyPatch()
    monkey.setenv("TOKI_AUTH_TOKEN", "demo-token")
    try:
        assert ctl.configured_auth_token() == "a-real-looking-token-value-here"
    finally:
        monkey.undo()


def test_a_real_token_is_left_alone(auth_home):
    auth_home.write_text("s3cret-token-value\n")
    assert ctl.configured_auth_token() == "s3cret-token-value"
    assert auth_home.read_text().strip() == "s3cret-token-value"


def test_the_environment_wins_when_it_is_real(auth_home):
    auth_home.write_text("from-the-file\n")
    monkey = pytest.MonkeyPatch()
    monkey.setenv("TOKI_AUTH_TOKEN", "from-the-environment")
    try:
        assert ctl.configured_auth_token() == "from-the-environment"
    finally:
        monkey.undo()


def test_no_token_anywhere_is_not_a_placeholder(auth_home):
    assert ctl.configured_auth_token() is None
    generated = ctl.auth_token()
    assert generated and generated != "demo-token"


def test_the_daemon_and_the_clients_agree(auth_home):
    """One implementation, so they cannot drift."""
    from tokirecur.api import app as api_app

    auth_home.write_text("demo-token\n")
    assert ctl.configured_auth_token() == api_app._get_auth_token()
    assert ctl.auth_token() == api_app._get_auth_token()
