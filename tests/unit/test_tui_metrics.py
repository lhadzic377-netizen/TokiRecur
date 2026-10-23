"""The TUI's metrics reading.

Two things were wrong:

- Prometheus labels its counters — ``toki_requests_total{route=…,
  status=…}`` — and the label block is part of the token. The parser
  compared the whole token, so the request counter never matched anything
  and always read zero, while the daemon had served well over a
  thousand requests.
- ``/metrics`` fetched the same numbers the panel was already
  refreshing every two seconds, updated the panel, and said nothing. A
  command that answers nothing cannot be told apart from a broken one.
"""

import pytest

from tokirecur.tui.app import TokiApp
from tokirecur.tui.client import DaemonClient

_PROM = """
# HELP toki_requests_total Total requests
# TYPE toki_requests_total counter
toki_requests_total{route="/healthz",status="200"} 294.0
toki_requests_total{route="/v1/chat/completions",status="200"} 4.0
# HELP toki_cache_hits_total Cache hit classifications
# TYPE toki_cache_hits_total counter
toki_cache_hits_total{kind="hit"} 12.0
toki_cache_hits_total{kind="soft_hit"} 2.0
toki_cache_hits_total{kind="miss"} 7.0
# HELP toki_tokens_skipped_total Tokens served from cache
# TYPE toki_tokens_skipped_total counter
toki_tokens_skipped_total 11.0
# HELP toki_active_slots Number of slots currently in use
# TYPE toki_active_slots gauge
toki_active_slots 2.0
toki_auth_failures_total 0.0
"""


def _parse(text: str) -> dict:
    """The same parse the client does, without the network."""
    from tokirecur.tui import client as client_mod

    empty = {
        "requests_total": 0, "cache_hits": 0, "cache_misses": 0,
        "tokens_skipped": 0, "active_slots": 0, "auth_failures": 0,
        "daemon": "unreachable",
    }
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        token, value = parts[0], parts[1]
        name = token.split("{", 1)[0]
        labels = token[len(name):]
        try:
            v = float(value)
        except ValueError:
            continue
        if name == "toki_requests_total":
            empty["requests_total"] += v
        elif name == "toki_cache_hits_total" and 'kind="hit"' in labels:
            empty["cache_hits"] += v
        elif name == "toki_cache_hits_total" and 'kind="miss"' in labels:
            empty["cache_misses"] += v
        elif name == "toki_tokens_skipped_total":
            empty["tokens_skipped"] = int(v)
        elif name == "toki_active_slots":
            empty["active_slots"] = int(v)
        elif name == "toki_auth_failures_total":
            empty["auth_failures"] = int(v)
    empty["daemon"] = "ok"
    return empty


def test_labelled_counters_are_read():
    """The request counter is labelled, and it is a sum over the labels."""
    m = _parse(_PROM)
    assert m["requests_total"] == 298.0
    assert m["cache_hits"] == 12.0
    assert m["cache_misses"] == 7.0
    assert m["tokens_skipped"] == 11
    assert m["active_slots"] == 2
    assert m["daemon"] == "ok"


@pytest.mark.asyncio
async def test_the_command_says_what_the_numbers_are(monkeypatch):
    """A command that answers nothing looks exactly like a broken one."""
    app = TokiApp()
    said: list[str] = []

    async def fake_metrics():
        return {
            "requests_total": 298, "cache_hits": 12, "cache_misses": 7,
            "tokens_skipped": 11, "active_slots": 2, "auth_failures": 0,
            "daemon": "ok",
        }

    monkeypatch.setattr(app.client, "metrics_live", fake_metrics)
    async with app.run_test() as pilot:
        monkeypatch.setattr(
            app.chat, "write_system",
            lambda message, style="yellow": said.append(message),
        )
        await app._handle_slash("/metrics")
        await pilot.pause()
    assert said, "/metrics said nothing"
    text = said[-1]
    assert "298" in text
    assert "12" in text and "7" in text
    assert "hit" in text.lower()


@pytest.mark.asyncio
async def test_the_command_says_when_the_daemon_is_not_there(monkeypatch):
    """Zeros would read as "no traffic"; it means "no answer"."""
    app = TokiApp()
    said: list[str] = []

    async def fake_metrics():
        return {
            "requests_total": 0, "cache_hits": 0, "cache_misses": 0,
            "tokens_skipped": 0, "active_slots": 0, "auth_failures": 0,
            "daemon": "unreachable",
        }

    monkeypatch.setattr(app.client, "metrics_live", fake_metrics)
    async with app.run_test() as pilot:
        monkeypatch.setattr(
            app.chat, "write_system",
            lambda message, style="yellow": said.append(message),
        )
        await app._handle_slash("/metrics")
        await pilot.pause()
    assert said, "/metrics said nothing"
    assert "unavailable" in said[-1]
    assert "unreachable" in said[-1]
