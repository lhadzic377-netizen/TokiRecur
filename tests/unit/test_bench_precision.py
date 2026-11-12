"""A dead server must not score a perfect precision retention.

_retry_post returns {} on persistent failure; three empty strings used to
look 'identical' (len(set(['','',''])) == 1) and '' == '' cross-matched,
so an unreachable server awarded 100% precision. Failed requests must
count as non-matches.
"""

from __future__ import annotations

import tokirecur.bench.cli as bench


def test_dead_server_scores_zero_not_perfect(monkeypatch):
    monkeypatch.setattr(bench, "_retry_post", lambda *a, **k: {})
    result = bench.precision_retention("http://dead", 16)
    assert result["baseline_deterministic"] == 0.0
    assert result["recycled_deterministic"] == 0.0
    assert result["byte_identical_to_baseline"] == 0.0


def test_healthy_deterministic_server_scores_one(monkeypatch):
    monkeypatch.setattr(
        bench, "_retry_post", lambda *a, **k: {"content": "stable answer"},
    )
    result = bench.precision_retention("http://ok", 16)
    assert result["baseline_deterministic"] == 1.0
    assert result["recycled_deterministic"] == 1.0
    assert result["byte_identical_to_baseline"] == 1.0


def test_partial_failure_lowers_score(monkeypatch):
    # Baseline succeeds, recycled fails → cross-identity must be 0 even
    # though the baseline alone is deterministic.
    def fake_post(client, base_url, payload, **kw):
        if payload.get("cache_prompt"):
            return {}
        return {"content": "answer"}

    monkeypatch.setattr(bench, "_retry_post", fake_post)
    result = bench.precision_retention("http://flaky", 16)
    assert result["baseline_deterministic"] == 1.0
    assert result["recycled_deterministic"] == 0.0
    assert result["byte_identical_to_baseline"] == 0.0
