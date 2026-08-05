"""Integration tests for the hardened daemon (auth, metrics, SSE, rate limit)."""


import pytest


@pytest.fixture
def auth_daemon_app(llama_server):
    """Daemon with auth and rate limiting enabled."""
    import os as _os
    import tempfile
    from pathlib import Path

    from tokirecur.api.app import create_app
    from tokirecur.backend import LlamaServerBackend
    from tokirecur.index import BruteForceIndex

    _os.environ["TOKI_AUTH_TOKEN"] = "secret-token-123"
    _os.environ["TOKI_RATE_LIMIT_BURST"] = "5"
    _os.environ["TOKI_RATE_LIMIT_RATE"] = "0"  # no refill -> deterministic 429s

    backend = LlamaServerBackend(llama_server, timeout=120.0)
    tmp_db = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp_db.close()
    index = BruteForceIndex(tmp_db.name)
    cache_dir = Path(tempfile.mkdtemp())
    return create_app(
        backend=backend, index=index, cache_dir=cache_dir,
        similarity_threshold=0.85,
        enable_auth=True,
        enable_rate_limit=True,
    )


@pytest.fixture
def hardened_client(auth_daemon_app):
    import os as _os

    from fastapi.testclient import TestClient
    with TestClient(auth_daemon_app) as c:
        yield c
    _os.environ.pop("TOKI_AUTH_TOKEN", None)
    _os.environ.pop("TOKI_RATE_LIMIT_BURST", None)
    _os.environ.pop("TOKI_RATE_LIMIT_RATE", None)


class TestAuth:
    def test_missing_token_returns_401(self, hardened_client):
        r = hardened_client.post("/v1/completions", json={
            "prompt": "Hello", "max_tokens": 4,
        })
        assert r.status_code == 401

    def test_wrong_token_returns_401(self, hardened_client):
        r = hardened_client.post(
            "/v1/completions",
            json={"prompt": "Hello", "max_tokens": 4},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert r.status_code == 401

    def test_correct_token_returns_200(self, hardened_client):
        r = hardened_client.post(
            "/v1/completions",
            json={"prompt": "Hello", "max_tokens": 4},
            headers={"Authorization": "Bearer secret-token-123"},
        )
        assert r.status_code == 200

    def test_healthz_does_not_require_auth(self, hardened_client):
        r = hardened_client.get("/healthz")
        assert r.status_code == 200

    def test_metrics_does_not_require_auth(self, hardened_client):
        r = hardened_client.get("/metrics")
        assert r.status_code == 200


class TestMetrics:
    def test_metrics_endpoint_returns_prometheus_format(self, hardened_client):
        # Make one completion to populate metrics.
        hardened_client.post(
            "/v1/completions",
            json={"prompt": "Hello", "max_tokens": 4},
            headers={"Authorization": "Bearer secret-token-123"},
        )
        r = hardened_client.get("/metrics")
        assert r.status_code == 200
        body = r.text
        # Check for our custom metrics.
        assert "toki_requests_total" in body
        assert "toki_request_duration_seconds" in body
        assert "toki_uptime_seconds" in body

    def test_metrics_records_completion_stats(self, hardened_client):
        # Make a completion.
        hardened_client.post(
            "/v1/completions",
            json={"prompt": "Hello", "max_tokens": 4},
            headers={"Authorization": "Bearer secret-token-123"},
        )
        body = hardened_client.get("/metrics").text
        # Should see at least one request in the counter.
        assert 'toki_requests_total{route="/v1/completions"' in body
        # Should see cache classification counter (might be 0 but counter exists).
        assert "toki_cache_hits_total" in body

    def test_metrics_records_auth_failures(self, hardened_client):
        # Trigger an auth failure.
        hardened_client.post("/v1/completions", json={"prompt": "Hi", "max_tokens": 2})
        body = hardened_client.get("/metrics").text
        # Check that the metric exists and is non-zero (could be any
        # value due to other tests in the suite incrementing it).
        assert "toki_auth_failures_total" in body
        # Find the line with the actual value (not the HELP/TYPE comment).
        for line in body.splitlines():
            if line.startswith("toki_auth_failures_total "):
                # Parse value, ensure it's >= 1.
                value = float(line.split()[1])
                assert value >= 1.0, f"expected counter >= 1, got {value}"
                return
        raise AssertionError("toki_auth_failures_total value not found")


class TestRateLimit:
    def test_rate_limit_returns_429_after_burst(self, hardened_client):
        # Burst is 5, refill is slow. Send 10 requests with the same token.
        # First 5 should succeed; subsequent should be 429.
        headers = {"Authorization": "Bearer secret-token-123"}
        statuses = []
        for _ in range(10):
            r = hardened_client.post(
                "/v1/completions",
                json={"prompt": "Hi", "max_tokens": 2},
                headers=headers,
            )
            statuses.append(r.status_code)
        assert 200 in statuses
        assert 429 in statuses, f"expected rate limit to trigger, got {statuses}"

    def test_rate_limit_per_session(self, hardened_client):
        """Two different sessions should NOT share rate limit state."""
        headers_a = {"Authorization": "Bearer secret-token-123", "X-Session-Id": "session-A"}
        headers_b = {"Authorization": "Bearer secret-token-123", "X-Session-Id": "session-B"}
        # Burn through session-A's burst.
        for _ in range(6):
            hardened_client.post(
                "/v1/completions",
                json={"prompt": "Hi", "max_tokens": 2},
                headers=headers_a,
            )
        # session-B should still have full quota.
        r = hardened_client.post(
            "/v1/completions",
            json={"prompt": "Hi", "max_tokens": 2},
            headers=headers_b,
        )
        assert r.status_code == 200


class TestStreamingSSE:
    def test_completions_stream_returns_sse(self, hardened_client):
        with hardened_client.stream(
            "POST",
            "/v1/completions",
            json={"prompt": "Hello", "max_tokens": 4, "stream": True},
            headers={"Authorization": "Bearer secret-token-123"},
        ) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers.get("content-type", "")
            content = r.read().decode("utf-8")
        # SSE format: data: {...}\n\n chunks
        assert content.startswith("data: ")
        assert "[DONE]" in content

    def test_chat_completions_stream_returns_sse(self, hardened_client):
        with hardened_client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 4,
                "stream": True,
            },
            headers={"Authorization": "Bearer secret-token-123"},
        ) as r:
            assert r.status_code == 200
            content = r.read().decode("utf-8")
        assert "data: " in content
        assert "[DONE]" in content
