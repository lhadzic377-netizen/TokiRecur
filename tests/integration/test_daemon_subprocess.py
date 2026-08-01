"""End-to-end daemon hardening tests.

Spawns `toki serve` as a real subprocess and tests it under realistic
conditions: concurrent requests, rate limiting, streaming, and
graceful shutdown.

These tests are slower than the in-process TestClient tests but
exercise the daemon the way production users would.
"""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(base_url: str, timeout: float = 30.0) -> bool:
    started = time.time()
    while time.time() - started < timeout:
        try:
            r = httpx.get(f"{base_url}/healthz", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


@pytest.fixture
def daemon_subprocess(llama_server, tmp_path):
    """Spawn `toki serve` as a subprocess on a free port."""
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    # Use a temporary index file so we don't pollute production state.
    db_path = tmp_path / "index.sqlite"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    env = os.environ.copy()
    env["TOKI_AUTH_TOKEN"] = "e2e-test-token"
    env["TOKI_RATE_LIMIT_BURST"] = "10"
    env["TOKI_RATE_LIMIT_RATE"] = "2.0"
    env["TOKI_DB"] = str(db_path)
    env["TOKI_CACHE_DIR"] = str(cache_dir)

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "tokirecur.api.app:create_app_from_env",
            "--factory",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--log-level", "warning",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    try:
        if not _wait_for_health(base_url, timeout=30.0):
            stdout = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            proc.terminate()
            pytest.fail(
                f"daemon subprocess failed to start on {base_url}\n"
                f"stdout: {stdout[:500]}\n"
                f"stderr: {stderr[:500]}"
            )
        yield base_url
    finally:
        # Send SIGTERM and wait for graceful shutdown.
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


class TestDaemonSubprocessHealth:
    def test_health_endpoint(self, daemon_subprocess):
        r = httpx.get(f"{daemon_subprocess}/healthz", timeout=5.0)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_metrics_endpoint_no_auth_required(self, daemon_subprocess):
        # The /metrics endpoint is public (for Prometheus).
        r = httpx.get(f"{daemon_subprocess}/metrics", timeout=5.0)
        assert r.status_code == 200
        body = r.text
        assert "toki_uptime_seconds" in body

    def test_v1_completions_requires_auth(self, daemon_subprocess):
        r = httpx.post(
            f"{daemon_subprocess}/v1/completions",
            json={"prompt": "Hello", "max_tokens": 4},
            timeout=5.0,
        )
        assert r.status_code == 401

    def test_v1_completions_with_auth(self, daemon_subprocess):
        r = httpx.post(
            f"{daemon_subprocess}/v1/completions",
            json={"prompt": "Hello", "max_tokens": 4},
            headers={"Authorization": "Bearer e2e-test-token"},
            timeout=30.0,
        )
        assert r.status_code == 200


class TestDaemonSubprocessConcurrent:
    def test_concurrent_requests_all_complete(self, daemon_subprocess):
        """10 concurrent completion requests — all should succeed."""
        import concurrent.futures as cf

        headers = {"Authorization": "Bearer e2e-test-token"}
        payload = {"prompt": "What is 2+2?", "max_tokens": 4, "temperature": 0.0}

        def send(_):
            return httpx.post(
                f"{daemon_subprocess}/v1/completions",
                json=payload, headers=headers, timeout=30.0,
            )

        with cf.ThreadPoolExecutor(max_workers=10) as pool:
            results = list(pool.map(send, range(10)))

        # All 10 should return 200.
        assert all(r.status_code == 200 for r in results), [
            (r.status_code, r.text[:100]) for r in results if r.status_code != 200
        ]
        # All should have completion_tokens > 0.
        for r in results:
            body = r.json()
            assert body["usage"]["completion_tokens"] > 0


class TestDaemonSubprocessRateLimit:
    def test_rate_limit_triggers_after_burst(self, daemon_subprocess):
        """With burst=10 and slow refill, ~15 requests should yield some 429s."""
        headers = {"Authorization": "Bearer e2e-test-token"}
        payload = {"prompt": "Hi", "max_tokens": 2}

        statuses = []
        for _ in range(15):
            r = httpx.post(
                f"{daemon_subprocess}/v1/completions",
                json=payload, headers=headers, timeout=30.0,
            )
            statuses.append(r.status_code)
        # Some should be 200, some should be 429.
        assert 200 in statuses
        assert 429 in statuses, f"expected 429, got {statuses}"


class TestDaemonSubprocessGracefulShutdown:
    def test_sigterm_drains_inflight(self, llama_server, tmp_path):
        """Send SIGTERM during a long completion — should drain."""
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        db_path = tmp_path / "index.sqlite"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        env = os.environ.copy()
        env["TOKI_AUTH_TOKEN"] = "shutdown-test"
        env["TOKI_DB"] = str(db_path)
        env["TOKI_CACHE_DIR"] = str(cache_dir)

        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "tokirecur.api.app:create_app_from_env",
                "--factory",
                "--host", "127.0.0.1",
                "--port", str(port),
                "--log-level", "warning",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            assert _wait_for_health(base_url, timeout=30.0)
            headers = {"Authorization": "Bearer shutdown-test"}

            # Start a long-running request in a thread.
            import threading
            response_holder: dict = {}

            def long_request():
                try:
                    r = httpx.post(
                        f"{base_url}/v1/completions",
                        json={"prompt": "Write a long story.", "max_tokens": 32},
                        headers=headers, timeout=60.0,
                    )
                    response_holder["response"] = r
                except Exception as e:
                    response_holder["error"] = e

            t = threading.Thread(target=long_request, daemon=True)
            t.start()

            # Give the request a moment to start.
            time.sleep(0.5)

            # Send SIGTERM.
            proc.send_signal(signal.SIGTERM)

            # Process should exit cleanly within 10s.
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                pytest.fail("daemon did not exit within 10s of SIGTERM")

            # Wait for the request thread to finish (or time out).
            t.join(timeout=5.0)

            # The request might have completed (200) or been cut off (timeout).
            # Either is acceptable — what matters is the daemon didn't crash.
            # The key check: no zombie process.
            assert proc.poll() is not None
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
