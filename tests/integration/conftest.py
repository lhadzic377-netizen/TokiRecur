"""Pytest fixtures for integration tests.

Spins up a real llama-server against the bundled 9B model if not
already running. Integration tests skip gracefully if no GGUF is
found or the server can't start within the timeout.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest

DEFAULT_MODEL = Path.home() / "Downloads" / "Qwen3.5-9B-Q4_K_M.gguf"
# Fallback model for fast CI: smaller, often full-attention.
# Used when TOKI_TEST_MODEL_FAST is set or default model is unavailable.
FAST_MODEL = Path.home() / ".lmstudio" / "models" / "unsloth" / "Qwen3.5-0.8B-MTP-GGUF" / "Qwen3.5-0.8B-Q4_K_M.gguf"
STARTUP_TIMEOUT = 180.0


def _resolve_model() -> Path | None:
    """Pick a model based on env vars, falling back through candidates."""
    explicit = os.environ.get("TOKI_TEST_MODEL")
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    fast = os.environ.get("TOKI_TEST_FAST", "").lower() in ("1", "true", "yes")
    if fast and FAST_MODEL.exists():
        return FAST_MODEL
    if DEFAULT_MODEL.exists():
        return DEFAULT_MODEL
    if FAST_MODEL.exists():
        return FAST_MODEL
    return None


def _free_port() -> int:
    """Find a free localhost port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(base_url: str, timeout: float = STARTUP_TIMEOUT) -> bool:
    started = time.time()
    while time.time() - started < timeout:
        try:
            r = httpx.get(f"{base_url}/health", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


@pytest.fixture(scope="session")
def llama_server():
    """Provide a running llama-server. Spawns one on a free port if
    not already running. Skips the test session if no GGUF is found."""
    base_url = os.environ.get("TOKI_TEST_LLAMA_URL")
    if base_url:
        # External server already configured.
        if not _wait_for_server(base_url, timeout=10.0):
            pytest.skip(f"external llama-server at {base_url} not reachable")
        yield base_url
        return

    model = _resolve_model()
    if model is None:
        pytest.skip("no GGUF model found (set TOKI_TEST_MODEL or install one of: "
                    f"{DEFAULT_MODEL}, {FAST_MODEL})")

    if shutil.which("llama-server") is None:
        pytest.skip("llama-server not in PATH")

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    proc = subprocess.Popen(
        [
            "llama-server",
            "-m", str(model),
            "--host", "127.0.0.1",
            "--port", str(port),
            "-c", "2048",
            "-ngl", "99",
            "--parallel", "2",
            "--batch-size", "512",
            "--ubatch-size", "512",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        if not _wait_for_server(base_url):
            proc.terminate()
            pytest.fail(f"llama-server failed to start on {base_url}")
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def backend(llama_server):
    """Provide a LlamaServerBackend pointing at the test server."""
    from tokirecur.backend import LlamaServerBackend
    return LlamaServerBackend(llama_server, timeout=120.0)


@pytest.fixture
def engine(llama_server):
    """Provide a fresh RecyclingEngine with an in-memory index."""
    from tokirecur.core.recycling import RecyclingEngine
    from tokirecur.backend import LlamaServerBackend
    from tokirecur.index import BruteForceIndex
    import tempfile

    backend = LlamaServerBackend(llama_server, timeout=120.0)
    tmp_db = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp_db.close()
    index = BruteForceIndex(tmp_db.name)
    return RecyclingEngine(backend, index)


@pytest.fixture(scope="session")
def precision_server():
    """Dedicated llama-server for byte-identity tests.

    This build of llama-server leaves slots in a state where subsequent
    greedy generations become non-deterministic after certain request
    sequences (notably back-to-back native prefills with no common
    prefix). Byte-identity assertions are only meaningful on a server
    whose slots carry no history from other test files, so these tests
    get their own server instead of the shared session one.
    """
    model = _resolve_model()
    if model is None or shutil.which("llama-server") is None:
        pytest.skip("no GGUF model or llama-server available")

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [
            "llama-server",
            "-m", str(model),
            "--host", "127.0.0.1",
            "--port", str(port),
            "-c", "2048",
            "-ngl", "99",
            "--parallel", "2",
            "--batch-size", "512",
            "--ubatch-size", "512",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        if not _wait_for_server(base_url):
            proc.terminate()
            pytest.fail(f"precision llama-server failed to start on {base_url}")
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def precision_engine(precision_server):
    """Fresh RecyclingEngine against the dedicated precision server."""
    from tokirecur.core.recycling import RecyclingEngine
    from tokirecur.backend import LlamaServerBackend
    from tokirecur.index import BruteForceIndex
    import tempfile

    backend = LlamaServerBackend(precision_server, timeout=120.0)
    tmp_db = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp_db.close()
    index = BruteForceIndex(tmp_db.name)
    return RecyclingEngine(backend, index)


@pytest.fixture
def http_client(llama_server):
    """Provide an httpx client for direct API tests."""
    with httpx.Client(timeout=120.0, base_url=llama_server) as c:
        yield c
