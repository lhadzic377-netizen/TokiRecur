"""Security regression tests: path traversal, WS origin, auth timing.

These lock in the fixes for the release-readiness review findings:
  - DELETE /cache/{cache_id} rejects non-hex ids (path traversal guard)
  - the web WebSocket rejects cross-origin handshakes
  - bearer-token comparison is constant-time
  - the default auth token is never a well-known value
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tokirecur import ctl
from tokirecur.api.app import create_app
from tokirecur.cache.blob import blob_path
from tokirecur.web.server import _is_same_origin, create_web_app

# ── Path traversal: DELETE /cache/{cache_id} / blob_path ──────────────


class TestCacheDeleteTraversal:
    def test_blob_path_rejects_traversal(self, tmp_path):
        for bad in (
            "../secret",
            "../../etc/passwd",
            "..",
            "a/b",
            "a\\b",
            "..%2f",
            "cache_id ",
            "x" * 65,
            "",
        ):
            with pytest.raises(ValueError):
                blob_path(tmp_path, bad)

    def test_blob_path_accepts_hex(self, tmp_path):
        p = blob_path(tmp_path, "abc123")
        assert p.name == "abc123.trkv"
        assert p.parent == tmp_path

    def test_delete_endpoint_rejects_traversal(self, tmp_path):
        app = create_app(
            backend=_FakeBackend(),
            index=_FakeIndex(),
            cache_dir=tmp_path,
            enable_auth=False,
            enable_rate_limit=False,
        )
        with TestClient(app) as client:
            # Cases that reach the route with an invalid id must be
            # rejected with 400 by the validator.
            for bad in ("@", "a b", "x" * 65):
                resp = client.delete(f"/cache/{bad}")
                assert resp.status_code == 400, bad
            # Path-separator / parent forms are normalized by the HTTP
            # client/router and 404 before ever reaching the handler —
            # they cannot touch files outside cache_dir either.
            for bad in ("..", "a/b"):
                assert client.delete(f"/cache/{bad}").status_code == 404, bad

    def test_delete_endpoint_accepts_hex_id(self, tmp_path):
        app = create_app(
            backend=_FakeBackend(),
            index=_FakeIndex(),
            cache_dir=tmp_path,
            enable_auth=False,
            enable_rate_limit=False,
        )
        with TestClient(app) as client:
            resp = client.delete("/cache/abc123")
            assert resp.status_code == 404  # not in index, but not a 400

    def test_delete_endpoint_does_not_reach_outside_cache_dir(self, tmp_path):
        outside = tmp_path / "pwn"
        outside.mkdir()
        (outside / "evil.trkv").write_text("x")
        app = create_app(
            backend=_FakeBackend(),
            index=_FakeIndex(),
            cache_dir=tmp_path,
            enable_auth=False,
            enable_rate_limit=False,
        )
        with TestClient(app) as client:
            client.delete("/cache/%2e%2e/pwn/evil")
            assert (outside / "evil.trkv").exists()


# ── WebSocket cross-origin handshake ──────────────────────────────────


class TestWebSocketOrigin:
    def test_origin_check_helper_rejects_foreign_origin(self):
        class _Conn:
            headers = {"origin": "https://evil.example"}
            url = type("U", (), {"netloc": "127.0.0.1:7777"})()

        assert _is_same_origin(_Conn()) is False

    def test_origin_check_helper_accepts_matching_origin(self):
        class _Conn:
            headers = {"origin": "http://127.0.0.1:7777"}
            url = type("U", (), {"netloc": "127.0.0.1:7777"})()

        assert _is_same_origin(_Conn()) is True

    def test_ws_rejects_cross_origin(self):
        app = create_web_app(raw_url="http://raw:8088", daemon_url="http://daemon:8787")
        with TestClient(app) as client:
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(
                    "/ws/chat", headers={"origin": "https://evil.example"}
                ):
                    pass

    def test_ws_accepts_same_origin(self):
        app = create_web_app(raw_url="http://raw:8088", daemon_url="http://daemon:8787")
        with TestClient(app) as client:
            with client.websocket_connect(
                "/ws/chat", headers={"origin": "http://testserver"}
            ) as ws:
                ws.send_json({"mode": "raw", "prompt": "hi"})
                msg = ws.receive_json()
                assert msg["type"] == "start"


# ── Constant-time auth comparison ─────────────────────────────────────


class TestConstantTimeAuth:
    def test_wrong_token_gets_401(self, monkeypatch):
        monkeypatch.setenv("TOKI_AUTH_TOKEN", "s3cret-token-here")
        app = create_app(
            backend=_FakeBackend(),
            index=_FakeIndex(),
            cache_dir="/tmp/toki-security",
            enable_auth=True,
            enable_rate_limit=False,
        )
        with TestClient(app) as client:
            resp = client.get("/cache/stats", headers={"Authorization": "Bearer wrong"})
            assert resp.status_code == 401

    def test_right_token_accepted(self, monkeypatch):
        monkeypatch.setenv("TOKI_AUTH_TOKEN", "s3cret-token-here")
        app = create_app(
            backend=_FakeBackend(),
            index=_FakeIndex(),
            cache_dir="/tmp/toki-security",
            enable_auth=True,
            enable_rate_limit=False,
        )
        with TestClient(app) as client:
            resp = client.get(
                "/cache/stats", headers={"Authorization": "Bearer s3cret-token-here"}
            )
            assert resp.status_code == 200


# ── Default token is never well-known ─────────────────────────────────


class TestDefaultToken:
    def test_default_auth_is_random_not_known(self, monkeypatch):
        monkeypatch.delenv("TOKI_AUTH_TOKEN", raising=False)
        tok = ctl.auth_token()
        assert tok
        assert tok != "demo-token"
        assert len(tok) >= 32

    def test_generated_default_looks_like_a_token(self, monkeypatch):
        monkeypatch.delenv("TOKI_AUTH_TOKEN", raising=False)
        assert re.fullmatch(r"[A-Za-z0-9_-]+", ctl.auth_token())

    def test_env_var_wins_over_random_default(self, monkeypatch):
        monkeypatch.setenv("TOKI_AUTH_TOKEN", "my-configured-token")
        assert ctl.auth_token() == "my-configured-token"


# ── Fakes ─────────────────────────────────────────────────────────────


class _FakeBackend:
    def embed(self, text):
        return [0.0] * 4

    def complete_openai(self, messages, **kw):
        return {
            "text": "hi", "tool_calls": None, "finish_reason": "stop",
            "reasoning": "", "tokens_generated": 1, "prefill_tokens": 1,
            "ttft_ms": 1.0, "total_ms": 1.0, "prefill_tokens_per_sec": 0.0,
        }


class _FakeIndex:
    def __init__(self) -> None:
        self.entries: dict[str, object] = {}

    def search(self, embedding):
        return None

    def insert(self, entry) -> None:
        self.entries[entry.cache_id] = entry

    def delete(self, cache_id: str) -> bool:
        return self.entries.pop(cache_id, None) is not None

    def touch(self, cache_id: str) -> None:
        pass

    def stats(self) -> dict:
        return {"entries": len(self.entries), "bytes_on_disk": 0, "embedding_dim": 4}
