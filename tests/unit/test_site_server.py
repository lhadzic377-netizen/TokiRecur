"""Unit tests for the anonymous wishlist counter API (site/server.py).

The privacy contract: the endpoint stores a single integer and nothing
else — no emails, no IPs, no per-visitor data.
"""

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_SERVER_PATH = Path(__file__).resolve().parents[2] / "site" / "server.py"


@pytest.fixture()
def site_server(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("toki_site_server", _SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mod, "COUNT_FILE", tmp_path / "wishlist.json")
    mod._buckets.clear()
    yield mod


@pytest.fixture()
def client(site_server):
    with TestClient(site_server.app) as c:
        yield c


class TestWishlistCounter:
    def test_get_starts_at_zero(self, client):
        r = client.get("/api/wishlist")
        assert r.status_code == 200
        assert r.json() == {"count": 0}

    def test_toggle_up_and_down(self, client):
        assert client.post("/api/wishlist", json={"delta": 1}).json() == {"count": 1}
        assert client.post("/api/wishlist", json={"delta": 1}).json() == {"count": 2}
        assert client.post("/api/wishlist", json={"delta": -1}).json() == {"count": 1}
        assert client.get("/api/wishlist").json() == {"count": 1}

    def test_count_persists_across_restarts(self, site_server, tmp_path):
        with TestClient(site_server.app) as c:
            c.post("/api/wishlist", json={"delta": 1})
            c.post("/api/wishlist", json={"delta": 1})
        spec = importlib.util.spec_from_file_location("toki_site_server2", _SERVER_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.DATA_DIR = tmp_path
        mod.COUNT_FILE = tmp_path / "wishlist.json"
        with TestClient(mod.app) as c:
            assert c.get("/api/wishlist").json() == {"count": 2}

    def test_never_goes_below_zero(self, client):
        assert client.post("/api/wishlist", json={"delta": -1}).json() == {"count": 0}
        assert client.get("/api/wishlist").json() == {"count": 0}

    def test_rejects_zero_and_big_deltas(self, client):
        assert client.post("/api/wishlist", json={"delta": 0}).status_code == 422
        assert client.post("/api/wishlist", json={"delta": 5}).status_code == 422
        assert client.post("/api/wishlist", json={"delta": -2}).status_code == 422
        assert client.post("/api/wishlist", json={}).status_code == 422

    def test_returns_no_visitor_identifiers(self, client):
        body = client.get("/api/wishlist").json()
        assert set(body) == {"count"}

    def test_rate_limit_blocks_bursts(self, site_server):
        site_server.TOGGLE_RATE_PER_MINUTE = 3
        site_server._buckets.clear()
        with TestClient(site_server.app) as c:
            assert c.post("/api/wishlist", json={"delta": 1}).status_code == 200
            assert c.post("/api/wishlist", json={"delta": 1}).status_code == 200
            assert c.post("/api/wishlist", json={"delta": 1}).status_code == 200
            assert c.post("/api/wishlist", json={"delta": 1}).status_code == 429

    def test_serves_the_landing_page(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "Toki Recur" in r.text
