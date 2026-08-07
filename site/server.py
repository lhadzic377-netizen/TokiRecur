"""Toki Recur website server, static site + anonymous wishlist counter.

Serves the static `site/` files and a single tiny API:

    GET  /api/wishlist  -> {"count": N}
    POST /api/wishlist  -> {"delta": +1|-1} -> {"count": N}

Privacy contract: no emails, no names, no IPs, no cookies, no logs of
signups. The only thing persisted is a single integer (the running
total), stored as `data/wishlist.json` next to the site. Visitor state
lives in their own localStorage; the endpoint just applies a +/-1
delta, so a fresh browser toggling twice moves the number by at most
one.

Run:  uvicorn server:app --host 127.0.0.1 --port 8891
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

SITE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("TOKI_WL_DATA_DIR", SITE_DIR / "data"))
COUNT_FILE = DATA_DIR / "wishlist.json"

# Crude in-memory spam brake: allow at most N toggles per minute per
# client IP. No IPs are persisted, the counter is the only state.
TOGGLE_RATE_PER_MINUTE = 10


class WishlistDelta(BaseModel):
    delta: int = Field(ge=-1, le=1)

    @field_validator("delta")
    @classmethod
    def _nonzero(cls, v: int) -> int:
        if v == 0:
            raise ValueError("delta must be +1 or -1")
        return v


def _read_count() -> int:
    try:
        return int(COUNT_FILE.read_text().strip() or "0")
    except (OSError, ValueError):
        return 0


_lock = threading.Lock()
_buckets: dict[str, list[float]] = {}


def _rate_ok(key: str) -> bool:
    import time

    now = time.time()
    recent = [t for t in _buckets.get(key, []) if now - t < 60.0]
    if len(recent) >= TOGGLE_RATE_PER_MINUTE:
        _buckets[key] = recent
        return False
    recent.append(now)
    _buckets[key] = recent
    return True


app = FastAPI(title="Toki Recur site", docs_url=None, redoc_url=None)


@app.get("/api/wishlist")
async def wishlist_get() -> dict:
    return {"count": _read_count()}


@app.post("/api/wishlist")
async def wishlist_post(request: Request, body: WishlistDelta) -> JSONResponse:
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_ok(client_ip):
        return JSONResponse({"count": _read_count(), "detail": "slow down"}, status_code=429)
    with _lock:
        count = max(0, _read_count() + body.delta)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = COUNT_FILE.with_suffix(".tmp")
        tmp.write_text(str(count))
        tmp.replace(COUNT_FILE)
    return JSONResponse({"count": count})


app.mount("/", StaticFiles(directory=SITE_DIR, html=True), name="site")
