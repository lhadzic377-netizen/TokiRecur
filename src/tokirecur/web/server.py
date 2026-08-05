"""Toki Recur web UI — production-ready chat interface.

Serves a single-page chat app with mode toggle (raw / cached / compare),
streaming responses, and live metrics. Listens on `127.0.0.1:7777` by
default — an uncommon port unlikely to conflict with other services.

Run via `toki web` (added in cli/__init__.py).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ..daemon.ratelimit import RateLimiter

# Configure logging once. Honor TOKI_LOG_LEVEL env var (default INFO).
logging.basicConfig(
    level=os.environ.get("TOKI_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_RAW_URL = "http://127.0.0.1:8088"
DEFAULT_DAEMON_URL = "http://127.0.0.1:8787"
DEFAULT_PORT = 7777

STATIC_DIR = Path(__file__).parent / "static"


def _parse_prometheus_metrics(text: str) -> dict:
    """Parse a few representative counters out of Prometheus text format.

    Returns a small dict with the most useful numbers for the UI.
    Labels are stripped from the metric name (`name{kind="hit"}`
    parses as `toki_cache_hits_total` + kind=hit).
    """
    out = {
        "requests_total": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "tokens_skipped": 0,
        "active_slots": 0,
        "auth_failures": 0,
    }
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name, value = parts[0], parts[1]
        base_name = name.split("{")[0]
        try:
            v = float(value)
        except ValueError:
            continue
        if base_name == "toki_requests_total":
            out["requests_total"] += v
        elif base_name == "toki_cache_hits_total":
            if 'kind="hit"' in name:
                out["cache_hits"] += v
            elif 'kind="miss"' in name:
                out["cache_misses"] += v
        elif base_name == "toki_tokens_skipped_total":
            out["tokens_skipped"] = int(v)
        elif base_name == "toki_active_slots":
            out["active_slots"] = int(v)
        elif base_name == "toki_auth_failures_total":
            out["auth_failures"] = int(v)
    return out


def _forward_headers(request: Request) -> dict[str, str]:
    """Build headers for proxied daemon calls.

    Uses the client's Authorization header when present (legacy setups
    that manage their own token), otherwise the web server's own
    configured token — the default when the UI is behind the login
    gate. The daemon therefore never sees an unauthenticated call from
    the web UI.
    """
    auth = request.headers.get("authorization")
    if auth:
        return {"Authorization": auth}
    token = _configured_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _token_headers(token: str | None) -> dict[str, str]:
    """Build Authorization headers from a token supplied in the
    WebSocket payload. Returns empty dict if token is falsy.
    """
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


# ── Web UI authentication ─────────────────────────────────────────────
#
# The web UI itself is gated by the same bearer token the daemon uses.
# After a successful login the server issues an HMAC-signed, HttpOnly,
# SameSite session cookie. All stateful endpoints (including the
# WebSocket) require that cookie, so an anonymous client can neither
# drive the daemon through the UI nor read slot/metrics state.
#
# When no token is configured, the UI stays open (localhost-only
# deployment) and the legacy client-supplied token path remains usable.

AUTH_COOKIE = "toki_session"
SESSION_TTL_SECONDS = 12 * 3600
# Brute-force protection: per-IP cap on /login attempts. Without this,
# an attacker can grind the token in a tight loop.
LOGIN_RATE_LIMIT_PER_MINUTE = 10


def _configured_token() -> str | None:
    """Web/daemon token lookup, matching the daemon: env var first,
    then ~/.config/tokirecur/auth, then /etc/toki-recur/auth."""
    tok = os.environ.get("TOKI_AUTH_TOKEN")
    if tok:
        return tok
    for path in (
        Path.home() / ".config" / "tokirecur" / "auth",
        Path("/etc/toki-recur/auth"),
    ):
        try:
            if path.is_file():
                content = path.read_text().strip()
                if content:
                    return content
        except OSError:
            pass
    return None


def _sign_session(token: str, ttl: int = SESSION_TTL_SECONDS) -> str:
    payload = json.dumps({"exp": int(time.time()) + ttl}, separators=(",", ":")).encode()
    b64 = base64.urlsafe_b64encode(payload).decode()
    sig = hmac.new(token.encode(), b64.encode(), hashlib.sha256).hexdigest()
    return f"{b64}.{sig}"


def _valid_session(cookie: str | None, token: str | None) -> bool:
    """Verify an HMAC-signed session cookie (constant-time)."""
    if not cookie or not token:
        return False
    try:
        b64, sig = cookie.rsplit(".", 1)
        expected = hmac.new(token.encode(), b64.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return False
        payload = json.loads(base64.urlsafe_b64decode(b64.encode()))
        return payload.get("exp", 0) > time.time()
    except Exception:
        return False


def _is_same_origin(request: Request) -> bool:
    """Reject requests whose Origin doesn't match the Host we serve."""
    origin = request.headers.get("origin")
    if not origin:
        return True  # no Origin header (same-origin navigations, curl, etc.)
    try:
        parsed = urlparse(origin)
        return parsed.netloc == request.url.netloc and parsed.scheme in ("http", "https")
    except Exception:
        return False


def create_web_app(
    *,
    raw_url: str = DEFAULT_RAW_URL,
    daemon_url: str = DEFAULT_DAEMON_URL,
    tls: bool = False,
) -> FastAPI:
    """Build the FastAPI app with the given default backend URLs.

    ``tls`` marks the deployment as serving over HTTPS so session
    cookies get the Secure flag. It is set automatically when the CLI
    is given cert/key paths.
    """
    app = FastAPI(title="Toki Recur", version="0.1.0")
    app.state.raw_url = raw_url
    app.state.daemon_url = daemon_url
    web_token = _configured_token()
    secure_cookies = tls or os.environ.get("TOKI_WEB_TLS") == "1"
    # Per-IP token bucket for /login. Keyed by remote address; entries
    # older than ~10 min fall out naturally as their tokens refill
    # back to capacity and stop triggering 429s.
    login_limiter = RateLimiter(
        burst=LOGIN_RATE_LIMIT_PER_MINUTE,
        refill_rate=LOGIN_RATE_LIMIT_PER_MINUTE / 60.0,
    )

    @app.middleware("http")
    async def security_and_no_store(request: Request, call_next):
        """Enforce same-origin + auth, then serve everything no-store.

        A stale mix of old HTML + new styles (or vice versa) has
        repeatedly broken the chat input for real users, and every
        response is therefore served no-store so a reload always gets
        one consistent, current copy of the UI.
        """
        if not _is_same_origin(request):
            return JSONResponse(
                {"detail": "cross-origin requests are not allowed"},
                status_code=403,
            )
        path = request.url.path
        if web_token and path != "/login" and not path.startswith("/static"):
            cookie = request.cookies.get(AUTH_COOKIE)
            if not _valid_session(cookie, web_token):
                return RedirectResponse("/login", status_code=307)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        # Defence-in-depth CSP. The static UI is self-hosted so a tight
        # policy is possible: only same-origin scripts/styles, no
        # inline event handlers, no eval, no framing by third parties.
        # `unsafe-inline` is allowed for styles only because the static
        # stylesheet uses inline `style="..."` attributes on a handful
        # of dynamically sized elements.
        if response.headers.get("Content-Type", "").startswith("text/html"):
            response.headers.setdefault(
                "Content-Security-Policy",
                (
                    "default-src 'self'; "
                    "script-src 'self'; "
                    "style-src 'self' 'unsafe-inline'; "
                    "img-src 'self' data:; "
                    "connect-src 'self'; "
                    "frame-ancestors 'none'; "
                    "base-uri 'self'; "
                    "form-action 'self'"
                ),
            )
        return response

    @app.get("/", response_class=HTMLResponse)
    async def root() -> HTMLResponse:
        index = STATIC_DIR / "index.html"
        body = index.read_text(encoding="utf-8") if index.exists() else (
            "<h1>Toki Recur</h1><p>UI not found.</p>"
        )
        return HTMLResponse(body, headers={"Cache-Control": "no-store"})

    @app.get("/login", response_class=HTMLResponse)
    async def login_page() -> HTMLResponse:
        page = STATIC_DIR / "login.html"
        body = page.read_text(encoding="utf-8") if page.exists() else "<h1>Token required</h1>"
        return HTMLResponse(body, headers={"Cache-Control": "no-store"})

    @app.post("/login")
    async def login(request: Request) -> JSONResponse:
        """Verify the supplied token (constant-time) and issue a cookie.

        Rate-limited per client IP to blunt brute-force attempts."""
        client_ip = (request.client.host if request.client else "unknown")
        if not login_limiter.check(f"login:{client_ip}"):
            return JSONResponse(
                {"ok": False, "detail": "too many login attempts; slow down"},
                status_code=429,
            )
        data = {}
        try:
            data = await request.json()
        except Exception:
            pass
        provided = str(data.get("token", ""))
        if web_token and hmac.compare_digest(provided, web_token):
            resp = JSONResponse({"ok": True})
            resp.set_cookie(
                AUTH_COOKIE,
                _sign_session(web_token),
                max_age=SESSION_TTL_SECONDS,
                httponly=True,
                samesite="strict",
                secure=secure_cookies,
            )
            return resp
        return JSONResponse({"ok": False, "detail": "invalid token"}, status_code=401)

    @app.post("/logout")
    async def logout() -> JSONResponse:
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(AUTH_COOKIE)
        return resp

    @app.get("/config")
    async def config() -> dict:
        return {
            "raw_url": app.state.raw_url,
            "daemon_url": app.state.daemon_url,
            "auth_enabled": bool(web_token),
        }

    @app.get("/v1/cache/state")
    async def cache_state(request: Request) -> dict:
        """Proxy to the toki daemon's cache state endpoint.

        Forwards the Authorization header so the daemon's bearer-token
        auth works correctly. Returns empty data + a `daemon_status`
        field if the daemon is unreachable or auth fails (so the
        frontend can show a useful error instead of "no slots").
        """
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(
                    f"{app.state.daemon_url}/v1/cache/state",
                    headers=_forward_headers(request),
                )
                if r.status_code == 200:
                    body = r.json()
                    body["daemon_status"] = "ok"
                    return body
                if r.status_code == 401:
                    return {
                        "slots": [], "sessions": {}, "spill_count": 0,
                        "max_context_tokens": 0,
                        "daemon_status": "auth_required",
                    }
        except Exception:
            pass
        return {
            "slots": [], "sessions": {}, "spill_count": 0,
            "max_context_tokens": 0, "daemon_status": "unreachable",
        }

    @app.post("/v1/cache/reset")
    async def cache_reset(request: Request) -> dict:
        """Proxy to the toki daemon's reset endpoint."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(
                    f"{app.state.daemon_url}/v1/cache/reset",
                    headers=_forward_headers(request),
                )
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 401:
                    return {"status": "error", "detail": "auth required"}
        except Exception as e:
            return {"status": "error", "detail": str(e)}
        return {"status": "error", "detail": "daemon unreachable"}

    @app.get("/v1/health/llama")
    async def health_llama() -> dict:
        """Probe the llama-server backend. Returns reachable / unreachable."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{app.state.raw_url}/health")
                if r.status_code == 200:
                    return {"status": "ok"}
        except Exception:
            pass
        return {"status": "unreachable"}

    @app.get("/v1/health/daemon")
    async def health_daemon(request: Request) -> dict:
        """Probe the toki daemon. Returns ok / auth_required / unreachable."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(
                    f"{app.state.daemon_url}/healthz",
                    headers=_forward_headers(request),
                )
                if r.status_code == 200:
                    return {"status": "ok"}
                if r.status_code == 401:
                    return {"status": "auth_required"}
        except Exception:
            pass
        return {"status": "unreachable"}

    @app.post("/v1/precision")
    async def precision_check(request: Request) -> dict:
        """Proxy the daemon's precision-retention check.

        The check runs 3 prompts × 3 runs (baseline + recycled) against
        llama-server and reports byte-identity stats. Proxying the
        daemon's endpoint keeps auth and the logic in one place. Note:
        byte-identity results are only meaningful against a dedicated
        llama-server with no prior slot history (see the caveat shown in
        the UI).
        """
        data = {}
        try:
            data = await request.json()
        except Exception:
            pass
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(
                    f"{app.state.daemon_url}/v1/precision",
                    json=data,
                    headers=_forward_headers(request),
                )
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 401:
                    return {"error": "auth required — set the daemon token"}
        except Exception as e:
            return {"error": f"daemon unreachable: {e}"}
        return {"error": "daemon unreachable"}

    @app.get("/v1/metrics/live")
    async def metrics_live(request: Request) -> dict:
        """Proxy the daemon's Prometheus /metrics endpoint and parse
        out a few key counters for the UI.

        Returns a small JSON dict with the most useful numbers instead
        of the full Prometheus text format.
        """
        out = {
            "requests_total": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "tokens_skipped": 0,
            "active_slots": 0,
            "auth_failures": 0,
        }
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(
                    f"{app.state.daemon_url}/metrics",
                    headers=_forward_headers(request),
                )
                if r.status_code != 200:
                    return out
                return _parse_prometheus_metrics(r.text)
        except Exception:
            return out

    @app.websocket("/ws/chat")
    async def chat_ws(websocket: WebSocket) -> None:
        # No anonymous sockets: when the UI requires auth, the handshake
        # must carry a valid session cookie (sent automatically by the
        # browser for same-origin WS upgrades). This closes the hole
        # where the auth token previously travelled inside every payload.
        if web_token and not _valid_session(
            websocket.cookies.get(AUTH_COOKIE), web_token
        ):
            await websocket.close(code=4401)
            return
        await websocket.accept()
        try:
            while True:
                msg = await websocket.receive_text()
                try:
                    payload = json.loads(msg)
                except Exception:
                    await websocket.send_json({"type": "error", "message": "invalid JSON"})
                    continue
                mode = payload.get("mode", "raw")
                prompt = payload.get("prompt", "").strip()
                system = payload.get("system") or None
                max_tokens = int(payload.get("max_tokens", 2048))
                temperature = float(payload.get("temperature", 0.0))
                history = payload.get("history", [])
                session_id = payload.get("session_id") or None
                # The server uses its own configured credential for
                # daemon calls — the client-supplied token is ignored
                # (and never sent over the socket anymore). When the web
                # UI runs with no token of its own, fall back to the
                # legacy client-supplied token so existing setups keep
                # working.
                auth_token = web_token or (payload.get("auth_token") or None)
                if not prompt:
                    await websocket.send_json({"type": "error", "message": "empty prompt"})
                    continue

                if mode == "compare":
                    await _do_compare(
                        websocket, prompt, system, max_tokens, temperature, history,
                        session_id=session_id, auth_token=auth_token,
                        raw_url=app.state.raw_url, daemon_url=app.state.daemon_url,
                    )
                else:
                    await _do_single(
                        websocket, prompt, system, max_tokens, temperature, history,
                        mode=mode,
                        session_id=session_id, auth_token=auth_token,
                        raw_url=app.state.raw_url,
                        daemon_url=app.state.daemon_url,
                    )
        except WebSocketDisconnect:
            return
        except Exception as e:  # noqa: BLE001
            logger.exception("ws error")
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def create_web_app_from_env() -> FastAPI:
    """Factory for `uvicorn tokirecur.web.server:create_web_app_from_env
    --factory`.

    Reads configuration from environment variables:
      TOKI_WEB_RAW_URL     default http://127.0.0.1:8088
      TOKI_WEB_DAEMON_URL  default http://127.0.0.1:8787
      TOKI_WEB_TLS         "1" to set the Secure flag on cookies
    """
    return create_web_app(
        raw_url=os.environ.get("TOKI_WEB_RAW_URL", DEFAULT_RAW_URL),
        daemon_url=os.environ.get("TOKI_WEB_DAEMON_URL", DEFAULT_DAEMON_URL),
        tls=os.environ.get("TOKI_WEB_TLS") == "1",
    )


def _build_messages(
    history: list[dict],
    system: str | None,
    current_prompt: str,
) -> list[dict[str, str]]:
    """Build OpenAI-format messages from history + the current turn."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    for m in history:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role in ("user", "assistant", "system"):
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": current_prompt})
    return messages


def _completion_payload(
    *,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    mode: str,
    raw_url: str,
    daemon_url: str,
    session_id: str | None,
) -> tuple[str, dict]:
    """Return (endpoint, payload) for a single completion path.

    Both paths use llama-server's OpenAI-compatible /v1/chat/completions
    so the chat template is applied correctly, and `cache_prompt` reuses
    the in-memory KV prefix across turns (raw mode). 'cached' routes
    through the daemon so session_id and max_slots reach the engine for
    slot pinning.
    """
    base = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        # Disable Qwen3.5's thinking mode by default. Without this the
        # model burns every token on reasoning and returns empty content.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if mode == "raw":
        base["cache_prompt"] = True
        return f"{raw_url}/v1/chat/completions", base
    base["session_id"] = session_id
    base["max_slots"] = 4
    return f"{daemon_url}/v1/chat/completions", base


async def _iter_sse_chunks(r) -> AsyncIterator[dict]:
    """Parse an OpenAI-compatible SSE stream.

    Yields {"content": str} for token chunks and, for the final
    usage chunk, {"usage": {...}, "cache": {...}}. The usage chunk
    carries no content, so both kinds are easy to tell apart.
    """
    buffer = ""
    async for chunk_text in r.aiter_text():
        buffer += chunk_text
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            if not line.strip():
                continue
            if line.startswith("data: "):
                line = line[6:]
            if line.strip() in ("[DONE]", ""):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("usage") or obj.get("cache"):
                yield {"usage": obj.get("usage"), "cache": obj.get("cache")}
                continue
            chunk = ""
            if obj.get("content"):
                chunk = obj["content"]
            elif obj.get("response"):
                chunk = obj["response"]
            else:
                choices = obj.get("choices") or []
                if choices and isinstance(choices, list):
                    delta = choices[0].get("delta") or {}
                    chunk = delta.get("content", "")
            if chunk:
                yield {"content": chunk}


async def _watch_abort(websocket: WebSocket, cancelled: asyncio.Event) -> None:
    """Consume WS messages while a stream is in flight.

    Sets `cancelled` when the client requests an abort, so the active
    stream can stop and report partial output. Also fires on disconnect.
    """
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if msg.get("type") == "abort":
                cancelled.set()
                return
    except WebSocketDisconnect:
        cancelled.set()


async def _do_single(
    ws: WebSocket,
    prompt: str,
    system: str | None,
    max_tokens: int,
    temperature: float,
    history: list[dict],
    *,
    mode: str,
    session_id: str | None = None,
    auth_token: str | None = None,
    raw_url: str,
    daemon_url: str,
) -> None:
    """Stream a single-path completion (raw or cached).

    For cached mode, the daemon's OpenAI-compatible endpoint is used
    (/v1/chat/completions) so session_id and max_slots reach the
    engine. For raw mode, llama-server's own OpenAI endpoint is hit
    with cache_prompt so multi-turn KV is reused.
    """
    messages = _build_messages(history, system, prompt)
    await ws.send_json({
        "type": "start",
        "mode": mode,
        "target": "raw" if mode == "raw" else "cached",
        "session_id": session_id,
    })

    endpoint, payload = _completion_payload(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        mode=mode,
        raw_url=raw_url,
        daemon_url=daemon_url,
        session_id=session_id,
    )
    auth_hdrs = _token_headers(auth_token)
    started = time.perf_counter()
    first_token_at: float | None = None
    text_chunks: list[str] = []
    cached_metadata: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    error_msg: str | None = None
    aborted = False
    cancelled = asyncio.Event()
    watcher = asyncio.create_task(_watch_abort(ws, cancelled))
    try:
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                async with client.stream(
                    "POST",
                    endpoint,
                    json=payload,
                    headers=auth_hdrs,
                ) as r:
                    if r.status_code != 200:
                        body = await r.aread()
                        error_msg = (
                            f"upstream {r.status_code}: "
                            f"{body[:200].decode('utf-8', errors='replace')}"
                        )
                        logger.warning(
                            "upstream %s returned %d: %s",
                            endpoint, r.status_code, body[:200],
                        )
                    else:
                        async for ev in _iter_sse_chunks(r):
                            if cancelled.is_set():
                                aborted = True
                                break
                            if "content" in ev:
                                if first_token_at is None:
                                    first_token_at = time.perf_counter()
                                text_chunks.append(ev["content"])
                                await ws.send_json({"type": "token", "content": ev["content"]})
                            else:
                                usage = ev.get("usage") or usage
                                cached_metadata = ev.get("cache") or cached_metadata
        except Exception as e:  # noqa: BLE001
            logger.exception("stream error to %s", endpoint)
            error_msg = f"{type(e).__name__}: {e}"
    finally:
        watcher.cancel()

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    ttft_ms = (
        (first_token_at - started) * 1000.0 if first_token_at is not None else 0.0
    )
    metrics = {
        "ttft_ms": ttft_ms,
        "total_ms": elapsed_ms,
        # Prefer the real token count from the usage chunk; fall back to
        # counting streamed chunks (llama-server raw may omit usage).
        "tokens_generated": usage.get("completion_tokens", len(text_chunks)),
    }
    if usage.get("prompt_tokens") is not None:
        metrics["prompt_tokens"] = usage["prompt_tokens"]

    if error_msg:
        # Send the error so the client renders a useful message, then a
        # synthetic `done` so the busy state is cleared.
        await ws.send_json({"type": "error", "message": error_msg})
        done_msg: dict = {
            "type": "done",
            "mode": mode,
            "text": "",
            "metrics": metrics,
            "error": error_msg,
        }
    else:
        done_msg = {
            "type": "done",
            "mode": mode,
            "text": "".join(text_chunks),
            "metrics": metrics,
        }
        if aborted:
            done_msg["aborted"] = True
    if cached_metadata:
        done_msg["cache"] = cached_metadata
    await ws.send_json(done_msg)


async def _do_compare(
    ws: WebSocket,
    prompt: str,
    system: str | None,
    max_tokens: int,
    temperature: float,
    history: list[dict],
    *,
    session_id: str | None = None,
    auth_token: str | None = None,
    raw_url: str,
    daemon_url: str,
) -> None:
    """Run both paths, stream each, and report a side-by-side comparison.

    The raw side hits llama-server's OpenAI endpoint with cache_prompt.
    The cached side hits the daemon's OpenAI-compatible endpoint so
    session_id and max_slots reach the engine for slot pinning.
    """
    messages = _build_messages(history, system, prompt)

    cancelled = asyncio.Event()
    watcher = asyncio.create_task(_watch_abort(ws, cancelled))

    async def run_one(
        label: str, endpoint: str, payload: dict, token: str | None
    ) -> tuple[str, dict, dict, str | None, bool]:
        """Returns (text, metrics, cache, error_message, aborted)."""
        await ws.send_json({"type": "compare_start", "label": label, "target": endpoint})
        started = time.perf_counter()
        first_token_at: float | None = None
        text_chunks: list[str] = []
        usage: dict = {}
        cache: dict = {}
        error_msg: str | None = None
        aborted = False
        auth_hdrs = _token_headers(token)
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                async with client.stream("POST", endpoint, json=payload, headers=auth_hdrs) as r:
                    if r.status_code != 200:
                        body = await r.aread()
                        error_msg = (
                            f"{label}: status {r.status_code}: "
                            f"{body[:200].decode('utf-8', errors='replace')}"
                        )
                    else:
                        async for ev in _iter_sse_chunks(r):
                            if cancelled.is_set():
                                aborted = True
                                break
                            if "content" in ev:
                                if first_token_at is None:
                                    first_token_at = time.perf_counter()
                                text_chunks.append(ev["content"])
                                await ws.send_json({
                                    "type": "compare_token",
                                    "label": label,
                                    "content": ev["content"],
                                })
                            else:
                                usage = ev.get("usage") or usage
                                cache = ev.get("cache") or cache
        except Exception as e:  # noqa: BLE001
            error_msg = f"{label}: {type(e).__name__}: {e}"
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        ttft_ms = (
            (first_token_at - started) * 1000.0 if first_token_at is not None else 0.0
        )
        metrics = {
            "ttft_ms": ttft_ms,
            "total_ms": elapsed_ms,
            "tokens_generated": usage.get("completion_tokens", len(text_chunks)),
        }
        if usage.get("prompt_tokens") is not None:
            metrics["prompt_tokens"] = usage["prompt_tokens"]
        await ws.send_json({
            "type": "compare_done",
            "label": label,
            "text": "".join(text_chunks) if not error_msg else "",
            "metrics": metrics,
            "cache": cache or None,
            "error": error_msg,
            "aborted": aborted or None,
        })
        return "".join(text_chunks), metrics, cache, error_msg, aborted

    try:
        raw_endpoint, raw_payload = _completion_payload(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            mode="raw",
            raw_url=raw_url,
            daemon_url=daemon_url,
            session_id=session_id,
        )
        cached_endpoint, cached_payload = _completion_payload(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            mode="cached",
            raw_url=raw_url,
            daemon_url=daemon_url,
            session_id=session_id,
        )

        # Run raw first (llama-server directly), then cached (daemon).
        raw_text, raw_m, raw_cache, raw_err, raw_aborted = await run_one(
            "raw", raw_endpoint, raw_payload, None,
        )
        if cancelled.is_set():
            await ws.send_json({
                "type": "summary",
                "raw": raw_m,
                "cached": None,
                "speedup": None,
                "identical": False,
                "first_diff_char": -1,
                "diff_chars": 0,
                "cached_error": None,
                "raw_error": None,
                "aborted": True,
            })
            return
        cached_text, cached_m, cached_cache, cached_err, cached_aborted = await run_one(
            "cached", cached_endpoint, cached_payload, auth_token,
        )
    finally:
        watcher.cancel()

    # HONEST summary: if cached failed, don't fake a speedup.
    if cached_err or not cached_text:
        # Cached path failed. Show raw timing + error instead of fake speedup.
        await ws.send_json({
            "type": "summary",
            "raw": raw_m,
            "cached": cached_m,
            "speedup": None,
            "identical": False,
            "first_diff_char": -1,
            "diff_chars": 0,
            "cached_error": cached_err or "no output received from cached path",
            "raw_error": raw_err,
            "aborted": cached_aborted or None,
        })
        return

    if raw_err or not raw_text:
        # Raw failed (unusual). Show cached timing only.
        await ws.send_json({
            "type": "summary",
            "raw": raw_m,
            "cached": cached_m,
            "speedup": None,
            "identical": False,
            "first_diff_char": -1,
            "diff_chars": 0,
            "raw_error": raw_err or "no output received from raw path",
            "cached_error": None,
            "aborted": raw_aborted or None,
        })
        return

    speedup = raw_m.get("total_ms", 1) / max(cached_m.get("total_ms", 1), 1e-9)
    identical = raw_text == cached_text
    first_diff = 0
    if not identical:
        for i in range(min(len(raw_text), len(cached_text))):
            if raw_text[i] != cached_text[i]:
                first_diff = i
                break
        else:
            first_diff = min(len(raw_text), len(cached_text))
    diff_chars = sum(
        1 for a, b in zip(raw_text, cached_text, strict=False) if a != b
    ) + abs(len(raw_text) - len(cached_text))

    await ws.send_json({
        "type": "summary",
        "raw": raw_m,
        "cached": cached_m,
        "speedup": speedup,
        "identical": identical,
        "first_diff_char": first_diff if not identical else -1,
        "diff_chars": diff_chars if not identical else 0,
        "cached_error": None,
        "raw_error": None,
        "aborted": cached_aborted or raw_aborted or None,
    })
