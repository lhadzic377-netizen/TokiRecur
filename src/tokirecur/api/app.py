"""FastAPI application exposing the recycling engine.

OpenAI-compatible endpoints:
    POST /v1/completions
    POST /v1/chat/completions
    POST /v1/responses   (OpenAI Responses API shim → chat completions)

Admin endpoints:
    GET  /cache/stats
    GET  /v1/cache/state
    POST /v1/cache/reset
    GET  /healthz
    GET  /metrics

Auth: bearer-token via `TOKI_AUTH_TOKEN` env var. If unset, the daemon
refuses to bind to non-loopback addresses.

Rate limiting: per-session_id (or client IP) token bucket, default 60
requests/minute.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from ..core.recycling import (
    LlamaBackend,
    RecyclingEngine,
)
from ..daemon.metrics import METRICS
from ..daemon.ratelimit import RateLimiter
from ..index import Index

logger = logging.getLogger(__name__)

# Thinking-mode headroom. Many GGUF chat templates (Qwen3 "Thinking",
# unsloth "Thinking-2507", DeepSeek-R1 distill, etc.) burn hundreds of
# tokens on reasoning before producing visible content. If the client
# sends a small `max_tokens`, llama-server cuts the model off inside
# reasoning and returns `content: ""`. The daemon therefore floors the
# budget so thinking models always answer; non-thinking models stop on
# EOS regardless. Verbose thinking models (e.g. Qwen3.5-35B-A3B reasons
# ~570 tokens for a trivial prompt) need a much higher floor than the
# 512 used when thinking is disabled.
MAX_TOKENS_FLOOR = 512
THINKING_TOKENS_FLOOR = 2048

_FLOOR_WARNED = False


def _default_chat_template_kwargs(reasoning: str) -> dict:
    """Qwen-style chat templates need an explicit ``enable_thinking``
    kwarg. ``off`` disables thinking so the response contains only
    visible content; anything else (``auto``/``on``) enables it so the
    model emits reasoning the clients can render. Other templates
    ignore the field.
    """
    return {"enable_thinking": reasoning != "off"}


def _max_context_ceiling() -> int | None:
    """Upper bound for max_tokens: the real slot context, passed by the
    stack as TOKI_MAX_CONTEXT_TOKENS. Keeps clients from requesting
    more output tokens than the slot has room for (which llama-server
    rejects with 400 "not enough tokens")."""
    try:
        value = int(os.environ.get("TOKI_MAX_CONTEXT_TOKENS", ""))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _active_model_id() -> str | None:
    """The model_id of the currently-active spec, or None if no model
    is active. Used to normalize the ``model`` field on incoming
    OpenAI requests so a client sending ``"unsloth/..."`` (legacy
    form) gets a consistent identity back from the daemon.
    """
    try:
        from ..ctl import get_model, read_active
        name = read_active()
        if not name:
            return None
        return get_model(name).model_id
    except Exception:  # noqa: BLE001
        return None


def _resolve_max_tokens(requested: int, reasoning: str = "auto") -> int:
    """Floor at the thinking/non-thinking budget without capping larger
    client requests, then clamp to the real slot context when the stack
    provides one. Logs a one-time warning when a client's budget is
    silently raised.
    """
    global _FLOOR_WARNED
    requested = int(requested)
    floor = THINKING_TOKENS_FLOOR if reasoning != "off" else MAX_TOKENS_FLOOR
    if requested < floor and not _FLOOR_WARNED:
        _FLOOR_WARNED = True
        logger.warning(
            "client requested max_tokens=%d below the %d floor; "
            "budget raised so thinking models can finish reasoning",
            requested, MAX_TOKENS_FLOOR,
        )
    resolved = max(requested, floor)
    ceiling = _max_context_ceiling()
    if ceiling:
        resolved = min(resolved, ceiling)
    return resolved


class CompletionRequest(BaseModel):
    model: str = "default"
    prompt: str
    # Hard bounds instead of trusting the client: llama.cpp truncates
    # out-of-range values silently, which made overrides look broken.
    max_tokens: int = Field(256, ge=1, le=65536)
    temperature: float = Field(0.0, ge=-2.0, le=2.0)
    max_slots: int = Field(4, ge=1, le=32)
    session_id: str | None = None
    skip_cache: bool = False
    stream: bool = False


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    model: str
    created: int
    choices: list[dict[str, Any]]
    usage: dict[str, int]
    cache: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """One OpenAI chat message.

    ``content`` may be a plain string, ``None`` (tool-call assistant
    turns) or a list (multi-part content, rare). Agentic clients like
    opencode also send ``tool_calls`` on assistant messages and
    ``tool_call_id`` + ``name`` on ``tool`` messages — all preserved
    and forwarded to llama-server verbatim.
    """

    role: str
    content: str | list | None = None
    tool_calls: list | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    # Hard bounds instead of trusting the client: llama.cpp truncates
    # out-of-range values silently, which made overrides look broken.
    max_tokens: int = Field(256, ge=1, le=65536)
    temperature: float = Field(0.0, ge=-2.0, le=2.0)
    max_slots: int = Field(4, ge=1, le=32)
    session_id: str | None = None
    stream: bool = False
    # Tool calling (agentic clients): forwarded to llama-server, which
    # applies the model's chat template. When tools are present the
    # response may contain tool_calls instead of text, and the stream
    # relays llama-server's OpenAI deltas verbatim.
    tools: list | None = None
    tool_choice: str | dict | None = None
    # Pass-through for chat-template kwargs. If unset, the daemon
    # defaults to disabling Qwen3.5's thinking mode so the response
    # contains visible content instead of internal reasoning tokens.
    # Other models ignore this field.
    chat_template_kwargs: dict | None = None


class ResponsesItem(BaseModel):
    """One entry of the Responses API ``input`` array. The fields are
    intentionally loose: the OpenAI SDKs evolve item shapes faster than
    this shim, and unknown/extra fields should pass through untouched.
    """

    type: str
    role: str | None = None
    content: Any = None  # str | list of input_text/input_image parts
    call_id: str | None = None
    output: Any = None  # function_call_output payload (any JSON)
    name: str | None = None
    arguments: str | None = None


class ResponsesReasoning(BaseModel):
    effort: str = "auto"


class ResponsesRequest(BaseModel):
    model: str = "default"
    input: str | list[ResponsesItem]
    instructions: str | None = None  # mapped to a system message
    max_output_tokens: int | None = Field(None, ge=1, le=65536)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    stream: bool = False
    tools: list | None = None
    tool_choice: str | dict | None = None
    reasoning: ResponsesReasoning | None = None
    stream_options: dict | None = None
    store: bool = False
    previous_response_id: str | None = None
    metadata: dict | None = None
    # toki extensions (same knobs as /v1/chat/completions)
    max_slots: int = Field(4, ge=1, le=32)
    session_id: str | None = None


class CacheLookupRequest(BaseModel):
    prompt: str


class PrecisionRequest(BaseModel):
    prompts: list[str] | None = None
    max_tokens: int | None = Field(32, ge=1, le=512)


def _get_auth_token() -> str | None:
    # Read order: env var → ~/.config/tokirecur/auth → /etc/toki-recur/auth.
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


def _check_auth(request: Request) -> None:
    """Validate bearer token. Raises 401 if missing/invalid."""
    expected = _get_auth_token()
    if expected is None:
        # No token configured — accept everything on loopback, reject
        # on non-loopback.
        client_host = request.client.host if request.client else ""
        if client_host not in ("127.0.0.1", "::1", "localhost"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="TOKI_AUTH_TOKEN not set; refusing non-localhost request",
            )
        return
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        METRICS.auth_failures.inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    provided = auth[len("Bearer "):].strip()
    if provided != expected:
        METRICS.auth_failures.inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid token",
        )


def _get_rate_key(request: Request, session_id: str | None) -> str:
    if session_id:
        return f"session:{session_id}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


def create_app(
    *,
    backend: LlamaBackend,
    index: Index,
    cache_dir: Path,
    similarity_threshold: float = 0.85,
    memory_budget_mb: int = 8192,
    enable_auth: bool = True,
    enable_rate_limit: bool = True,
    max_context_tokens: int | None = None,
    default_reasoning: str = "auto",
) -> FastAPI:
    started_at = time.time()
    rate_limiter = RateLimiter() if enable_rate_limit else None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Toki Recur daemon starting")
        if _get_auth_token() == "demo-token":
            logger.warning(
                "auth token is the well-known default 'demo-token'; set "
                "TOKI_AUTH_TOKEN (or ~/.config/tokirecur/auth) to a real "
                "secret before exposing the daemon beyond localhost",
            )
        yield
        logger.info("Toki Recur daemon shutting down")

    app = FastAPI(title="Toki Recur", version="0.2.0", lifespan=lifespan)
    engine = RecyclingEngine(backend, index, similarity_threshold=similarity_threshold)

    # The engine's serve_* methods are synchronous and mutate shared slot
    # state. Run them in a threadpool (so concurrent requests don't stall
    # the event loop) but serialize them behind a lock (so state mutations
    # never interleave).
    engine_lock = asyncio.Lock()

    async def _run_engine(call, *args, **kwargs):
        async with engine_lock:
            return await asyncio.to_thread(call, *args, **kwargs)

    if max_context_tokens:
        # Match the engine's per-slot recycling budget to llama-server's
        # context window (llama.cpp truncates n_predict to what fits in
        # n_ctx, so a budget that ignores n_ctx silently caps responses).
        engine._max_context_tokens = max_context_tokens
    app.state.backend = backend
    app.state.index = index
    app.state.cache_dir = cache_dir
    app.state.memory_budget_mb = memory_budget_mb
    app.state.started_at = started_at

    # ---- Middleware ----
    @app.middleware("http")
    async def observe_and_authorize(request: Request, call_next):
        # Skip auth on health/metrics (monitoring needs access).
        path = request.url.path
        public = path in ("/healthz", "/metrics")
        if enable_auth and not public:
            try:
                _check_auth(request)
            except HTTPException as e:
                METRICS.observe_request(path, e.status_code, 0.0)
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=e.status_code,
                    content={"detail": e.detail},
                    headers=e.headers or {},
                )
        # Rate limit (also skip for public). Cheap admin/read endpoints
        # that UIs poll frequently (cache state, cache reset, precision)
        # are exempt from rate limiting — the limit protects inference,
        # not monitoring. Auth still applies to them.
        rate_exempt = path.startswith("/v1/cache/") or path == "/v1/precision"
        if rate_limiter is not None and not public and not rate_exempt:
            session_id = request.headers.get("x-session-id")
            key = _get_rate_key(request, session_id)
            if not rate_limiter.check(key):
                METRICS.rate_limit_rejections.inc()
                METRICS.observe_request(path, 429, 0.0)
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=429,
                    content={"detail": "rate limit exceeded"},
                )
        # Dispatch and time.
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - t0
        METRICS.observe_request(path, response.status_code, elapsed)
        return response

    # ---- Public endpoints ----

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "ts": time.time()}

    @app.get("/metrics")
    async def metrics() -> Response:
        METRICS.uptime_seconds.set(time.time() - started_at)
        # active_slots = slots with non-zero KV context (the engine
        # evicts spilled slots from its map, so this tracks "live").
        state = engine.slot_state()
        METRICS.active_slots.set(
            sum(1 for s in state.get("slots", []) if s.get("context_tokens", 0) > 0)
        )
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    # ---- Admin endpoints ----

    @app.get("/cache/stats")
    async def cache_stats() -> dict:
        return {
            "index": index.stats(),
            "cache_dir": str(cache_dir),
            "memory_budget_mb": memory_budget_mb,
        }

    @app.post("/cache/lookup")
    async def cache_lookup(req: CacheLookupRequest) -> dict:
        emb = await asyncio.to_thread(backend.embed, req.prompt)
        match = index.search(emb)
        if match is None:
            return {"match": None}
        return {
            "match": {
                "cache_id": match.cache_id,
                "similarity": match.similarity,
                "prompt_preview": match.prompt[:200],
            }
        }

    @app.get("/v1/cache/state")
    async def cache_state() -> dict:
        return engine.slot_state()

    @app.post("/v1/cache/reset")
    async def cache_reset() -> dict:
        engine.reset_state()
        return {"status": "reset"}

    @app.get("/v1/health/llama")
    async def health_llama() -> dict:
        """Probe the upstream llama-server. Returns ok/unreachable."""
        try:
            info = backend.cache_info(0)  # cheap call that hits the server
            if info.get("model_id"):
                return {"status": "ok"}
        except Exception:
            pass
        return {"status": "unreachable"}

    @app.post("/v1/precision")
    async def precision_check(req: PrecisionRequest) -> dict:
        """Run the precision retention check against llama-server directly."""
        try:
            data = req.prompts or [
                "List three Python web frameworks in order.",
                "What is the capital of France?",
                "Explain the recycling algorithm in plain English.",
            ]
            max_tokens = req.max_tokens or 32
            baseline_identical = 0
            recycled_identical = 0
            cross_identical = 0
            per_prompt = []
            # We don't have direct llama-server access here; reuse
            # `backend.complete()`. Baseline must run with cache_prompt
            # disabled or it would hit llama-server's slot cache and the
            # precision comparison would silently degenerate to "same
            # path vs same path".
            for idx, prompt in enumerate(data):
                base_runs = []
                rec_runs = []
                for _ in range(3):
                    try:
                        r = backend.complete(
                            prompt, slot_id=0, max_tokens=max_tokens,
                            temperature=0.0, cache_prompt=False,
                        )
                        base_runs.append(r.get("text", ""))
                    except Exception:
                        base_runs.append("")
                slot = 1 + idx
                for _ in range(3):
                    try:
                        r = backend.complete(
                            prompt, slot_id=slot, max_tokens=max_tokens,
                            temperature=0.0,
                        )
                        rec_runs.append(r.get("text", ""))
                    except Exception:
                        rec_runs.append("")
                base_match = len(set(base_runs)) == 1 and base_runs[0] != ""
                rec_match = len(set(rec_runs)) == 1 and rec_runs[0] != ""
                cross_match = base_runs[0] != "" and base_runs[0] == rec_runs[0]
                if base_match:
                    baseline_identical += 1
                if rec_match:
                    recycled_identical += 1
                if cross_match:
                    cross_identical += 1
                per_prompt.append({
                    "prompt": prompt[:50],
                    "base_match": base_match,
                    "rec_match": rec_match,
                    "cross_match": cross_match,
                })
            n = len(data)
            return {
                "baseline_deterministic": (
                    baseline_identical / n if n else 0.0
                ),
                "recycled_deterministic": (
                    recycled_identical / n if n else 0.0
                ),
                "byte_identical_to_baseline": (
                    cross_identical / n if n else 0.0
                ),
                "prompts_tested": n,
                "per_prompt": per_prompt,
            }
        except Exception as e:
            return {
                "baseline_deterministic": 0.0,
                "recycled_deterministic": 0.0,
                "byte_identical_to_baseline": 0.0,
                "prompts_tested": 0,
                "per_prompt": [],
                "error": str(e),
            }

    @app.get("/v1/metrics/live")
    async def metrics_live() -> dict:
        """Parse this daemon's own Prometheus /metrics into a small JSON dict."""
        # We don't have direct access to /metrics from inside the app,
        # so we re-emit the key counters based on internal state.
        try:
            from prometheus_client import REGISTRY
            out = {
                "requests_total": 0, "cache_hits": 0, "cache_misses": 0,
                "tokens_skipped": 0, "active_slots": 0, "auth_failures": 0,
            }
            state = engine.slot_state()
            out["active_slots"] = len(state.get("slots", []))
            for metric in REGISTRY.collect():
                if metric.name == "toki_requests_total":
                    out["requests_total"] = sum(s.value for s in metric.samples)
                elif metric.name == "toki_cache_hits_total":
                    for s in metric.samples:
                        kind = s.labels.get("kind", "")
                        if kind == "hit":
                            out["cache_hits"] = int(s.value)
                        elif kind == "miss":
                            out["cache_misses"] = int(s.value)
                elif metric.name == "toki_tokens_skipped_total":
                    out["tokens_skipped"] = int(sum(s.value for s in metric.samples))
                elif metric.name == "toki_auth_failures_total":
                    out["auth_failures"] = int(sum(s.value for s in metric.samples))
            return out
        except Exception:
            return {
                "requests_total": 0, "cache_hits": 0, "cache_misses": 0,
                "tokens_skipped": 0, "active_slots": 0, "auth_failures": 0,
            }

    # ---- Completion endpoints ----

    async def _stream_completion(
        prompt: str,
        messages: list | None,
        max_tokens: int,
        temperature: float,
        max_slots: int,
        session_id: str | None,
        model_name: str,
        completion_id: str,
        is_chat: bool,
        chat_template_kwargs: dict | None = None,
        tools: list | None = None,
        tool_choice: str | dict | None = None,
    ) -> Any:
        """Generator that yields SSE-formatted OpenAI chunks.

        If ``messages`` is provided, routes through the engine's
        ``serve_chat_stream`` path (OpenAI-format → llama-server
        ``/v1/chat/completions``) and streams tokens as they arrive.
        Otherwise falls back to ``serve_native`` with a flat prompt
        string.

        In tool mode (``tools`` set) the model may emit tool calls
        instead of text: ``delta.tool_calls`` chunks are relayed
        verbatim and the terminal chunk carries the real finish reason
        (``tool_calls`` / ``stop``) instead of a hardcoded "stop".
        """
        import json as _json
        object_type = "chat.completion.chunk" if is_chat else "text_completion"
        # Thinking-mode headroom: never let llama-server cut a model off
        # inside its reasoning tokens. Cheap for non-thinking models
        # (they stop on EOS). Native completions can't think, so they
        # get the non-thinking floor.
        max_tokens = _resolve_max_tokens(
            max_tokens, reasoning=default_reasoning if is_chat else "off"
        )

        def chunk_with(
            delta: dict,
            finish_reason: str | None = None,
            usage: dict | None = None,
            cache: dict | None = None,
        ) -> str:
            payload = {
                'id': completion_id,
                'object': object_type,
                'created': int(time.time()),
                'model': model_name,
                'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}],
            }
            if usage is not None:
                payload['usage'] = usage
            if cache is not None:
                payload['cache'] = cache
            return f"data: {_json.dumps(payload)}\n\n"

        # First chunk: role for chat, empty for completion.
        yield chunk_with({"role": "assistant", "content": ""} if is_chat else {"content": ""})

        finish_reason: str | None = None

        if messages is not None and is_chat:
            # Default thinking mode comes from the model's registry
            # `reasoning` setting (threaded via TOKI_REASONING): 'off'
            # disables Qwen-style thinking, anything else enables it so
            # the response includes reasoning_content deltas.
            if chat_template_kwargs is None:
                chat_template_kwargs = _default_chat_template_kwargs(default_reasoning)
            try:
                result = None
                async for item in engine.serve_chat_stream(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_slots=max_slots,
                    session_id=session_id,
                    chat_template_kwargs=chat_template_kwargs,
                    tools=tools,
                    tool_choice=tool_choice,
                ):
                    kind = item["kind"]
                    if kind == "delta":
                        yield chunk_with({"content": item["delta"]})
                    elif kind == "reasoning":
                        # Streamed in a separate SSE delta so OpenAI-aware
                        # clients can render it dimmed; clients that don't
                        # know the field simply ignore it.
                        yield chunk_with({"reasoning_content": item["delta"]})
                    elif kind == "tool_calls":
                        # OpenAI delta-format tool-call chunk, verbatim
                        # from llama-server so SDK-side accumulation
                        # (by index) works unchanged.
                        yield chunk_with({"tool_calls": item["tool_calls"]})
                    elif kind == "finish":
                        if item.get("finish_reason"):
                            finish_reason = item["finish_reason"]
                    else:  # result
                        result = item["result"]
            except Exception as e:
                logger.exception("streaming chat completion failed")
                yield chunk_with(
                    {"content": ""}, finish_reason="error",
                )
                yield chunk_with(
                    {"content": f"[daemon error: {e}]"},
                )
                yield "data: [DONE]\n\n"
                return
            if result is None:
                yield "data: [DONE]\n\n"
                return
        else:
            # Non-chat (native) streaming: blocking call, approximate
            # token-by-token emission (native endpoint has no streaming
            # support in the engine yet).
            result = await _run_engine(
                engine.serve_native,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                max_slots=max_slots,
                session_id=session_id,
            )
            text = result.output_text
            chunk_size = 4  # characters per chunk
            for i in range(0, len(text), chunk_size):
                chunk_text = text[i:i + chunk_size]
                yield chunk_with({"content": chunk_text})

        # Final chunk: finish + usage + cache verdict (OpenAI-style: usage
        # arrives in a final chunk so clients can render prompt/completion
        # token counts and the cache hit/miss verdict). The finish reason
        # is the real one (e.g. "tool_calls" in tool mode), not always
        # "stop".
        yield chunk_with({}, finish_reason=finish_reason or "stop")
        yield chunk_with(
            {},
            usage={
                "prompt_tokens": result.prefilled_tokens,
                "completion_tokens": result.generated_tokens,
                "total_tokens": result.prefilled_tokens + result.generated_tokens,
            },
            cache={
                "hit": result.hit,
                "soft_hit": result.soft_hit,
                "cache_id": result.cache_id,
                "reuse_depth": result.reuse_depth,
                "embedding_similarity": result.embedding_similarity,
                "skipped_tokens": result.skipped_tokens,
                "ttft_ms": result.ttft_ms,
                "total_ms": result.total_ms,
                "slot_id": result.slot_id,
                "spilled": result.spilled,
            },
        )
        yield "data: [DONE]\n\n"
        # Record metrics.
        state = engine.slot_state()
        slot_ctx = next(
            (s["context_tokens"] for s in state["slots"] if s["slot_id"] == result.slot_id),
            0,
        )
        METRICS.observe_completion(
            ttft_ms=result.ttft_ms,
            prefill_tokens=result.prefilled_tokens,
            generated_tokens=result.generated_tokens,
            skipped_tokens=result.skipped_tokens,
            hit=result.hit,
            soft_hit=result.soft_hit,
            slot_id=result.slot_id,
            context_tokens=slot_ctx,
            spilled=result.spilled,
        )

    def _normalize_responses_tools(tools: list | None) -> list | None:
        """Normalize Responses-API tools to llama-server's chat shape.

        - flat function tools (``{"type": "function", "name": ...}``)
          are rewrapped with the function fields nested under
          ``function``, which llama-server's OpenAI parser requires;
        - ``namespace`` tools (codex multi-agent) are flattened into
          their nested function tools;
        - non-function tool types (``web_search``, ...) are dropped —
          llama-server cannot parse them and they carry no model
          decision for the local stack anyway.
        """
        if not tools:
            return tools
        out: list[dict] = []
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                nested = tool.get("tools") if isinstance(tool, dict) else None
                if isinstance(nested, list):
                    out.extend(_normalize_responses_tools(nested) or [])
                continue
            if "function" in tool:
                out.append(tool)
                continue
            if not tool.get("name"):
                continue
            fn = {k: tool[k] for k in (
                "name", "description", "parameters", "strict")
                if tool.get(k) is not None}
            fn.setdefault("parameters", {})
            out.append({"type": "function", "function": fn})
        return out or None

    def _normalize_responses_tool_choice(tool_choice: Any) -> Any:
        if (isinstance(tool_choice, dict)
                and tool_choice.get("type") == "function"
                and tool_choice.get("name")
                and "function" not in tool_choice):
            return {"type": "function",
                    "function": {"name": tool_choice["name"]}}
        return tool_choice

    def _responses_messages(req: ResponsesRequest) -> list[dict]:
        """Map a Responses API request to OpenAI chat messages.

        ``input`` may be a plain string or a list of items:
        messages (role + content parts), function_call_output (→ tool
        message) and function_call (→ assistant tool_calls; consecutive
        items are merged into one assistant message so llama-server's
        tool template sees a valid turn).
        """
        messages: list[dict] = []
        if req.instructions:
            messages.append({"role": "system", "content": req.instructions})

        if isinstance(req.input, str):
            messages.append({"role": "user", "content": req.input})
            return messages

        pending_tool_calls: list[dict] = []

        def flush_tool_calls() -> None:
            nonlocal pending_tool_calls
            if pending_tool_calls:
                messages.append({
                    "role": "assistant", "content": None,
                    "tool_calls": pending_tool_calls,
                })
                pending_tool_calls = []

        for item in req.input:
            if item.type == "message":
                flush_tool_calls()
                parts = item.content if isinstance(item.content, list) else []
                text = "".join(
                    p["text"] for p in parts
                    if isinstance(p, dict) and isinstance(p.get("text"), str)
                )
                role = item.role or "user"
                if role == "developer":
                    # llama-server maps developer → system internally,
                    # which breaks Qwen3.5's "system first" template
                    # when instructions already produced a system
                    # message. Fold it into the leading system message.
                    role = "system"
                if (role == "system" and messages
                        and messages[-1]["role"] == "system" and text):
                    messages[-1]["content"] = (
                        messages[-1]["content"] + "\n\n" + text
                    )
                else:
                    messages.append({"role": role, "content": text})
            elif item.type == "function_call_output":
                flush_tool_calls()
                output = item.output
                if not isinstance(output, str):
                    output = _json.dumps(output)
                messages.append({
                    "role": "tool", "tool_call_id": item.call_id or "",
                    "content": output,
                })
            elif item.type == "function_call":
                pending_tool_calls.append({
                    "id": item.call_id or "", "type": "function",
                    "function": {"name": item.name or "",
                                 "arguments": item.arguments or ""},
                })
        flush_tool_calls()
        return messages

    def _responses_output_items(result: Any, ts: int) -> list[dict]:
        """Responses-format output items for a finished completion.

        Tool mode emits only function_call items (the Responses API has
        no message item when the model asks for tools); text mode emits
        one completed message item.
        """
        output: list[dict] = []
        if result.reasoning:
            output.append({
                "id": f"rs-{ts}", "type": "reasoning", "status": "completed",
                "summary": [{"type": "summary_text", "text": result.reasoning}],
            })
        if result.tool_calls:
            for i, tc in enumerate(result.tool_calls):
                function = tc.get("function") or {}
                output.append({
                    "id": f"fc-{i}-{ts}", "type": "function_call",
                    "status": "completed",
                    "call_id": tc.get("id") or f"call_{i}_{ts}",
                    "name": function.get("name") or "",
                    "arguments": function.get("arguments") or "",
                })
            return output
        output.append({
            "id": f"msg-{ts}", "type": "message", "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": result.output_text,
                         "annotations": []}],
        })
        return output

    async def _stream_responses(req: ResponsesRequest) -> Any:
        """Responses-API SSE event stream over the chat pipeline.

        Codex requires ``wire_api = "responses"`` since v0.146; this
        shim translates the Responses request into chat messages, feeds
        them through the same cache-aware engine path, and re-emits the
        result as Responses events (response.created → output_text
        deltas → function_call_arguments deltas → response.completed).
        """
        messages = _responses_messages(req)
        max_tokens = req.max_output_tokens or 256
        reasoning_mode = (
            "off"
            if req.reasoning and req.reasoning.effort == "none"
            else default_reasoning
        )
        ctk = _default_chat_template_kwargs(reasoning_mode)
        resp_id = f"resp-{int(time.time() * 1000)}"
        msg_id = f"msg-{resp_id}"
        created = int(time.time())

        def event(name: str, payload: dict) -> str:
            # Codex's SSE parser (rust) requires a top-level "type"
            # matching the event name on every event; the real OpenAI
            # Responses stream includes it too.
            payload.setdefault("type", name)
            return f"event: {name}\ndata: {_json.dumps(payload)}\n\n"

        yield event("response.created", {
            "id": resp_id, "object": "response", "created_at": created,
            "status": "in_progress", "model": req.model,
        })
        yield event("response.in_progress", {
            "id": resp_id, "object": "response", "created_at": created,
            "status": "in_progress", "model": req.model,
        })
        yield event("response.output_item.added", {
            "id": resp_id, "object": "response", "created_at": created,
            "output_index": 0,
            "item": {"id": msg_id, "type": "message", "role": "assistant",
                     "status": "in_progress", "content": []},
        })
        yield event("response.content_part.added", {
            "id": resp_id, "object": "response", "created_at": created,
            "item_id": msg_id, "output_index": 0, "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        })

        finish_reason: str | None = None
        result: Any = None
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_state: dict[int, dict] = {}  # index → accumulated tool call
        try:
            async for item in engine.serve_chat_stream(
                messages,
                max_tokens=_resolve_max_tokens(max_tokens,
                                               reasoning=reasoning_mode),
                temperature=req.temperature,
                max_slots=req.max_slots,
                session_id=req.session_id,
                chat_template_kwargs=ctk,
                tools=_normalize_responses_tools(req.tools),
                tool_choice=_normalize_responses_tool_choice(req.tool_choice),
            ):
                kind = item["kind"]
                if kind == "delta":
                    text_parts.append(item["delta"])
                    yield event("response.output_text.delta", {
                        "id": resp_id, "object": "response",
                        "created_at": created,
                        "item_id": msg_id, "output_index": 0,
                        "content_index": 0, "delta": item["delta"],
                    })
                elif kind == "reasoning":
                    reasoning_parts.append(item["delta"])
                    yield event("response.reasoning_summary_text.delta", {
                        "id": resp_id, "object": "response",
                        "created_at": created,
                        "item_id": msg_id, "output_index": 0,
                        "summary": [{"type": "summary_text",
                                     "text": item["delta"]}],
                    })
                elif kind == "tool_calls":
                    for chunk in item["tool_calls"]:
                        idx = chunk.get("index", 0)
                        state = tool_state.setdefault(idx, {
                            "call_id": chunk.get("id"),
                            "name": (chunk.get("function") or {}).get("name"),
                            "args": "",
                        })
                        if chunk.get("id") and not state["call_id"]:
                            state["call_id"] = chunk["id"]
                        name = (chunk.get("function") or {}).get("name")
                        if name and not state["name"]:
                            state["name"] = name
                            fc_id = f"fc-{idx}-{resp_id}"
                            yield event("response.output_item.added", {
                                "id": resp_id, "object": "response",
                                "created_at": created, "output_index": idx,
                                "item": {"id": fc_id, "type": "function_call",
                                         "call_id": state["call_id"]
                                         or f"call_{idx}_{resp_id}",
                                         "name": state["name"],
                                         "arguments": "",
                                         "status": "in_progress"},
                            })
                        args = (chunk.get("function") or {}).get("arguments")
                        if args:
                            state["args"] += args
                            yield event("response.function_call_arguments.delta", {
                                "id": resp_id, "object": "response",
                                "created_at": created,
                                "item_id": f"fc-{idx}-{resp_id}",
                                "output_index": idx, "delta": args,
                            })
                elif kind == "finish":
                    if item.get("finish_reason"):
                        finish_reason = item["finish_reason"]
                else:  # result
                    result = item["result"]
        except Exception as e:
            logger.exception(
                "responses streaming failed; messages=%s", messages)
            yield event("response.failed", {
                "id": resp_id, "object": "response", "created_at": created,
                "status": "failed",
                "error": {"code": "server_error", "message": str(e)},
            })
            return
        if result is None:
            yield event("response.failed", {
                "id": resp_id, "object": "response", "created_at": created,
                "status": "failed",
                "error": {"code": "server_error", "message": "no result"},
            })
            return

        text = "".join(text_parts)
        reasoning = "".join(reasoning_parts)
        yield event("response.output_text.done", {
            "id": resp_id, "object": "response", "created_at": created,
            "item_id": msg_id, "output_index": 0, "content_index": 0,
            "text": text,
        })
        yield event("response.content_part.done", {
            "id": resp_id, "object": "response", "created_at": created,
            "item_id": msg_id, "output_index": 0, "content_index": 0,
            "part": {"type": "output_text", "text": text, "annotations": []},
        })
        yield event("response.output_item.done", {
            "id": resp_id, "object": "response", "created_at": created,
            "output_index": 0,
            "item": {"id": msg_id, "type": "message",
                     "status": "completed", "role": "assistant",
                     "content": [{"type": "output_text", "text": text,
                                  "annotations": []}]},
        })

        tool_calls = result.tool_calls or []
        for idx, tc in enumerate(tool_calls):
            state = tool_state.get(idx, {})
            function = tc.get("function") or {}
            fc_id = f"fc-{idx}-{resp_id}"
            yield event("response.function_call_arguments.done", {
                "id": resp_id, "object": "response", "created_at": created,
                "item_id": fc_id, "output_index": idx,
                "arguments": state.get("args") or "",
            })
            yield event("response.output_item.done", {
                "id": resp_id, "object": "response", "created_at": created,
                "output_index": idx,
                "item": {"id": fc_id, "type": "function_call",
                         "status": "completed",
                         "call_id": state.get("call_id") or tc.get("id")
                         or f"call_{idx}_{resp_id}",
                         "name": function.get("name") or "",
                         "arguments": function.get("arguments") or ""},
            })

        if reasoning:
            yield event("response.reasoning_summary_text.done", {
                "id": resp_id, "object": "response", "created_at": created,
                "item_id": msg_id, "output_index": 0,
                "summary": [{"type": "summary_text", "text": reasoning}],
            })

        usage = {
            "input_tokens": result.prefilled_tokens,
            "output_tokens": result.generated_tokens,
            "total_tokens": result.prefilled_tokens + result.generated_tokens,
        }
        cache = {
            "hit": result.hit, "soft_hit": result.soft_hit,
            "cache_id": result.cache_id, "reuse_depth": result.reuse_depth,
            "embedding_similarity": result.embedding_similarity,
            "skipped_tokens": result.skipped_tokens,
            "ttft_ms": result.ttft_ms, "total_ms": result.total_ms,
            "slot_id": result.slot_id, "spilled": result.spilled,
        }
        # codex parses the terminal event's payload from the nested
        # `response` key (see codex-rs codex-api/src/sse/responses.rs
        # ResponseCompleted); top-level fields are ignored, which made
        # codex treat the stream as closed before response.completed.
        yield event("response.completed", {
            "response": {
                "id": resp_id, "object": "response", "created_at": created,
                "status": "completed", "model": req.model,
                "incomplete_details": (
                    {"reason": "max_output_tokens"}
                    if finish_reason == "length" else None
                ),
                "output": _responses_output_items(result, int(time.time() * 1000)),
                "usage": usage, "token_details": None, "metadata": {},
                "cache": cache,
            },
        })
        # Note: no `data: [DONE]` terminator here — the Responses SSE
        # format ends with response.completed; codex's parser rejects
        # the chat-format [DONE] marker as an unparseable event.
        state = engine.slot_state()
        slot_ctx = next(
            (s["context_tokens"] for s in state["slots"]
             if s["slot_id"] == result.slot_id),
            0,
        )
        METRICS.observe_completion(
            ttft_ms=result.ttft_ms,
            prefill_tokens=result.prefilled_tokens,
            generated_tokens=result.generated_tokens,
            skipped_tokens=result.skipped_tokens,
            hit=result.hit,
            soft_hit=result.soft_hit,
            slot_id=result.slot_id,
            context_tokens=slot_ctx,
            spilled=result.spilled,
        )

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest) -> Response:
        try:
            if req.stream:
                completion_id = f"cmpl-{int(time.time() * 1000)}"
                return StreamingResponse(
                    _stream_completion(
                        req.prompt,
                        None,
                        req.max_tokens,
                        req.temperature,
                        req.max_slots,
                        req.session_id,
                        req.model,
                        completion_id,
                        is_chat=False,
                    ),
                    media_type="text/event-stream",
                )
            result = await _run_engine(
                engine.serve_native,
                req.prompt,
                max_tokens=_resolve_max_tokens(req.max_tokens, reasoning="off"),
                temperature=req.temperature,
                max_slots=req.max_slots,
                session_id=req.session_id,
            )
            # Record metrics.
            state = engine.slot_state()
            slot_ctx = next(
                (s["context_tokens"] for s in state["slots"] if s["slot_id"] == result.slot_id),
                0,
            )
            METRICS.observe_completion(
                ttft_ms=result.ttft_ms,
                prefill_tokens=result.prefilled_tokens,
                generated_tokens=result.generated_tokens,
                skipped_tokens=result.skipped_tokens,
                hit=result.hit,
                soft_hit=result.soft_hit,
                slot_id=result.slot_id,
                context_tokens=slot_ctx,
                spilled=result.spilled,
            )
            return CompletionResponse(
                id=f"cmpl-{int(time.time() * 1000)}",
                model=req.model,
                created=int(time.time()),
                choices=[
                    {
                        "text": result.output_text,
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
                usage={
                    "prompt_tokens": result.prefilled_tokens,
                    "completion_tokens": result.generated_tokens,
                    "total_tokens": result.prefilled_tokens + result.generated_tokens,
                },
                cache={
                    "hit": result.hit,
                    "cache_id": result.cache_id,
                    "reuse_depth": result.reuse_depth,
                    "embedding_similarity": result.embedding_similarity,
                    "skipped_tokens": result.skipped_tokens,
                    "ttft_ms": result.ttft_ms,
                    "total_ms": result.total_ms,
                    "slot_id": result.slot_id,
                    "spilled": result.spilled,
                },
            )
        except Exception as e:
            logger.exception("completion failed")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> Response:
        # Resolve the model name the client sent. hermes/openclaw/etc.
        # can ship a model id that doesn't match the active spec (e.g.
        # the historical "unsloth/<repo>-GGUF" form vs the spec's
        # short "qwen3.5-0.8b" form). We accept any model name when
        # only one model is loaded — the daemon is single-tenant.
        # Override `req.model` with the active spec's model_id so
        # downstream consumers (and the response) see a consistent name.
        active_model = _active_model_id()
        if active_model and req.model and req.model != active_model:
            logger.debug(
                "chat_completions: client model %r != active %r; "
                "using active", req.model, active_model,
            )
            req = req.model_copy(update={"model": active_model})
        # Pass OpenAI-format messages through (including tool_calls /
        # tool_call_id fields for agentic clients). llama-server applies
        # the chat template appropriate for the loaded model.
        messages = [m.model_dump(exclude_none=True) for m in req.messages]
        try:
            if req.stream:
                completion_id = f"chatcmpl-{int(time.time() * 1000)}"
                return StreamingResponse(
                    _stream_completion(
                        None,  # no flat prompt
                        messages,
                        req.max_tokens,
                        req.temperature,
                        req.max_slots,
                        req.session_id,
                        req.model,
                        completion_id,
                        is_chat=True,
                        chat_template_kwargs=req.chat_template_kwargs,
                        tools=req.tools,
                        tool_choice=req.tool_choice,
                    ),
                    media_type="text/event-stream",
                )
            # Default thinking mode from the model's `reasoning` setting;
            # the client can override with explicit chat_template_kwargs.
            ctk = req.chat_template_kwargs
            if ctk is None:
                ctk = _default_chat_template_kwargs(default_reasoning)
            result = await _run_engine(
                engine.serve_chat,
                messages,
                max_tokens=_resolve_max_tokens(req.max_tokens, reasoning=default_reasoning),
                temperature=req.temperature,
                max_slots=req.max_slots,
                session_id=req.session_id,
                chat_template_kwargs=ctk,
                tools=req.tools,
                tool_choice=req.tool_choice,
            )
            state = engine.slot_state()
            slot_ctx = next(
                (s["context_tokens"] for s in state["slots"] if s["slot_id"] == result.slot_id),
                0,
            )
            METRICS.observe_completion(
                ttft_ms=result.ttft_ms,
                prefill_tokens=result.prefilled_tokens,
                generated_tokens=result.generated_tokens,
                skipped_tokens=result.skipped_tokens,
                hit=result.hit,
                soft_hit=result.soft_hit,
                slot_id=result.slot_id,
                context_tokens=slot_ctx,
                spilled=result.spilled,
            )
            # OpenAI chat response shape. In tool mode the assistant
            # message carries tool_calls (and null content) and the
            # finish reason is "tool_calls".
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": result.output_text,
            }
            if result.reasoning:
                assistant_message["reasoning_content"] = result.reasoning
            if result.tool_calls:
                assistant_message["content"] = None
                assistant_message["tool_calls"] = result.tool_calls
            return CompletionResponse(
                id=f"chatcmpl-{int(time.time() * 1000)}",
                object="chat.completion",
                model=req.model,
                created=int(time.time()),
                choices=[
                    {
                        "message": assistant_message,
                        "index": 0,
                        "finish_reason": result.finish_reason or "stop",
                    }
                ],
                usage={
                    "prompt_tokens": result.prefilled_tokens,
                    "completion_tokens": result.generated_tokens,
                    "total_tokens": result.prefilled_tokens + result.generated_tokens,
                },
                cache={
                    "hit": result.hit,
                    "cache_id": result.cache_id,
                    "reuse_depth": result.reuse_depth,
                    "embedding_similarity": result.embedding_similarity,
                    "skipped_tokens": result.skipped_tokens,
                    "ttft_ms": result.ttft_ms,
                    "total_ms": result.total_ms,
                    "slot_id": result.slot_id,
                    "spilled": result.spilled,
                },
            )
        except Exception as e:
            logger.exception("chat completion failed")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/v1/responses")
    async def responses(req: ResponsesRequest) -> Response:
        """OpenAI Responses API shim (Codex and other 2025-era clients).

        Codex ≥ v0.146 refuses ``wire_api = "chat"``, so this endpoint
        maps a Responses request onto the chat-completions pipeline —
        same cache lookup, thinking mode, tool relay and rate limiting —
        and speaks the Responses wire format (non-streaming JSON body or
        SSE events with ``event:`` lines). ``previous_response_id`` and
        ``store`` are ignored: toki is stateless, like the other
        OpenAI-compatible endpoints.
        """
        try:
            if req.stream:
                return StreamingResponse(
                    _stream_responses(req),
                    media_type="text/event-stream",
                )
            messages = _responses_messages(req)
            max_tokens = req.max_output_tokens or 256
            reasoning_mode = (
                "off"
                if req.reasoning and req.reasoning.effort == "none"
                else default_reasoning
            )
            ctk = _default_chat_template_kwargs(reasoning_mode)
            result = await _run_engine(
                engine.serve_chat,
                messages,
                max_tokens=_resolve_max_tokens(max_tokens,
                                               reasoning=reasoning_mode),
                temperature=req.temperature,
                max_slots=req.max_slots,
                session_id=req.session_id,
                chat_template_kwargs=ctk,
                tools=_normalize_responses_tools(req.tools),
                tool_choice=_normalize_responses_tool_choice(req.tool_choice),
            )
            state = engine.slot_state()
            slot_ctx = next(
                (s["context_tokens"] for s in state["slots"]
                 if s["slot_id"] == result.slot_id),
                0,
            )
            METRICS.observe_completion(
                ttft_ms=result.ttft_ms,
                prefill_tokens=result.prefilled_tokens,
                generated_tokens=result.generated_tokens,
                skipped_tokens=result.skipped_tokens,
                hit=result.hit,
                soft_hit=result.soft_hit,
                slot_id=result.slot_id,
                context_tokens=slot_ctx,
                spilled=result.spilled,
            )
            created = int(time.time())
            ts = int(time.time() * 1000)
            return Response(
                content=_json.dumps({
                    "id": f"resp-{ts}", "object": "response",
                    "created_at": created, "status": "completed",
                    "error": None,
                    "incomplete_details": (
                        {"reason": "max_output_tokens"}
                        if result.finish_reason == "length" else None
                    ),
                    "model": req.model,
                    "output": _responses_output_items(result, ts),
                    "output_text": result.output_text,
                    "usage": {
                        "input_tokens": result.prefilled_tokens,
                        "output_tokens": result.generated_tokens,
                        "total_tokens": (
                            result.prefilled_tokens + result.generated_tokens
                        ),
                    },
                    "token_details": None,
                    "metadata": {},
                    "cache": {
                        "hit": result.hit,
                        "cache_id": result.cache_id,
                        "reuse_depth": result.reuse_depth,
                        "embedding_similarity": result.embedding_similarity,
                        "skipped_tokens": result.skipped_tokens,
                        "ttft_ms": result.ttft_ms,
                        "total_ms": result.total_ms,
                        "slot_id": result.slot_id,
                        "spilled": result.spilled,
                    },
                }),
                media_type="application/json",
            )
        except Exception as e:
            logger.exception("responses completion failed")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/cache/list")
    async def cache_list() -> dict:
        return {
            "entries": index.entries(),
            "cache_dir": str(cache_dir),
        }

    @app.delete("/cache/{cache_id}")
    async def cache_delete(cache_id: str) -> dict:
        from ..cache.blob import blob_path

        removed = index.delete(cache_id)
        blob = blob_path(cache_dir, cache_id)
        if blob.exists():
            blob.unlink()
        if not removed:
            raise HTTPException(
                status_code=404,
                detail=f"cache_id {cache_id!r} not found",
            )
        return {"status": "deleted", "cache_id": cache_id}

    return app


def create_app_from_env() -> FastAPI:
    """Factory suitable for `uvicorn tokirecur.api.app:create_app_from_env --factory`.

    Reads configuration from environment variables:
      TOKI_BACKEND_URL          default http://127.0.0.1:8088
      TOKI_DB                   default ~/.tokirecur/index.sqlite
      TOKI_CACHE_DIR            default ~/.tokirecur/blobs
      TOKI_SIMILARITY_THRESHOLD default 0.85
      TOKI_MEMORY_BUDGET_MB     default 8192
      TOKI_MAX_CONTEXT_TOKENS   per-slot recycling budget; set to
                                (llama-server n_ctx - max response tokens).
                                Default 6144 (for -c 8192).
      TOKI_AUTH_TOKEN           optional; required for non-loopback binding
      TOKI_REASONING            the model's `reasoning` registry setting
                                (auto|on|off); decides whether the daemon
                                defaults to enabling thinking mode.
    """
    from ..backend import LlamaServerBackend
    from ..index import BruteForceIndex

    backend = LlamaServerBackend(
        os.environ.get("TOKI_BACKEND_URL", "http://127.0.0.1:8088"),
        cache_dir=os.environ.get("TOKI_CACHE_DIR"),
    )
    index = BruteForceIndex(
        os.environ.get("TOKI_DB", str(Path.home() / ".tokirecur/index.sqlite"))
    )
    return create_app(
        backend=backend,
        index=index,
        cache_dir=Path(os.environ.get("TOKI_CACHE_DIR", str(Path.home() / ".tokirecur/blobs"))),
        similarity_threshold=float(os.environ.get("TOKI_SIMILARITY_THRESHOLD", "0.85")),
        memory_budget_mb=int(os.environ.get("TOKI_MEMORY_BUDGET_MB", "8192")),
        max_context_tokens=_env_int("TOKI_MAX_CONTEXT_TOKENS"),
        default_reasoning=os.environ.get("TOKI_REASONING", "auto"),
    )


def _env_int(name: str) -> int | None:
    """Parse an integer env var, returning None when unset/invalid."""
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None
