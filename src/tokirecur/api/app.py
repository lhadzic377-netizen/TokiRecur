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
import hmac
import json as _json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from .. import tap
from ..backend import _normalize_messages
from ..core.recycling import (
    LlamaBackend,
    RecyclingEngine,
)
from ..daemon.metrics import METRICS
from ..daemon.ratelimit import RateLimiter
from ..index import Index
from .anthropic import (
    AnthropicMessagesRequest,
    approximate_input_text,
    chat_template_kwargs,
    error_event,
    response_message,
    stream_message,
    to_openai_messages,
    to_openai_tool_choice,
    to_openai_tools,
)
from .anthropic import (
    session_id as anthropic_session_id,
)

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
MAX_REQUEST_TOKENS = 262144

# Sampling defaults for every request the daemon serves. Greedy decode
# (temperature 0.0) with no repetition penalty is the loop engine behind
# agent harnesses — identical context reproduces identical tokens and
# the model re-sends the same tool call forever. These defaults break
# that lock; they are decode-time only, so the KV cache is untouched.
# The active model's profile comes in as TOKI_SAMPLING (JSON, set by ctl
# at daemon spawn from DEFAULT_SAMPLING + the registry `sampling:` block).
# A client value for any field always wins over the default.
DEFAULT_SAMPLING = {"temperature": 0.4, "repeat_penalty": 1.15,
                    "top_k": 40, "min_p": 0.05}

_FLOOR_WARNED = False


def _sampling_defaults() -> dict[str, Any]:
    """Active model's sampling profile: TOKI_SAMPLING env (JSON) merged
    over the built-in defaults, so a partial registry block still gets
    the full profile."""
    merged = dict(DEFAULT_SAMPLING)
    try:
        raw = os.environ.get("TOKI_SAMPLING")
        if raw:
            from_env = _json.loads(raw)
            if isinstance(from_env, dict):
                merged.update({k: v for k, v in from_env.items()
                               if v is not None})
    except (ValueError, TypeError):
        logger.warning("ignoring malformed TOKI_SAMPLING: %r",
                       os.environ.get("TOKI_SAMPLING"))
    return merged


def _resolve_sampling(
    *,
    defaults: dict[str, Any] | None = None,
    temperature: float | None = None,
    repeat_penalty: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    default_temperature: float = 0.4,
) -> tuple[float, dict[str, Any]]:
    """Resolve a request's effective sampling profile.

    Returns ``(temperature, sampling)`` where ``sampling`` carries only
    the non-None repetition/top-k/top-p/min-p values (temperature is a
    separate positional parameter all the way to llama-server). Client
    values win; None falls back to the model defaults.
    """
    d = defaults if defaults is not None else _sampling_defaults()
    temp = temperature if temperature is not None else \
        d.get("temperature", default_temperature)
    sampling: dict[str, Any] = {}
    for key, value in (
        ("repeat_penalty", repeat_penalty),
        ("top_p", top_p),
        ("top_k", top_k),
        ("min_p", min_p),
    ):
        if value is None:
            value = d.get(key)
        if value is not None:
            sampling[key] = value
    return temp, sampling


def _responses_sampling_defaults() -> dict[str, Any]:
    """Sampling profile for the Responses (codex) endpoint: the model's
    repetition controls but WITHOUT the default temperature, so OpenAI's
    temperature=1.0 default applies when the client sends nothing
    (codex streams temperature=0.0 by default, which combined with no
    penalties is the loop amplifier on agents)."""
    return {k: v for k, v in _sampling_defaults().items()
            if k != "temperature"}


def _default_chat_template_kwargs(reasoning: str) -> dict:
    """Qwen-style chat templates need an explicit ``enable_thinking``
    kwarg. ``off`` disables thinking so the response contains only
    visible content; anything else (``auto``/``on``) enables it so the
    model emits reasoning the clients can render. Other templates
    ignore the field.
    """
    return {"enable_thinking": reasoning != "off"}


def _active_vision_stack() -> dict | None:
    """A vision-capable stack that can caption images right now.

    The daemon serves one model, and that model is usually text-only — so an
    image sent through it reaches a llama-server with no projector and dies
    with ``all input arrays must have the same shape``. Images therefore go
    through a *vision* stack instead. Returns ``{"llama_port", "model_id",
    "name"}`` for the first registered stack whose projector file is present
    and whose llama-server answers, or None when there is none.

    ``TOKI_VISION_MODEL`` pins a registry name; otherwise registry order
    wins, which is stable.
    """
    try:
        from ..ctl import load_registry
    except Exception:
        return None
    try:
        specs = load_registry()
    except Exception:
        return None
    pinned = (os.environ.get("TOKI_VISION_MODEL") or "").strip()
    ordered = sorted(
        specs,
        key=lambda s: (0 if getattr(s, "name", "") == pinned else 1),
    )
    import httpx

    for spec in ordered:
        projector = getattr(spec, "mmproj", "") or ""
        if not projector or not Path(projector).exists():
            continue
        port = getattr(spec, "llama_port", 0) or 0
        if not port:
            continue
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
            if response.status_code != 200:
                continue
        except Exception:
            continue
        return {
            "llama_port": port,
            "model_id": getattr(spec, "model_id", "") or getattr(spec, "name", ""),
            "name": getattr(spec, "name", ""),
        }
    return None


def _active_model_sees_images() -> bool:
    """True when the currently served model has its own projector, in which
    case images pass through natively and need no captioning."""
    try:
        from ..ctl import get_model, read_active
        name = read_active()
        if not name:
            return False
        spec = get_model(name)
        projector = getattr(spec, "mmproj", "") or ""
        return bool(projector and Path(projector).exists())
    except Exception:
        return False


async def _caption_image(data_url: str, stack: dict) -> str | None:
    """One image to one or two sentences, via the vision stack's llama-server
    directly (not the daemon — the daemon is the text path that cannot carry
    images in the first place)."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"http://127.0.0.1:{stack['llama_port']}/v1/chat/completions",
                json={
                    "model": stack["model_id"],
                    "stream": False,
                    "temperature": 0,
                    "max_tokens": 150,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text",
                             "text": "Describe what is visible in one or two sentences. "
                                     "Do not speculate about anything you cannot see."},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }],
                },
            )
            if response.status_code != 200:
                return None
            data = response.json()
            message = (data.get("choices") or [{}])[0].get("message") or {}
            text = (message.get("content") or "").strip()
            return text or None
    except Exception as e:
        logger.warning("vision caption failed: %s", e)
        return None


async def _inline_images(messages: list[dict]) -> list[dict]:
    """Replace ``image_url`` content parts with caption text.

    The active model is usually text-only, so forwarding pixels guarantees a
    500 from a projector-less llama-server. Captioning first keeps the turn
    alive on every model; when the active model sees natively, messages pass
    through untouched. At most 8 images per request are captioned — beyond
    that the turn would take minutes, so the rest are marked omitted.
    """
    has_image = any(
        isinstance(m.get("content"), list)
        and any(isinstance(p, dict) and p.get("type") == "image_url"
                for p in m["content"])
        for m in messages
    )
    if not has_image:
        return messages
    if _active_model_sees_images():
        return messages
    stack = _active_vision_stack()
    out: list[dict] = []
    captioned = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        parts: list[dict] = []
        for part in content:
            if not (isinstance(part, dict) and part.get("type") == "image_url"):
                parts.append(part)
                continue
            url = ((part.get("image_url") or {}).get("url") or "").strip()
            if not url:
                parts.append({"type": "text", "text": "[attached image omitted: empty]"})
                continue
            if stack is None:
                parts.append({"type": "text",
                              "text": "[attached image omitted: no vision model is "
                                      "registered — the model cannot see it]"})
                continue
            if captioned >= 8:
                parts.append({"type": "text",
                              "text": "[attached image omitted: too many images in one turn]"})
                continue
            captioned += 1
            caption = await _caption_image(url, stack)
            parts.append({"type": "text",
                          "text": f"[attached image: {caption}]" if caption
                                 else "[attached image omitted: captioning failed]"})
        out.append({**message, "content": parts})
    return out


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


def _max_output_ceiling() -> int | None:
    """Hard per-turn output budget in tokens (TOKI_MAX_OUTPUT_TOKENS,
    set from the registry's per-model ``max_output_tokens`` at spawn).
    Guards against runaway single generations: registry ``max_tokens``
    is advertising (the context window) and client-side caps are not
    always sent as max_tokens. None = no cap."""
    try:
        value = int(os.environ.get("TOKI_MAX_OUTPUT_TOKENS", ""))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _max_slots_ceiling() -> int | None:
    """The server's real ``--parallel`` slot count (TOKI_MAX_SLOTS, set
    by the stack's fit planner). The recycling engine round-robins over
    the client-requested slot range; a request above the real count
    would 400 on llama-server, so the daemon clamps."""
    try:
        value = int(os.environ.get("TOKI_MAX_SLOTS", ""))
    except (TypeError, ValueError):
        return None
    return value if 1 <= value <= 32 else None


def _resolve_max_slots(requested: int) -> int:
    """Clamp the client's slot request to the server's real slot count."""
    ceiling = _max_slots_ceiling()
    if ceiling and requested > ceiling:
        return ceiling
    return requested


def _context_overflow_status(e: Exception) -> int | None:
    """413 for llama.cpp's context-window overflow errors, else None.

    Clients (opencode, hermes) surface the status + detail to the user;
    a bare llama.cpp 400 "not enough tokens" was the old behavior that
    read as a generic failure and sent hermes into a compress/retry
    loop against a window it believed to be 256k.

    llama.cpp words this error two ways, and both are live: the older
    "not enough tokens in the context window" and the current "request
    (N tokens) exceeds the available context size (M tokens)". Missing
    the second one turned every overflow into a 500, which is the
    status a client reads as "the provider is gone" - MiniMax Code
    compacts and retries on a 413, and gives up on a 500.
    """
    msg = str(e)
    if (
        "not enough tokens in the context window" in msg
        or "exceeds the context window" in msg
        or "exceeds the available context size" in msg
        or "exceed_context_size_error" in msg
    ):
        return 413
    return None


def _openai_error(
    status: int, message: str, code: str | None = None,
) -> JSONResponse:
    """An error body in the shape OpenAI clients actually read.

    FastAPI's default is ``{"detail": ...}``. Every OpenAI-compatible
    client looks for ``error.message`` instead, so one of ours arrives
    as an empty body - MiniMax Code prints "413 status code (no body)"
    and retries, which hides the one message that would have told it
    to compact and try again. The ``detail`` key stays for the clients
    (and the dashboard) that read it.
    """
    return JSONResponse(
        status_code=status,
        content={
            "detail": message,
            "error": {
                "message": message,
                "type": (
                    "invalid_request_error" if status < 500 else "server_error"
                ),
                "code": code,
            },
        },
    )


def _prompt_text(messages: list[dict]) -> str:
    """The prompt's text, whatever shape the content came in."""
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if isinstance(part.get("text"), str):
                        parts.append(part["text"])
                elif isinstance(part, str):
                    parts.append(part)
        elif isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def _prompt_would_overflow(
    engine: Any, messages: list[dict], tools: list | None = None,
    ceiling: int | None = None,
) -> bool:
    """Whether the prompt alone cannot fit the slot's context window.

    The check has to run before the response headers go out. A client
    that streams gets the 413 as a status only while the turn has not
    started; once the stream is open, an overflow arrives as a stream
    that dies mid-turn, and the client concludes the provider is gone.
    MiniMax Code compacts and retries on a 413 and gives up on a 500.

    The window is the one the stack actually gave llama-server
    (``TOKI_MAX_CONTEXT_TOKENS``), not a number the client believes.

    The size is the one llama-server would really build: the chat
    template plus the tool schemas, which on an agent client are often
    more tokens than the conversation. Asking the server is the only
    way to be right - a count taken from the message text alone is short
    by the whole tool block.
    """
    if ceiling is None:
        ceiling = _max_context_ceiling()
    if not ceiling:
        return False
    text = _prompt_text(messages)
    tools_text = _json.dumps(tools) if tools else ""
    if not text and not tools_text:
        return False
    # Cheap filter first - roughly four characters to a token - so the
    # server is only asked when the estimate is near the ceiling.
    rough = (len(text) + len(tools_text)) // 4
    if rough + 1024 < ceiling:
        return False
    count = None
    try:
        prompt = engine.backend.apply_template(messages, tools)
        count = len(engine.backend.tokenize(prompt))
    except Exception:
        # A server that cannot render the template is not a reason to
        # refuse the request; the estimate is the best we have.
        count = rough
    # The template adds a little on top of the text it renders.
    return count + 64 > ceiling


def _tool_call_parse_failure(e: Exception) -> bool:
    """True when the model wrote a tool call the app could not read.

    A small model can emit a call whose arguments are one quote short. The
    engine then refuses the whole turn, and the raw error is a JSON blob
    with a column number in it — which tells whoever is reading it
    nothing they can act on. It is the model's mistake, not the reader's.
    """
    msg = str(e).lower()
    return "failed to parse tool call" in msg or "tool call arguments" in msg


def _overflow_detail(ceiling: int | None = None) -> str:
    """Human + machine-parseable explanation for a context overflow."""
    if ceiling is None:
        ceiling = _max_context_ceiling()
    window = f"{ceiling} tokens" if ceiling else "the daemon's configured window"
    return (
        f"request exceeds the model's real context window ({window}). "
        "The daemon only advertises what the KV cache actually fits."
    )


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


def _resolve_effort(
    model_id: str, requested: str | None, served: str | None = None,
) -> str | None:
    """The reasoning effort this turn runs at, or None for the model's own.

    A client may ask for an effort outright ("high"), or name a model
    variant ("qwen3.8-27b:high") and let the variant carry it. A
    variant's effort only applies when it names the model this daemon
    actually serves: the daemon is single-tenant, so an effort meant for
    another model would be answered by this one.

    An effort the level vocabulary does not know is ignored rather than
    guessed at — a template that does not understand "high" would answer
    with whatever it defaults to, and saying so is better than pretending.
    """
    try:
        from ..ctl import EFFORT_LEVELS, load_registry, resolve_model_id
    except Exception:
        return None
    try:
        specs = load_registry()
    except Exception:
        return None
    if not served:
        # The backend has not answered yet, or is not reachable. The
        # active model is the next best answer to "which model is this".
        try:
            from ..ctl import read_active
            served = read_active()
        except Exception:
            served = None
    if requested:
        wanted = requested.strip().lower()
        if wanted not in EFFORT_LEVELS:
            return None
        if wanted == "none":
            # Thinking off is always an answer the model can give.
            return wanted
        # The model may not take every level: Qwen3.8 raises on "max",
        # and a level its template refuses is a 500, not a default. The
        # model's own setting stands instead.
        served_spec = _served_spec(specs, served)
        levels = getattr(served_spec, "reasoning_efforts", None) or []
        if levels and wanted not in levels:
            return None
        return wanted
    spec, effort = resolve_model_id(specs, model_id)
    if spec is None or effort is None:
        return None
    if served and not _same_model(spec, served):
        return None
    return effort


def _stack_effort() -> str | None:
    """The effort the stack is being served at.

    The launcher pins it; every client that does not ask for a level of
    its own gets it. That is what makes the harnesses need no effort
    in their model ids: the effort is a property of the stack, like the
    model, and a client that wants a different one names it.
    """
    try:
        from ..ctl import read_active_effort
        return read_active_effort()
    except Exception:
        return None


def _served_spec(specs: list[Any], served: str | None) -> Any:
    """The registry spec for the model the daemon is serving."""
    if not served:
        return None
    wanted = served.strip().lower()
    for spec in specs:
        if wanted in {
            str(getattr(spec, "model_id", "")).strip().lower(),
            str(getattr(spec, "name", "")).strip().lower(),
        }:
            return spec
    return None


def _same_model(spec: Any, served: str) -> bool:
    """Whether the served id names this spec.

    The served id comes from the backend, which reports llama-server's own
    alias for the weights ("Qwen3.8-27B-Q4_K_M"); the registry spells it
    its own way ("qwen3.8-27b-q4_k_m"). Same model, different case — so
    the comparison is case-blind, and the name counts too.
    """
    wanted = served.strip().lower()
    return wanted in {
        str(getattr(spec, "model_id", "")).strip().lower(),
        str(getattr(spec, "name", "")).strip().lower(),
    }


def _resolve_max_tokens(requested: int, reasoning: str = "auto") -> int:
    """Floor at the thinking/non-thinking budget without capping larger
    client requests, then clamp to the real slot context when the stack
    provides one, then to the per-model output budget
    (TOKI_MAX_OUTPUT_TOKENS) when configured. Logs a one-time warning
    when a client's budget is silently raised.
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
    output_cap = _max_output_ceiling()
    if output_cap:
        resolved = min(resolved, max(output_cap, floor))
    return resolved


class CompletionRequest(BaseModel):
    model: str = "default"
    prompt: str
    # Hard bounds instead of trusting the client: llama.cpp truncates
    # out-of-range values silently, which made overrides look broken.
    max_tokens: int = Field(256, ge=1, le=MAX_REQUEST_TOKENS)
    # None = the active model's sampling defaults (temperature 0.4,
    # repeat_penalty 1.15, top_k 40, min_p 0.05) — greedy-with-no-
    # penalty is the loop amplifier behind agent harnesses.
    temperature: float | None = Field(None, ge=-2.0, le=2.0)
    repeat_penalty: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    top_k: int | None = Field(None, ge=0, le=100)
    min_p: float | None = Field(None, ge=0.0, le=1.0)
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
    max_tokens: int = Field(256, ge=1, le=MAX_REQUEST_TOKENS)
    # None = the active model's sampling defaults (temperature 0.4,
    # repeat_penalty 1.15, top_k 40, min_p 0.05). A client value always
    # wins. Greedy decode with no repetition penalty is the loop engine
    # behind agent harnesses — these defaults break the lock.
    temperature: float | None = Field(None, ge=-2.0, le=2.0)
    repeat_penalty: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    top_k: int | None = Field(None, ge=0, le=100)
    min_p: float | None = Field(None, ge=0.0, le=1.0)
    max_slots: int = Field(4, ge=1, le=32)
    session_id: str | None = None
    stream: bool = False
    # Tool calling (agentic clients): forwarded to llama-server, which
    # applies the model's chat template. When tools are present the
    # response may contain tool_calls instead of text, and the stream
    # relays llama-server's OpenAI deltas verbatim.
    tools: list | None = None
    tool_choice: str | dict | None = None
    # Reasoning effort for this turn: "none", "minimal", "low", "medium",
    # "high", "xhigh", "max". llama-server hands it to the model's chat
    # template, so the model steers its own depth — the levels mean
    # whatever that template says they mean. "none" is thinking off.
    # A client that instead names a variant ("qwen3.8-27b:high") gets
    # the same thing: the variant's effort applies.
    reasoning_effort: str | None = None
    # Pass-through for chat-template kwargs. If unset, the daemon
    # defaults to disabling Qwen3.5's thinking mode so the response
    # contains visible content instead of internal reasoning tokens.
    # Other models ignore this field.
    chat_template_kwargs: dict | None = None


class ResponsesItem(BaseModel):
    """One entry of the Responses API ``input`` array. The fields are
    intentionally loose: the OpenAI SDKs evolve item shapes faster than
    this shim, and unknown/extra fields should pass through untouched.

    ``type`` is optional because clients that count tokens (MiniMax
    Code) send ``{"role": ..., "content": ...}`` with no type at all.
    """

    type: str | None = None
    role: str | None = None
    content: Any = None  # str | list of input_text/input_image parts
    call_id: str | None = None
    output: Any = None  # function_call_output payload (any JSON)
    name: str | None = None
    arguments: str | None = None


class ResponsesReasoning(BaseModel):
    effort: str = "auto"


class ResponsesInputTokensRequest(BaseModel):
    """The body of ``POST /v1/responses/input_tokens``.

    Only the fields that carry the prompt are read; the rest of a
    Responses request is ignored, and ``input`` may be missing.
    """
    model: str = "default"
    input: str | list[ResponsesItem] = ""
    instructions: str | None = None

class ResponsesRequest(BaseModel):
    model: str = "default"
    input: str | list[ResponsesItem]
    instructions: str | None = None  # mapped to a system message
    max_output_tokens: int | None = Field(None, ge=1, le=MAX_REQUEST_TOKENS)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    # Repetition controls (toki extension, same knobs as chat): None
    # falls back to the model's sampling profile. Codex sends no
    # penalties by default, which is the loop amplifier on agents.
    repeat_penalty: float | None = Field(None, ge=0.0, le=2.0)
    top_k: int | None = Field(None, ge=0, le=100)
    min_p: float | None = Field(None, ge=0.0, le=1.0)
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
    """The token this daemon accepts, or None when none is configured.

    One implementation, shared with the control plane: the clients and the
    daemon have to agree on the token or every request is a 401, and they
    used to be two separate copies of the same lookup. A token file that
    holds a well-known placeholder is replaced with a real one on the way
    past — a placeholder protects nothing.
    """
    from ..ctl import configured_auth_token

    return configured_auth_token()


def _backend_model_id(backend) -> str | None:
    """Best-effort model id from the backend for the health report.

    llama-server's ``/v1/models`` uses ``{"models": [...]}`` (not the
    OpenAI ``{"data": [...]}`` shape), so use whichever is present.
    """
    try:
        m = backend._client.get(f"{backend.base_url}/v1/models", timeout=3.0)
        data = m.json()
        items = data.get("models") or data.get("data") or []
        if items:
            return items[0].get("id") or items[0].get("model") or items[0].get("name")
    except Exception:
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
    candidates: list[str] = []
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        candidates.append(auth[len("Bearer "):].strip())
    api_key = request.headers.get("x-api-key", "").strip()
    if api_key:
        candidates.append(api_key)
    provided = next((value for value in candidates if value), "")
    if not provided:
        METRICS.auth_failures.inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing Authorization: Bearer <token> or x-api-key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not any(
        hmac.compare_digest(value.encode("utf-8"), expected.encode("utf-8"))
        for value in candidates
    ):
        METRICS.auth_failures.inc()
        # A wrong token is almost never a wrong server: it is a client
        # that was never given ours. Claude Code is the usual one — it
        # sends ANTHROPIC_AUTH_TOKEN when it has not approved the API
        # key we handed it, and that token belongs to another provider.
        # Say so, because "invalid token" alone sends people to /login.
        logger.warning(
            "auth failed on %s: the request carried a token the daemon "
            "does not accept. If this is Claude Code, the local API key "
            "was not approved by it: run `toki model use <model>` to "
            "re-point the client and approve the key.",
            request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "invalid token — the daemon only accepts the token in its "
                "auth file. If this is Claude Code, run `toki model use "
                "<model>` to re-point it and approve the local key."
            ),
        )


def _get_rate_key(request: Request, session_id: str | None) -> str:
    if session_id:
        return f"session:{session_id}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


# How long a stream may stay silent before the daemon says something.
# A backend prefilling a long prompt can be quiet for minutes; a client
# with a stream-idle timeout reads that as a dead connection.
STREAM_HEARTBEAT_SECONDS = 15.0


async def _aiter_with_heartbeat(source, interval: float | None = None):
    """Iterate ``source``, yielding ``None`` when it goes quiet.

    A backend prefilling a long prompt says nothing for minutes. A
    client with a stream-idle timeout — codex, most SDKs — reads that
    silence as a dead connection and reconnects, so on a machine that
    needs a minute to prefill the harness can never finish a turn.

    The pending ``__anext__`` is never cancelled while waiting: a
    timeout here must not tear down the upstream request (a cancelled
    upstream leaves llama-server's slot flagged busy).
    """
    if interval is None:
        interval = STREAM_HEARTBEAT_SECONDS
    iterator = source.__aiter__()
    pending: asyncio.Future | None = None
    while True:
        if pending is None:
            pending = asyncio.ensure_future(iterator.__anext__())
        done, _ = await asyncio.wait({pending}, timeout=interval)
        if not done:
            yield None
            continue
        try:
            item = pending.result()
        except StopAsyncIteration:
            return
        pending = None
        yield item


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
        if _get_auth_token() is None:
            logger.warning(
                "no auth token configured; the daemon accepts loopback "
                "requests only. Set TOKI_AUTH_TOKEN (or "
                "~/.config/tokirecur/auth) before exposing it beyond localhost",
            )
        watcher = asyncio.create_task(_watch_slots())
        try:
            yield
        finally:
            watcher.cancel()
        logger.info("Toki Recur daemon shutting down")

    app = FastAPI(title="Toki Recur", version="0.2.0", lifespan=lifespan)
    engine = RecyclingEngine(backend, index, similarity_threshold=similarity_threshold)
    # The window this daemon really serves: what the stack gave
    # llama-server, not what a client believes. The overflow check
    # needs it, and the env var is only set when the stack spawned the
    # daemon.
    window = max_context_tokens or _max_context_ceiling()
    app.state.max_context_tokens = window

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
    # Measured prefill rate per model (tokens per second). It is what
    # turns "the client timed out" into "this prompt needs eight
    # minutes on this machine".
    app.state.prefill_rates = {}

    # ---- Middleware ----
    @app.middleware("http")
    async def observe_and_authorize(request: Request, call_next):
        # Skip auth on health/metrics (monitoring needs access) and on the
        # Anthropic connectivity probe, which clients send before they have
        # credentials and which must not look like an auth failure.
        path = request.url.path
        public = path in ("/healthz", "/metrics", "/anthropic/api/hello")
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
        """Probe the upstream llama-server. Returns ok/unreachable.

        Probes the backend's own ``/health`` endpoint (llama-server
        answers 200 once it can serve) — ``cache_info`` is not a
        reliable liveness signal because llama-server's ``/v1/models``
        shape differs and idle slots omit the model field. The backend
        URL is included so clients can confirm this daemon is wired to
        the RIGHT llama-server (not a stale daemon left pointing at an
        old port) before trusting it.
        """
        try:
            r = backend._client.get(f"{backend.base_url}/health", timeout=3.0)
            if r.status_code == 200:
                return {
                    "status": "ok",
                    "backend": backend.base_url,
                    "model_id": _backend_model_id(backend),
                }
        except Exception:
            pass
        return {"status": "unreachable", "backend": backend.base_url}

    # A wedged backend answers /health with 200 while serving nothing:
    # llama-server keeps a cancelled task's slot flagged busy, so every
    # completion queues forever.
    #
    # Telling "wedged" from "busy" is harder than it looks. llama-server
    # reports no live progress for a slot that is prefilling — its
    # prompt counter moves in 2048-token blocks, which on a slow CPU is
    # minutes apart — so a short sample cannot tell the two apart, and
    # guessing wrong refuses valid requests. So the daemon watches the
    # backend in the background, remembers when it last produced
    # anything, and only calls a stall a wedge when every slot has
    # produced NOTHING at all for a long time.
    WEDGE_AFTER_SECONDS = 90.0
    _SLOT_POLL_SECONDS = 15.0

    async def _watch_slots() -> None:
        """Track the backend's slots in the background."""
        state = getattr(app.state, "slot_watch", None)
        if state is None:
            state = {"snapshot": [], "last_progress": time.time()}
            app.state.slot_watch = state
        previous = None
        while True:
            try:
                snapshot = await backend.slot_snapshot()
            except Exception:
                snapshot = None
            if snapshot:
                busy = [s for s in snapshot if s.get("busy")]
                marks = [(s.get("processed"), s.get("decoded")) for s in busy]
                if not busy or marks != previous:
                    # Anything moving at all — a token decoded, a prompt
                    # block processed, a slot freed — counts as progress.
                    state["last_progress"] = time.time()
                previous = marks
                state["snapshot"] = snapshot
            await asyncio.sleep(_SLOT_POLL_SECONDS)

    def _backend_wedged() -> str | None:
        """Why the backend cannot serve anything, or None if it can.

        The stall must be total: every slot busy, none of them having
        produced a single token, and no movement for the whole window.
        A backend that is merely prefilling a long prompt is working.
        """
        state = getattr(app.state, "slot_watch", None)
        if not state:
            return None
        snapshot = state.get("snapshot") or []
        if not snapshot:
            return None
        busy = [s for s in snapshot if s.get("busy")]
        if not busy or len(busy) != len(snapshot):
            return None
        if any(s.get("processed") or s.get("decoded") for s in busy):
            return None
        idle_for = time.time() - float(state.get("last_progress", time.time()))
        if idle_for < WEDGE_AFTER_SECONDS:
            return None
        tasks = ", ".join(str(s.get("task")) for s in busy)
        return (
            f"llama-server ({backend.base_url}) is wedged: "
            f"{len(busy)}/{len(snapshot)} slots busy with no progress "
            f"for {int(idle_for)}s (task ids {tasks}). A cancelled request "
            f"left the slots flagged busy. Restart the stack: "
            f"`toki model use <model>`."
        )

    # ---- OpenAI model metadata (context discovery) ----

    def _record_stream_prefill(
        items: Any, model_name: str,
    ) -> Any:
        """Remember the prefill rate of a streamed turn.

        The rate has to come from the traffic that matters: Claude Code
        only ever streams, so a daemon that learned its rate from the
        non-streaming path would never know how slow the client is
        waiting for.
        """

        async def wrapper() -> Any:
            async for item in items:
                if isinstance(item, dict) and item.get("kind") == "result":
                    result = item.get("result")
                    if result is not None:
                        _record_prefill_rate(model_name, result)
                yield item

        return wrapper()

    def _measured_prefill(result: Any) -> dict[str, Any] | None:
        """What the prefill of this request actually cost."""
        prefilled = int(getattr(result, "prefilled_tokens", 0) or 0)
        ttft_ms = float(getattr(result, "ttft_ms", 0) or 0)
        if prefilled <= 0 or ttft_ms <= 0:
            return None
        return {
            "prefill_tokens": prefilled,
            "prefill_ms": round(ttft_ms, 1),
            "prefill_tokens_per_second": round(prefilled / (ttft_ms / 1000.0), 1),
        }

    def _record_prefill_rate(model_name: str, result: Any) -> None:
        """Remember how fast this machine prefills for this model."""
        prefilled = int(getattr(result, "prefilled_tokens", 0) or 0)
        ttft_ms = float(getattr(result, "ttft_ms", 0) or 0)
        if prefilled <= 0 or ttft_ms <= 0:
            return
        rate = prefilled / (ttft_ms / 1000.0)
        if rate <= 0:
            return
        rates = getattr(app.state, "prefill_rates", None)
        if rates is None:
            rates = {}
            app.state.prefill_rates = rates
        rates[model_name] = rate

    def _prompt_token_estimate(messages: list[dict[str, Any]]) -> int:
        """Rough prompt size in tokens, from the characters.

        Four characters per token is the usual rule of thumb. The real
        count is only known after the prefill; this number exists to
        warn *before* the wait, so a rough one is enough.
        """
        chars = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text")
                        if isinstance(text, str):
                            chars += len(text)
        return chars // 4

    def _prefill_estimate(
        messages: list[dict[str, Any]], model_name: str,
    ) -> dict[str, Any] | None:
        """How long this prompt will take to prefill, if we know.

        The rate comes from the last completed request on this daemon.
        Without a measurement there is nothing honest to say, so the
        estimate is None.
        """
        rates = getattr(app.state, "prefill_rates", None) or {}
        rate = rates.get(model_name)
        if not rate or rate <= 0:
            return None
        tokens = _prompt_token_estimate(messages)
        if tokens <= 0:
            return None
        return {
            "prompt_tokens_estimated": tokens,
            "prefill_tokens_per_second": round(rate, 1),
            "prefill_seconds_estimated": round(tokens / rate, 1),
        }

    def _warn_slow_prefill(estimate: dict[str, Any] | None) -> None:
        """Say the number, before the client decides to give up."""
        if not estimate:
            return
        threshold = float(os.environ.get("TOKI_SLOW_PREFILL_SECONDS", "20"))
        seconds = float(estimate.get("prefill_seconds_estimated") or 0)
        if seconds < threshold:
            return
        logger.warning(
            "slow prefill expected: ~%s tokens at %.0f tok/s is about "
            "%.0f s. A client with a ten-minute budget gives up in the "
            "middle of it — use a smaller model, a smaller context, or "
            "Claude Code with --bare.",
            estimate.get("prompt_tokens_estimated"),
            float(estimate.get("prefill_tokens_per_second") or 0),
            seconds,
        )

    def _served_model_id() -> str | None:
        """The id THIS daemon's llama-server actually serves, cached.

        The global active marker names the stack the CLI routes chat to,
        not this daemon. When a companion stack flips the marker, a
        marker-derived id made this daemon advertise — and echo into chat
        responses — a model it was not serving (an Ornith stack that
        answered as lfm2.5 all session). The backend is the only honest
        source, and its id is fixed for the daemon's lifetime, so probe
        once and cache. While the backend has not answered yet, fall back
        to the marker but retry the probe at most every 10s instead of
        stalling every request on a 3s timeout.
        """
        cached, ts = getattr(app.state, "served_model_id", (None, 0.0))
        now = time.time()
        if cached:
            return cached
        if now - ts < 10.0:
            return _active_model_id()
        # Negative-cache first: a probe that fails must not be retried on
        # every request; a success overwrites it immediately below.
        app.state.served_model_id = (None, now)
        mid = _backend_model_id(backend)
        if mid:
            app.state.served_model_id = (mid, now)
            return mid
        return _active_model_id()

    def _model_info_entry() -> dict:
        """OpenAI-format model entry advertising the REAL context window.

        Clients probe GET /v1/models (hermes's model_metadata detector,
        vLLM-style tools) to learn how much context they may use. The
        daemon answers with the effective slot context (TOKI_MAX_CONTEXT
        _TOKENS), never the registry's raw ctx_size — that mismatch is
        what made hermes believe 256k and send oversized prompts.
        Top-level `max_model_len`/`context_length` are used because
        llama.cpp's own `meta.n_ctx` field is ignored by those probes.
        """
        ctx = _max_context_ceiling()
        # The id a client is CONFIGURED with is the registry's model_id
        # (that is what the writers put in every client config), while
        # the backend reports the id of the weights it loaded. They
        # differ — "minicpm5-1b-q4_k_m" vs "MiniCPM5-1B-Q4_K_M" — and a
        # harness that looks up its configured id in /v1/models (codex
        # does) then finds nothing and falls back to guessed metadata.
        # Advertise the registry id; the served id stays listed too.
        registry_id = _active_model_id()
        served_id = _served_model_id()
        entry: dict[str, Any] = {
            "id": registry_id or served_id or "default",
            "object": "model",
            "created": int(app.state.started_at),
            "owned_by": "tokirecur",
        }
        if ctx:
            entry.update({
                "max_model_len": ctx,
                "context_length": ctx,
                "max_tokens": ctx,
                "meta": {"n_ctx": ctx},
            })
        # The effort the stack is pinned at, if any: a client that wants
        # to start from the stack's setting can read it here instead of
        # guessing one.
        effort = _stack_effort()
        if effort:
            entry["reasoning_effort"] = effort
        return entry

    def _model_info_entries() -> list[dict[str, Any]]:
        """The model this daemon serves, and its effort variants.

        A client that asks "what models do you have?" should see the
        variants: they are the same weights with the reasoning steered,
        and a harness that can only pick a model id picks them by id.
        """
        entry = _model_info_entry()
        entries = [entry]
        # The id of the weights themselves, when it differs from the id
        # the clients are configured with.
        served = _served_model_id()
        if served and served != entry["id"]:
            alias = dict(entry)
            alias["id"] = served
            entries.append(alias)
        try:
            from ..ctl import expand_variants, load_registry
            specs = load_registry()
        except Exception:
            return entries
        served = _served_model_id()
        for variant in expand_variants(specs):
            if variant.effort is None:
                continue
            if served and not _same_model(variant.spec, served):
                continue
            sibling = dict(entry)
            sibling["id"] = variant.model_id
            entries.append(sibling)
        return entries

    @app.get("/v1/models")
    async def models_list() -> dict:
        return {"object": "list", "data": _model_info_entries()}

    @app.get("/v1/models/{model}")
    async def models_get(model: str) -> dict:
        return _model_info_entry()

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
        reasoning_effort: str | None = None,
        tools: list | None = None,
        tool_choice: str | dict | None = None,
        sampling: dict | None = None,
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
            max_tokens,
            reasoning=("off" if reasoning_effort == "none" else default_reasoning)
            if is_chat else "off",
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
                async for item in _aiter_with_heartbeat(
                    engine.serve_chat_stream(
                        messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        max_slots=max_slots,
                        session_id=session_id,
                        chat_template_kwargs=chat_template_kwargs,
                        reasoning_effort=reasoning_effort,
                        tools=tools,
                        tool_choice=tool_choice,
                        sampling=sampling,
                    )
                ):
                    if item is None:
                        # The backend is busy (a long prefill). Say so on
                        # the wire so the client's idle timer does not
                        # fire and reconnect mid-turn.
                        yield chunk_with({})
                        continue
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
                if _context_overflow_status(e):
                    err_msg = f"[context overflow: {_overflow_detail()}]"
                elif _tool_call_parse_failure(e):
                    # The model wrote a call the app could not read. Say so
                    # in words a person can use, instead of quoting a JSON
                    # parse error with a column number in it.
                    err_msg = (
                        "[malformed tool call: the model wrote a call the app "
                        "could not read — asking again usually gets it right]"
                    )
                else:
                    err_msg = f"[daemon error: {e}]"
                yield chunk_with(
                    {"content": ""}, finish_reason="error",
                )
                yield chunk_with(
                    {"content": err_msg},
                )
                yield "data: [DONE]\n\n"
                tap.record(
                    endpoint="chat.completions" if is_chat else "completions",
                    model=model_name,
                    messages=messages,
                    prompt=prompt if not is_chat else None,
                    effort=reasoning_effort,
                    session_id=session_id,
                    params={"temperature": temperature, "max_tokens": max_tokens},
                    error=f"{type(e).__name__}: {e}",
                )
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
                sampling=sampling,
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
        tap.record(
            endpoint="chat.completions" if is_chat else "completions",
            model=model_name,
            result=result,
            messages=messages,
            prompt=prompt if not is_chat else None,
            effort=reasoning_effort,
            session_id=session_id,
            params={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "sampling": sampling,
                "chat_template_kwargs": chat_template_kwargs,
            },
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
            if item.type == "message" or (
                item.type is None and (item.role or item.content)
            ):
                flush_tool_calls()
                if isinstance(item.content, str):
                    # A plain string is a message body; the parts form
                    # is only one way to say it.
                    text = item.content
                else:
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
        messages = await _inline_images(messages)
        max_tokens = req.max_output_tokens or 256
        reasoning_mode = (
            "off"
            if req.reasoning and req.reasoning.effort == "none"
            else default_reasoning
        )
        ctk = _default_chat_template_kwargs(reasoning_mode)
        # Temperature keeps OpenAI's responses default (1.0) when the
        # client sends nothing; repetition controls fall back to the
        # model's profile (codex sends no penalties by default, which
        # is the loop amplifier on agents).
        temperature, sampling = _resolve_sampling(
            defaults=_responses_sampling_defaults(),
            temperature=req.temperature,
            repeat_penalty=req.repeat_penalty,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=req.min_p,
            default_temperature=1.0,
        )
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
            async for item in _aiter_with_heartbeat(
                engine.serve_chat_stream(
                    messages,
                    max_tokens=_resolve_max_tokens(max_tokens,
                                                   reasoning=reasoning_mode),
                    temperature=temperature,
                    max_slots=_resolve_max_slots(req.max_slots),
                    session_id=req.session_id,
                    chat_template_kwargs=ctk,
                    tools=_normalize_responses_tools(req.tools),
                    tool_choice=_normalize_responses_tool_choice(
                        req.tool_choice),
                    sampling=sampling,
                )
            ):
                if item is None:
                    # Still prefilling: keep the stream visibly alive.
                    yield event("response.in_progress", {
                        "id": resp_id, "object": "response",
                        "created_at": created,
                        "status": "in_progress", "model": req.model,
                    })
                    continue
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
        tap.record(
            endpoint="responses",
            model=req.model,
            result=result,
            messages=messages,
            effort=None if reasoning_mode != "off" else "none",
            session_id=req.session_id,
            params={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "sampling": sampling,
                "chat_template_kwargs": ctk,
            },
        )

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest) -> Response:
        wedged = _backend_wedged()
        if wedged:
            raise HTTPException(status_code=503, detail=wedged)
        temperature, sampling = _resolve_sampling(
            temperature=req.temperature,
            repeat_penalty=req.repeat_penalty,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=req.min_p,
        )
        try:
            if req.stream:
                completion_id = f"cmpl-{int(time.time() * 1000)}"
                return StreamingResponse(
                    _stream_completion(
                        req.prompt,
                        None,
                        req.max_tokens,
                        temperature,
                        _resolve_max_slots(req.max_slots),
                        req.session_id,
                        req.model,
                        completion_id,
                        is_chat=False,
                        sampling=sampling,
                    ),
                    media_type="text/event-stream",
                )
            result = await _run_engine(
                engine.serve_native,
                req.prompt,
                max_tokens=_resolve_max_tokens(req.max_tokens, reasoning="off"),
                temperature=temperature,
                max_slots=_resolve_max_slots(req.max_slots),
                session_id=req.session_id,
                sampling=sampling,
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
            tap.record(
                endpoint="completions",
                model=req.model,
                result=result,
                prompt=req.prompt,
                session_id=req.session_id,
                params={
                    "temperature": temperature,
                    "max_tokens": req.max_tokens,
                    "sampling": sampling,
                },
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
            if _context_overflow_status(e):
                raise HTTPException(status_code=413,
                                    detail=_overflow_detail()) from e
            logger.exception("completion failed")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> Response:
        wedged = _backend_wedged()
        if wedged:
            raise HTTPException(status_code=503, detail=wedged)
        # Resolve the model name the client sent. hermes/openclaw/etc.
        # can ship a model id that doesn't match the active spec (e.g.
        # the historical "unsloth/<repo>-GGUF" form vs the spec's
        # short "qwen3.5-0.8b" form). We accept any model name when
        # only one model is loaded — the daemon is single-tenant.
        # Override `req.model` with the id this daemon actually serves
        # (not the global marker: a flipped marker made responses echo
        # a model this stack was not running) so downstream consumers
        # see a consistent, honest name.
        # The effort this turn runs at. Resolved before the model name is
        # overridden below, because a variant carries its effort in the
        # model id ("qwen3.8-27b:high").
        active_model = _served_model_id()
        effort = _resolve_effort(req.model, req.reasoning_effort, active_model)
        if effort is None:
            effort = _stack_effort()
        # Thinking follows the effort: "none" is thinking off, every other
        # level turns it on. A client that sent its own chat_template_kwargs
        # keeps them.
        ctk = req.chat_template_kwargs
        if effort is not None:
            ctk = dict(ctk or {})
            ctk.setdefault("enable_thinking", effort != "none")
        # The token floor follows too: a turn with thinking off does not
        # need the headroom a thinking turn does.
        reasoning_mode = "off" if effort == "none" else default_reasoning

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
        # Text-only active model + attached images: caption first, or the
        # projector-less backend 500s on pixels it cannot represent.
        messages = await _inline_images(messages)
        messages = _normalize_messages(messages)
        # A prompt that cannot fit the window is an answer, not a
        # stream: the client needs the status before the first byte, or
        # it sees a stream that dies mid-turn and concludes the
        # provider is gone.
        if _prompt_would_overflow(engine, messages, req.tools, window):
            return _openai_error(
                413, _overflow_detail(window),
                code="context_length_exceeded",
            )
        temperature, sampling = _resolve_sampling(
            temperature=req.temperature,
            repeat_penalty=req.repeat_penalty,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=req.min_p,
        )
        try:
            if req.stream:
                completion_id = f"chatcmpl-{int(time.time() * 1000)}"
                return StreamingResponse(
                    _stream_completion(
                        None,  # no flat prompt
                        messages,
                        req.max_tokens,
                        temperature,
                        _resolve_max_slots(req.max_slots),
                        req.session_id,
                        req.model,
                        completion_id,
                        is_chat=True,
                        chat_template_kwargs=ctk,
                        reasoning_effort=effort,
                        tools=req.tools,
                        tool_choice=req.tool_choice,
                        sampling=sampling,
                    ),
                    media_type="text/event-stream",
                )
            # Default thinking mode from the model's `reasoning` setting;
            # the client can override with explicit chat_template_kwargs.
            if ctk is None:
                ctk = _default_chat_template_kwargs(default_reasoning)
            result = await _run_engine(
                engine.serve_chat,
                messages,
                max_tokens=_resolve_max_tokens(req.max_tokens, reasoning=reasoning_mode),
                temperature=temperature,
                max_slots=_resolve_max_slots(req.max_slots),
                session_id=req.session_id,
                chat_template_kwargs=ctk,
                reasoning_effort=effort,
                tools=req.tools,
                tool_choice=req.tool_choice,
                sampling=sampling,
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
            # Remember how fast this machine prefills, so the next
            # request can say how long it will take before it waits.
            _record_prefill_rate(req.model, result)
            tap.record(
                endpoint="chat.completions",
                model=req.model,
                result=result,
                messages=messages,
                effort=effort,
                session_id=req.session_id,
                params={
                    "temperature": temperature,
                    "max_tokens": req.max_tokens,
                    "sampling": sampling,
                    "chat_template_kwargs": ctk,
                },
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
            if _context_overflow_status(e):
                return _openai_error(
                    413, _overflow_detail(), code="context_length_exceeded",
                )
            logger.exception("chat completion failed")
            return _openai_error(500, str(e))

    async def _tap_anthropic_stream(
        items: Any, *, model_name: str, messages: list[dict],
        session: str | None, params: dict,
    ) -> Any:
        """Relay the engine's stream, recording the finished turn.

        The Anthropic path hands the engine's items to ``stream_message``,
        which consumes them — so the tap has to sit between the two to
        still see the final result.
        """
        result = None
        try:
            async for item in items:
                if item.get("kind") == "result":
                    result = item.get("result")
                yield item
        except Exception as e:
            tap.record(
                endpoint="messages",
                model=model_name,
                messages=messages,
                effort=None,
                session_id=session,
                params=params,
                error=f"{type(e).__name__}: {e}",
            )
            raise
        if result is not None:
            tap.record(
                endpoint="messages",
                model=model_name,
                result=result,
                messages=messages,
                effort=None,
                session_id=session,
                params=params,
            )

    @app.api_route("/anthropic/api/hello", methods=["GET", "HEAD"])
    async def anthropic_hello() -> Response:
        """Anthropic-style connectivity probe.

        Claude Code sends this before it has credentials; answering 401 makes a
        working endpoint look broken, so return a static OK instead.
        """
        return JSONResponse({"status": "ok"})

    @app.post("/anthropic/v1/messages")
    @app.post("/v1/messages")
    async def anthropic_messages(req: AnthropicMessagesRequest) -> Response:
        wedged = _backend_wedged()
        if wedged:
            raise HTTPException(status_code=503, detail=wedged)
        # Same honesty rule as chat_completions: name the model this
        # daemon serves, not whatever the global marker currently says.
        active_model = _served_model_id()
        model_name = active_model or req.model
        messages = to_openai_messages(req)
        messages = await _inline_images(messages)
        tools = to_openai_tools(req.tools)
        tool_choice = to_openai_tool_choice(req.tool_choice)
        template_kwargs = chat_template_kwargs(req, default_reasoning)
        reasoning_mode = default_reasoning if template_kwargs.get("enable_thinking") else "off"
        max_tokens = _resolve_max_tokens(req.max_tokens, reasoning=reasoning_mode)
        temperature, sampling = _resolve_sampling(
            temperature=req.temperature,
            top_p=req.top_p,
        )
        session = anthropic_session_id(req)
        # How long this prompt will take to prefill, said out loud
        # before the wait rather than after the client gives up.
        estimate = _prefill_estimate(messages, model_name)
        _warn_slow_prefill(estimate)

        if req.stream:
            async def event_stream() -> Any:
                try:
                    items = engine.serve_chat_stream(
                        messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        max_slots=_resolve_max_slots(2),
                        session_id=session,
                        chat_template_kwargs=template_kwargs,
                        tools=tools,
                        tool_choice=tool_choice,
                        sampling=sampling,
                    )
                    items = _tap_anthropic_stream(
                        items,
                        model_name=model_name,
                        messages=messages,
                        session=session,
                        params={
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "sampling": sampling,
                            "chat_template_kwargs": template_kwargs,
                        },
                    )
                    items = _record_stream_prefill(items, model_name)
                    async for event in stream_message(
                        items, model_name, estimate=estimate,
                    ):
                        yield event
                except Exception as e:
                    logger.exception("Anthropic messages streaming failed")
                    yield error_event(str(e))

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        try:
            result = await _run_engine(
                engine.serve_chat,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                max_slots=_resolve_max_slots(2),
                session_id=session,
                chat_template_kwargs=template_kwargs,
                tools=tools,
                tool_choice=tool_choice,
                sampling=sampling,
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
            tap.record(
                endpoint="messages",
                model=model_name,
                result=result,
                messages=messages,
                effort=None,
                session_id=session,
                params={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "sampling": sampling,
                    "chat_template_kwargs": template_kwargs,
                },
            )
            _record_prefill_rate(model_name, result)
            body = response_message(result, model_name)
            prefill = _measured_prefill(result)
            if prefill:
                # The numbers, not a guess: how long this prompt took
                # to prefill and how fast this machine does it.
                body["toki_prefill"] = prefill
            return JSONResponse(body)
        except Exception as e:
            if _context_overflow_status(e):
                raise HTTPException(status_code=413, detail=_overflow_detail()) from e
            logger.exception("Anthropic messages completion failed")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/anthropic/v1/messages/count_tokens")
    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(req: AnthropicMessagesRequest) -> Response:
        text = approximate_input_text(req)
        try:
            count = len(engine.backend.tokenize(text))
        except Exception:
            count = max(1, len(text) // 4)
        return JSONResponse({"input_tokens": count})

    @app.post("/v1/responses/input_tokens")
    async def responses_input_tokens(req: ResponsesInputTokensRequest) -> Response:
        """Count the tokens of a Responses request.

        Clients that show the prompt size ask for this before the turn:
        MiniMax Code does, and without the route it gets a 404, retries
        it, and never shows the number.
        """
        messages = _responses_messages(
            ResponsesRequest(model=req.model, input=req.input,
                             instructions=req.instructions)
        )
        text = "\n".join(str(m.get("content") or "") for m in messages)
        try:
            count = len(engine.backend.tokenize(text))
        except Exception:
            # A backend that cannot tokenize must not fail the probe.
            count = max(1, len(text) // 4)
        return JSONResponse({
            "object": "response.input_tokens",
            "input_tokens": count,
        })

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
        messages = _responses_messages(req)
        messages = await _inline_images(messages)
        # Before the stream starts: a client needs the status, not a
        # stream that dies mid-turn.
        if _prompt_would_overflow(engine, messages, req.tools, window):
            return _openai_error(
                413, _overflow_detail(window),
                code="context_length_exceeded",
            )
        try:
            if req.stream:
                return StreamingResponse(
                    _stream_responses(req),
                    media_type="text/event-stream",
                )
            max_tokens = req.max_output_tokens or 256
            reasoning_mode = (
                "off"
                if req.reasoning and req.reasoning.effort == "none"
                else default_reasoning
            )
            ctk = _default_chat_template_kwargs(reasoning_mode)
            temperature, sampling = _resolve_sampling(
                defaults=_responses_sampling_defaults(),
                temperature=req.temperature,
                repeat_penalty=req.repeat_penalty,
                top_p=req.top_p,
                top_k=req.top_k,
                min_p=req.min_p,
                default_temperature=1.0,
            )
            result = await _run_engine(
                engine.serve_chat,
                messages,
                max_tokens=_resolve_max_tokens(max_tokens,
                                               reasoning=reasoning_mode),
                temperature=temperature,
                max_slots=_resolve_max_slots(req.max_slots),
                session_id=req.session_id,
                chat_template_kwargs=ctk,
                tools=_normalize_responses_tools(req.tools),
                tool_choice=_normalize_responses_tool_choice(req.tool_choice),
                sampling=sampling,
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
            tap.record(
                endpoint="responses",
                model=req.model,
                result=result,
                messages=messages,
                effort=None if reasoning_mode != "off" else "none",
                session_id=req.session_id,
                params={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "sampling": sampling,
                    "chat_template_kwargs": ctk,
                },
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
            if _context_overflow_status(e):
                return _openai_error(
                    413, _overflow_detail(), code="context_length_exceeded",
                )
            logger.exception("responses completion failed")
            return _openai_error(500, str(e))

    @app.get("/cache/list")
    async def cache_list() -> dict:
        return {
            "entries": index.entries(),
            "cache_dir": str(cache_dir),
        }

    @app.delete("/cache/{cache_id}")
    async def cache_delete(cache_id: str) -> dict:
        from ..cache.blob import blob_path

        try:
            blob = blob_path(cache_dir, cache_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="invalid cache_id",
            ) from None
        removed = index.delete(cache_id)
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
