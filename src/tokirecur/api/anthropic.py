from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field


class AnthropicMessagesRequest(BaseModel):
    model: str = "default"
    max_tokens: int = Field(4096, ge=1, le=262144)
    messages: list[dict[str, Any]]
    system: str | list[dict[str, Any]] | None = None
    temperature: float | None = Field(None, ge=-2.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    stop_sequences: list[str] | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None
    output_config: dict[str, Any] | None = None
    context_management: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for block in value:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif block.get("type") == "tool_result":
            parts.append(_text(block.get("content")))
    return "\n".join(part for part in parts if part)


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _image_part(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source")
    if not isinstance(source, dict):
        return None
    if source.get("type") == "base64" and source.get("data"):
        media_type = source.get("media_type", "image/png")
        url = f"data:{media_type};base64,{source['data']}"
    elif source.get("type") == "url" and source.get("url"):
        url = source["url"]
    else:
        return None
    return {"type": "image_url", "image_url": {"url": url}}


def _flush_user_parts(parts: list[dict[str, Any]], out: list[dict[str, Any]]) -> None:
    if not parts:
        return
    if all(part.get("type") == "text" for part in parts):
        content: Any = "\n".join(part.get("text", "") for part in parts)
    else:
        content = parts
    out.append({"role": "user", "content": content})
    parts.clear()


def to_openai_messages(request: AnthropicMessagesRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = _text(request.system)
    if system:
        messages.append({"role": "system", "content": system})

    for raw in request.messages:
        role = str(raw.get("role", "user"))
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            continue

        if role == "assistant":
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text" and isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
                elif block_type == "thinking" and isinstance(block.get("thinking"), str):
                    thinking_parts.append(block["thinking"])
                elif block_type == "tool_use":
                    tool_calls.append({
                        "id": block.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": block.get("name") or "",
                            "arguments": _json_text(block.get("input") or {}),
                        },
                    })
            if text_parts or thinking_parts or tool_calls:
                assistant: dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) or None,
                }
                if thinking_parts:
                    assistant["reasoning_content"] = "\n".join(thinking_parts)
                if tool_calls:
                    assistant["tool_calls"] = tool_calls
                messages.append(assistant)
            continue

        parts: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
            elif block_type == "image":
                image = _image_part(block)
                if image is not None:
                    parts.append(image)
            elif block_type == "tool_result":
                _flush_user_parts(parts, messages)
                result = block.get("content")
                result_text = _text(result) if isinstance(result, list) else _json_text(result)
                if block.get("is_error"):
                    result_text = f"Error: {result_text}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") or "",
                    "content": result_text,
                })
        _flush_user_parts(parts, messages)
    return messages


def to_openai_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name")
        if not name:
            continue
        converted.append({
            "type": "function",
            "function": {
                "name": name,
                "description": function.get("description", ""),
                "parameters": function.get("input_schema") or function.get("parameters") or {},
            },
        })
    return converted or None


def to_openai_tool_choice(choice: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
    if isinstance(choice, str):
        return choice
    if not isinstance(choice, dict):
        return None
    kind = choice.get("type")
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    if kind in {"auto", "required"}:
        return kind
    return None


def session_id(request: AnthropicMessagesRequest) -> str | None:
    metadata = request.metadata or {}
    value = metadata.get("session_id")
    if isinstance(value, str) and value:
        return value
    user_id = metadata.get("user_id")
    if isinstance(user_id, str):
        try:
            decoded = json.loads(user_id)
        except (TypeError, ValueError):
            return None
        value = decoded.get("session_id") if isinstance(decoded, dict) else None
        if isinstance(value, str) and value:
            return value
    return None


def chat_template_kwargs(
    request: AnthropicMessagesRequest, default_reasoning: str,
) -> dict[str, Any]:
    thinking = request.thinking or {}
    kind = thinking.get("type")
    if kind == "disabled":
        enabled = False
    elif kind in {"enabled", "adaptive"}:
        enabled = True
    else:
        enabled = default_reasoning != "off"
    return {"enable_thinking": enabled}


def _tool_input(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    arguments = function.get("arguments", "")
    if isinstance(arguments, dict):
        return arguments
    try:
        value = json.loads(arguments or "{}")
    except (TypeError, ValueError):
        return {"_raw": arguments}
    return value if isinstance(value, dict) else {"_raw": value}


def _stop_reason(result: Any) -> str:
    reason = getattr(result, "finish_reason", None)
    if getattr(result, "tool_calls", None):
        return "tool_use"
    if reason in {"length", "max_tokens"}:
        return "max_tokens"
    if reason == "stop_sequence":
        return "stop_sequence"
    return "end_turn"


def response_message(result: Any, model: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    reasoning = getattr(result, "reasoning", "") or ""
    if reasoning:
        content.append({
            "type": "thinking",
            "thinking": reasoning,
            "signature": "toku-local",
        })
    text = getattr(result, "output_text", "") or ""
    tool_calls = getattr(result, "tool_calls", None) or []
    if text or not tool_calls:
        content.append({"type": "text", "text": text})
    for call in tool_calls:
        function = call.get("function") or {}
        content.append({
            "type": "tool_use",
            "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
            "name": function.get("name") or "",
            "input": _tool_input(call),
        })
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _stop_reason(result),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(getattr(result, "prefilled_tokens", 0) or 0),
            "output_tokens": int(getattr(result, "generated_tokens", 0) or 0),
        },
    }


def sse_event(name: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}\ndata: {data}\n\n"


def error_event(message: str) -> str:
    return sse_event("error", {
        "type": "error",
        "error": {"type": "api_error", "message": message},
    })


STREAM_PING_INTERVAL = 10.0


def ping_event() -> str:
    return sse_event("ping", {"type": "ping"})


async def _with_keepalive(
    items: AsyncIterator[dict[str, Any]], interval: float = STREAM_PING_INTERVAL,
) -> AsyncIterator[dict[str, Any] | None]:
    """Yield engine items, or None when the engine has been idle for `interval`.

    Anthropic clients (Claude Code in particular) abort a stream that stays
    silent, and the local engine can spend 45-90s in prompt evaluation before
    its first token, so the connection needs Anthropic's `ping` keepalive.

    A background task owns the engine iterator so that a timed-out wait never
    cancels in-flight generation; the task is cancelled on disconnect instead.
    """
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    done = object()

    async def pump() -> None:
        try:
            async for item in items:
                await queue.put(("item", item))
        finally:
            await queue.put(("done", done))

    task = asyncio.create_task(pump())
    try:
        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=interval)
            except (TimeoutError, asyncio.TimeoutError):
                yield None
                continue
            if kind == "done":
                return
            yield payload
    finally:
        if not task.done():
            task.cancel()


async def stream_message(
    items: AsyncIterator[dict[str, Any]], model: str,
    ping_interval: float = STREAM_PING_INTERVAL,
    estimate: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    """The Anthropic event stream for one turn.

    ``estimate`` is the daemon's projection of how long the prefill
    will take, carried in the first event so a client (or a human
    reading the stream) can see the number instead of guessing from
    the silence.
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    message: dict[str, Any] = {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": model,
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    if estimate:
        message["toki_prefill_estimate"] = estimate
    yield sse_event("message_start", {
        "type": "message_start",
        "message": message,
    })

    next_index = 0
    text_index: int | None = None
    thinking_index: int | None = None
    tool_indices: dict[int, int] = {}
    open_indices: list[int] = []
    finish_reason: str | None = None
    result: Any = None

    async for item in _with_keepalive(items, ping_interval):
        if item is None:
            yield ping_event()
            continue
        kind = item.get("kind")
        if kind == "delta":
            if text_index is None:
                text_index = next_index
                next_index += 1
                open_indices.append(text_index)
                yield sse_event("content_block_start", {
                    "type": "content_block_start",
                    "index": text_index,
                    "content_block": {"type": "text", "text": ""},
                })
            yield sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": text_index,
                "delta": {"type": "text_delta", "text": item.get("delta", "")},
            })
        elif kind == "reasoning":
            if thinking_index is None:
                thinking_index = next_index
                next_index += 1
                open_indices.append(thinking_index)
                yield sse_event("content_block_start", {
                    "type": "content_block_start",
                    "index": thinking_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                })
            yield sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": thinking_index,
                "delta": {"type": "thinking_delta", "thinking": item.get("delta", "")},
            })
        elif kind == "tool_calls":
            for call in item.get("tool_calls", []):
                if not isinstance(call, dict):
                    continue
                call_index = int(call.get("index", 0))
                if call_index not in tool_indices:
                    block_index = next_index
                    next_index += 1
                    tool_indices[call_index] = block_index
                    open_indices.append(block_index)
                    function = call.get("function") or {}
                    yield sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                            "name": function.get("name") or f"tool_{call_index}",
                            "input": {},
                        },
                    })
                arguments = (call.get("function") or {}).get("arguments")
                if arguments:
                    yield sse_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": tool_indices[call_index],
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": arguments,
                        },
                    })
        elif kind == "finish":
            finish_reason = item.get("finish_reason") or finish_reason
        elif kind == "result":
            result = item.get("result")

    if result is None:
        yield error_event("local engine returned no result")
        return

    if thinking_index is None and getattr(result, "reasoning", ""):
        thinking_index = next_index
        next_index += 1
        open_indices.append(thinking_index)
        yield sse_event("content_block_start", {
            "type": "content_block_start",
            "index": thinking_index,
            "content_block": {"type": "thinking", "thinking": ""},
        })
        yield sse_event("content_block_delta", {
            "type": "content_block_delta",
            "index": thinking_index,
            "delta": {
                "type": "thinking_delta",
                "thinking": result.reasoning,
            },
        })

    if text_index is None:
        text_index = next_index
        next_index += 1
        open_indices.append(text_index)
        yield sse_event("content_block_start", {
            "type": "content_block_start",
            "index": text_index,
            "content_block": {"type": "text", "text": ""},
        })
        if getattr(result, "output_text", ""):
            yield sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": text_index,
                "delta": {
                    "type": "text_delta",
                    "text": result.output_text,
                },
            })

    for call_index, call in enumerate(getattr(result, "tool_calls", None) or []):
        source_index = call.get("index", call_index)
        if source_index in tool_indices:
            continue
        block_index = next_index
        next_index += 1
        tool_indices[source_index] = block_index
        open_indices.append(block_index)
        function = call.get("function") or {}
        yield sse_event("content_block_start", {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {
                "type": "tool_use",
                "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": function.get("name") or f"tool_{source_index}",
                "input": {},
            },
        })
        arguments = function.get("arguments") or ""
        if arguments:
            yield sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "input_json_delta", "partial_json": arguments},
            })

    for index in open_indices:
        yield sse_event("content_block_stop", {
            "type": "content_block_stop",
            "index": index,
        })

    yield sse_event("message_delta", {
        "type": "message_delta",
        "delta": {
            "stop_reason": _stop_reason(result) if finish_reason is None else (
                "tool_use" if finish_reason == "tool_calls" else
                "max_tokens" if finish_reason in {"length", "max_tokens"} else
                "end_turn"
            ),
            "stop_sequence": None,
        },
        "usage": {"output_tokens": int(getattr(result, "generated_tokens", 0) or 0)},
    })
    yield sse_event("message_stop", {"type": "message_stop"})


def approximate_input_text(request: AnthropicMessagesRequest) -> str:
    parts = [_text(request.system)]
    for message in request.messages:
        parts.append(_text(message.get("content")))
    return "\n".join(part for part in parts if part)


__all__ = [
    "AnthropicMessagesRequest",
    "STREAM_PING_INTERVAL",
    "approximate_input_text",
    "chat_template_kwargs",
    "error_event",
    "ping_event",
    "response_message",
    "session_id",
    "sse_event",
    "stream_message",
    "to_openai_messages",
    "to_openai_tool_choice",
    "to_openai_tools",
]
