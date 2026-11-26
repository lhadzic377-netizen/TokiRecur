"""The Gemini wire format.

Antigravity's CLI (``agy``) does not speak the OpenAI API: it speaks
the Gemini one - ``POST /v1beta/models/<model>:generateContent`` with
``contents``, ``systemInstruction`` and ``generationConfig``. This
module is the translation in both directions: the request as the engine
wants it, and the engine's answer as Gemini wants it.

The shapes that matter:

* ``contents`` are the turns; a turn is a role (``user`` / ``model``)
  and a list of parts. A text part carries ``text``, a tool call
  carries ``functionCall`` and a tool answer carries
  ``functionResponse``.
* the answer is a list of ``candidates``; each has ``content`` (the
  parts), ``finishReason`` and an index.
* the stream is Server-Sent Events whose ``data:`` payload is one
  candidate chunk - and, unlike the OpenAI stream, there is no
  ``[DONE]`` sentinel: the stream simply ends.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from pydantic import BaseModel, Field


class GeminiGenerateContentRequest(BaseModel):
    """One Gemini turn."""

    contents: list[dict[str, Any]] = Field(default_factory=list)
    systemInstruction: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    toolConfig: dict[str, Any] | None = None
    generationConfig: dict[str, Any] | None = None
    safetySettings: list[dict[str, Any]] | None = None
    cachedContent: str | None = None
    # MemT native memory (the memt-native-memory branch of llama.cpp).
    memory_id: str | None = None
    memory_commit: bool | None = None
    memory_documents: list[str] | None = None


def _text(value: Any) -> str:
    """The text of a Gemini content block."""
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""
    parts = value.get("parts")
    if not isinstance(parts, list):
        return ""
    out: list[str] = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            out.append(part["text"])
    return "\n".join(out)


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _role(value: Any) -> str:
    role = str(value or "user").lower()
    if role in {"model", "assistant"}:
        return "assistant"
    return "user"


def to_openai_messages(request: GeminiGenerateContentRequest) -> list[dict[str, Any]]:
    """The Gemini turn as the engine's message list."""
    messages: list[dict[str, Any]] = []
    system = _text(request.systemInstruction)
    if system:
        messages.append({"role": "system", "content": system})

    for content in request.contents:
        if not isinstance(content, dict):
            continue
        role = _role(content.get("role"))
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue

        if role == "assistant":
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                # The thought flag has to be checked FIRST: a part with
                # both a thought flag and text is the model's reasoning,
                # and reading it as plain text put the thinking back
                # into the assistant's visible content.
                if part.get("thought") is True and isinstance(part.get("text"), str):
                    thinking_parts.append(part["text"])
                elif isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                call = part.get("functionCall")
                if isinstance(call, dict) and call.get("name"):
                    tool_calls.append({
                        "id": call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": str(call.get("name")),
                            "arguments": _json_text(call.get("args") or {}),
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

        user_text_parts: list[str] = []
        tool_results: list[dict[str, Any]] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("text"), str):
                user_text_parts.append(part["text"])
            response = part.get("functionResponse")
            if isinstance(response, dict):
                name = str(response.get("name") or "")
                payload = response.get("response")
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": response.get("id") or name,
                    "content": _json_text(payload if payload is not None else {}),
                })
        # The tool answers come first: they are the answer to the tool
        # calls the previous turn made.
        messages.extend(tool_results)
        if user_text_parts:
            messages.append({
                "role": "user", "content": "\n".join(user_text_parts),
            })
    return messages


def to_openai_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Gemini's function declarations as the engine's tools."""
    if not tools:
        return None
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        declarations = tool.get("functionDeclarations")
        if not isinstance(declarations, list):
            declarations = [tool]
        for declaration in declarations:
            if not isinstance(declaration, dict):
                continue
            name = declaration.get("name")
            if not name:
                continue
            converted.append({
                "type": "function",
                "function": {
                    "name": str(name),
                    "description": declaration.get("description", ""),
                    "parameters": declaration.get("parameters") or {},
                },
            })
    return converted or None


def to_openai_tool_choice(config: dict[str, Any] | None) -> str | None:
    """Gemini's tool config as the engine's choice."""
    if not isinstance(config, dict):
        return None
    modes = config.get("functionCallingConfig")
    if not isinstance(modes, dict):
        return None
    mode = str(modes.get("mode") or "AUTO").upper()
    if mode == "ANY":
        return "required"
    if mode == "NONE":
        return "none"
    return "auto"


def chat_template_kwargs(
    request: GeminiGenerateContentRequest, default_reasoning: str,
) -> dict[str, Any]:
    """Whether the model should think, in the engine's terms."""
    config = request.generationConfig or {}
    thinking = config.get("thinkingConfig")
    if isinstance(thinking, dict):
        budget = thinking.get("thinkingBudget")
        if isinstance(budget, int):
            if budget == 0:
                # Thinking off.
                return {"enable_thinking": False}
            if budget > 0:
                return {"enable_thinking": True}
            # A negative budget is "think as much as the model wants" -
            # the model's own default, not an instruction to stop.
    return {"enable_thinking": default_reasoning != "off"}


def max_output_tokens(request: GeminiGenerateContentRequest, fallback: int) -> int:
    config = request.generationConfig or {}
    value = config.get("maxOutputTokens")
    if isinstance(value, int) and value > 0:
        return value
    return fallback


def _finish_reason(result: Any) -> str:
    reason = getattr(result, "finish_reason", None)
    if getattr(result, "tool_calls", None):
        return "STOP"
    if reason in {"length", "max_tokens"}:
        return "MAX_TOKENS"
    return "STOP"


def _parts(result: Any) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    reasoning = getattr(result, "reasoning", "") or ""
    if reasoning:
        parts.append({"text": reasoning, "thought": True})
    text = getattr(result, "output_text", "") or ""
    if text:
        parts.append({"text": text})
    for call in getattr(result, "tool_calls", None) or []:
        function = call.get("function") or {}
        arguments = function.get("arguments") or "{}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except (TypeError, ValueError):
                arguments = {"_raw": arguments}
        parts.append({
            "functionCall": {
                "name": function.get("name") or "",
                "args": arguments if isinstance(arguments, dict) else {},
            },
        })
    if not parts:
        parts.append({"text": ""})
    return parts


def response_body(result: Any, model: str) -> dict[str, Any]:
    """The engine's answer as Gemini's response."""
    prompt_tokens = int(getattr(result, "prefilled_tokens", 0) or 0)
    output_tokens = int(getattr(result, "generated_tokens", 0) or 0)
    return {
        "candidates": [{
            "content": {"role": "model", "parts": _parts(result)},
            "finishReason": _finish_reason(result),
            "index": 0,
        }],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": output_tokens,
            "totalTokenCount": prompt_tokens + output_tokens,
        },
        "modelVersion": model,
    }


def chunk_body(text: str, model: str, *, thought: bool = False) -> dict[str, Any]:
    """One streamed candidate chunk."""
    part: dict[str, Any] = {"text": text}
    if thought:
        part["thought"] = True
    return {
        "candidates": [{
            "content": {"role": "model", "parts": [part]},
            "index": 0,
        }],
        "modelVersion": model,
    }


def sse_chunk(payload: dict[str, Any]) -> str:
    """One Server-Sent Event.

    Gemini's stream has no ``[DONE]`` sentinel: the stream ends when
    the server closes it, and a client that reads ``[DONE]`` as JSON
    fails on it.
    """
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {data}\n\n"


def error_body(message: str) -> dict[str, Any]:
    """Gemini's error shape."""
    return {
        "error": {
            "code": 500,
            "message": message,
            "status": "INTERNAL",
        },
    }


__all__ = [
    "GeminiGenerateContentRequest",
    "chat_template_kwargs",
    "chunk_body",
    "error_body",
    "max_output_tokens",
    "response_body",
    "sse_chunk",
    "to_openai_messages",
    "to_openai_tool_choice",
    "to_openai_tools",
]
