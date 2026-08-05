"""Interactive chat REPL for Toki Recur.

Launched via `toki chat` or `toki chat --mode {raw,cached,compare}`.

Supports streaming output, conversation history, slash commands, and
A/B comparison between cached and raw inference paths.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.panel import Panel

console = Console()


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str
    metrics: dict[str, Any] = field(default_factory=dict)


def _auth_headers() -> dict[str, str]:
    """Bearer header for the toki daemon when a token is configured.

    Same read order as the daemon: TOKI_AUTH_TOKEN env var, then
    ~/.config/tokirecur/auth, then /etc/toki-recur/auth.
    """
    token = os.environ.get("TOKI_AUTH_TOKEN")
    if not token:
        for path in (
            Path.home() / ".config" / "tokirecur" / "auth",
            Path("/etc/toki-recur/auth"),
        ):
            try:
                if path.is_file():
                    content = path.read_text().strip()
                    if content:
                        token = content
                        break
            except OSError:
                pass
    return {"Authorization": f"Bearer {token}"} if token else {}


def stream_completion(
    base_url: str,
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    system: str | None = None,
    history: list[dict] | None = None,
    timeout: float = 120.0,
    api: str = "native",
) -> tuple[str, dict[str, Any]]:
    """Send a streaming completion. Returns (full_text, metrics).

    ``api="native"`` talks to llama-server's own ``/completion`` endpoint.
    ``api="openai"`` talks to the toki daemon's OpenAI-compatible
    ``/v1/chat/completions`` (recycling engine, SSE stream).

    metrics keys: ttft_ms, total_ms, tokens_generated, prefill_tokens,
                 hit (bool), reuse_depth, cache_id.
    """
    if api == "openai":
        return _stream_openai(
            base_url, prompt, max_tokens=max_tokens, temperature=temperature,
            system=system, history=history, timeout=timeout,
        )
    payload: dict[str, Any] = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if system:
        payload["system_prompt"] = system  # llama-server native param

    metrics: dict[str, Any] = {"ttft_ms": 0.0, "prefill_tokens": 0, "tokens_generated": 0}
    chunks: list[str] = []
    started = time.perf_counter()
    first_token_time: float | None = None
    try:
        with httpx.stream("POST", f"{base_url}/completion", json=payload,
                          timeout=timeout, headers=_auth_headers()) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                # llama-server SSE format: `data: {"response":..., ...}`
                if line.startswith("data: "):
                    line = line[6:]
                if line.strip() in ("[DONE]", ""):
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    obj = None
                if obj and "response" in obj:
                    tok = obj.get("response", "")
                    if first_token_time is None and tok:
                        first_token_time = time.perf_counter()
                    chunks.append(tok)
                    sys.stdout.write(tok)
                    sys.stdout.flush()
                elif obj and obj.get("stop"):
                    metrics["prefill_tokens"] = int(obj.get("tokens_evaluated", 0))
                    metrics["tokens_generated"] = int(obj.get("tokens_predicted", 0))
                    break
    except httpx.HTTPError as e:
        sys.stdout.write(f"\n[error: {e}]")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if first_token_time is not None:
        metrics["ttft_ms"] = (first_token_time - started) * 1000.0
    metrics["total_ms"] = elapsed_ms
    return "".join(chunks), metrics


def _openai_messages(
    prompt: str,
    *,
    system: str | None = None,
    history: list[dict] | None = None,
) -> list[dict]:
    """Build OpenAI chat messages, system first, history, then user."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return messages


def _stream_openai(
    base_url: str,
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    system: str | None = None,
    history: list[dict] | None = None,
    timeout: float = 120.0,
) -> tuple[str, dict[str, Any]]:
    """Stream from the toki daemon's OpenAI-compatible endpoint.

    Parses the daemon's SSE chunks (``choices[0].delta.content``) and
    the terminal chunk carrying ``usage``/``cache`` verdicts.
    """
    payload: dict[str, Any] = {
        "model": "default",
        "messages": _openai_messages(prompt, system=system, history=history),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    metrics: dict[str, Any] = {"ttft_ms": 0.0, "prefill_tokens": 0, "tokens_generated": 0}
    chunks: list[str] = []
    started = time.perf_counter()
    first_token_time: float | None = None
    try:
        with httpx.stream("POST", f"{base_url}/v1/chat/completions", json=payload,
                          timeout=timeout, headers=_auth_headers()) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                if line.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(line)
                except Exception:
                    obj = None
                if not obj:
                    continue
                cache = obj.get("cache") or {}
                usage = obj.get("usage") or {}
                if usage:
                    metrics["prefill_tokens"] = int(usage.get("prompt_tokens", 0))
                    metrics["tokens_generated"] = int(usage.get("completion_tokens", 0))
                if cache:
                    metrics["hit"] = bool(cache.get("hit"))
                    metrics["reuse_depth"] = cache.get("reuse_depth", 0)
                    metrics["cache_id"] = cache.get("cache_id")
                    if cache.get("ttft_ms") is not None:
                        metrics["ttft_ms"] = float(cache["ttft_ms"])
                    if cache.get("total_ms") is not None:
                        metrics["total_ms"] = float(cache["total_ms"])
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                tok = delta.get("content", "") or ""
                if tok:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    chunks.append(tok)
                    sys.stdout.write(tok)
                    sys.stdout.flush()
    except httpx.HTTPError as e:
        sys.stdout.write(f"\n[error: {e}]")
    if first_token_time is not None:
        metrics["ttft_ms"] = first_token_time - started
    metrics["total_ms"] = (time.perf_counter() - started) * 1000.0
    return "".join(chunks), metrics


def completion_once(
    base_url: str,
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    system: str | None = None,
    history: list[dict] | None = None,
    timeout: float = 120.0,
    api: str = "native",
) -> tuple[str, dict[str, Any]]:
    """Send a non-streaming completion. Used for compare mode and benchmarking.

    ``api="openai"`` targets the toki daemon's ``/v1/chat/completions``.
    """
    if api == "openai":
        payload: dict[str, Any] = {
            "model": "default",
            "messages": _openai_messages(prompt, system=system, history=history),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        started = time.perf_counter()
        try:
            r = httpx.post(f"{base_url}/v1/chat/completions", json=payload,
                           timeout=timeout, headers=_auth_headers())
            r.raise_for_status()
            body = r.json()
            cache = body.get("cache") or {}
            usage = body.get("usage") or {}
            return body.get("choices", [{}])[0].get("message", {}).get("content", ""), {
                "ttft_ms": float(cache.get("ttft_ms", 0.0)),
                "total_ms": (time.perf_counter() - started) * 1000.0,
                "tokens_generated": int(usage.get("completion_tokens", 0)),
                "prefill_tokens": int(usage.get("prompt_tokens", 0)),
                "hit": bool(cache.get("hit")),
                "reuse_depth": int(cache.get("reuse_depth", 0)),
                "cache_id": cache.get("cache_id"),
            }
        except httpx.HTTPError as e:
            return f"[error: {e}]", {}
    payload: dict[str, Any] = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if system:
        payload["system_prompt"] = system
    started = time.perf_counter()
    try:
        r = httpx.post(f"{base_url}/completion", json=payload, timeout=timeout,
                       headers=_auth_headers())
        r.raise_for_status()
        body = r.json()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        timings = body.get("timings", {}) or {}
        return body.get("content", ""), {
            "ttft_ms": float(timings.get("prompt_ms", 0.0)),
            "total_ms": elapsed_ms,
            "prefill_tokens": int(body.get("tokens_evaluated", 0)),
            "tokens_generated": int(body.get("tokens_predicted", 0)),
        }
    except httpx.HTTPError as e:
        return "", {"error": str(e), "total_ms": (time.perf_counter() - started) * 1000.0}


def build_prompt(history: list[Message], system: str | None) -> str:
    """Build a flat prompt string from the conversation history."""
    parts: list[str] = []
    if system:
        parts.append(system.strip())
    for m in history:
        if m.role == "user":
            parts.append(f"\n\nUser: {m.content}")
        elif m.role == "assistant":
            parts.append(f"\n\nAssistant: {m.content}")
    parts.append("\n\nAssistant:")
    return "".join(parts)


def chat_session(
    *,
    raw_url: str,
    daemon_url: str,
    initial_mode: str = "raw",
    initial_system: str | None = None,
    initial_model: str = "default",
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> int:
    """Run the chat REPL. Returns process exit code."""
    history: list[Message] = []
    mode = initial_mode  # "raw" | "cached" | "compare"
    system = initial_system

    console.print(
        Panel(
            f"[bold cyan]Toki Recur chat[/bold cyan]\n"
            f"mode = [yellow]{mode}[/yellow]\n"
            f"raw   = {raw_url}\n"
            f"daemon= {daemon_url}\n"
            f"[dim]slash commands: /help, /quit, /mode, /clear, /system, /metrics, /raw, /cached, /compare[/dim]",
            border_style="cyan",
        )
    )

    last_metrics: dict[str, Any] = {}

    while True:
        try:
            line = input("\n[bold cyan]you ❯[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye.[/dim]")
            return 0
        line = line.rstrip()
        if not line:
            continue

        # Slash commands.
        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            if cmd in ("/quit", "/exit"):
                return 0
            if cmd == "/help":
                console.print(
                    "[dim]/quit /exit   exit chat\n"
                    "/mode raw|cached|compare   switch mode\n"
                    "/system <text>   set system prompt (empty = clear)\n"
                    "/clear           clear conversation history\n"
                    "/metrics         show last request metrics\n"
                    "/raw <msg>       send next message via raw path\n"
                    "/cached <msg>    send next message via cache\n"
                    "/compare <msg>   send same prompt raw + cached, show side-by-side\n"
                    "/max <n>         set max tokens for next request\n"
                    "/temp <f>        set temperature for next request[/dim]"
                )
                continue
            if cmd == "/mode":
                if arg in ("raw", "cached", "compare"):
                    mode = arg
                    console.print(f"[green]mode = {mode}[/green]")
                else:
                    console.print("[red]usage: /mode raw|cached|compare[/red]")
                continue
            if cmd == "/system":
                system = arg or None
                console.print(f"[green]system prompt {'set' if system else 'cleared'}[/green]")
                continue
            if cmd == "/clear":
                history.clear()
                console.print("[green]conversation cleared[/green]")
                continue
            if cmd == "/metrics":
                if not last_metrics:
                    console.print("[yellow]no metrics yet[/yellow]")
                else:
                    console.print_json(data=last_metrics)
                continue
            if cmd == "/max":
                try:
                    max_tokens = int(arg)
                    console.print(f"[green]max_tokens = {max_tokens}[/green]")
                except ValueError:
                    console.print("[red]usage: /max <int>[/red]")
                continue
            if cmd == "/temp":
                try:
                    temperature = float(arg)
                    console.print(f"[green]temperature = {temperature}[/green]")
                except ValueError:
                    console.print("[red]usage: /temp <float>[/red]")
                continue
            if cmd == "/raw":
                if not arg:
                    console.print("[red]usage: /raw <message>[/red]")
                    continue
                line = arg
                mode = "raw"
            elif cmd == "/cached":
                if not arg:
                    console.print("[red]usage: /cached <message>[/red]")
                    continue
                line = arg
                mode = "cached"
            elif cmd == "/compare":
                if not arg:
                    console.print("[red]usage: /compare <message>[/red]")
                    continue
                line = arg
                mode = "compare"
            else:
                console.print(f"[red]unknown command: {cmd}[/red]  type /help")
                continue

        # Regular user message.
        history.append(Message(role="user", content=line))

        # Build the prompt that gets sent.
        if mode == "compare":
            _do_compare(
                history,
                raw_url=raw_url,
                daemon_url=daemon_url,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            # For compare, history is updated with the daemon response (best-effort).
            continue

        prompt = build_prompt(history, system)
        # Strip the trailing "Assistant:" so we just generate the response.
        prompt_for_send = prompt[: -len("\n\nAssistant:")] if prompt.endswith("\n\nAssistant:") else prompt

        # Pick endpoint.
        if mode == "raw":
            target = raw_url
            label = "[dim](raw)[/dim]"
        else:
            target = daemon_url
            label = "[dim](cached)[/dim]"

        # Stream.
        sys.stdout.write(f"\n[bold green]assistant[/bold green] {label}: ")
        sys.stdout.flush()
        text, metrics = stream_completion(
            target,
            prompt_for_send,
            max_tokens=max_tokens,
            temperature=temperature,
            api="native" if target == raw_url else "openai",
        )
        sys.stdout.write("\n")
        last_metrics = metrics
        # Show inline metrics.
        ttft = metrics.get("ttft_ms", 0)
        total = metrics.get("total_ms", 0)
        gen = metrics.get("tokens_generated", 0)
        pref = metrics.get("prefill_tokens", 0)
        sys.stdout.write(
            f"  [dim]ttft={ttft:.0f}ms  total={total:.0f}ms  "
            f"prefill={pref}  generated={gen}[/dim]\n"
        )
        # Divergence warning if compare mode had a hit (not applicable here).
        history.append(Message(role="assistant", content=text, metrics=metrics))


def _do_compare(
    history: list[Message],
    *,
    raw_url: str,
    daemon_url: str,
    system: str | None,
    max_tokens: int,
    temperature: float,
) -> None:
    """Run the last user message both ways (raw + cached) and show a side-by-side."""
    if not history or history[-1].role != "user":
        console.print("[red]no user message to compare[/red]")
        return
    prompt = build_prompt(history, system)
    prompt_for_send = prompt[: -len("\n\nAssistant:")] if prompt.endswith("\n\nAssistant:") else prompt

    console.print("\n[bold cyan]compare mode[/bold cyan] — running both paths")
    # Run raw.
    console.print("[dim]running raw...[/dim]")
    with console.status("[dim]running raw (no cache)...[/dim]"):
        raw_text, raw_m = completion_once(
            raw_url, prompt_for_send, max_tokens=max_tokens, temperature=temperature,
            api="native",
        )
    # Run cached.
    console.print("[dim]running cached...[/dim]")
    cached_text, cached_m = completion_once(
        daemon_url, prompt_for_send, max_tokens=max_tokens, temperature=temperature,
        api="openai",
    )

    # Side-by-side.
    raw_panel = Panel(
        raw_text[:1500] + ("\n[dim]…[/dim]" if len(raw_text) > 1500 else ""),
        title=f"[bold]raw[/bold]   ttft={raw_m.get('ttft_ms', 0):.0f}ms  total={raw_m.get('total_ms', 0):.0f}ms",
        border_style="red",
    )
    cached_panel = Panel(
        cached_text[:1500] + ("\n[dim]…[/dim]" if len(cached_text) > 1500 else ""),
        title=f"[bold]cached[/bold] ttft={cached_m.get('ttft_ms', 0):.0f}ms  total={cached_m.get('total_ms', 0):.0f}ms",
        border_style="green",
    )
    console.print(raw_panel)
    console.print(cached_panel)

    # Comparison summary.
    raw_total = raw_m.get("total_ms", 1)
    cached_total = cached_m.get("total_ms", 1)
    speedup = raw_total / max(cached_total, 1e-9)
    raw_prefill = raw_m.get("prefill_tokens", 0)
    cached_prefill = cached_m.get("prefill_tokens", 0)
    skipped = max(0, raw_prefill - cached_prefill)
    identical = raw_text == cached_text
    diff_chars = sum(1 for a, b in zip(raw_text, cached_text, strict=False) if a != b)
    diff_chars += abs(len(raw_text) - len(cached_text))

    verdict_color = "green" if speedup > 1.05 else "yellow" if speedup > 0.95 else "red"
    console.print(
        Panel(
            f"speedup: [{verdict_color}]{speedup:.3f}x[/{verdict_color}]\n"
            f"prefill tokens saved: {skipped}\n"
            f"outputs byte-identical: {'yes' if identical else 'no'}\n"
            f"differing chars: {diff_chars}",
            title="compare summary",
            border_style=verdict_color,
        )
    )

    # Divergence warning.
    if not identical:
        first_diff = 0
        for i in range(min(len(raw_text), len(cached_text))):
            if raw_text[i] != cached_text[i]:
                first_diff = i
                break
        else:
            first_diff = min(len(raw_text), len(cached_text))
        console.print(
            f"[yellow]⚠ outputs diverge at char {first_diff} "
            f"({diff_chars} chars total differ) — "
            f"this is the cache's restore noise. "
            f"Same prompt, different bytes.[/yellow]"
        )

    # Add the daemon response to history.
    history.append(Message(role="assistant", content=cached_text, metrics=cached_m))


def main(
    *,
    raw_url: str = "http://127.0.0.1:8088",
    daemon_url: str = "http://127.0.0.1:8787",
    mode: str = "raw",
    system: str | None = None,
    model: str = "default",
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> int:
    return chat_session(
        raw_url=raw_url,
        daemon_url=daemon_url,
        initial_mode=mode,
        initial_system=system,
        initial_model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
