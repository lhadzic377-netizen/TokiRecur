"""Top-level CLI entry point for the tokirecur daemon and admin commands."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel


@click.group()
@click.option(
    "-v", "--verbose",
    count=True,
    help="Increase verbosity (use -vv for debug).",
)
@click.option(
    "--db",
    type=click.Path(path_type=Path),
    default=lambda: Path(os.environ.get("TOKI_DB", str(Path.home() / ".tokirecur/index.sqlite"))),
    help="Path to the index SQLite database.",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=lambda: Path(os.environ.get("TOKI_CACHE_DIR", str(Path.home() / ".tokirecur/blobs"))),
    help="Directory for KV cache blobs.",
)
@click.pass_context
def cli(ctx: click.Context, verbose: int, db: Path, cache_dir: Path) -> None:
    """Toki Recur — KV cache reuse for LLM inference."""
    level = logging.WARNING - 10 * verbose
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    db.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ctx.ensure_object(dict)
    ctx.obj["db"] = db
    ctx.obj["cache_dir"] = cache_dir


@cli.command()
@click.option(
    "--host", default="127.0.0.1",
    help="Bind host (default: 127.0.0.1).",
)
@click.option(
    "--port", default=8787, type=int,
    help="Bind port (default: 8787).",
)
@click.option(
    "--backend-url", default="http://127.0.0.1:8080",
    help="Backend llama-server URL (default: http://127.0.0.1:8080).",
)
@click.option(
    "--similarity-threshold", default=0.85, type=float,
    help="Cosine similarity threshold (default: 0.85).",
)
@click.option(
    "--memory-budget-mb", default=8192, type=int,
    help="Cache memory budget in MB (default: 8192).",
)
@click.pass_context
def serve(
    ctx: click.Context,
    host: str,
    port: int,
    backend_url: str,
    similarity_threshold: float,
    memory_budget_mb: int,
) -> None:
    """Start the tokirecur daemon."""
    import uvicorn  # noqa: F401
    from ..api.app import create_app
    from ..backend import LlamaServerBackend
    from ..index import BruteForceIndex

    backend = LlamaServerBackend(backend_url)
    index = BruteForceIndex(ctx.obj["db"])
    app = create_app(
        backend=backend,
        index=index,
        cache_dir=ctx.obj["cache_dir"],
        similarity_threshold=similarity_threshold,
        memory_budget_mb=memory_budget_mb,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


@cli.command()
@click.option(
    "--backend-url", default="http://127.0.0.1:8080",
    help="Backend llama-server URL.",
)
@click.pass_context
def stats(ctx: click.Context, backend_url: str) -> None:
    """Print index statistics."""
    from ..index import BruteForceIndex
    index = BruteForceIndex(ctx.obj["db"])
    import json
    click.echo(json.dumps(index.stats(), indent=2))


@cli.command()
@click.option(
    "--target-mb", type=int, required=True,
    help="Target cache size in MB.",
)
@click.pass_context
def evict(ctx: click.Context, target_mb: int) -> None:
    """Evict LRU entries until the cache is under target_mb."""
    from ..index import BruteForceIndex
    index = BruteForceIndex(ctx.obj["db"])
    target_bytes = target_mb * 1024 * 1024
    n = index.evict_lru(target_bytes)
    click.echo(f"evicted {n} entries")


@cli.command(
    "bench",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],  # disable Click's --help so argparse handles it
    },
)
@click.pass_context
def bench(ctx: click.Context) -> None:
    """Run the benchmark suite. Pass `--help` to see all options."""
    from ..bench.cli import main as bench_main
    sys.argv = ["toki bench"] + ctx.args
    raise SystemExit(bench_main(ctx.args))


@cli.command(
    "run",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    },
)
@click.option("--raw-url", default="http://127.0.0.1:8088",
              help="llama-server URL (default: http://127.0.0.1:8088)")
@click.option("--daemon-url", default="http://127.0.0.1:8787",
              help="toki daemon URL (default: http://127.0.0.1:8787)")
@click.option("--mode", default="raw", type=click.Choice(["raw", "cached", "compare"]),
              help="Inference path (default: raw)")
@click.option("--system", default=None, help="System prompt")
@click.option("--model", default="default", help="Model name (informational)")
@click.option("--max-tokens", "-m", default=256, type=int)
@click.option("--temperature", "-t", default=0.0, type=float)
@click.pass_context
def run(ctx: click.Context, raw_url: str, daemon_url: str, mode: str, system: str | None,
        model: str, max_tokens: int, temperature: float) -> None:
    """Start an interactive chat session against the model.

    With --mode raw, sends directly to llama-server (bypasses cache).
    With --mode cached, sends through the toki daemon (recycling engine).
    With --mode compare, runs both ways and shows side-by-side diff.

    In-session slash commands:
      /mode raw|cached|compare   /system <text>   /clear
      /metrics   /max <n>   /temp <f>   /raw <msg>   /cached <msg>
      /compare <msg>   /quit
    """
    if "--help" in ctx.args or "-h" in ctx.args:
        click.echo(
            "Usage: toki run [OPTIONS]\n\n"
            "Interactive chat REPL. Type /help inside for slash commands.\n\n"
            "  --raw-url TEXT         llama-server URL (default: http://127.0.0.1:8088)\n"
            "  --daemon-url TEXT      toki daemon URL (default: http://127.0.0.1:8787)\n"
            "  --mode [raw|cached|compare]   inference path (default: raw)\n"
            "  --system TEXT          system prompt\n"
            "  --max-tokens INT       max tokens to generate (default: 256)\n"
            "  --temperature FLOAT    sampling temperature (default: 0.0)\n"
        )
        return
    from .chat import main as chat_main
    raise SystemExit(
        chat_main(
            raw_url=raw_url,
            daemon_url=daemon_url,
            mode=mode,
            system=system,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    )


@cli.command("chat")
@click.option("--raw-url", default="http://127.0.0.1:8088")
@click.option("--daemon-url", default="http://127.0.0.1:8787")
@click.option("--mode", default="raw", type=click.Choice(["raw", "cached", "compare"]))
@click.option("--system", default=None)
@click.option("--max-tokens", "-m", default=256, type=int)
@click.option("--temperature", "-t", default=0.0, type=float)
def chat_cmd(raw_url: str, daemon_url: str, mode: str, system: str | None,
             max_tokens: int, temperature: float) -> None:
    """Start an interactive chat session (alias for `toki run`)."""
    from .chat import main as chat_main
    raise SystemExit(
        chat_main(
            raw_url=raw_url,
            daemon_url=daemon_url,
            mode=mode,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    )


@cli.command("dashboard")
@click.option(
    "--reports",
    type=click.Path(path_type=Path),
    default=lambda: Path(os.environ.get("TOKI_REPORTS_DIR", str(Path.home() / "Downloads" / "toki-reports"))),
    help="Directory containing benchmark JSON reports",
)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=7778, type=int, show_default=True)
def dashboard(reports: Path, host: str, port: int) -> None:
    """Launch the benchmark dashboard — visualises reports from `toki bench`."""
    from ..dashboard import main as dash_main
    sys.argv = ["toki dashboard", "--reports", str(reports), "--host", host, "--port", str(port)]
    raise SystemExit(dash_main(["--reports", str(reports), "--host", host, "--port", str(port)]))


@cli.command("tui")
@click.option(
    "--daemon-url",
    default=lambda: os.environ.get("TOKI_DAEMON_URL", "http://127.0.0.1:8787"),
    show_default=True,
    help="Toki daemon URL.",
)
@click.option(
    "--raw-url",
    default=lambda: os.environ.get("TOKI_LLAMA_URL", "http://127.0.0.1:8088"),
    show_default=True,
    help="llama-server URL (used for raw mode and precision check).",
)
@click.option(
    "--auth-token",
    default=lambda: os.environ.get("TOKI_AUTH_TOKEN"),
    help="Bearer token for the daemon. Defaults to $TOKI_AUTH_TOKEN.",
)
def tui(daemon_url: str, raw_url: str, auth_token: str | None) -> None:
    """Launch the production TUI chat client.

    A terminal-based chat interface for talking to a local LLM through
    the toki daemon. Three modes (raw / cached / compare), live metrics,
    session pinning, and precision-check panel.

    \b
    Keys:
      Ctrl+1  raw       Ctrl+L  clear chat    Ctrl+U  new session
      Ctrl+2  cached    Ctrl+R  reprobe       Ctrl+M  toggle panels
      Ctrl+3  compare   Ctrl+P  precision     Ctrl+C  quit

    \b
    Slash commands:
      /help /quit /clear /mode /session /session-new
      /precision /metrics /slots /reset /reprobe
    """
    import sys
    if not sys.stdout.isatty():
        click.echo("error: toki tui requires an interactive terminal (TTY).", err=True)
        click.echo("hint: run from your terminal, or use `toki run` for non-interactive chat.", err=True)
        raise SystemExit(2)
    try:
        from ..tui.app import run as tui_run
        tui_run(
            daemon_url=daemon_url, raw_url=raw_url, auth_token=auth_token,
        )
    except ImportError as e:
        click.echo(f"error: {e}", err=True)
        click.echo("hint: install textual with `pip install textual`.", err=True)
        raise SystemExit(1)


@cli.command("diff")
@click.option("--raw-url", default="http://127.0.0.1:8088")
@click.option("--daemon-url", default="http://127.0.0.1:8787")
@click.option("--system", default=None, help="System prompt")
@click.option("--max-tokens", "-m", default=256, type=int)
@click.option("--temperature", "-t", default=0.0, type=float)
@click.argument("prompt", nargs=-1)
def diff(raw_url: str, daemon_url: str, system: str | None, max_tokens: int,
         temperature: float, prompt: tuple[str, ...]) -> None:
    """Run a single prompt both ways (raw + cached) and show side-by-side diff.

    Usage: toki diff --system "<long system>" "<user prompt>"
    """
    from .chat import completion_once
    con = Console()
    if not prompt:
        con.print("[red]usage: toki diff [OPTIONS] PROMPT[/red]")
        raise SystemExit(2)
    full_prompt = " ".join(prompt)
    with con.status("[dim]running raw...[/dim]"):
        raw_text, raw_m = completion_once(
            raw_url, full_prompt, max_tokens=max_tokens,
            temperature=temperature, system=system,
        )
    with con.status("[dim]running cached...[/dim]"):
        cached_text, cached_m = completion_once(
            daemon_url, full_prompt, max_tokens=max_tokens,
            temperature=temperature, system=system,
        )
    con.print(Panel(
        raw_text, title=f"[bold red]raw[/bold red]  ttft={raw_m.get('ttft_ms', 0):.0f}ms  total={raw_m.get('total_ms', 0):.0f}ms",
        border_style="red",
    ))
    con.print(Panel(
        cached_text, title=f"[bold green]cached[/bold green] ttft={cached_m.get('ttft_ms', 0):.0f}ms  total={cached_m.get('total_ms', 0):.0f}ms",
        border_style="green",
    ))
    speedup = raw_m.get("total_ms", 1) / max(cached_m.get("total_ms", 1), 1e-9)
    identical = raw_text == cached_text
    skipped = max(0, raw_m.get("prefill_tokens", 0) - cached_m.get("prefill_tokens", 0))
    color = "green" if speedup > 1.05 else "yellow" if speedup > 0.95 else "red"
    con.print(Panel(
        f"speedup: [{color}]{speedup:.3f}x[/{color}]\n"
        f"prefill tokens saved: {skipped}\n"
        f"outputs byte-identical: {'yes' if identical else 'no'}",
        title="diff summary",
        border_style=color,
    ))
    if not identical:
        first_diff = 0
        for i in range(min(len(raw_text), len(cached_text))):
            if raw_text[i] != cached_text[i]:
                first_diff = i
                break
        con.print(f"[yellow]⚠ outputs diverge at char {first_diff}[/yellow]")


@cli.command()
def doctor() -> None:
    """Check llama-server health, ROCm runtime, GPU availability."""
    import shutil
    import subprocess
    rows = []
    rows.append(("llama-server binary", shutil.which("llama-server") or "missing"))
    for cmd, label in [
        (["rocminfo"], "rocminfo"),
        (["hipconfig", "--version"], "hipconfig"),
    ]:
        try:
            out = subprocess.check_output(cmd, text=True, timeout=5).strip().splitlines()
            rows.append((label, out[0] if out else "ok"))
        except Exception as e:
            rows.append((label, f"missing ({e})"))
    try:
        import httpx
        r = httpx.get("http://127.0.0.1:8080/health", timeout=2.0)
        rows.append(("llama-server :8080", "ok" if r.status_code == 200 else "down"))
    except Exception:
        rows.append(("llama-server :8080", "not reachable"))
    try:
        import httpx
        r = httpx.get("http://127.0.0.1:8787/healthz", timeout=2.0)
        rows.append(("toki daemon :8787", "ok" if r.status_code == 200 else "down"))
    except Exception:
        rows.append(("toki daemon :8787", "not running"))

    width = max(len(k) for k, _ in rows) + 2
    for k, v in rows:
        click.echo(f"{k.ljust(width)}{v}")


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=7777, type=int, show_default=True,
              help="Unused localhost port (default: 7777)")
@click.option("--raw-url", default="http://127.0.0.1:8088", show_default=True,
              help="Backend llama-server URL")
@click.option("--daemon-url", default="http://127.0.0.1:8787", show_default=True,
              help="Toki daemon URL")
@click.option("--tls-cert", type=click.Path(path_type=Path), default=None,
              help="Path to a TLS certificate (PEM). Enables HTTPS + wss.")
@click.option("--tls-key", type=click.Path(path_type=Path), default=None,
              help="Path to the TLS private key (PEM). Requires --tls-cert.")
@click.option("--detach", is_flag=True,
              help="Run in background, return PID")
def web(host: str, port: int, raw_url: str, daemon_url: str,
        tls_cert: Path | None, tls_key: Path | None, detach: bool) -> None:
    """Launch the production-ready web chat UI on http(s)://HOST:PORT.

    Default port is 7777 (an uncommon port unlikely to conflict with
    other localhost services like 3000, 5000, 8080, etc.).

    Pass --tls-cert and --tls-key to serve over HTTPS (wss:// for the
    chat socket). The web UI reads TOKI_AUTH_TOKEN (or the auth file)
    and gates the whole app behind a login; over TLS the session cookie
    is marked Secure.

    The UI lets you toggle between three inference modes (raw, cached,
    compare) with a single click, stream responses, and see live metrics
    per message. Open the URL in any modern browser.
    """
    if bool(tls_cert) != bool(tls_key):
        raise click.UsageError("--tls-cert and --tls-key must be given together")
    if tls_cert is None and tls_key is None:
        # Honor env-file settings (used by the systemd unit, which
        # can't pass flags). Explicit flags take precedence.
        tls_cert = Path(os.environ["TOKI_WEB_TLS_CERT"]) if os.environ.get("TOKI_WEB_TLS_CERT") else None
        tls_key = Path(os.environ["TOKI_WEB_TLS_KEY"]) if os.environ.get("TOKI_WEB_TLS_KEY") else None
        if bool(tls_cert) != bool(tls_key):
            raise click.UsageError(
                "TOKI_WEB_TLS_CERT and TOKI_WEB_TLS_KEY must both be set")
    if tls_cert and not tls_cert.is_file():
        raise click.UsageError(f"TLS certificate not found: {tls_cert}")
    if tls_key and not tls_key.is_file():
        raise click.UsageError(f"TLS key not found: {tls_key}")

    import uvicorn  # noqa: F401
    from ..web import create_web_app

    app = create_web_app(raw_url=raw_url, daemon_url=daemon_url, tls=bool(tls_cert))
    scheme = "https" if tls_cert else "http"

    if detach:
        import subprocess
        log_path = Path(os.environ.get("TOKI_WEB_LOG", "/tmp/toki/web.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "uvicorn",
            "tokirecur.web.server:create_web_app_from_env",
            "--factory",
            "--host", host,
            "--port", str(port),
            "--log-level", "info",
        ]
        if tls_cert:
            cmd += ["--ssl-certfile", str(tls_cert), "--ssl-keyfile", str(tls_key)]
        proc = subprocess.Popen(
            cmd,
            env={**os.environ,
                 "TOKI_WEB_RAW_URL": raw_url,
                 "TOKI_WEB_DAEMON_URL": daemon_url,
                 "TOKI_WEB_TLS": "1" if tls_cert else "0"},
            stdout=open(log_path, "ab"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        click.echo(f"toki web started (pid={proc.pid})  {scheme}://{host}:{port}")
        click.echo(f"log: {log_path}")
        return

    click.echo(f"  → Toki Recur web UI: {scheme}://{host}:{port}")
    click.echo(f"  → raw_url: {raw_url}    daemon_url: {daemon_url}")
    click.echo("  → Ctrl+C to stop")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        ssl_certfile=str(tls_cert) if tls_cert else None,
        ssl_keyfile=str(tls_key) if tls_key else None,
    )


@cli.command()
@click.option(
    "--model",
    default=lambda: os.environ.get(
        "TOKI_MODEL",
        str(Path.home() / "Downloads/Qwen3.5-9B-Q4_K_M.gguf"),
    ),
    type=click.Path(path_type=Path),
    help="Path to GGUF model (default: $TOKI_MODEL or ~/Downloads/Qwen3.5-9B-Q4_K_M.gguf)",
)
@click.option("--llama-port", default=8088, type=int)
@click.option("--daemon-port", default=8787, type=int)
@click.option("--ctx-size", default=8192, type=int)
@click.option("--n-gpu-layers", default=99, type=int)
@click.option("--detach/--foreground", default=False, help="Run in background (default: foreground)")
@click.option(
    "--skip-llama",
    is_flag=True,
    help="Don't start llama-server; assume it's already running.",
)
@click.option(
    "--extra-llama-args",
    default="",
    help="Extra args to pass to llama-server (space-separated).",
)
def up(
    model: Path,
    llama_port: int,
    daemon_port: int,
    ctx_size: int,
    n_gpu_layers: int,
    detach: bool,
    skip_llama: bool,
    extra_llama_args: str,
) -> None:
    """Start the full Toki Recur stack: llama-server + daemon.

    Foreground by default (Ctrl+C stops both). Pass --detach to run in
    the background; use `toki down` to stop."""
    from .. import lifecycle

    if not model.exists() and not skip_llama:
        click.echo(f"error: model not found: {model}", err=True)
        click.echo("hint: set TOKI_MODEL or pass --model", err=True)
        raise SystemExit(2)

    if not skip_llama:
        try:
            pid = lifecycle.start_llama_server(
                model=str(model),
                port=llama_port,
                ctx_size=ctx_size,
                n_gpu_layers=n_gpu_layers,
                extra_args=extra_llama_args.split() if extra_llama_args else None,
            )
            click.echo(f"llama-server started (pid={pid})  http://127.0.0.1:{llama_port}")
        except Exception as e:
            click.echo(f"failed to start llama-server: {e}", err=True)
            raise SystemExit(1)
    else:
        click.echo(f"skipping llama-server start (assumed on :{llama_port})")

    if detach:
        pid = lifecycle.start_daemon(
            host="127.0.0.1",
            port=daemon_port,
            backend_url=f"http://127.0.0.1:{llama_port}",
            wait=True,
        )
        click.echo(f"toki daemon started (pid={pid})  http://127.0.0.1:{daemon_port}")
        click.echo(f"logs: {lifecycle.LLAMA_LOG}  {lifecycle.DAEMON_LOG}")
        return

    # Foreground: run uvicorn in this process so Ctrl+C works cleanly.
    try:
        lifecycle.start_daemon(
            host="127.0.0.1",
            port=daemon_port,
            backend_url=f"http://127.0.0.1:{llama_port}",
            wait=False,
        )
    except Exception as e:
        click.echo(f"failed to spawn toki daemon: {e}", err=True)
        raise SystemExit(1)

    click.echo(f"toki daemon  http://127.0.0.1:{daemon_port}")
    click.echo(f"logs: {lifecycle.LLAMA_LOG}  {lifecycle.DAEMON_LOG}")
    click.echo("Ctrl+C to stop everything.")

    try:
        import uvicorn
        uvicorn.run(
            "tokirecur.api.app:create_app_from_env",
            factory=True,
            host="127.0.0.1",
            port=daemon_port,
            log_level="info",
        )
    except KeyboardInterrupt:
        click.echo("\nshutting down...")
    finally:
        lifecycle.stop_all()


@cli.command()
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def down(yes: bool) -> None:
    """Stop the Toki Recur stack (llama-server + daemon)."""
    from .. import lifecycle
    s = lifecycle.status()
    if not s["llama_server"]["alive"] and not s["toki_daemon"]["alive"]:
        click.echo("nothing to stop (no processes tracked).")
        return
    if not yes:
        click.echo("will stop:")
        if s["llama_server"]["alive"]:
            click.echo(f"  llama-server (pid={s['llama_server']['pid']})")
        if s["toki_daemon"]["alive"]:
            click.echo(f"  toki daemon   (pid={s['toki_daemon']['pid']})")
        if not click.confirm("proceed?", default=True):
            raise SystemExit(0)
    lifecycle.stop_all()
    click.echo("stopped.")


@cli.command()
def status() -> None:
    """Show the state of the Toki Recur stack."""
    from .. import lifecycle
    import json
    s = lifecycle.status()

    def row(label: str, data: dict, port: int) -> None:
        if data["alive"] and data["reachable"]:
            state = "[green]running[/green]"
        elif data["alive"] and not data["reachable"]:
            state = "[yellow]stuck (alive but unreachable)[/yellow]"
        elif data["pid"] and not data["alive"]:
            state = "[red]dead (stale pidfile)[/red]"
        else:
            state = "[dim]stopped[/dim]"
        pid = f"pid={data['pid']}" if data["pid"] else "no pidfile"
        click.echo(f"  {label.ljust(15)} {state}  {pid}  http://127.0.0.1:{port}")

    click.echo("Toki Recur stack:")
    row("llama-server", s["llama_server"], 8088)
    row("toki daemon", s["toki_daemon"], 8787)
    if s["llama_server"]["alive"] and s["toki_daemon"]["alive"]:
        click.echo("use `toki bench` to run the benchmark, `toki down` to stop.")
    elif s["llama_server"]["alive"] or s["toki_daemon"]["alive"]:
        click.echo("stack is partial; `toki down` then `toki up` to reset.")
    else:
        click.echo("nothing running. `toki up` to start.")


@cli.command()
@click.pass_context
def interactive(ctx: click.Context) -> None:
    """Start the interactive shell (the default when no subcommand is given)."""
    from .. import repl
    raise SystemExit(repl.run())


def main() -> int:
    # Known subcommands. Anything else (no args, or a flag) prints help
    # and shows stack status instead of running a multi-minute benchmark.
    if len(sys.argv) == 1:
        # No args — show help + status, exit fast.
        click.echo(cli.get_help(click.Context(cli)))
        click.echo("")
        click.echo("Current stack status:")
        try:
            from .. import lifecycle
            s = lifecycle.status()
            ll = "running" if s["llama_server"]["alive"] else "stopped"
            da = "running" if s["toki_daemon"]["alive"] else "stopped"
            click.echo(f"  llama-server :8088  {ll}")
            click.echo(f"  toki daemon  :8787  {da}")
            if not s["llama_server"]["alive"] and not s["toki_daemon"]["alive"]:
                click.echo("")
                click.echo("Run `toki up` to start the full stack.")
        except Exception:
            pass
        return 0
    # Backward-compat: `toki --workload X --max-tokens N` is the
    # default-benchmark shortcut. Detect flags and route accordingly.
    if sys.argv[1] not in _SUBCOMMANDS and sys.argv[1].startswith("-"):
        return run_default_benchmark()
    if sys.argv[1] not in _SUBCOMMANDS:
        # Truly unknown subcommand — show help.
        click.echo(cli.get_help(click.Context(cli)))
        return 0 if sys.argv[1] in ("--help", "-h") else 2
    return cli(obj={})


_SUBCOMMANDS = {
    "bench", "doctor", "down", "evict", "interactive", "run",
    "serve", "stats", "status", "up", "shell",
    "chat", "diff", "web", "dashboard", "tui",
}


def run_default_benchmark() -> int:
    """One-shot: start llama-server, run benchmark, print results, clean up.

    Resolves tunables from CLI flags if present, otherwise from $TOKI_*
    environment variables, otherwise from sensible defaults:

        --model           $TOKI_MODEL or ~/Downloads/Qwen3.5-9B-Q4_K_M.gguf
        --llama-port      $TOKI_LLAMA_PORT or 8088
        --ctx-size        $TOKI_CTX_SIZE or 8192
        --n-gpu-layers    $TOKI_NGPU or 99
        --workload        $TOKI_WORKLOAD or agent_long_context
        --max-tokens      $TOKI_MAX_TOKENS or 64
        --keep-running    leave llama-server running after the bench
    """
    import argparse
    from .. import lifecycle

    parser = argparse.ArgumentParser(
        prog="toki",
        description="Run the full Toki Recur benchmark end-to-end.",
        add_help=True,
    )
    parser.add_argument("--model", type=Path,
        default=Path(os.environ.get("TOKI_MODEL",
            str(Path.home() / "Downloads/Qwen3.5-9B-Q4_K_M.gguf"))))
    parser.add_argument("--llama-port", type=int,
        default=int(os.environ.get("TOKI_LLAMA_PORT", "8088")))
    parser.add_argument("--ctx-size", type=int,
        default=int(os.environ.get("TOKI_CTX_SIZE", "8192")))
    parser.add_argument("--n-gpu-layers", type=int,
        default=int(os.environ.get("TOKI_NGPU", "99")))
    parser.add_argument("--workload", default=os.environ.get("TOKI_WORKLOAD",
        "agent_100"),
        choices=["agent_long_context", "agent_25", "agent_50", "agent_100",
                 "agent_long_output", "code_review", "summarization",
                 "rag_chunks", "mixed", "no_overlap",
                 "realistic_mixed", "stress_adversarial"])
    parser.add_argument("--workload-file", type=Path, default=None,
        help="Load custom workload from JSON file (overrides --workload)")
    parser.add_argument("--max-tokens", type=int,
        default=int(os.environ.get("TOKI_MAX_TOKENS", "64")))
    parser.add_argument("--keep-running", action="store_true",
        help="Leave llama-server running after the benchmark.")
    parser.add_argument("--output-json", type=Path, default=None,
        help="Write full JSON report to this path.")
    args = parser.parse_args(sys.argv[1:])

    if not args.model.exists():
        click.echo(f"error: model not found: {args.model}", err=True)
        click.echo("hint: set TOKI_MODEL or pass --model", err=True)
        return 2

    # Phase 1: ensure llama-server is running.
    started_by_us = False
    if not lifecycle.llama_reachable(args.llama_port):
        click.echo(f"[1/3] starting llama-server on :{args.llama_port}...")
        try:
            lifecycle.start_llama_server(
                model=str(args.model),
                port=args.llama_port,
                ctx_size=args.ctx_size,
                n_gpu_layers=args.n_gpu_layers,
            )
            started_by_us = True
        except Exception as e:
            click.echo(f"failed to start llama-server: {e}", err=True)
            return 1
    else:
        click.echo(f"[1/3] llama-server already running on :{args.llama_port}")

    # Phase 2: run the benchmark.
    click.echo(f"[2/3] running benchmark ({args.workload}, max-tokens={args.max_tokens})...")
    from ..bench.cli import main as bench_main
    bench_argv = [
        "--model", str(args.model),
        "--base-url", f"http://127.0.0.1:{args.llama_port}",
        "--workload", args.workload,
        "--max-tokens", str(args.max_tokens),
    ]
    if args.workload_file:
        bench_argv += ["--workload-file", str(args.workload_file)]
    if args.output_json:
        bench_argv += ["--output-json", str(args.output_json)]
    rc = 0
    try:
        bench_main(bench_argv)
    except SystemExit as e:
        rc = e.code or 0

    # Phase 3: clean up if we started llama-server.
    if started_by_us and not args.keep_running:
        click.echo(f"[3/3] stopping llama-server...")
        lifecycle.stop_pid(lifecycle.LLAMA_PID, "llama-server")
    else:
        click.echo(f"[3/3] leaving llama-server running (use `toki down` to stop)")

    return rc


if __name__ == "__main__":
    sys.exit(main())
