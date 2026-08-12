"""Top-level CLI entry point for the tokirecur daemon and admin commands."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from pathlib import Path

import click
import httpx
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
    import httpx
    import uvicorn  # noqa: F401

    from ..api.app import create_app
    from ..backend import LlamaServerBackend

    # 0.2.17: every daemon start is a repair point. Rewrite the four
    # agent configs for the active model so stale/legacy configs (the
    # "model '<name>' not found" 404 class) heal themselves without any
    # user command. Best-effort and non-fatal by design.
    from ..ctl import heal_cli_configs
    from ..index import BruteForceIndex
    healed = heal_cli_configs()
    if healed:
        click.echo("agent configs synced for the active model")

    # Fail loudly-but-kindly: a daemon with no llama-server behind it
    # looks exactly like "the server doesn't start". Ping the backend
    # before binding so the user gets a hint instead of silence.
    try:
        r = httpx.get(f"{backend_url}/health", timeout=2.0)
        if r.status_code >= 500:
            raise httpx.HTTPStatusError(
                f"backend unhealthy: {r.status_code}", request=r.request, response=r
            )
        click.echo(f"backend: {backend_url} reachable ({r.status_code})")
    except Exception as e:
        click.echo(
            f"WARNING: llama-server not reachable at {backend_url} "
            f"({type(e).__name__}: {e})",
            err=True,
        )
        click.echo(
            "  Start it with `toki up` or `toki model use <name>` "
            "before sending requests, or pass --backend-url.",
            err=True,
        )

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
    """Launch the TUI chat client (Windows: the CLI chat instead).

    A terminal-based chat interface for talking to a local LLM through
    the toki daemon. Three modes (raw / cached / compare), live metrics,
    session pinning, and precision-check panel.

    On Windows the Textual TUI is replaced by the plain Ollama-style
    CLI chat (``clichat``): the classic Windows console renders TUIs
    slowly and glitches. Linux and macOS keep the full TUI. (Bare
    ``toki`` on Windows opens the CLI launcher menu — ``clilauncher``.)

    \b
    Keys (TUI):
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
        click.echo("hint: run from an interactive terminal.", err=True)
        raise SystemExit(2)
    if os.name == "nt":
        from ..clichat import run_cli_chat

        click.echo(
            "note: the TUI needs a capable terminal; on Windows you get "
            "the CLI chat instead (Linux/macOS keep the full TUI).",
            err=True,
        )
        try:
            raise SystemExit(run_cli_chat(
                daemon_url=daemon_url, raw_url=raw_url, auth_token=auth_token,
            ))
        except SystemExit:
            raise
    try:
        from ..tui.app import run as tui_run
        tui_run(
            daemon_url=daemon_url, raw_url=raw_url, auth_token=auth_token,
        )
    except ImportError as e:
        click.echo(f"error: {e}", err=True)
        click.echo("hint: install textual with `pip install textual`.", err=True)
        raise SystemExit(1) from None
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
    from .completions import completion_once
    con = Console()
    if not prompt:
        con.print("[red]usage: toki diff [OPTIONS] PROMPT[/red]")
        raise SystemExit(2)
    full_prompt = " ".join(prompt)
    with con.status("[dim]running raw...[/dim]"):
        raw_text, raw_m = completion_once(
            raw_url, full_prompt, max_tokens=max_tokens,
            temperature=temperature, system=system, api="native",
        )
    with con.status("[dim]running cached...[/dim]"):
        cached_text, cached_m = completion_once(
            daemon_url, full_prompt, max_tokens=max_tokens,
            temperature=temperature, system=system, api="openai",
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
@click.option("--fix", is_flag=True,
              help="Auto-repair detected issues: install the CUDA "
                   "llama-server build when the bin dir is CPU but an "
                   "NVIDIA GPU is present, and restart the active model.")
def doctor(fix: bool) -> None:
    """Check GPU, llama-server build, and stack health."""
    from .. import diagnose

    click.echo(diagnose.run())
    # Stack health (existing checks, unchanged in spirit).
    import httpx

    rows = []
    try:
        r = httpx.get("http://127.0.0.1:8080/health", timeout=2.0)
        rows.append(("llama-server :8080", "ok" if r.status_code == 200 else "down"))
    except Exception:
        rows.append(("llama-server :8080", "not reachable"))
    try:
        r = httpx.get("http://127.0.0.1:8787/healthz", timeout=2.0)
        rows.append(("toki daemon :8787", "ok" if r.status_code == 200 else "down"))
    except Exception:
        rows.append(("toki daemon :8787", "not running"))
    for k, v in rows:
        click.echo(f"{k.ljust(24)}{v}")

    if not fix:
        return

    # --- Auto-repair ---
    from .. import ctl
    from . import _detect_gpu_backend as _detect_gpu_backend  # noqa: F401

    actions: list[str] = []

    # 1) If the available llama-server is a CPU build but an NVIDIA GPU
    #    is present, install the CUDA build and point the active model
    #    at it (or install into the bin dir if no model is active).
    from ..diagnose import binary_backend, gpu_hardware, installed_build
    hw = gpu_hardware()
    builds = installed_build()
    state_bin = builds.get("state_bin")
    if hw.get("backend") == "CUDA" and state_bin and binary_backend(state_bin) == "CPU":
        from ..ctl import read_active
        from . import install_llama
        active = read_active()
        try:
            if active:
                install_llama.callback(tag=None, dest=None, name=active,
                                       backend="cuda", gfx=None)
                actions.append(f"installed CUDA build for {active}")
            else:
                install_llama.callback(tag=None, dest=None, name=None,
                                       backend="cuda", gfx=None)
                actions.append("installed CUDA build into state bin")
        except (RuntimeError, OSError, SystemExit) as e:
            click.echo(f"fix: install-llama failed: {e}", err=True)

    # 2) If the active model has a stale/empty stack, restart it with the
    #    freshly installed build.
    from ..ctl import model_status, read_active, use_model
    active = read_active()
    if active:
        try:
            spec = ctl.get_model(active)
        except Exception:
            spec = None
        if spec is not None:
            st = model_status(spec)
            if not st["llama"]["reachable"]:
                try:
                    use_model(active)
                    actions.append(f"restarted {active}")
                except (KeyError, FileNotFoundError, RuntimeError, OSError) as e:
                    click.echo(f"fix: use_model failed: {e}", err=True)

    # 3) Heal the four agent configs for the active model (0.2.17):
    #    legacy repo-path model ids are normalized and any stale
    #    endpoint is rewritten to the active model's daemon. This is
    #    the one-command fix for the "model '<name>' not found" 404.
    from ..ctl import heal_cli_configs
    healed_paths = heal_cli_configs()
    if healed_paths:
        names = ", ".join(f"{k}:{p.name}" for k, p in healed_paths.items())
        actions.append(f"synced agent configs ({names})")
        spec = None
        try:
            spec = ctl.get_model(active) if active else None
        except Exception:
            spec = None
        if spec is not None:
            import httpx as _httpx
            try:
                r = _httpx.get(f"{spec.daemon_url}/healthz", timeout=2.0)
                state = "ok" if r.status_code == 200 else f"down ({r.status_code})"
            except Exception:
                state = "not reachable"
            actions.append(f"agent endpoints -> {spec.daemon_url}/v1 ({state}; "
                           f"start the stack with `toki model use {active}` if down)")

    if actions:
        click.echo("fix:")
        for a in actions:
            click.echo(f"  - {a}")
        click.echo("rerun `toki doctor` to confirm.")
    else:
        click.echo("fix: nothing to do.")


@cli.command()
@click.option("--repo", default=None,
              help="Path to the tokirecur git checkout (default: walk "
                   "parents of the installed package until .git is found).")
def self_upgrade(repo: str | None) -> None:
    """Upgrade toki to the latest git HEAD: git pull + pip install.

    Run from the repo directory on a Windows install that was created
    via ``pip install`` from a git checkout. Collapses the standard
    5-step recovery into one command:

        git fetch origin && git reset --hard origin/master
        pip install --force-reinstall .

    Refuses to run if the installed package isn't a git checkout.
    """
    import subprocess
    import sys

    repo_path = Path(repo) if repo else _find_repo_dir()
    if repo_path is None:
        click.echo(
            "error: cannot find a tokirecur git checkout. Pass --repo "
            "<path> or run from inside the repo.",
            err=True,
        )
        raise SystemExit(1) from None
    if not (repo_path / ".git").is_dir():
        click.echo(
            f"error: {repo_path} is not a git checkout (no .git/)",
            err=True,
        )
        raise SystemExit(1) from None
    if not (repo_path / "pyproject.toml").is_file():
        click.echo(
            f"error: {repo_path} has no pyproject.toml",
            err=True,
        )
        raise SystemExit(1) from None

    def _run(cmd: list[str]) -> tuple[int, str]:
        r = subprocess.run(cmd, cwd=repo_path, capture_output=True,
                           text=True, timeout=120)
        return r.returncode, (r.stdout + r.stderr).strip()

    click.echo(f"repo: {repo_path}")
    rc, out = _run(["git", "fetch", "origin"])
    if rc != 0:
        click.echo(f"git fetch failed: {out}", err=True)
        raise SystemExit(1) from None
    rc, out = _run(["git", "reset", "--hard", "origin/master"])
    if rc != 0:
        click.echo(f"git reset failed: {out}", err=True)
        raise SystemExit(1) from None
    rc, out = _run(["git", "rev-parse", "--short", "HEAD"])
    new_head = out.splitlines()[-1] if out else "?"
    click.echo(f"HEAD: {new_head}")

    # Editable installs don't need a pip install (the .pth/egg-link
    # already points at the source tree we just updated).
    from ..ctl import STATE_ROOT  # noqa: F401  (imports side effects)
    rc, out = _run([sys.executable, "-m", "pip", "show", "tokirecur"])
    is_editable = "Editable project location" in out
    if is_editable:
        click.echo("install: editable — nothing to reinstall")
    else:
        rc, out = _run([sys.executable, "-m", "pip", "install",
                        "--force-reinstall", "."])
        if rc != 0:
            click.echo(f"pip install failed: {out}", err=True)
            raise SystemExit(1) from None
        click.echo("pip install: ok")
    click.echo(
        "done. A new Python process is needed for the upgrade to "
        "take effect — restart your shell or rerun toki."
    )


def _find_repo_dir() -> Path | None:
    """Walk parents of this module looking for a directory containing
    ``.git`` and a ``pyproject.toml`` named tokirecur."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / ".git").is_dir() and (parent / "pyproject.toml").is_file():
            try:
                text = (parent / "pyproject.toml").read_text()
            except OSError:
                continue
            if "name = \"tokirecur\"" in text or "name = 'tokirecur'" in text:
                return parent
    return None


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
    chat socket). The UI is always open (no login page); the daemon
    stays protected because the server authenticates every proxied
    call with its own configured token (TOKI_AUTH_TOKEN or the auth
    file).

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

    app = create_web_app(raw_url=raw_url, daemon_url=daemon_url)
    scheme = "https" if tls_cert else "http"

    if detach:
        import subprocess
        log_path = Path(os.environ.get(
            "TOKI_WEB_LOG", str(Path.home() / ".local/state/tokirecur/web.log")))
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
                 "TOKI_WEB_DAEMON_URL": daemon_url},
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

    # 0.2.17: `toki up` is a repair point — heal the four agent configs
    # for the active registry model before the stack comes up.
    from ..ctl import heal_cli_configs
    healed = heal_cli_configs()
    if healed:
        click.echo("agent configs synced for the active model")

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
            raise SystemExit(1) from None
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
        raise SystemExit(1) from None
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
    """Stop everything: every model stack (llama-server + daemon) and
    the legacy stack."""
    from .. import lifecycle
    from ..ctl import load_registry, model_status, stop_model
    s = lifecycle.status()
    targets: list[str] = []
    if s["llama_server"]["alive"] or s["toki_daemon"]["alive"]:
        targets.append("legacy stack")
    for spec in load_registry():
        st = model_status(spec)
        if st["llama"]["reachable"] or st["daemon"]["reachable"]:
            targets.append(f"{spec.name} (llama :{spec.llama_port}, daemon :{spec.daemon_port})")
    if not targets:
        click.echo("nothing to stop (all stacks are down).")
        return
    if not yes:
        click.echo("will stop:")
        for t in targets:
            click.echo(f"  {t}")
        if not click.confirm("proceed?", default=True):
            raise SystemExit(0)
    stopped: list[str] = []
    if s["llama_server"]["alive"] or s["toki_daemon"]["alive"]:
        lifecycle.stop_all()
        stopped.append("legacy stack")
    for spec in load_registry():
        st = model_status(spec)
        if st["llama"]["reachable"] or st["daemon"]["reachable"]:
            try:
                stop_model(spec, force=True)
                stopped.append(spec.name)
            except RuntimeError as e:
                click.echo(f"warning: could not stop {spec.name}: {e}", err=True)
    if stopped:
        click.echo("stopped: " + ", ".join(stopped))
    else:
        click.echo("stopped.")


@cli.command()
def status() -> None:
    """Show the state of the Toki Recur stack (all registered models)."""
    from rich.console import Console

    from ..ctl import load_registry, model_status, read_active
    from ..tui import art

    specs = load_registry()
    active = read_active()

    def state(data: dict) -> tuple[str, str]:
        if data["reachable"]:
            return "[green]running[/green]", f"pid={data['pid'] or '?'}"
        pid = data.get("pid")
        if pid:
            return "[red]dead (stale pidfile)[/red]", f"pid={pid}"
        return "[dim]stopped[/dim]", "no pidfile"

    con = Console()
    con.print(art.WORDMARK, style=f"bold {art.YELLOW}", highlight=False)
    con.print(art.TAGLINE, style="dim", highlight=False)
    con.print("")
    con.print("")
    con.print("Toki Recur stack:")
    up = 0
    for spec in specs:
        s = model_status(spec)
        lstate, lpid = state(s["llama"])
        dstate, dpid = state(s["daemon"])
        if s["llama"]["reachable"] or s["daemon"]["reachable"]:
            up += 1
        marker = "  (active)" if spec.name == active else ""
        con.print(
            f"  {spec.name:<22} llama :{spec.llama_port:5d} {lstate}  "
            f"daemon :{spec.daemon_port:5d} {dstate}{marker}",
        )
    if up:
        con.print(f"{up} model stack(s) running; `toki model use <name>` to switch, "
                  "`toki model stop <name>` to stop.")
    else:
        con.print("nothing running. `toki up` to start, `toki model use <name>` for one.")


@cli.command()
@click.pass_context
def interactive(ctx: click.Context) -> None:
    """Start the interactive shell (the default when no subcommand is given)."""
    from .. import repl
    raise SystemExit(repl.run())


# ── Multi-model plug & play ──────────────────────────────────────────────


@cli.group()
def model() -> None:
    """Manage the plug-and-play model stacks (9b / 35b / laguna).

    Each model gets its own llama-server + toki daemon on dedicated
    ports; `toki model use <name>` starts it, stops competing big-model
    stacks, and points opencode, openclaw and hermes at it.
    """


@model.command()
def list() -> None:
    """List the models in the registry."""
    from ..ctl import load_registry
    for spec in load_registry():
        flags = []
        if spec.always_on:
            flags.append("always on")
        if spec.boot:
            flags.append("boot")
        extra = f" ({', '.join(flags)})" if flags else ""
        click.echo(f"  {spec.name.ljust(8)} {spec.label}{extra}")
        click.echo(f"            llama :{spec.llama_port}  daemon :{spec.daemon_port}  "
                   f"ctx {spec.ctx_size}  kv {spec.kv}  -> {spec.gguf}")


@model.command()
def status() -> None:  # noqa: F811

    """Show per-model stack status."""
    from ..ctl import gpu_state, load_registry, model_status, read_active
    active = read_active()
    click.echo(f"active model: {active or 'none'}")
    for spec in load_registry():
        s = model_status(spec)
        llama = s["llama"]
        daemon = s["daemon"]
        ctx = f"slot_ctx={llama['slot_ctx']}" if llama["slot_ctx"] else ""
        llama_state = "up" if llama["reachable"] else "down"
        daemon_state = "up" if daemon["reachable"] else "down"
        mark = "*" if spec.name == active else " "
        gpu = gpu_state(spec)
        gpu_bits = []
        if gpu["backend"]:
            gpu_bits.append(gpu["backend"])
            if gpu["layers"]:
                gpu_bits.append(gpu["layers"] + " layers")
        elif llama_state == "up":
            gpu_bits.append("CPU")
        gpu_col = f"  gpu={'/'.join(gpu_bits)}" if gpu_bits else ""
        click.echo(
            f"  {mark} {spec.name.ljust(8)} llama={llama_state} ({ctx})  "
            f"daemon={daemon_state}  pids={llama['pid']}/{daemon['pid']}"
            f"{gpu_col}"
        )


@model.command()
@click.argument("name")
@click.option(
    "--keep-others", is_flag=True,
    help="Do not stop the other big-model stacks (uses more RAM).",
)
@click.option(
    "--wait", is_flag=True,
    help="Stay attached (for systemd units). The stack is torn down "
         "when this process is stopped.",
)
@click.option(
    "--sync-only", is_flag=True,
    help="Only rewrite the opencode/openclaw/hermes/codex configs; do "
         "not start or stop any stack (installer-friendly heal).",
)
@click.option(
    "--mode", type=click.Choice(["raw", "managed"]), default=None,
    help="Inference path for all four clients: managed (through the "
         "toki daemon — context truth, KV recycling) or raw (direct "
         "llama-server). Persisted; default is managed.",
)
def use(name: str, keep_others: bool, wait: bool, sync_only: bool,
        mode: str | None) -> None:
    """Start the model stack and select it in opencode/openclaw/hermes."""
    from ..ctl import _effective_ctx, model_status, sync_cli_configs, use_model, write_client_mode
    if mode:
        for tool in ("opencode", "openclaw", "hermes", "codex"):
            write_client_mode(tool, mode)
    try:
        spec = use_model(name, stop_others=not keep_others, sync_only=sync_only)
    except (KeyError, FileNotFoundError, RuntimeError) as e:
        click.echo(f"error: {e}", err=True)
        raise SystemExit(1) from None
    paths = sync_cli_configs(spec)
    effective = _effective_ctx(spec)
    if not sync_only:
        try:
            live_slot = model_status(spec)["llama"].get("slot_ctx")
        except Exception:
            live_slot = None
        if live_slot and live_slot != effective:
            click.echo(
                f"note: llama is already serving {live_slot} tokens and "
                f"will keep doing so until restarted — run `toki model "
                f"stop {name}` then `toki model use {name}` to apply "
                f"{effective}"
            )
    if effective < spec.ctx_size:
        click.echo(
            f"note: context clamped to {effective} tokens (fits this "
            f"machine's RAM; configured {spec.ctx_size})"
        )
    if effective < 24000:
        click.echo(
            "note: effective context is under 24k — on a small-RAM box "
            "consider `toki model use laguna` (or 0.8b) instead of a "
            "9B-class model at full window"
        )
    if effective < 64000:
        click.echo(
            "warning: hermes refuses models under 64k context. This model "
            f"advertises {effective} — either raise its ctx_size in the "
            "launcher settings (0.8b's true window is 262144) or pick "
            "another model for hermes sessions"
        )
    if sync_only:
        click.echo(f"model '{name}' configs synced (stack not started)")
    else:
        click.echo(f"model '{name}' is up: llama :{spec.llama_port}  daemon :{spec.daemon_port}")
    for tool, path in paths.items():
        click.echo(f"  {tool:9s} -> {path}")
    if mode:
        click.echo(f"mode: {mode} for all clients (codex stays on the daemon)")
    click.echo("switched. per-run override: opencode --model, openclaw agent --model, "
               "hermes -m --provider")
    if wait:
        # systemd KillMode=control-group reaps the spawned llama-server /
        # daemon when this process exits, so stay alive until stopped.
        #
        # Also tell the kernel to auto-reap any child that dies while we
        # sleep — without this, a llama-server crash (OOM, model file
        # gone, etc.) leaves a defunct (Z) child attached to us, and
        # systemd keeps showing the unit as "active" with zombie
        # grandchildren attached. SIG_IGN on SIGCHLD is the POSIX way
        # to opt into the kernel's auto-reap.
        import signal
        import time
        # SIGCHLD does not exist on Windows; the auto-reap trick is a
        # POSIX-only nicety, skip it elsewhere.
        if hasattr(signal, "SIGCHLD"):
            signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        while True:
            time.sleep(3600)


@model.command("mode")
@click.argument("client", type=click.Choice(["opencode", "openclaw", "hermes", "codex"]))
@click.argument("mode", type=click.Choice(["raw", "managed"]))
def model_mode(client: str, mode: str) -> None:
    """Set one client's inference path.

    managed = through the toki daemon (context truth + KV recycling;
    required by codex, which speaks the Responses API the daemon shims).
    raw = straight to llama-server (no daemon in the path).
    """
    from ..ctl import write_client_mode
    write_client_mode(client, mode)
    click.echo(f"client '{client}' -> {mode} (applied on next sync/stack start)")


@model.command()
@click.argument("name")
@click.option("--force", "-f", is_flag=True,
              help="Stop the stack even if it was not started by `toki model`.")
def stop(name: str, force: bool) -> None:
    """Stop a model's llama-server + daemon."""
    from ..ctl import get_model, stop_model
    try:
        stop_model(get_model(name), force=force)
    except (KeyError, RuntimeError) as e:
        click.echo(f"error: {e}", err=True)
        raise SystemExit(1) from None
    click.echo(f"model '{name}' stopped.")


@model.command()
@click.argument("name")
@click.option("--force", "-f", is_flag=True,
              help="Allow removing the active model and stop unmanaged stacks.")
@click.option("--purge", is_flag=True,
              help="Also delete the model weights (~/models/<name>/) and runtime "
                   "state (slot-save, llama.log, daemon.log, pid files). "
                   "Re-adding the model after this requires re-downloading.")
def remove(name: str, force: bool, purge: bool) -> None:
    """Drop a model from the registry.

    Stops the stack first so the launcher and opencode / openclaw / hermes
    never point at a model that is no longer registered. By default the
    model weights and runtime state are kept on disk so you can re-add the
    same name with `toki model pull` without re-downloading; pass --purge
    to free the disk. Refuses to remove the active model without --force.
    """
    from ..ctl import remove_model
    try:
        summary = remove_model(name, force=force, purge=purge)
    except (KeyError, RuntimeError) as e:
        click.echo(f"error: {e}", err=True)
        raise SystemExit(1) from None
    extras = []
    if summary["was_active"]:
        extras.append("was the active model — cleared the active marker")
    if summary["purged_weights"]:
        extras.append("weights deleted")
    elif purge:
        extras.append("weights kept (path lives outside ~/models)")
    if summary["purged_state"]:
        extras.append("state deleted")
    tail = f"  ({'; '.join(extras)})" if extras else ""
    click.echo(f"removed '{summary['removed']}' from the registry.{tail}")


@model.command()
def repair() -> None:
    """Backfill binary/env on legacy pulled entries.

    Entries written before `pull_defaults` existed are missing the
    machine's tuned llama.cpp path and ROCm env vars, which makes
    pulled stacks fall back to the old Debian build. This rewrites
    those entries in place to inherit from the first reference spec
    that already has them. Idempotent: no-op if nothing needs repair.

    This also runs automatically on every `toki model use`.
    """
    from ..ctl import backfill_binary_env

    healed = backfill_binary_env()
    if not healed:
        click.echo("nothing to repair.")
        return
    click.echo(f"repaired: {', '.join(healed)}")
    click.echo("re-run `toki model use <name>` (already restarted stacks will pick up the new binary on their next launch).")


@model.command()
@click.argument("repo")
@click.option("--name", default=None,
              help="Registry name (defaults to a lowercase slug of REPO).")
@click.option("--file", "gguf_file", default=None,
              help="Exact GGUF filename in the repo (needed when the repo "
                   "contains several; otherwise you are prompted).")
@click.option("--ctx", "ctx_size", default=None, type=int,
              help="Context size for the registry entry (default: "
                   "auto-sized to fit this machine and the model).")
@click.option("--kv", default="f16", help="KV cache type (f16, q8_0, ...).")
@click.option("--endpoint", default=None,
              help="HF mirror base URL, e.g. http://localhost:8090 "
                   "(overrides HF_ENDPOINT).")
@click.option("--use", "use_it", is_flag=True,
              help="Start the model stack and select it everywhere.")
@click.option("--reasoning", "reasoning", default="auto",
              type=click.Choice(["auto", "on", "off"]),
              help="How to handle thinking-mode models: 'auto' streams "
                   "reasoning with headroom (default); 'on' forces "
                   "thinking on; 'off' passes --reasoning off to "
                   "llama-server at launch.")
def pull(repo: str, name: str | None, gguf_file: str | None, ctx_size: int,
         kv: str, endpoint: str | None, use_it: bool,
         reasoning: str) -> None:
    """Download a GGUF from Hugging Face and register it.

    Uses huggingface_hub, so `HF_ENDPOINT` (or --endpoint) transparently
    routes downloads through a mirror such as DingoSpeed. The registry
    name defaults to a lowercase slug of the repo. Ports are auto-assigned
    after the existing models. When the repo has several GGUF files you
    get a numbered chooser; with no interactive input the recommended
    quant is used.
    """
    from ..ctl import pull_model, use_model
    try:
        spec = pull_model(repo, name, file=gguf_file, ctx_size=ctx_size,
                          kv=kv, endpoint=endpoint, reasoning=reasoning,
                          choose_file=_pick_gguf_interactive)
    except (ValueError, RuntimeError, KeyError) as e:
        click.echo(f"error: {e}", err=True)
        raise SystemExit(1) from None
    click.echo(f"pulled '{spec.name}' -> {spec.gguf}")
    click.echo(f"  llama :{spec.llama_port}  daemon :{spec.daemon_port}  "
               f"ctx {spec.ctx_size}  kv {spec.kv}")
    if use_it:
        use_model(spec.name)
        click.echo(f"model '{spec.name}' is up and active everywhere.")
    else:
        click.echo(
            f"run `toki model use {spec.name}` to start it (or re-run with --use)."
        )


@model.command("fix-model-ids")
def fix_model_ids() -> None:
    """Rewrite repo-path model_ids (e.g. ``unsloth/Qwen3.5-0.8B-GGUF``)
    to short, stable identifiers derived from the GGUF filename.

    Pre-0.2.15 `toki model pull` set ``model_id`` to the full Hugging
    Face repo id. hermes and other agents compare the request's
    ``model`` field against this string and 404 when it doesn't match
    the loaded alias. Running this once normalises every spec and
    re-writes the opencode/openclaw/hermes/codex configs.
    """
    from ..ctl import normalize_model_ids
    changes = normalize_model_ids()
    if not changes:
        click.echo("no model_ids needed fixing.")
        return
    for name, old, new in changes:
        click.echo(f"  {name}: {old!r} -> {new!r}")
    # Re-sync the agent configs so the new short names land in
    # hermes/openclaw/opencode/codex immediately.
    from ..ctl import get_model, read_active, sync_cli_configs
    active = read_active()
    if active:
        try:
            spec = get_model(active)
            sync_cli_configs(spec)
            click.echo(f"  re-synced agent configs for '{active}'.")
        except Exception as e:
            click.echo(f"  warning: could not re-sync agent configs: {e}",
                      err=True)


def _detect_gpu_backend(os_name: str) -> str:
    """Best-guess GPU backend for this machine, without admin access."""
    import shutil as _shutil

    if os_name == "macos":
        # llama.cpp macOS assets are always the Metal build (which also
        # runs on CPU-only Macs).
        return "metal"
    if os_name == "windows":
        if _shutil.which("nvidia-smi") or os.environ.get("CUDA_PATH"):
            return "cuda"
        if _shutil.which("hipconfig") or Path(
            "C:/Program Files/AMD/ROCm"
        ).exists():
            return "rocm"
        if _shutil.which("vulkaninfo"):
            # An Intel HD/UHD/Iris iGPU-only box must not get the Vulkan
            # build: llama.cpp's Vulkan backend still probes/initializes
            # the device at startup on those old drivers, which pegs the
            # CPU without ever producing tokens (the same class of bug
            # as `_default_ngl` refusing to offload to them). The plain
            # CPU build is what ollama uses there — and works.
            from ..ctl import _intel_igpu_only
            if not _intel_igpu_only():
                return "vulkan"
            return "cpu"
        return "cpu"
    # Linux
    if _shutil.which("hipconfig") or Path("/opt/rocm").exists():
        return "rocm"
    if _shutil.which("vulkaninfo") or Path(
        "/usr/lib/x86_64-linux-gnu/libvulkan.so.1"
    ).exists():
        return "vulkan"
    return "cpu"


def _asset_patterns(os_name: str, arch: str) -> list[tuple[str, str]]:
    """Ordered (backend, asset-substring) pairs for this OS/arch.

    Earlier entries win when several assets match. Generic "ubuntu-x64"
    is deliberately last so GPU builds are preferred when both exist.
    """
    if os_name == "macos":
        return [("metal", f"macos-{arch}"), ("metal", "macos-universal")]
    if os_name == "windows":
        if arch == "x64":
            return [
                ("cuda", "win-cuda-13.3-x64"),
                ("cuda", "win-cuda-12.4-x64"),
                ("rocm", "win-hip-radeon-x64"),
                ("vulkan", "win-vulkan-x64"),
                ("cpu", "win-cpu-x64"),
            ]
        # arm64 Windows: only CPU (and Adreno OpenCL) builds exist.
        return [
            ("vulkan", "win-opencl-adreno-arm64"),
            ("cpu", "win-cpu-arm64"),
        ]
    if arch == "x64":
        return [
            ("rocm", "ubuntu-rocm-"),
            ("vulkan", "ubuntu-vulkan-x64"),
            ("cpu", "ubuntu-x64"),
        ]
    return [
        ("vulkan", "ubuntu-vulkan-arm64"),
        ("cpu", "ubuntu-arm64"),
    ]


# Number of times to retry a streaming download from the GitHub
# release-asset CDN before giving up. The body of a release asset is
# typically tens of MB; transient TCP resets / TLS RSTs / captive-portal
# interference are common enough that one retry is usually enough, but
# we budget three attempts with exponential backoff to ride out the
# rare multi-second outage without a wall of traceback.
_DOWNLOAD_ATTEMPTS = 3
# Retryable transport errors during a streaming GET. pygithub is built
# on `requests`+`urllib3` and surfaces these via httpx when the
# underlying response stream tears down mid-body.
_RETRYABLE_STREAM_ERRORS: tuple[type[BaseException], ...] = (
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.TimeoutException,
    ConnectionError,
    OSError,  # catches ECONNRESET / EPIPE / ETXTBSY-after-rename races
)


def _safe_copy_replace(src: Path, dst: Path) -> None:
    """Copy ``src`` over ``dst`` safely even when ``dst`` is the active
    executable of a running process.

    On POSIX, ``shutil.copy2`` (and ``os.replace`` / ``os.rename``) over
    an mmap'd executable returns ETXTBSY (errno 26 — "Text file busy").
    The fix is to rename the old file aside first; the unlinked inode
    stays alive for any process that has it open, while the directory
    entry now points at a fresh inode that accepts writes.

    On Windows the running executable is locked differently and rename
    may fail; ``shutil.copy2`` will then succeed if no process holds
    the destination open. The behaviour we expose is: POSIX uses
    rename-then-copy; Windows falls back to direct copy.

    Raises a clear ``click.ClickException`` on ETXTBSY after the rename
    path (should be unreachable on POSIX — kept as a safety net) and
    surfaces the original OSError otherwise so the caller can decide.
    """
    if dst.exists():
        backup = dst.with_name(dst.name + f".stale.{os.getpid()}")
        try:
            dst.rename(backup)
        except OSError as exc:
            # Windows: rename across an open executable can fail with
            # PermissionError. Fall back to a direct copy (which will
            # succeed if nothing has dst open, and fail with a clear
            # ETXTBSY if something does).
            if exc.errno not in (13, 22):  # EACCES / EINVAL on Windows
                raise
        else:
            # Leave the backup in place; the running daemon's mmap keeps
            # it alive. ``_sweep_stale_backups`` cleans it up once the
            # owning pid has exited.
            shutil.copy2(src, dst)
            return
    shutil.copy2(src, dst)


def _sweep_stale_backups(bin_dir: Path, exe_name: str) -> None:
    """Remove ``<exe_name>.stale.<pid>`` backups whose owning pid has
    exited. Live pid → leave alone (the running daemon still needs the
    inode); dead pid → unlink. Bound the cleanup to a single bin_dir so
    it's cheap and never touches anything outside the install dir.
    """
    pattern = f"{exe_name}.stale.*"
    for stale in bin_dir.glob(pattern):
        suffix = stale.name.rsplit(".", 1)[-1]
        try:
            pid = int(suffix)
        except ValueError:
            # Not a pid; remove it as an unowned orphan.
            try:
                stale.unlink()
            except OSError:
                pass
            continue
        try:
            os.kill(pid, 0)  # signal 0 → check alive, never delivers
        except (ProcessLookupError, OSError):
            # Dead (or permission denied — safe to ignore, leave it).
            try:
                stale.unlink()
            except OSError:
                pass


def _stream_download_with_retry(
    api, url: str, dest: Path, *, label: str = "download",
) -> None:
    """Stream ``url`` into ``dest`` with retry-on-disconnect. Replaces
    ``dest`` between attempts (truncate) so a half-written previous
    attempt doesn't poison the next one. Surfaces a clear
    ``click.ClickException`` after the retry budget is exhausted —
    never the raw 60-line traceback.
    """
    last_err: BaseException | None = None
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            # Truncate between attempts so we never append a partial
            # body to bytes from a previous interrupted response.
            with open(dest, "wb") as fh:
                with api.stream("GET", url) as resp:
                    resp.raise_for_status()
                    for chunk in resp.iter_bytes(1 << 20):
                        fh.write(chunk)
            return
        except _RETRYABLE_STREAM_ERRORS as exc:
            last_err = exc
            if attempt >= _DOWNLOAD_ATTEMPTS:
                break
            wait = 2 ** attempt  # 2s, 4s
            click.echo(
                f"{label}: stream interrupted ({type(exc).__name__}); "
                f"retrying in {wait}s ({attempt}/{_DOWNLOAD_ATTEMPTS - 1})…"
            )
            time.sleep(wait)
    # All attempts exhausted; surface a single actionable error.
    host = url.split("/releases/", 1)[0] + "/…"
    raise click.ClickException(
        f"{label}: failed to download after {_DOWNLOAD_ATTEMPTS} attempts "
        f"({type(last_err).__name__}: {last_err}).\n"
        f"  source: {host}\n"
        f"  target: {dest}\n"
        f"  likely causes:\n"
        f"    - transient network / DNS / captive-portal interruption\n"
        f"    - GitHub unauthenticated rate-limit on the release asset CDN\n"
        f"    - corporate proxy stripping chunked transfer-encoding\n"
        f"  fixes:\n"
        f"    - retry: re-run `toki model install-llama`\n"
        f"    - auth:  export TOKI_GITHUB_TOKEN=<PAT> then re-run\n"
        f"    - VPN:   disable / change VPN / proxy and re-run\n"
    )


def _pick_asset(
    assets: list[dict],
    patterns: list[tuple[str, str]],
) -> tuple[dict | None, str | None]:
    """Return (best asset, backend) or (None, None)."""
    best: tuple[tuple[int, int], dict, str] | None = None
    for a in assets:
        low = a.get("name", "").lower()
        for i, (backend, frag) in enumerate(patterns):
            if frag in low:
                # Prefer the archive that actually contains llama-server:
                # CUDA/ROCm releases also ship runtime archives named
                # cudart-llama-bin-* / cuda-cublas-llama-bin-* that match
                # the same fragment but hold no binaries.
                rank = 0 if low.startswith("llama-") else 1
                score = (rank, i)
                if best is None or score < best[0]:
                    best = (score, a, backend)
                break
    if best is None:
        return None, None
    _, asset, backend = best
    return asset, backend


def _cuda_runtime_archive_name(main_asset_name: str) -> str | None:
    """Name of the cudart runtime archive paired with a CUDA main asset.

    llama.cpp's CUDA 13.x Windows builds split cudart/cublas out of the
    main archive into a separate `cudart-llama-bin-win-cuda-13.3-x64.zip`
    that must be unpacked alongside ``llama-server.exe`` (the 12.4 build
    statically links cudart, so it doesn't need this).
    """
    import re
    m = re.search(r"win-cuda-(\d+\.\d+)-x64", main_asset_name)
    if not m:
        return None
    return f"cudart-llama-bin-win-cuda-{m.group(1)}-x64.zip"


def _download_and_unpack_zip(api, asset: dict, bin_dir: Path,
                              progress_prefix: str = "") -> None:
    """Download a llama.cpp release archive and flatten its files into bin_dir.

    The archives nest files under a single top-level folder; this helper
    strips that folder so ``llama-server.exe``, ``ggml-cuda.dll``,
    ``cudart64_13.dll`` etc. land directly in bin_dir.
    """
    import tempfile
    import zipfile
    with tempfile.TemporaryDirectory(prefix="toki-llama-") as tmp:
        tmp_dir = Path(tmp)
        zip_path = tmp_dir / asset["name"]
        _stream_download_with_retry(
            api, asset["browser_download_url"], zip_path,
            label=f"{progress_prefix}download {asset['name']}",
        )
        if zip_path.suffix == ".zip":
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_dir)
        else:
            import tarfile
            with tarfile.open(zip_path) as tf:
                try:
                    tf.extractall(tmp_dir, filter="data")
                except TypeError:
                    tf.extractall(tmp_dir)
        # Clear stale backups before writing the sidecar files; same
        # rationale as in install_llama — a previous interrupted install
        # may have left .stale.<pid> orphans to clean up.
        _sweep_stale_backups(bin_dir, Path(asset["name"]).stem)
        for src in tmp_dir.rglob("*"):
            if not src.is_file() or src == zip_path:
                continue
            rel = src.relative_to(tmp_dir)
            target_name = "/".join(rel.parts[1:]) if len(rel.parts) > 1 else rel.name
            target = bin_dir / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            _safe_copy_replace(src, target)


def _detect_gfx_version() -> str | None:
    """Parse `rocminfo` for the first gfxNNN target (e.g. gfx1151)."""
    import re
    import subprocess

    rocminfo = shutil.which("rocminfo")
    if not rocminfo:
        return None
    try:
        out = subprocess.run(
            [rocminfo], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"\bgfx(\d{4})\b", out)
    if not m:
        return None
    digits = m.group(1)
    # gfx1151 -> 11.5.1 (ROCm's HSA_OVERRIDE_GFX_VERSION spelling).
    return f"{digits[0:2]}.{digits[2]}.{digits[3]}"


def _rocm_env(gfx: str | None) -> dict[str, str]:
    env = {"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"}
    if gfx:
        env["HSA_OVERRIDE_GFX_VERSION"] = gfx
    return env


@model.command("install-llama")
@click.option("--tag", default=None,
              help="llama.cpp release tag to install, e.g. b6400 "
                   "(default: latest).")
@click.option("--to", "dest", default=None,
              help="Install directory (default: the toki state bin dir).")
@click.option("--name", default=None,
              help="Registry model to point at the installed binary, e.g. 35b.")
@click.option("--backend", default="auto",
              type=click.Choice(["auto", "rocm", "cuda", "vulkan",
                                 "metal", "mlx", "cpu"]),
              help="GPU backend to install (default: auto-detect).")
@click.option("--gfx", "gfx", default=None,
              help="HSA_OVERRIDE_GFX_VERSION for ROCm builds, e.g. 11.5.1 "
                   "(default: detected from rocminfo).")
def install_llama(tag: str | None, dest: str | None, name: str | None,
                  backend: str, gfx: str | None) -> None:
    """Download the llama-server binary for this OS/arch and GPU backend.

    Fetches the matching llama.cpp release asset (ROCm, CUDA, Vulkan,
    Metal or CPU; x86_64 and aarch64 both supported), extracts
    llama-server and its sidecar libraries into the toki state bin dir,
    self-checks the binary, and optionally rewrites a registry model's
    ``binary`` (plus the ROCm env block) to point at it.

    ``--backend`` defaults to auto-detect: Linux prefers ROCm when a
    HIP toolchain is present, then Vulkan, then CPU; Windows prefers
    CUDA, then ROCm/HIP, then Vulkan, then CPU; macOS always uses the
    Metal build. When the requested backend has no asset for this arch
    (e.g. ROCm on arm64 Linux), the next available backend is used and
    a note is printed.
    """
    import platform
    import subprocess
    import tempfile
    import zipfile

    from ..ctl import set_binary

    system = sys.platform
    if system.startswith("linux"):
        os_name = "ubuntu"
    elif system == "darwin":
        os_name = "macos"
    elif system.startswith("win"):
        os_name = "windows"
    else:
        click.echo(f"error: unsupported platform '{system}'", err=True)
        raise SystemExit(1) from None
    # llama.cpp release assets use x64/arm64 naming regardless of the
    # Python platform.machine() spelling.
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64"):
        arch = "x64"
    elif machine in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        click.echo(f"error: unsupported architecture '{machine}'", err=True)
        raise SystemExit(1) from None

    if backend == "auto":
        backend = _detect_gpu_backend(os_name)
        click.echo(f"backend: auto-detected {backend} ({os_name}/{arch})")
    elif backend == "metal" and os_name != "macos":
        click.echo("error: --backend metal is only available on macOS", err=True)
        raise SystemExit(1) from None
    elif backend == "rocm" and os_name == "macos":
        click.echo("error: --backend rocm is not available on macOS", err=True)
        raise SystemExit(1) from None
    elif backend == "cuda" and os_name != "windows":
        click.echo("error: --backend cuda is only available on Windows "
                   "(llama.cpp ships no Linux CUDA assets)", err=True)
        raise SystemExit(1) from None
    elif backend == "mlx" and os_name != "macos":
        click.echo("error: --backend mlx is only available on macOS "
                   "(Apple's MLX framework)", err=True)
        raise SystemExit(1) from None
    else:
        click.echo(f"backend: {backend} ({os_name}/{arch})")

    patterns = _asset_patterns(os_name, arch)
    requested_patterns = [p for p in patterns if p[0] == backend]
    # When the requested backend has no asset for this arch, fall back
    # through the priority list so installs never dead-end on a missing
    # build (e.g. ROCm on arm64).
    patterns = requested_patterns or patterns

    if backend == "cuda" and os_name == "windows":
        from ..diagnose import (
            CUDA_12_WINDOWS_DRIVER,
            cuda_driver_verdict,
            driver_version,
        )

        too_old, prefer_cuda12 = cuda_driver_verdict(driver_version())
        if too_old:
            click.echo(
                "warning: NVIDIA driver is older than the "
                f"{CUDA_12_WINDOWS_DRIVER[0]}.{CUDA_12_WINDOWS_DRIVER[1]} "
                "floor for CUDA 12.4+ llama.cpp builds — this binary will "
                "fail to load with 'CUDA driver version is insufficient'. "
                "Update the NVIDIA driver first, then re-run this install.",
                err=True,
            )
        if prefer_cuda12:
            # Drivers below the 12.8/13.x floor run the CUDA 12.4 build
            # fine; prefer that asset so installs don't fail at load time.
            cuda12 = [p for p in patterns if "12.4" in p[1]]
            if cuda12:
                patterns = cuda12 + [p for p in patterns if p not in cuda12]

    state_dir = Path(os.environ.get(
        "TOKI_STATE_DIR", str(Path.home() / ".local" / "state" / "tokirecur"),
    ))
    bin_dir = Path(dest) if dest else state_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    exe_name = "llama-server.exe" if os_name == "windows" else "llama-server"

    # MLX has no release binary — it's a pip package. Install it, write
    # a shim into the bin dir, and point the registry at mlx_lm.server.
    if backend == "mlx":
        click.echo("installing mlx-lm (Apple MLX inference server)...")
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "mlx-lm"],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            click.echo(
                f"mlx-lm install failed: {proc.stderr.strip() or proc.stdout.strip()}",
                err=True,
            )
            raise SystemExit(1) from None
        click.echo("mlx-lm: ok")
        # Write a small launcher shim so binary_backend's path probe
        # finds something on disk; mlx_lm.server is also resolved via
        # `shutil.which` for the actual spawn.
        shim = bin_dir / "mlx_lm.server"
        shim.write_text(
            "#!/usr/bin/env sh\n"
            f"exec {sys.executable} -m mlx_lm.server \"$@\"\n"
        )
        if os.name != "nt":
            shim.chmod(0o755)
        click.echo(f"shim: {shim}")
        if name:
            try:
                spec = set_binary(name, "mlx_lm.server", env=None)
                click.echo(f"registry: '{spec.name}' now uses mlx_lm.server")
            except KeyError as e:
                click.echo(f"error: {e}", err=True)
                raise SystemExit(1) from None
        else:
            click.echo(
                f"registry: pass --name <model> to point a model at it "
                f"(or edit `binary:` in "
                f"{os.environ.get('TOKI_MODEL_REGISTRY', str(Path.home() / '.config/tokirecur/models.yaml'))})"
            )
        return

    api = httpx.Client(timeout=30.0, follow_redirects=True)
    # Authenticate when a token is available (CI provides GITHUB_TOKEN;
    # local users can set TOKI_GITHUB_TOKEN) to avoid the unauthenticated
    # API rate limit when several machines fetch concurrently.
    github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("TOKI_GITHUB_TOKEN")
    headers = {"Authorization": f"Bearer {github_token}"} if github_token else {}
    try:
        if tag:
            release = api.get(
                "https://api.github.com/repos/ggml-org/llama.cpp"
                f"/releases/tags/{tag}",
                headers=headers,
            ).json()
            if "assets" not in release:
                raise RuntimeError(f"release lookup failed: {release.get('message')}")
        else:
            # llama.cpp auto-publishes releases continuously and
            # occasionally ships a tag with zero assets for a few
            # minutes; walk newest -> oldest and use the first release
            # that carries an asset for our OS/arch/backend.
            releases = api.get(
                "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10",
                headers=headers,
            ).json()
            import builtins

            # builtins.list: this module defines a click command named
            # `list` that shadows the builtin inside this function.
            if not isinstance(releases, builtins.list) or not releases:
                raise RuntimeError("no releases found")
            release = next(
                (r for r in releases
                 if _pick_asset(r.get("assets") or [], patterns)[0]),
                None,
            )
            if release is None:
                raise RuntimeError(
                    "no recent llama.cpp release carries an asset for "
                    f"{os_name}/{arch} backend '{backend}'"
                )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"could not reach GitHub: {e}") from e

    tag_name = release["tag_name"]
    assets = release["assets"]
    asset, matched_backend = _pick_asset(assets, patterns)
    if asset is None:
        names = ", ".join(a.get("name", "") for a in assets[:12])
        raise RuntimeError(
            f"no asset matching {backend} for {os_name}/{arch} in {tag_name}; "
            f"available (first 12): {names}"
        )
    if matched_backend != backend:
        click.echo(
            f"note: no {backend} build for {os_name}/{arch}; "
            f"installing {matched_backend} instead"
        )
        backend = matched_backend
    click.echo(f"downloading {asset['name']} ({asset['size'] / 1048576:.0f} MB)...")

    with tempfile.TemporaryDirectory(prefix="toki-llama-") as tmp:
        zip_path = Path(tmp) / asset["name"]
        _stream_download_with_retry(
            api, asset["browser_download_url"], zip_path,
            label=f"download {asset['name']}",
        )
        # Windows assets are .zip; Linux/macOS are .tar.gz (and older
        # releases used .zip everywhere, so accept both).
        if zip_path.suffix == ".zip":
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
        else:
            import tarfile

            with tarfile.open(zip_path) as tf:
                try:
                    tf.extractall(tmp, filter="data")
                except TypeError:  # Python < 3.12
                    tf.extractall(tmp)

        matches = sorted(Path(tmp).rglob(exe_name))
        if not matches:
            raise RuntimeError(f"no {exe_name} found in the release archive")
        src = matches[0]
        dst = bin_dir / exe_name
        # Clear any stale backups from previous interrupted installs
        # whose owning pid has since exited; leaves live-pid orphans
        # alone (the running daemon's mmap still depends on them).
        _sweep_stale_backups(bin_dir, exe_name)
        _safe_copy_replace(src, dst)
        for side in src.parent.iterdir():
            if side.is_file() and side != src:
                _safe_copy_replace(side, bin_dir / side.name)

        # CUDA 13.x ships cudart/cublas as a separate runtime archive
        # (the main archive has no cudart64_*.dll inside ggml-cuda.dll);
        # without it llama-server can't load CUDA and falls back to CPU.
        if matched_backend == "cuda":
            cudart_name = _cuda_runtime_archive_name(asset["name"])
            if cudart_name:
                cudart = next(
                    (a for a in assets if a.get("name") == cudart_name),
                    None,
                )
                if cudart:
                    click.echo(
                        f"downloading {cudart['name']} "
                        f"({cudart['size'] / 1048576:.0f} MB)..."
                    )
                    _download_and_unpack_zip(
                        api, cudart, bin_dir,
                        progress_prefix="cudart: ",
                    )
    if os_name != "windows":
        dst.chmod(dst.stat().st_mode | 0o111)
    click.echo(f"installed: {dst}")

    # Self-check: the binary must actually execute before we call it good.
    try:
        probe = subprocess.run(
            [str(dst), "--version"],
            capture_output=True, text=True, timeout=30,
        )
        if probe.returncode == 0:
            click.echo(f"self-check: {probe.stdout.strip().splitlines()[0] if probe.stdout.strip() else 'ok'}")
        else:
            click.echo(
                f"warning: {dst} --version failed (rc={probe.returncode}): "
                f"{probe.stderr.strip() or probe.stdout.strip()}",
                err=True,
            )
    except (OSError, subprocess.SubprocessError) as e:
        click.echo(f"warning: could not self-check {dst}: {e}", err=True)

    if name:
        env = None
        if backend in ("rocm",):
            gfx_value = gfx or _detect_gfx_version()
            env = _rocm_env(gfx_value)
            if gfx_value:
                click.echo(f"rocm: HSA_OVERRIDE_GFX_VERSION={gfx_value}")
            else:
                click.echo("rocm: no gfx target detected; set --gfx <ver> for "
                           "HSA_OVERRIDE_GFX_VERSION")
        try:
            spec = set_binary(name, str(dst), env=env)
        except KeyError as e:
            click.echo(f"error: {e}", err=True)
            raise SystemExit(1) from None
        click.echo(f"registry: '{spec.name}' now uses {dst}")
        if env:
            click.echo(f"registry: '{spec.name}' env updated: {env}")
    else:
        click.echo(f"registry: pass --name <model> to point a model at it "
                   f"(or edit `binary:` in "
                   f"{os.environ.get('TOKI_MODEL_REGISTRY', str(Path.home() / '.config/tokirecur/models.yaml'))})")


def _pick_gguf_interactive(files: list[tuple[str, int | None]]) -> str:
    """Numbered chooser; falls back to the recommended quant on EOF."""
    from ..ctl import recommend_quant

    names = [name for name, _ in files]
    recommended = recommend_quant(names)
    click.echo("Several GGUF files in the repo:")
    for i, (name, size) in enumerate(files, 1):
        mark = "  (recommended)" if name == recommended else ""
        sz = f"  ({size / 1048576:.1f} MB)" if size else ""
        click.echo(f"  {i:>2}. {name}{sz}{mark}")
    try:
        idx = click.prompt(
            "Pick a file",
            type=click.IntRange(1, len(files)),
            default=names.index(recommended) + 1,
            show_default=False,
        )
    except click.exceptions.Abort:
        idx = names.index(recommended) + 1
        click.echo(f"Using recommended: {recommended}")
    return names[idx - 1]


@cli.group()
def service() -> None:
    """Manage systemd plug-and-play boot units for the model stacks.

    Installs `toki-model@.service` template units for every model marked
    always_on in the registry, chains them in registry order (so the
    last one, by default laguna, is the active model the clients point
    at), and enables user linger so the stacks come up at boot.
    """


@service.command()
def install() -> None:
    """Write units, enable linger, and start always-on model stacks."""
    from ..service import install as svc_install
    raise SystemExit(svc_install())


@service.command()
def uninstall() -> None:
    """Disable and remove the toki-model boot units."""
    from ..service import uninstall as svc_uninstall
    raise SystemExit(svc_uninstall())


@service.command()
def status() -> None:  # noqa: F811

    """Show linger state and per-model boot unit status."""
    from ..service import status as svc_status
    raise SystemExit(svc_status())


@service.command("logs")
@click.argument("name")
def service_logs(name: str) -> None:
    """Tail journal logs for a model's boot unit."""
    from ..service import logs as svc_logs
    raise SystemExit(svc_logs(name))


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("tokirecur")
    except Exception:
        return "0.1.0"  # running from source without an install


def main() -> int:
    # `--version` must be handled here: the dispatch below routes unknown
    # flags into the argparse-based benchmark runner.
    if len(sys.argv) > 1 and sys.argv[1] in ("--version", "-V"):
        click.echo(f"toki, version {_version()}")
        return 0
    # `--help` must show the full command reference, not the legacy
    # benchmark parser's help.
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        click.echo(cli.get_help(click.Context(cli)))
        return 0
    # Known subcommands. Anything else (no args, or a flag) prints help
    # and shows stack status instead of running a multi-minute benchmark.
    if len(sys.argv) == 1:
        # No args — open the interactive launcher selector (TTY) or show
        # help + status (scripts).
        if sys.stdin.isatty() and sys.stdout.isatty():
            from ..selector import run as selector_run
            return selector_run()
        click.echo(cli.get_help(click.Context(cli)))
        click.echo("")
        from rich.console import Console

        from ..tui import art
        con = Console()
        con.print(art.WORDMARK, style=f"bold {art.YELLOW}", highlight=False)
        con.print("")
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
    "bench", "doctor", "down", "evict", "interactive",
    "serve", "stats", "status", "up", "shell",
    "diff", "web", "dashboard", "tui", "model", "service",
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
        click.echo("[3/3] stopping llama-server...")
        lifecycle.stop_pid(lifecycle.LLAMA_PID, "llama-server")
    else:
        click.echo("[3/3] leaving llama-server running (use `toki down` to stop)")

    return rc


if __name__ == "__main__":
    sys.exit(main())
