"""Top-level CLI entry point for the tokirecur daemon and admin commands."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
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
        click.echo("model-capable CLI configs synced for the active model")

    # Fail loudly-but-kindly: a daemon with no llama-server behind it
    # looks exactly like "the server doesn't start". Ping the backend
    # before binding so the user gets a hint instead of silence. If the
    # configured backend is dead but the active model has a live llama,
    # point at it instead — a daemon that proxies a corpse helps nobody.
    from .. import ctl
    live = None
    try:
        active = ctl.read_active()
        if active:
            spec = ctl.get_model(active)
            if ctl.llama_reachable(spec):
                live = spec.llama_url
    except Exception:
        pass
    try:
        r = httpx.get(f"{backend_url}/health", timeout=2.0)
        if r.status_code >= 500:
            raise httpx.HTTPStatusError(
                f"backend unhealthy: {r.status_code}", request=r.request, response=r
            )
        click.echo(f"backend: {backend_url} reachable ({r.status_code})")
    except Exception as e:
        if live and live != backend_url:
            click.echo(
                f"WARNING: configured backend {backend_url} unreachable "
                f"({type(e).__name__}: {e})",
                err=True,
            )
            click.echo(
                f"  falling back to the active model's live llama: {live}",
                err=True,
            )
            backend_url = live
        else:
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
@click.option("--json", "as_json", is_flag=True,
              help="Machine output for the desktop selection panel: RAM, GPU, and per-model fit.")
def hardware(as_json: bool) -> None:
    """Show what this box can hold: RAM, GPU, and every registered model's footprint."""
    import json
    import socket
    from ..ctl import _available_gb, _estimate_need, _spec_is_image, load_registry
    from ..hardware import profile_box
    try:
        prof = profile_box(force=False)
    except Exception:
        prof = None
    total_gb, avail_gb = 0.0, _available_gb()
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb, avail_gb = mem.total / 1e9, mem.available / 1e9
    except Exception:
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        total_gb = int(line.split()[1]) / 1048576.0
                        break
        except OSError:
            pass
    gpus = (prof.gpus if prof else []) or []
    gpu = gpus[0] if gpus else None
    def _up(port: int) -> bool:
        if not port:
            return False
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=0.2)
            s.close()
            return True
        except OSError:
            return False
    try:
        specs = load_registry()
    except Exception:
        specs = []
    rows = []
    for spec in specs:
        need = round(_estimate_need(spec), 1)
        up = _up(spec.llama_port)
        rows.append({
            "name": spec.name,
            "kind": "image" if _spec_is_image(spec) else ("decider" if any(
                k in f"{spec.name} {spec.model_id}".lower()
                for k in ("clef", "decision", "decider", "laya")) else (
                "vision" if spec.mmproj else "chat")),
            "need_gb": need,
            "up": up,
            # Running stacks already hold their memory: they always "fit".
            "fits": True if up else (need <= avail_gb if avail_gb != float("inf") else True),
        })
    doc = {
        "ram_total_gb": round(total_gb, 1),
        "ram_avail_gb": round(avail_gb, 1) if avail_gb != float("inf") else None,
        "gpu": None if gpu is None else {
            "name": gpu.name,
            "vendor": gpu.vendor,
            "vram_gb": gpu.vram_gb,
            "shared_ram": bool(gpu.shared_ram),
            "note": gpu.reason,
        },
        "models": rows,
    }
    if as_json:
        click.echo(json.dumps(doc))
        return
    avail = f"{doc['ram_avail_gb']:.0f}" if doc['ram_avail_gb'] is not None else "?"
    click.echo(f"RAM {doc['ram_total_gb']:.0f} GB total, {avail} GB available")
    if doc["gpu"]:
        g = doc["gpu"]
        click.echo(f"GPU {g['name']} — {g['note']}")
        if g["shared_ram"]:
            click.echo("  shared memory: fit is judged on system RAM, not the VRAM carve")
    for m in rows:
        state = "up" if m["up"] else ("fits" if m["fits"] else "won't fit")
        click.echo(f"  {m['name']:<38} ~{m['need_gb']:>5} GB  {state}")


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

    rows = []
    # The ACTIVE model's ports, not the install defaults: every model in
    # the registry gets its own pair, and probing 8080/8787 reported a
    # stack that was up as down.
    llama_port, daemon_port = diagnose.active_stack_ports()
    try:
        r = httpx.get(f"http://127.0.0.1:{llama_port}/health", timeout=2.0)
        rows.append((f"llama-server :{llama_port}", "ok" if r.status_code == 200 else "down"))
    except Exception:
        rows.append((f"llama-server :{llama_port}", "not reachable"))
    try:
        r = httpx.get(f"http://127.0.0.1:{daemon_port}/healthz", timeout=2.0)
        rows.append((f"toki daemon :{daemon_port}", "ok" if r.status_code == 200 else "down"))
    except Exception:
        rows.append((f"toki daemon :{daemon_port}", "not running"))
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

    # 3) Heal model-capable CLI configs for the active model:
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

        git fetch origin && git reset --hard origin/main
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
    rc, out = _run(["git", "reset", "--hard", "origin/main"])
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
    default=None,
    type=click.Path(path_type=Path),
    help="Path to GGUF model (default: the active model's in the registry)",
)
@click.option("--llama-port", default=None, type=int)
@click.option("--daemon-port", default=None, type=int)
@click.option("--ctx-size", default=None, type=int)
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
    model: Path | None,
    llama_port: int | None,
    daemon_port: int | None,
    ctx_size: int | None,
    n_gpu_layers: int,
    detach: bool,
    skip_llama: bool,
    extra_llama_args: str,
) -> None:
    """Start the full Toki Recur stack: llama-server + daemon.

    With no arguments it starts the active model from the registry, on the
    ports the registry gave it, with the names the clients ask for. The old
    default was a hard-coded path to a starter download that most machines
    no longer have, so the command failed for everyone who had not kept it.
    """
    from .. import lifecycle

    # No --model, no --ports: take the active model from the registry, the
    # way the launcher and `toki model use` do. The old default was a
    # hard-coded path to a starter download that most machines no longer
    # have, so the command failed for everyone who had not kept it.
    from .. import ctl as ctl_mod

    spec = None
    if model is None:
        active = ctl_mod.read_active()
        if active:
            try:
                spec = ctl_mod.get_model(active)
            except Exception:
                spec = None
        if spec is None:
            click.echo(
                "error: no active model — run `toki model use <name>` first, "
                "or pass --model",
                err=True,
            )
            raise SystemExit(2)
        model = Path(ctl_mod.resolve_gguf_path(spec))
        if llama_port is None:
            llama_port = spec.llama_port
        if daemon_port is None:
            daemon_port = spec.daemon_port
        if ctx_size is None:
            ctx_size = spec.ctx_size
        click.echo(f"active model: {spec.name}")

    if model is None or not model.exists():
        if not skip_llama:
            click.echo(f"error: model not found: {model}", err=True)
            click.echo("hint: pass --model, or set TOKI_MODEL", err=True)
            raise SystemExit(2)
        model = Path(model) if model else Path("")

    if llama_port is None:
        llama_port = 8088
    if daemon_port is None:
        daemon_port = 8787
    if ctx_size is None:
        ctx_size = 8192

    if spec is None and model is not None:
        # The model was named by path. If it is in the registry, use that
        # entry's names, so the stack answers to the same identifiers the
        # clients are configured with — a stack that answers only to a file
        # path is a stack the clients cannot see.
        try:
            for candidate in ctl_mod.load_registry():
                try:
                    same = ctl_mod.resolve_gguf_path(candidate) == model
                except FileNotFoundError:
                    same = False
                if same:
                    spec = candidate
                    break
        except Exception:
            spec = None

    # `toki up` is a repair point — heal model-capable CLI configs
    # for the active registry model before the stack comes up.
    from ..ctl import heal_cli_configs
    healed = heal_cli_configs()
    if healed:
        click.echo("model-capable CLI configs synced for the active model")

    if not skip_llama:
        try:
            pid = lifecycle.start_llama_server(
                model=str(model),
                port=llama_port,
                ctx_size=ctx_size,
                n_gpu_layers=n_gpu_layers,
                extra_args=extra_llama_args.split() if extra_llama_args else None,
                # The names a client asks for. Without them the stack
                # answers to the file's path, and a client that lists
                # models or matches on the id is told it does not exist.
                alias=(
                    ",".join(ctl_mod.model_aliases(spec)) if spec is not None else None
                ),
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
    stacks, and points the supported agent clients at it.
    """


def _model_kind(s) -> str:
    """chat | vision | decider | image | tts | stt. The GGUF's own header
    speaks first: an audiocpp container or whisper architecture is TTS/STT
    no matter what its filename claims (a Fish Drama TTS quant pulled as a
    chat stack is how this was learned). sd-server stays image and mmproj
    stays vision without consulting the file — a stack that knows what it
    is keeps its answer. Names are the last resort, then chat. Metadata
    reads are header-only and never raise: a missing file just falls
    through to the older signals."""
    if (getattr(s, "binary", "") or "").endswith("sd-server"):
        return "image"
    try:
        from ..ggufmeta import classify_kind, read_gguf_kind

        gguf = getattr(s, "gguf", "") or ""
        if gguf:
            meta = read_gguf_kind(gguf)
            kind = classify_kind(meta.architecture, meta.audio_category)
            if kind:
                return kind
    except Exception:
        pass
    hay = f"{s.name} {s.model_id}".lower()
    from ..ctl import IMAGE_KEYS
    if any(k in hay for k in IMAGE_KEYS):
        return "image"
    if any(k in hay for k in ("clef", "decision", "decider", "laya")):
        return "decider"
    if bool(getattr(s, "mmproj", "")):
        return "vision"
    return "chat"


def _model_json_row(s, active_name: str | None) -> dict:
    """One registry row for machine clients (desktop selection panel)."""
    size_gb: float | None = None
    try:
        from ..ctl import resolve_gguf_path
        size_gb = round(resolve_gguf_path(s).stat().st_size / 1e9, 1)
    except Exception:
        pass
    return {
        "name": s.name,
        "label": s.label,
        "llama_port": s.llama_port,
        "daemon_port": s.daemon_port,
        "model_id": s.model_id,
        "ctx_size": s.ctx_size,
        "kv": s.kv,
        "kind": _model_kind(s),
        "size_gb": size_gb,
        "vision": bool(getattr(s, "mmproj", "")),
        "mmproj_present": bool(
            getattr(s, "mmproj", "") and Path(s.mmproj).exists()
        ),
        "active": s.name == active_name,
    }


@model.command("list")
@click.option("--json", "as_json", is_flag=True,
              help="Machine-readable output for tools that need to discover "
                   "stacks (e.g. a desktop client finding its vision model).")
def list_models(as_json: bool) -> None:
    """List the models in the registry."""
    import json as _json

    from ..ctl import load_registry, read_active
    if as_json:
        # A flat, stable shape: enough to find a stack and talk to it, and
        # deliberately no secrets (the auth token is read from the auth file
        # by the caller, never printed here).
        active_name = read_active()
        click.echo(_json.dumps([
            _model_json_row(s, active_name)
            for s in load_registry()
        ], indent=2))
        return
    for spec in load_registry():
        flags = []
        if spec.always_on:
            flags.append("always on")
        if spec.boot:
            flags.append("boot")
        if spec.mmproj:
            # Loud, because the failure without it is silence: a vision model
            # with no projector answers every image as though it were blank.
            flags.append("vision" if Path(spec.mmproj).exists() else "VISION, projector missing")
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
    help="Only rewrite model-capable CLI configs; do not start or stop any "
         "stack (installer-friendly heal).",
)
@click.option(
    "--force-restart", is_flag=True,
    help="Restart the model's own stack even when it is already "
         "running (used after llama-server reinstall so the new "
         "binary takes effect).",
)
@click.option(
    "--mode", type=click.Choice(["raw", "managed"]), default=None,
    help="Inference path for model-capable clients: managed (through the "
         "toki daemon — context truth, KV recycling) or raw (direct "
         "llama-server). Persisted; default is managed.",
)
def use(name: str, keep_others: bool, wait: bool, sync_only: bool,
        mode: str | None, force_restart: bool) -> None:
    """Start the model stack and select it in the agent clients."""
    from ..ctl import _effective_ctx, model_status, sync_cli_configs, use_model, write_client_mode
    if mode:
        for tool in (
            "opencode", "mimo", "openclaw", "hermes", "codex", "qwen",
            "droid", "cline", "dsh", "grok", "kilo", "pi", "copilot", "kimi", "omp",
        ):
            write_client_mode(tool, mode)
    try:
        spec = use_model(name, stop_others=not keep_others,
                         sync_only=sync_only, force_restart=force_restart)
    except (KeyError, FileNotFoundError, RuntimeError) as e:
        click.echo(f"error: {e}", err=True)
        raise SystemExit(1) from None
    paths = sync_cli_configs(spec)
    effective = _effective_ctx(spec)
    if not sync_only:
        try:
            if (spec.binary or "").endswith("sd-server"):
                click.echo(f"image stack on :{spec.llama_port} (sd-server)")
                return
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
    click.echo("switched. per-run override: opencode --model, mimo --model, "
               "openclaw agent --model, hermes -m --provider")
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
        # SIGCHLD does not exist on Windows; the auto-reap trick is a
        # POSIX-only nicety, skip it elsewhere.
        if hasattr(signal, "SIGCHLD"):
            signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        while True:
            time.sleep(3600)


@model.command("mode")
@click.argument("client", type=click.Choice([
    "opencode", "mimo", "openclaw", "hermes", "codex", "qwen", "droid",
    "cline", "dsh", "grok", "kilo", "pi", "copilot", "kimi", "omp",
]))
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
def load(name: str) -> None:
    """Load a model's stack beside the running ones (no switch, no config churn).

    Starts llama-server + daemon (or sd-server for image models) with
    --keep-others semantics: nothing already running is touched. The RAM
    gate still applies, so a stack that will not fit is refused with the
    numbers instead of dying mid-load.
    """
    from ..ctl import get_model, start_model
    try:
        spec = get_model(name)
        start_model(spec)
    except (KeyError, FileNotFoundError, RuntimeError) as e:
        click.echo(f"error: {e}", err=True)
        raise SystemExit(1) from None
    click.echo(f"model '{name}' loaded (running beside the others).")


@model.command()
@click.argument("name")
def unload(name: str) -> None:
    """Unload a model's stack to free its memory.

    Stops the model's servers and releases its weights + KV. The row stays
    registered with its ports, so `toki model load` brings it straight back.
    (Single-model llama-server builds expose no in-place weight unload —
    upstream removed POST /unload and /models/unload needs router mode —
    so freeing memory means stopping the stack. The separate verb keeps the
    intent — manual memory management — distinct from lifecycle `stop`.)
    """
    from ..ctl import _estimate_need, get_model, stop_model
    try:
        spec = get_model(name)
        freed = _estimate_need(spec)
        stop_model(spec)
    except (KeyError, RuntimeError) as e:
        click.echo(f"error: {e}", err=True)
        raise SystemExit(1) from None
    click.echo(f"model '{name}' unloaded (~{freed:.0f} GB freed).")


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

    Stops the stack first so the launcher and model-capable CLI clients
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
    re-writes the model-capable CLI configs.
    """
    from ..ctl import normalize_model_ids
    changes = normalize_model_ids()
    if not changes:
        click.echo("no model_ids needed fixing.")
        return
    for name, old, new in changes:
        click.echo(f"  {name}: {old!r} -> {new!r}")
    # Re-sync the model-capable CLI configs so the new short names land
    # in every supported harness immediately.
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


def _force_cpu_when_no_usable_gpu(os_name: str, arch: str, backend: str) -> str:
    """If a GPU backend is requested on a box with no usable GPU,
    return ``"cpu"`` instead.

    A GPU is "usable" when the hardware profile says so (``use_for_llm``
    is True — i.e. not shared-RAM, at least 4 GB VRAM, not lava pipe).
    On boxes where every GPU fails that check (Intel iGPU only, AMD APU
    only, no GPU at all) the Vulkan / CUDA / ROCm build still initialises
    the GPU and syncs on every generation token. Measured on a Core
    Ultra 155U: tg64 = 0.78 t/s with the Vulkan build vs ~50-70 t/s
    with the CPU build. The CPU build skips the GPU entirely.

    The rule is generic: any box where no GPU is usable gets the CPU
    build. No per-vendor special-casing.
    """
    if backend not in ("vulkan", "cuda", "rocm"):
        return backend
    try:
        from ..hardware import profile_box
        prof = profile_box(force=False)
    except Exception:
        return backend
    if prof is None or any(g.use_for_llm for g in prof.gpus):
        return backend
    click.echo(
        f"note: no usable GPU on this box; switching --backend "
        f"{backend} -> cpu. The {backend} build would sync with "
        f"the GPU on every token and slash generation throughput.",
        err=True,
    )
    return "cpu"


_KNOWN_BACKENDS = ("cuda", "rocm", "vulkan", "metal", "mlx", "cpu")


def _env_backend_override() -> str | None:
    """Respect TOKI_BACKEND when set to a known backend; warn (stderr)
    and ignore unknown values."""
    env_backend = os.environ.get("TOKI_BACKEND", "").strip().lower()
    if env_backend in _KNOWN_BACKENDS:
        return env_backend
    if env_backend:
        click.echo(
            f"warning: TOKI_BACKEND={env_backend} is not a known "
            f"backend; ignoring and auto-detecting", err=True,
        )
    return None


def _detect_gpu_backend(os_name: str) -> str:
    """Pick the best backend for this box.

    Order of preference:
      1. Cached hardware profile (real measurements from a previous
         install/doctor). Always wins.
      2. Live profile probe — what the current hardware.py sees.
      3. Heuristics — last resort fallback when profiling fails
         (offline box, no tooling). Kept narrow so it can't surprise
         anyone: NVIDIA + ROCm tools → those backends, everything
         else → CPU. Vulkan/CUDA/Metal all require a probe to pick.
    """
    try:
        from ..hardware import profile_box
        profile = profile_box()
        return profile.backend_for(model_size_class="any")
    except Exception:
        pass
    # Heuristic fallback: only the high-confidence signals.
    import shutil as _shutil
    if os_name == "macos":
        return "metal"
    if _shutil.which("nvidia-smi") or os.environ.get("CUDA_PATH"):
        return "cuda"
    if _shutil.which("hipconfig") or Path(
        "C:/Program Files/AMD/ROCm"
    ).exists() or Path("/opt/rocm").exists():
        return "rocm"
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
                # llama.cpp renamed the ROCm asset from win-hip-radeon-x64
                # to win-rocm-<ver>-x64 in mid-2026; match both so old and
                # new releases resolve to the ROCm build.
                ("rocm", "win-rocm-"),
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


# Runtime libs that ONLY belong to GPU builds. When a CPU build is
# installed they must be pruned from the bin dir — otherwise
# ``binary_backend`` / ``toki doctor`` still report "Vulkan" from a
# stale .so and the CPU binary itself may try to dlopen one (the
# Vulkan runtime on an iGPU box is what syncs per-token and tanked
# generation to 0.78 t/s).
_GPU_RUNTIME_LIBS = (
    "libggml-vulkan.so", "libggml-cuda.so", "libggml-rocm.so",
    "libggml-metal.so", "ggml-vulkan.dll", "ggml-cuda.dll",
    "ggml-rocm.dll", "ggml-metal.dll",
)


def _prune_foreign_gpu_libs(bin_dir: Path, backend: str) -> None:
    """Remove GPU runtime libs (and their .stale.* backups) that don't
    belong to ``backend``.

    Installing the CPU build over a previous Vulkan build would
    otherwise leave ``libggml-vulkan.so`` behind; the running
    llama-server (CPU) doesn't load it, but every backend probe does
    and the next Vulkan install would resurrect it. A CPU box must
    end up with exactly zero GPU libs.
    """
    if backend == "cpu":
        for pat in _GPU_RUNTIME_LIBS:
            for lib in list(bin_dir.glob(pat)) + list(bin_dir.glob(pat + ".stale.*")):
                try:
                    lib.unlink()
                except OSError:
                    pass


def _stream_download_with_retry(
    api, url: str, dest: Path, *, label: str = "download",
) -> None:
    """Stream ``url`` into ``dest`` with retry-on-disconnect.

    Prefers ``curl`` because its ``--retry-all-errors`` semantics are
    battle-tested for the "Server disconnected without sending a
    response" pattern GitHub's release CDN shows under load. When curl
    is missing or also fails, falls back to the in-process httpx path
    with bounded exponential backoff. Surfaces a single actionable
    ``click.ClickException`` after the combined retry budget is
    exhausted — never the raw 60-line httpx/httpcore traceback.
    """
    # Pygithub stores the token on the underlying Requester; for the
    # plain httpx.Client path we read it from the same env vars so curl
    # can authenticate identically.
    token = (
        os.environ.get("GITHUB_TOKEN")
        or os.environ.get("TOKI_GITHUB_TOKEN")
    )

    last_err: BaseException | None = None
    curl_path = shutil.which("curl")
    if curl_path:
        try:
            _curl_download(
                curl_path, url, dest, label=label, token=token,
            )
            return
        except _RetryableDownloadError as exc:
            last_err = exc
            click.echo(
                f"{label}: curl path failed; falling back to httpx…"
            )

    # httpx fallback (in-process, no shell). Three attempts with
    # exponential backoff.
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
    _raise_download_error(label, url, dest, last_err)


class _RetryableDownloadError(RuntimeError):
    """Internal signal that the curl path failed and the caller should
    try the in-process fallback (or, if curl was the last resort,
    surface the actionable error to the user)."""


def _curl_download(
    curl_path: str, url: str, dest: Path, *,
    label: str, token: str | None,
) -> None:
    """Stream ``url`` to ``dest`` using ``curl`` with native retry.

    Raises ``_RetryableDownloadError`` on any non-zero exit, on a 0-byte
    body (curl ``--fail`` aborts mid-download on 4xx/5xx and leaves the
    target file empty), or on a binary that's not actually present.
    """
    cmd = [
        curl_path,
        "--fail",                # 4xx/5xx → non-zero exit, no body
        "--location",            # follow GitHub release redirects
        "--show-error",          # surface errors on stderr
        "--silent",              # suppress progress meter
        "--max-time", "600",     # 10-minute hard ceiling
        "--connect-timeout", "30",
        "--retry", "8",
        "--retry-delay", "2",
        "--retry-all-errors",    # retry on ANY error incl. TCP reset / TLS RST
        "--retry-max-time", "300",  # give up on retrying after 5 min
        "-o", str(dest),
        url,
    ]
    if token:
        cmd.insert(-1, "-H")
        cmd.insert(-1, f"Authorization: Bearer {token}")

    click.echo(f"{label}: using curl with --retry-all-errors…")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise _RetryableDownloadError(
            f"curl exited {result.returncode}: "
            f"{stderr or 'unknown error'}"
        )
    # curl --fail aborts on HTTP 4xx/5xx, leaving dest empty/partial
    # (or never creating it). Either way the request never got a body.
    if not dest.exists() or dest.stat().st_size == 0:
        raise _RetryableDownloadError(
            "curl wrote 0 bytes (likely 4xx/5xx — "
            "check the URL or your GitHub token)"
        )


def _raise_download_error(
    label: str, url: str, dest: Path, last_err: BaseException | None,
) -> None:
    """Surface a single actionable error after the combined retry budget
    is exhausted. Never the raw 60-line httpx/httpcore traceback."""
    try:
        if dest.exists():
            dest.unlink()
    except OSError:
        pass
    host = url.split("/releases/", 1)[0] + "/…"
    name = type(last_err).__name__ if last_err else "UnknownError"
    msg = str(last_err) if last_err else "no error captured"
    raise click.ClickException(
        f"{label}: failed to download ({name}: {msg}).\n"
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


def _find_local_llama_server() -> str | None:
    """Return the absolute path of a locally-built llama-server, or
    None when none is on disk.

    Search order:
      1. ``LLAMA_CPP_BLD`` env var (set by ``scripts/build_llama_cpp.sh``)
      2. ``$HOME/llama.cpp/build-tr/bin/llama-server`` (the script's default)
      3. ``$HOME/llama.cpp/build/bin/llama-server`` (manual builds)
      4. ``/usr/local/bin/llama-server`` and ``/opt/llama.cpp/build/bin/llama-server``

    Used by ``install-llama`` when llama.cpp no longer ships a release
    asset for the requested backend (Linux ROCm post-b10375, custom
    CUDA builds). The state-dir copy is then made via
    ``_safe_copy_replace`` so the rest of the launch path stays uniform
    — the registry, ``binary_backend`` and the warn checks all see the
    same path.
    """
    candidates: list[Path] = []
    bld = os.environ.get("LLAMA_CPP_BLD")
    if bld:
        candidates.append(Path(bld) / "bin" / "llama-server")
    home = Path.home()
    candidates.extend([
        home / "llama.cpp" / "build-tr" / "bin" / "llama-server",
        home / "llama.cpp" / "build" / "bin" / "llama-server",
        Path("/usr/local/bin/llama-server"),
        Path("/opt/llama.cpp/build/bin/llama-server"),
    ])
    for c in candidates:
        try:
            if c.is_file() and os.access(c, os.X_OK):
                return str(c.resolve())
        except OSError:
            continue
    return None


def _local_binary_backend(local_path: str) -> str | None:
    """Backend label reported by a locally-built llama-server (CUDA /
    ROCm / Vulkan / CPU). Used to record what backend the local build
    actually carries, so the post-install verify step has something
    concrete to compare against.
    """
    try:
        from ..diagnose import binary_backend
        return binary_backend(local_path)
    except Exception:
        return None


def _install_from_local(
    local_path: str, *, asset_name: str | None, bin_dir: Path,
    exe_name: str, backend: str, os_name: str,
) -> None:
    """Copy a locally-built llama-server into ``bin_dir`` using the
    same safe-copy path as a release asset, then run the post-install
    backend verify. Used by the local-build fallback so the rest of
    the launcher sees a uniform ``STATE_ROOT/bin/llama-server``.
    """
    src = Path(local_path)
    dst = bin_dir / exe_name
    _sweep_stale_backups(bin_dir, exe_name)
    _safe_copy_replace(src, dst)
    # Local builds don't ship separate libggml-*.so files alongside
    # the binary on every host — when they do (CUDA/ROCm Linux
    # dynamic builds), copy them too so the post-install verify
    # matches the dynamic build.
    src_dir = src.parent
    for side in src_dir.iterdir():
        if side.is_file() and side != src:
            _safe_copy_replace(side, bin_dir / side.name)
    # Even with a local ROCm/CUDA build, prune any stray CPU/GPU
    # libs from a previous install that don't match this backend.
    _prune_foreign_gpu_libs(bin_dir, backend)
    if os_name != "windows":
        try:
            dst.chmod(dst.stat().st_mode | 0o111)
        except OSError:
            pass
    click.echo(f"installed: {dst}")
    from ..diagnose import binary_backend
    landed = (binary_backend(str(dst)) or "unknown").lower()
    if landed == "unknown":
        click.echo(
            "verify: could not inspect the installed binary; "
            f"expected backend {backend}",
            err=True,
        )
    elif landed != backend:
        click.echo(
            f"error: expected backend {backend} but this build is "
            f"{landed} — local build does not match the requested "
            f"backend. Rebuild with the matching target "
            f"(./scripts/build_llama_cpp.sh) or rerun with "
            f"--backend {landed}.",
            err=True,
        )
        raise SystemExit(1)
    else:
        click.echo(f"verify: landed backend {landed} (expected {backend})")


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
              help="GPU backend to install (default: auto-detect from "
                   "the cached hardware profile).")
@click.option("--gfx", "gfx", default=None,
              help="HSA_OVERRIDE_GFX_VERSION for ROCm builds, e.g. 11.5.1 "
                   "(default: detected from rocminfo).")
@click.option("--profile/--no-profile", default=True,
              help="Re-profile hardware before deciding the backend. "
                   "Default: yes. Pass --no-profile to skip the probe on "
                   "a slow box or when you've already verified the "
                   "recommendation.")
@click.option("--force-cpu", is_flag=True, default=False,
              help="Install the CPU build even when the profile suggests "
                   "GPU offload. Use when you know the GPU isn't worth "
                   "the setup overhead for the model size you're running.")
def install_llama(tag: str | None, dest: str | None, name: str | None,
                  backend: str, gfx: str | None, profile: bool,
                  force_cpu: bool) -> None:
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
        # TOKI_BACKEND: hard user override, honoured before any
        # profile or heuristic decision. Valid values are the same
        # ones --backend accepts.
        env_backend = _env_backend_override()
        if env_backend is not None:
            backend = env_backend
            click.echo(f"backend: TOKI_BACKEND={backend} ({os_name}/{arch})")
        else:
            from ..hardware import profile_box
            prof = profile_box(force=False) if profile else None
            if prof is not None:
                recommended = prof.recommended
                backend = recommended.get("backend", _detect_gpu_backend(os_name))
                p_core = ",".join(str(i) for i in prof.cpu.p_core_ids) or "—"
                simd = ",".join(prof.cpu.simd) or "—"
                click.echo(
                    f"backend: profile-recommended {backend} ({os_name}/{arch})"
                )
                click.echo(
                    f"hardware profile: cpu={prof.cpu.name} simd={simd} "
                    f"p_cores={p_core} ram={prof.ram_gb:.0f}GB"
                )
                for g in prof.gpus:
                    click.echo(
                        f"  gpu: {g.vendor} {g.name} vram={g.vram_gb:.1f}GB "
                        f"use={'yes' if g.use_for_llm else 'no'} ({g.reason})"
                    )
            else:
                backend = _detect_gpu_backend(os_name)
                click.echo(f"backend: auto-detected {backend} ({os_name}/{arch})")
        if force_cpu and backend in ("cuda", "rocm", "vulkan", "metal"):
            click.echo(
                f"note: --force-cpu requested; ignoring profile's "
                f"recommendation ({backend}) and installing CPU instead"
            )
            backend = "cpu"
        backend = _force_cpu_when_no_usable_gpu(os_name, arch, backend)
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

    # Prefer a locally-built llama-server over a release asset when
    # the resolved backend is CPU and the user has clearly built a
    # GPU variant on this box (the build script ships a ROCm binary
    # by default on Linux). The profile's CPU recommendation can be
    # wrong when ``use_for_llm`` flipped yes for an APU with 0.5 GB
    # VRAM, or when the user upgraded hardware and didn't re-profile.
    # Honouring the explicit local build also gives Linux ROCm /
    # CUDA users post-b10375 (no release asset) what they built.
    if backend == "cpu":
        local = _find_local_llama_server()
        if local:
            from ..diagnose import binary_backend
            local_kind = (binary_backend(local) or "").lower()
            if local_kind in ("cuda", "rocm", "vulkan", "metal"):
                click.echo(
                    f"backend: local {local_kind} build at {local} "
                    f"(overrides profile's CPU recommendation; "
                    f"rebuild with ./scripts/build_llama_cpp.sh to refresh)",
                    err=True,
                )
                _install_from_local(local, asset_name=None,
                                    bin_dir=bin_dir, exe_name=exe_name,
                                    backend=local_kind, os_name=os_name)
                return

    # Refuse to silently downgrade a working GPU binary to CPU. When
    # the bin dir already has a llama-server built with a GPU backend,
    # overwriting it with a CPU build makes every existing model spec
    # that pins ``-ngl 99`` refuse to start (``_check_gpu_offload_binary_match``
    # is now strict). Refuse the downgrade unless the user explicitly
    # forces it via TOKI_INSTALL_LLAMA_FORCE_DOWNGRADE=1.
    existing = bin_dir / exe_name
    if existing.exists():
        from ..diagnose import binary_backend
        existing_kind = (binary_backend(str(existing)) or "").lower()
        if (existing_kind in ("cuda", "rocm", "vulkan", "metal")
                and backend == "cpu"
                and os.environ.get("TOKI_INSTALL_LLAMA_FORCE_DOWNGRADE") != "1"):
            raise click.ClickException(
                f"refusing to overwrite the existing {existing_kind} "
                f"llama-server at {existing} with a CPU build — every "
                f"model pinned to GPU offload would refuse to start. "
                f"Re-run with --backend {existing_kind} to refresh the "
                f"GPU build, or set TOKI_INSTALL_LLAMA_FORCE_DOWNGRADE=1 "
                f"to force the CPU replacement."
            )

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
                "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=100",
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
        # No release asset for the requested backend on this OS/arch —
        # the classic Linux ROCm post-b10375 case. Fall back to a
        # locally-built llama-server when one exists, so a user who
        # built with ./scripts/build_llama_cpp.sh doesn't end up with
        # a CPU binary and -ngl 99 refusing to start.
        local = _find_local_llama_server()
        if local and backend != "cpu":
            click.echo(
                f"note: no {backend} asset on llama.cpp releases for "
                f"{os_name}/{arch}; using locally-built llama-server at "
                f"{local} instead. (Build with "
                f"./scripts/build_llama_cpp.sh to refresh.)",
                err=True,
            )
            backend = _local_binary_backend(local) or backend
            return _install_from_local(local, asset_name=None,
                                       bin_dir=bin_dir, exe_name=exe_name,
                                       backend=backend, os_name=os_name)
        names = ", ".join(a.get("name", "") for a in assets[:12])
        raise RuntimeError(
            f"no asset matching {backend} for {os_name}/{arch} in {tag_name}; "
            f"available (first 12): {names}"
        )
    if matched_backend != backend:
        # Release fell back to a different backend than requested —
        # usually CPU when the user wanted a GPU backend that has no
        # release asset for this OS/arch. Try the local build before
        # silently downgrading to CPU.
        local = _find_local_llama_server()
        if local and matched_backend == "cpu" and backend != "cpu":
            click.echo(
                f"note: no {backend} asset on llama.cpp releases for "
                f"{os_name}/{arch}; using locally-built llama-server at "
                f"{local} instead of the CPU fallback. (Build with "
                f"./scripts/build_llama_cpp.sh to refresh.)",
                err=True,
            )
            backend = _local_binary_backend(local) or backend
            return _install_from_local(local, asset_name=None,
                                       bin_dir=bin_dir, exe_name=exe_name,
                                       backend=backend, os_name=os_name)
        click.echo(
            f"note: no {backend} build for {os_name}/{arch}; "
            f"installing {matched_backend} instead"
        )
        backend = matched_backend
    # llama.cpp stopped shipping Linux ROCm assets after b10375 (mid-2026);
    # the walk lands on that last build, which predates newer model support.
    # Warn so users on a Linux ROCm box know to build from source instead.
    newest_tag = releases[0]["tag_name"] if not tag else None
    if (backend == "rocm" and os_name == "ubuntu"
            and newest_tag and tag_name != newest_tag):
        click.echo(
            f"note: {tag_name} is the last llama.cpp release with a Linux "
            f"ROCm asset (llama.cpp now ships ROCm only for Windows). "
            f"Newer model support lands on master first — to get it on this "
            f"box, build from source: ./scripts/build_llama_cpp.sh",
            err=True,
        )
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
        # A CPU install must not leave GPU runtime libs behind: stale
        # .so files mislead backend probes and confuse the launcher.
        _prune_foreign_gpu_libs(bin_dir, backend)

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

    # Post-install verification: what backend actually landed? A CPU
    # box must have zero GPU libs; a GPU box must have its libs. This
    # is what the launcher / doctor will report, so surface it now
    # instead of letting a stale build pass silently.
    from ..diagnose import binary_backend
    landed = (binary_backend(str(dst)) or "unknown").lower()
    if landed == "unknown":
        click.echo(
            "verify: could not inspect the installed binary "
            f"(ldd unavailable); expected backend {backend}",
            err=True,
        )
    elif landed != backend:
        click.echo(
            f"error: expected backend {backend} but this build is "
            f"{landed} — install failed to land the right build. "
            f"Re-run with --backend {backend} to retry.",
            err=True,
        )
        raise SystemExit(1)
    else:
        click.echo(f"verify: landed backend {landed} (expected {backend})")

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
    "bench", "doctor", "down", "evict", "hardware", "interactive",
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
