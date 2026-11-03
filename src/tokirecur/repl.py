"""Interactive REPL for Toki Recur.

Entered with `toki` (no args) or `toki interactive`. Provides a status-aware
prompt and a small DSL for controlling the local stack.

Commands
--------
  up [--detach]            Start llama-server + daemon
  down [-y]                Stop llama-server + daemon
  status                   Show stack state
  doctor                   Check llama-server, ROCm, GPU
  bench [bench-args...]    Run the benchmark suite
  run "prompt"             One-shot completion via daemon
  stats                    Cache index stats
  list [n]                 List cached prompts (default 20)
  show <cache-id>          Show prompt details
  evict <target-mb>        LRU evict to target MB
  clear                    Wipe cache
  model <path>             Set GGUF model path
  port llama <n>           Set llama-server port (default 8088)
  port daemon <n>          Set toki daemon port (default 8787)
  ctx <n>                  Set context size (default 8192)
  ngpu <n>                 Set GPU layers (default 99)
  config                   Show current config
  compare <a.json> <b.json>   Compare two bench reports
  help [command]           Show help for a command (or all)
  exit / quit / q          Exit shell
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()




def _auth_headers() -> dict[str, str]:
    """The daemon's bearer token.

    The daemon refuses an unauthenticated request, so every command in
    this shell that talks to it needs the token: without it, `run` and
    `stats` answered 401 and the shell blamed the daemon for not being
    running.
    """
    token = os.environ.get("TOKI_AUTH_TOKEN")
    if not token:
        for path in (
            Path.home() / ".config" / "tokirecur" / "auth",
            Path("/etc/toki-recur/auth"),
        ):
            try:
                if path.is_file():
                    token = path.read_text().strip()
                    if token:
                        break
            except OSError:
                continue
    return {"Authorization": f"Bearer {token}"} if token else {}


def _why_it_failed(error: Exception, port: int) -> str:
    """The hint under an error: what is actually wrong."""
    text = str(error)
    if "401" in text or "Unauthorized" in text:
        return (
            f"the daemon on :{port} wants a token — set TOKI_AUTH_TOKEN or put one "
            "in ~/.config/tokirecur/auth"
        )
    if "Connection refused" in text or "ConnectError" in text:
        return f"is the daemon running on :{port}?"
    return f"is the daemon running on :{port}?"


def _active():
    """The active registry model, or None."""
    try:
        from . import ctl

        return ctl.active_spec()
    except Exception:
        return None


def _default_model() -> Path:
    """The active model's weights.

    It used to be a hard-coded path to a starter download, so the shell
    pointed at a file most machines no longer have and at ports nothing
    was on — and every command that talked to the model failed with
    "connection refused" while the daemon was plainly running.
    """
    env = os.environ.get("TOKI_MODEL")
    if env:
        return Path(env)
    spec = _active()
    if spec is not None:
        try:
            from . import ctl

            return ctl.resolve_gguf_path(spec)
        except Exception:
            pass
    return Path.home() / "Downloads" / "Qwen3.5-9B-Q4_K_M.gguf"


def _default_llama_port() -> int:
    spec = _active()
    return spec.llama_port if spec is not None else 8088


def _default_daemon_port() -> int:
    spec = _active()
    return spec.daemon_port if spec is not None else 8787


def _default_ctx_size() -> int:
    spec = _active()
    return spec.ctx_size if spec is not None else 8192


def _default_backend_url() -> str:
    return f"http://127.0.0.1:{_default_llama_port()}"


@dataclass
class ReplConfig:
    """User-tunable values that survive REPL commands."""

    model: Path = field(default_factory=_default_model)
    llama_port: int = field(default_factory=_default_llama_port)
    daemon_port: int = field(default_factory=_default_daemon_port)
    ctx_size: int = field(default_factory=_default_ctx_size)
    n_gpu_layers: int = 99
    extra_llama_args: str = ""
    backend_url: str = field(default_factory=_default_backend_url)

    def summary(self) -> str:
        return (
            f"model={self.model}  llama=: {self.llama_port}  daemon=: {self.daemon_port}  "
            f"ctx={self.ctx_size}  ngpu={self.n_gpu_layers}"
        )


# ---------------------------------------------------------------------------
# Banner + prompt
# ---------------------------------------------------------------------------


BANNER = """
[bold cyan]Toki Recur[/bold cyan] — interactive shell
[dim]type 'help' for commands, 'exit' to quit[/dim]
"""


def _status_indicator(cfg: ReplConfig) -> str:
    """Returns a colour-coded [llama:✓ daemon:✓] style tag."""
    from . import lifecycle
    s = lifecycle.status()
    llama = "✓" if s["llama_server"]["reachable"] else "✗" if s["llama_server"]["alive"] else "·"
    daemon = "✓" if s["toki_daemon"]["reachable"] else "✗" if s["toki_daemon"]["alive"] else "·"
    llama_color = "green" if llama == "✓" else "red" if llama == "✗" else "dim"
    daemon_color = "green" if daemon == "✓" else "red" if daemon == "✗" else "dim"
    return (
        f"[{llama_color}]llama:{llama}[/{llama_color}] "
        f"[{daemon_color}]daemon:{daemon}[/{daemon_color}]"
    )


def _prompt(cfg: ReplConfig) -> str:
    """Build the prompt string with live status."""
    ind = _status_indicator(cfg)
    return f"[bold cyan]toki[/bold cyan] {ind} [bold cyan]❯[/bold cyan] "


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_help(args: list[str], cfg: ReplConfig) -> None:
    if args:
        name = args[0]
        h = HELP.get(name)
        if h:
            console.print(Panel(h, title=f"help: {name}", border_style="cyan"))
        else:
            console.print(f"[red]no help for {name!r}[/red]")
        return
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="bold cyan")
    t.add_column()
    t.add_row("up [--detach]", "start llama-server + daemon")
    t.add_row("down [-y]", "stop llama-server + daemon")
    t.add_row("status", "show stack state")
    t.add_row("doctor", "check llama-server, ROCm, GPU")
    t.add_row("bench [args]", "run the benchmark suite")
    t.add_row("run \"prompt\"", "one-shot completion via daemon")
    t.add_row("stats", "cache index stats")
    t.add_row("list [n]", "list cached prompts (default 20)")
    t.add_row("show <id>", "show prompt details")
    t.add_row("evict <mb>", "LRU evict to target MB")
    t.add_row("clear", "wipe cache")
    t.add_row("model <path>", "set GGUF model path")
    t.add_row("port llama <n>", f"set llama-server port (current {cfg.llama_port})")
    t.add_row("port daemon <n>", f"set toki daemon port (current {cfg.daemon_port})")
    t.add_row("ctx <n>", f"set context size (current {cfg.ctx_size})")
    t.add_row("ngpu <n>", f"set GPU layers (current {cfg.n_gpu_layers})")
    t.add_row("config", "show current config")
    t.add_row("compare <a.json> <b.json>", "compare two bench reports")
    t.add_row("exit / quit / q", "exit shell")
    console.print(Panel(t, title="commands", border_style="cyan"))


def cmd_up(args: list[str], cfg: ReplConfig) -> None:
    from . import lifecycle
    detach = "--detach" in args
    if not cfg.model.exists():
        console.print(f"[red]model not found: {cfg.model}[/red]")
        console.print("[dim]hint: `model <path>` to set[/dim]")
        return
    try:
        pid = lifecycle.start_llama_server(
            model=str(cfg.model),
            port=cfg.llama_port,
            ctx_size=cfg.ctx_size,
            n_gpu_layers=cfg.n_gpu_layers,
            extra_args=shlex.split(cfg.extra_llama_args) if cfg.extra_llama_args else None,
        )
        console.print(f"[green]llama-server started[/green] (pid={pid})  http://127.0.0.1:{cfg.llama_port}")
    except Exception as e:
        console.print(f"[red]failed to start llama-server:[/red] {e}")
        return

    if detach:
        pid = lifecycle.start_daemon(
            host="127.0.0.1",
            port=cfg.daemon_port,
            backend_url=f"http://127.0.0.1:{cfg.llama_port}",
            wait=True,
        )
        console.print(f"[green]toki daemon started[/green] (pid={pid})  http://127.0.0.1:{cfg.daemon_port}")
    else:
        console.print("[yellow]start the daemon with: toki up --detach,")
        console.print("[yellow]or run `toki serve` in another shell[/yellow]")


def cmd_down(args: list[str], cfg: ReplConfig) -> None:
    from . import lifecycle
    yes = "-y" in args or "--yes" in args
    s = lifecycle.status()
    if not s["llama_server"]["alive"] and not s["toki_daemon"]["alive"]:
        console.print("[dim]nothing to stop[/dim]")
        return
    if not yes:
        if s["llama_server"]["alive"]:
            console.print(f"  llama-server (pid={s['llama_server']['pid']})")
        if s["toki_daemon"]["alive"]:
            console.print(f"  toki daemon   (pid={s['toki_daemon']['pid']})")
        if not _confirm("stop?"):
            return
    lifecycle.stop_all()
    console.print("[green]stopped[/green]")


def cmd_status(args: list[str], cfg: ReplConfig) -> None:
    from . import lifecycle
    s = lifecycle.status()

    def fmt(name: str, data: dict, port: int) -> None:
        if data["alive"] and data["reachable"]:
            state = "[green]running[/green]"
        elif data["alive"]:
            state = "[yellow]stuck (alive but unreachable)[/yellow]"
        elif data["pid"]:
            state = "[red]dead (stale pidfile)[/red]"
        else:
            state = "[dim]stopped[/dim]"
        pid = f"pid={data['pid']}" if data["pid"] else "no pidfile"
        console.print(f"  {name.ljust(15)} {state}  {pid}  http://127.0.0.1:{port}")

    console.print("Toki Recur stack:")
    fmt("llama-server", s["llama_server"], 8088)
    fmt("toki daemon", s["toki_daemon"], 8787)


def cmd_doctor(args: list[str], cfg: ReplConfig) -> None:
    import shutil
    rows = []
    rows.append(("llama-server binary", shutil.which("llama-server") or "[red]missing[/red]"))
    for cmd in (["rocminfo"], ["hipconfig", "--version"]):
        try:
            out = subprocess.check_output(cmd, text=True, timeout=5).strip().splitlines()
            rows.append((cmd[0], out[0] if out else "ok"))
        except Exception as e:
            rows.append((cmd[0], f"[red]{e}[/red]"))
    from . import lifecycle
    s = lifecycle.status()
    rows.append((
        "llama-server :8088",
        "[green]ok[/green]" if s["llama_server"]["reachable"]
        else "[red]not reachable[/red]",
    ))
    rows.append((
        "toki daemon :8787",
        "[green]ok[/green]" if s["toki_daemon"]["reachable"]
        else "[red]not running[/red]",
    ))
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="bold")
    t.add_column()
    for k, v in rows:
        t.add_row(k, v)
    console.print(Panel(t, title="doctor", border_style="cyan"))


def cmd_bench(args: list[str], cfg: ReplConfig) -> None:
    from .bench.cli import main as bench_main
    argv = ["toki", "bench"] + args
    if "--model" not in args and not any(a.startswith("--model=") for a in args):
        argv += ["--model", str(cfg.model)]
    if "--base-url" not in args and not any(a.startswith("--base-url=") for a in args):
        argv += ["--base-url", f"http://127.0.0.1:{cfg.llama_port}"]
    sys.argv = argv
    try:
        bench_main(argv[2:])
    except SystemExit as e:
        if e.code and e.code != 0:
            console.print(f"[red]bench exited with code {e.code}[/red]")


def cmd_run(args: list[str], cfg: ReplConfig) -> None:
    import json

    import httpx
    if not args:
        console.print("[red]usage: run \"prompt text\"[/red]")
        return
    prompt = " ".join(args)
    url = f"http://127.0.0.1:{cfg.daemon_port}/v1/completions"
    # Ask for the stream the code below reads. It used to ask for a
    # single JSON object and then parse it as server-sent events, so the
    # text never arrived and the whole envelope was printed instead.
    payload = {
        "prompt": prompt,
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": True,
    }
    try:
        with httpx.stream("POST", url, json=payload, timeout=600.0,
                           headers=_auth_headers()) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                # The daemon streams server-sent events. The useful part is
                # the text; the envelope around it — ids, cache counters,
                # token usage — was what the shell printed, so `run` showed
                # a wall of JSON instead of the answer.
                data = line[5:].strip() if line.startswith("data:") else line.strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        console.print(piece, end="")
            console.print()
    except Exception as e:
        console.print(f"[red]error:[/red] {e}")
        console.print(f"[dim]hint: {_why_it_failed(e, cfg.daemon_port)}[/dim]")


def cmd_stats(args: list[str], cfg: ReplConfig) -> None:
    import httpx
    url = f"http://127.0.0.1:{cfg.daemon_port}/cache/stats"
    try:
        r = httpx.get(url, timeout=5.0, headers=_auth_headers())
        r.raise_for_status()
        console.print(json.dumps(r.json(), indent=2))
    except Exception as e:
        console.print(f"[red]error:[/red] {e}")


def cmd_list(args: list[str], cfg: ReplConfig) -> None:
    import httpx
    n = int(args[0]) if args and args[0].isdigit() else 20
    url = f"http://127.0.0.1:{cfg.daemon_port}/cache/list?n={n}"
    try:
        r = httpx.get(url, timeout=5.0, headers=_auth_headers())
        if r.status_code == 404:
            console.print("[yellow]/cache/list endpoint not implemented yet (Phase 2)[/yellow]")
            return
        r.raise_for_status()
        rows = r.json()
        if not rows:
            console.print("[dim]cache is empty[/dim]")
            return
        t = Table(show_header=True, header_style="bold")
        t.add_column("cache_id", style="cyan")
        t.add_column("prompt preview", max_width=80)
        t.add_column("last_used", justify="right")
        for row in rows:
            t.add_row(
                row.get("cache_id", "?"),
                row.get("prompt_preview", "")[:80],
                time.strftime("%H:%M:%S", time.localtime(row.get("last_used_at", 0))),
            )
        console.print(t)
    except Exception as e:
        console.print(f"[red]error:[/red] {e}")


def cmd_show(args: list[str], cfg: ReplConfig) -> None:
    if not args:
        console.print("[red]usage: show <cache-id>[/red]")
        return
    cache_id = args[0]
    import httpx
    url = f"http://127.0.0.1:{cfg.daemon_port}/cache/{cache_id}"
    try:
        r = httpx.get(url, timeout=5.0, headers=_auth_headers())
        if r.status_code == 404:
            console.print(f"[yellow]no cache entry with id {cache_id!r}[/yellow]")
            return
        r.raise_for_status()
        console.print_json(r.json())
    except Exception as e:
        console.print(f"[red]error:[/red] {e}")


def cmd_evict(args: list[str], cfg: ReplConfig) -> None:
    if not args or not args[0].isdigit():
        console.print("[red]usage: evict <target-mb>[/red]")
        return
    target = int(args[0])
    from .index import BruteForceIndex
    db = Path(os.environ.get("TOKI_DB", str(Path.home() / ".tokirecur/index.sqlite")))
    idx = BruteForceIndex(db)
    n = idx.evict_lru(target * 1024 * 1024)
    console.print(f"[green]evicted {n} entries[/green]")


def cmd_clear(args: list[str], cfg: ReplConfig) -> None:
    if not _confirm("wipe entire cache? this cannot be undone"):
        return
    db = Path(os.environ.get("TOKI_DB", str(Path.home() / ".tokirecur/index.sqlite")))
    if db.exists():
        db.unlink()
        console.print("[green]cache index wiped[/green]")
    blob_dir = Path(os.environ.get("TOKI_CACHE_DIR", str(Path.home() / ".tokirecur/blobs")))
    if blob_dir.exists():
        for f in glob_files(blob_dir):
            f.unlink()
        console.print("[green]cache blobs wiped[/green]")


def cmd_model(args: list[str], cfg: ReplConfig) -> None:
    if not args:
        console.print(f"[cyan]model:[/cyan] {cfg.model}")
        return
    p = Path(args[0]).expanduser()
    if not p.exists():
        console.print(f"[red]not found: {p}[/red]")
        return
    cfg.model = p
    console.print(f"[green]model set:[/green] {p}")


def cmd_port(args: list[str], cfg: ReplConfig) -> None:
    if len(args) < 2 or not args[1].isdigit():
        console.print("[red]usage: port llama <n>  or  port daemon <n>[/red]")
        return
    target, value = args[0], int(args[1])
    if target == "llama":
        cfg.llama_port = value
        console.print(f"[green]llama-server port:[/green] {value}")
    elif target == "daemon":
        cfg.daemon_port = value
        console.print(f"[green]toki daemon port:[/green] {value}")
    else:
        console.print(f"[red]unknown target: {target!r} (use 'llama' or 'daemon')[/red]")


def cmd_ctx(args: list[str], cfg: ReplConfig) -> None:
    if not args or not args[0].isdigit():
        console.print(f"[cyan]ctx:[/cyan] {cfg.ctx_size}")
        return
    cfg.ctx_size = int(args[0])
    console.print(f"[green]ctx:[/green] {cfg.ctx_size}")


def cmd_ngpu(args: list[str], cfg: ReplConfig) -> None:
    if not args or not args[0].isdigit():
        console.print(f"[cyan]ngpu:[/cyan] {cfg.n_gpu_layers}")
        return
    cfg.n_gpu_layers = int(args[0])
    console.print(f"[green]ngpu:[/green] {cfg.n_gpu_layers}")


def cmd_config(args: list[str], cfg: ReplConfig) -> None:
    console.print(Panel(cfg.summary(), title="config", border_style="cyan"))


def cmd_compare(args: list[str], cfg: ReplConfig) -> None:
    if len(args) < 2:
        console.print("[red]usage: compare <a.json> <b.json>[/red]")
        return
    a, b = Path(args[0]), Path(args[1])
    if not a.exists() or not b.exists():
        console.print(f"[red]missing file: {a} or {b}[/red]")
        return
    ja = json.loads(a.read_text())
    jb = json.loads(b.read_text())
    sa, sb = ja.get("score", {}), jb.get("score", {})

    t = Table(title="score diff", show_header=True)
    t.add_column("metric", style="bold")
    t.add_column("a", justify="right")
    t.add_column("b", justify="right")
    t.add_column("Δ", justify="right")
    for k in ("speedup", "ttft_improvement_pct", "prefill_skip_pct",
              "cache_hit_rate_pct", "throughput_gain_pct", "composite"):
        va, vb = sa.get(k, 0), sb.get(k, 0)
        delta = vb - va
        color = "green" if delta > 0 else "red" if delta < 0 else "white"
        t.add_row(k, f"{va:.3f}", f"{vb:.3f}", f"[{color}]{delta:+.3f}[/{color}]")
    console.print(t)


def cmd_exit(args: list[str], cfg: ReplConfig) -> bool:
    return False  # returning False from handler signals exit


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

HELP: dict[str, str] = {
    "up": "Start llama-server and the toki daemon. Use --detach to background.",
    "down": "Stop the stack. Pass -y to skip confirmation.",
    "status": "Show whether llama-server and the daemon are running.",
    "doctor": "Diagnose llama-server, ROCm, and GPU.",
    "bench": "Run the benchmark suite. All `toki bench` flags are accepted.",
    "run": "Send a one-shot completion through the daemon: run \"prompt text\"",
    "stats": "Show cache index stats (entries, bytes, embedding dim).",
    "list": "List cached prompt IDs (default 20).",
    "show": "Show details for a cache entry: show <cache-id>",
    "evict": "LRU evict down to target MB: evict 4096",
    "clear": "Wipe the entire cache (irreversible).",
    "model": "Set or show the GGUF model path.",
    "port": "Set ports: port llama 8088  /  port daemon 8787",
    "ctx": "Set llama-server context size.",
    "ngpu": "Set number of GPU layers.",
    "config": "Show current config.",
    "compare": "Compare two bench reports: compare a.json b.json",
    "help": "Show help for a command (or all commands).",
    "exit": "Exit the shell.",
}

COMMANDS: dict[str, Callable[[list[str], ReplConfig], Any]] = {
    "help": cmd_help,
    "up": cmd_up,
    "down": cmd_down,
    "status": cmd_status,
    "doctor": cmd_doctor,
    "bench": cmd_bench,
    "run": cmd_run,
    "stats": cmd_stats,
    "list": cmd_list,
    "show": cmd_show,
    "evict": cmd_evict,
    "clear": cmd_clear,
    "model": cmd_model,
    "port": cmd_port,
    "ctx": cmd_ctx,
    "ngpu": cmd_ngpu,
    "config": cmd_config,
    "compare": cmd_compare,
    "exit": cmd_exit,
    "quit": cmd_exit,
    "q": cmd_exit,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _confirm(question: str) -> bool:
    try:
        ans = input(f"{question} [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def glob_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return list(directory.glob("*.trkv")) + list(directory.glob("*.bin"))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run() -> int:
    """Enter the REPL. Returns the process exit code."""
    cfg = ReplConfig()
    console.print(BANNER)

    # Try to enable history & completion if prompt_toolkit is available.
    use_ptk = False
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.history import FileHistory

        completer = WordCompleter(list(COMMANDS.keys()), ignore_case=True)
        history_path = Path.home() / ".tokirecur_history"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        session = PromptSession(
            history=FileHistory(str(history_path)),
            completer=completer,
        )
        use_ptk = True
    except ImportError:
        session = None

    while True:
        try:
            if use_ptk:
                from prompt_toolkit.formatted_text import HTML
                line = session.prompt(
                    HTML(_status_indicator_html(cfg) + " <b>❯</b> ")
                )
            else:
                line = input("toki> ")
        except KeyboardInterrupt:
            console.print("\n[dim]use 'exit' to quit[/dim]")
            continue
        except EOFError:
            console.print()
            break

        line = line.strip()
        if not line:
            continue

        try:
            args = shlex.split(line)
        except ValueError as e:
            console.print(f"[red]parse error:[/red] {e}")
            continue

        cmd, rest = args[0], args[1:]
        handler = COMMANDS.get(cmd)
        if handler is None:
            console.print(f"[red]unknown command: {cmd!r}[/red]  [dim]type 'help'[/dim]")
            continue
        try:
            result = handler(rest, cfg)
        except KeyboardInterrupt:
            console.print("\n[yellow]interrupted[/yellow]")
            continue
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]error:[/red] {e}")
            continue
        if result is False:
            break

    console.print("[dim]bye.[/dim]")
    return 0


def _status_indicator_html(cfg: ReplConfig) -> str:
    """HTML variant for prompt_toolkit (which doesn't support rich markup)."""
    from . import lifecycle
    s = lifecycle.status()
    llama = "✓" if s["llama_server"]["reachable"] else "✗" if s["llama_server"]["alive"] else "·"
    daemon = "✓" if s["toki_daemon"]["reachable"] else "✗" if s["toki_daemon"]["alive"] else "·"
    lstyle = "ansigreen" if llama == "✓" else "ansired" if llama == "✗" else "ansigray"
    dstyle = "ansigreen" if daemon == "✓" else "ansired" if daemon == "✗" else "ansigray"
    return (
        f"<aaaaff><b>toki</b></aaaaff> "
        f"<{lstyle}>llama:{llama}</{lstyle}> "
        f"<{dstyle}>daemon:{daemon}</{dstyle}>"
    )
