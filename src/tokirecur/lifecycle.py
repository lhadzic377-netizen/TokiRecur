"""Process lifecycle for the local llama-server managed by `toki up`.

We keep it simple: a state directory under ~/.local/state/tokirecur
(override with TOKI_STATE_DIR) with one PID file per managed process,
and start/stop/status helpers. No systemd, no docker, no
daemons-of-daemons — just os.fork + signal handling.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

STATE_DIR = Path(os.environ.get("TOKI_STATE_DIR", str(Path.home() / ".local/state/tokirecur")))
LLAMA_PID = STATE_DIR / "llama-server.pid"
DAEMON_PID = STATE_DIR / "tokid.pid"
LLAMA_LOG = STATE_DIR / "llama-server.log"
DAEMON_LOG = STATE_DIR / "tokid.log"

# systemd sets INVOCATION_ID in the environment of every service process;
# use it to decide whether to detach spawned children. Detaching is the
# right call for `toki up` (the launcher should be able to exit without
# killing the stack) but wrong under systemd (the wrapper is the cgroup
# owner; detached children survive cgroup kills and become orphaned
# zombies when the wrapper is reaped).
_UNDER_SYSTEMD = bool(os.environ.get("INVOCATION_ID"))
_DETACH = not _UNDER_SYSTEMD


def _state_dir() -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR


def _read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except (ValueError, OSError):
        return None


def _is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def _wait_for(url: str, timeout: float = 180.0) -> bool:
    """Poll `url` until it returns 2xx or timeout elapses."""
    started = time.time()
    while time.time() - started < timeout:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def find_llama_server() -> str | None:
    """Locate the llama-server binary.

    Priority: the toki state bin dir (where `toki model install-llama`
    puts builds) first, then PATH, then the classic systemd install
    path. This makes `toki up` use the build we actually installed
    instead of whatever random llama-server happens to be on PATH.
    """
    from shutil import which

    state_bin = STATE_DIR / "bin"
    if state_bin.exists():
        candidate = state_bin / ("llama-server.exe" if os.name == "nt" else "llama-server")
        if candidate.exists():
            return str(candidate)
    return which("llama-server") or which("/usr/local/bin/llama-server")


def llama_default_args(
    model: str,
    port: int,
    ctx_size: int,
    n_gpu_layers: int,
    alias: str | None = None,
) -> list[str]:
    # Ensure required dirs exist before llama-server validates them.
    slot_save_dir = STATE_DIR / "slot-save"
    slot_save_dir.mkdir(parents=True, exist_ok=True)
    binary = find_llama_server() or "llama-server"
    args = [
        binary,
        "-m", model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "-c", str(ctx_size),
        "-ngl", str(n_gpu_layers),
        "--parallel", "4",
        "--slot-save-path", str(slot_save_dir),
        "--embedding",
    ]
    # The names a client may ask for. Without them the stack answers to
    # the model's file path, and a client that lists models — or matches on
    # the id it was configured with — is told the model does not exist.
    if alias:
        args += ["--alias", alias]
    return args


def llama_reachable(port: int = 8088) -> bool:
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def start_llama_server(
    *,
    model: str,
    port: int = 8088,
    ctx_size: int = 8192,
    n_gpu_layers: int = 99,
    extra_args: list[str] | None = None,
    wait: bool = True,
    alias: str | None = None,
) -> int:
    """Start llama-server. Returns the PID. If already running, returns
    the existing PID without restarting."""
    _state_dir()

    if llama_reachable(port):
        # Already running. Find the matching PID (best-effort).
        existing = _read_pid(LLAMA_PID)
        if existing and _is_alive(existing):
            logger.info("llama-server already running on :%d (pid=%d)", port, existing)
            return existing

    args = llama_default_args(model, port, ctx_size, n_gpu_layers, alias=alias)
    if extra_args:
        args.extend(extra_args)
    logger.info("starting: %s", " ".join(args))

    _state_dir()
    log = open(LLAMA_LOG, "ab")
    proc = subprocess.Popen(
        args,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=_DETACH,
    )
    LLAMA_PID.write_text(str(proc.pid))
    logger.info("llama-server pid=%d log=%s", proc.pid, LLAMA_LOG)

    if wait:
        if not _wait_for(f"http://127.0.0.1:{port}/health", timeout=180.0):
            raise RuntimeError(
                f"llama-server failed to start within 180s; see {LLAMA_LOG}"
            )
    return proc.pid


def start_daemon(
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    backend_url: str = "http://127.0.0.1:8088",
    similarity_threshold: float = 0.85,
    memory_budget_mb: int = 8192,
    reasoning: str = "auto",
    sampling: str | None = None,
    wait: bool = True,
) -> int:
    """Start the Toki Recur daemon (uvicorn). Returns the PID."""
    _state_dir()

    existing = _read_pid(DAEMON_PID)
    if existing and _is_alive(existing):
        logger.info("toki daemon already running (pid=%d)", existing)
        return existing

    args = [
        sys.executable, "-m", "uvicorn",
        "tokirecur.api.app:create_app_from_env",
        "--factory",
        "--host", host,
        "--port", str(port),
        "--log-level", "info",
    ]
    logger.info("starting: %s", " ".join(args))
    log = open(DAEMON_LOG, "ab")
    daemon_env = {
        **os.environ, "TOKI_BACKEND_URL": backend_url,
        "TOKI_SIMILARITY_THRESHOLD": str(similarity_threshold),
        "TOKI_MEMORY_BUDGET_MB": str(memory_budget_mb),
        "TOKI_REASONING": reasoning,
    }
    if sampling is not None:
        daemon_env["TOKI_SAMPLING"] = sampling
    proc = subprocess.Popen(
        args,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=daemon_env,
        start_new_session=_DETACH,
    )
    DAEMON_PID.write_text(str(proc.pid))
    logger.info("toki daemon pid=%d log=%s", proc.pid, DAEMON_LOG)

    if wait:
        if not _wait_for(f"http://127.0.0.1:{port}/healthz", timeout=60.0):
            raise RuntimeError(
                f"toki daemon failed to start; see {DAEMON_LOG}"
            )
    return proc.pid


def _signal_proc(pid: int, sig: int) -> None:
    """Send ``sig`` to the process — to its whole process group on POSIX,
    and to the single process on Windows (where ``os.killpg`` does not
    exist and the pid is the only handle we have)."""
    if hasattr(os, "killpg") and hasattr(os, "getpgid"):
        try:
            os.killpg(os.getpgid(pid), sig)
            return
        except OSError:
            pass
    try:
        os.kill(pid, sig)
    except OSError:
        pass


def _hard_kill(pid: int) -> None:
    """Best-effort hard kill: ``taskkill /F`` on Windows (where Python's
    ``signal`` module has no SIGKILL), ``os.kill(SIGKILL)`` on POSIX."""
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/pid", str(pid), "/f"],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        return
    _signal_proc(pid, signal.SIGKILL)


def stop_pid(pid_file: Path, name: str, grace: float = 5.0) -> bool:
    pid = _read_pid(pid_file)
    if pid is None:
        logger.info("%s: no pid file, nothing to stop", name)
        return False
    if not _is_alive(pid):
        logger.info("%s: pid=%d not alive, cleaning up", name, pid)
        pid_file.unlink(missing_ok=True)
        return False
    _signal_proc(pid, signal.SIGTERM)
    # Wait for graceful shutdown
    started = time.time()
    while time.time() - started < grace:
        if not _is_alive(pid):
            pid_file.unlink(missing_ok=True)
            logger.info("%s: stopped (pid=%d)", name, pid)
            return True
        time.sleep(0.2)
    _hard_kill(pid)
    logger.warning("%s: force-killed (pid=%d)", name, pid)
    pid_file.unlink(missing_ok=True)
    return True


def stop_all() -> None:
    stop_pid(DAEMON_PID, "toki daemon")
    stop_pid(LLAMA_PID, "llama-server")


def status() -> dict:
    """Return a dict describing the state of both processes."""
    llama_pid = _read_pid(LLAMA_PID)
    daemon_pid = _read_pid(DAEMON_PID)
    return {
        "llama_server": {
            "pid": llama_pid,
            "alive": _is_alive(llama_pid) if llama_pid else False,
            "reachable": llama_reachable(8088),
            "log": str(LLAMA_LOG),
        },
        "toki_daemon": {
            "pid": daemon_pid,
            "alive": _is_alive(daemon_pid) if daemon_pid else False,
            "reachable": _health_daemon(8787),
            "log": str(DAEMON_LOG),
        },
    }


def _health_daemon(port: int) -> bool:
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False
