"""systemd plug-and-play management for the model stacks.

`toki service install` writes a `toki-model@.service` template unit,
enables boot units for every registry model marked ``boot: true``,
and enables user linger so the stacks come up at boot without a login
session. Ordering drop-ins chain the units in registry order so the
last always-on model (by default: laguna) ends up as the active model
the clients point at.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .ctl import load_registry

UNIT_DIR = Path.home() / ".config" / "systemd" / "user"
UNIT_NAME = "toki-model@.service"


def _systemd_available() -> bool:
    """systemd user units exist only on Linux with systemd installed."""
    return sys.platform.startswith("linux") and shutil.which("systemctl") is not None


def _resolve_token() -> str | None:
    """Same read order as the daemon: env var, then auth files."""
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


UNIT_TEMPLATE = """\
[Unit]
Description=Toki model %i (llama-server + daemon)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=%h/.local/bin/toki model use %i --keep-others --wait
Restart=on-failure
RestartSec=15
StartLimitBurst=5
TimeoutStartSec=1800
Environment=HOME=%h
Environment=PATH=/usr/local/bin:/usr/bin:%h/.local/bin
{auth_line}
[Install]
WantedBy=default.target
"""


def unit_text() -> str:
    token = _resolve_token()
    if token is None:
        print(
            "warning: no auth token configured; the daemon accepts "
            "loopback requests only. Set TOKI_AUTH_TOKEN (or "
            "~/.config/tokirecur/auth) before exposing it beyond localhost.",
            file=sys.stderr,
        )
    auth_line = f"Environment=TOKI_AUTH_TOKEN={token}\n" if token else ""
    return UNIT_TEMPLATE.format(auth_line=auth_line)


def _systemctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True)


def boot_models() -> list:
    return [spec for spec in load_registry() if spec.boot]


def drop_in_dir(instance: str) -> Path:
    return UNIT_DIR / f"toki-model@{instance}.service.d"


def install() -> int:
    models = boot_models()
    if not models:
        print("no models marked boot in the registry; nothing to install.")
        return 1
    UNIT_DIR.mkdir(parents=True, exist_ok=True)
    unit_path = UNIT_DIR / UNIT_NAME
    tmp = unit_path.with_name(unit_path.name + ".tmp")
    tmp.write_text(unit_text())
    os.replace(tmp, unit_path)
    print(f"wrote {unit_path}")

    prev: str | None = None
    for spec in models:
        if prev is not None:
            d = drop_in_dir(spec.name)
            d.mkdir(parents=True, exist_ok=True)
            ordering = d / "ordering.conf"
            tmp = ordering.with_name(ordering.name + ".tmp")
            tmp.write_text(
                f"[Unit]\nAfter=toki-model@{prev}.service\n"
            )
            os.replace(tmp, ordering)
            print(f"wrote {ordering}")
        prev = spec.name

    _systemctl("daemon-reload")
    for spec in models:
        unit = f"toki-model@{spec.name}.service"
        _systemctl("enable", unit)
        _systemctl("start", unit)
        print(f"enabled+started {unit}")
    proc = subprocess.run(["loginctl", "enable-linger", os.environ["USER"]],
                          capture_output=True, text=True)
    if proc.returncode == 0:
        print("linger enabled (stacks start at boot without login)")
    else:
        print(f"warning: could not enable linger: {proc.stderr.strip()}")
    print("next reboot will start:", ", ".join(s.name for s in models))
    return 0


def uninstall() -> int:
    models = boot_models()
    for spec in models:
        unit = f"toki-model@{spec.name}.service"
        _systemctl("disable", "--now", unit)
        print(f"disabled {unit}")
    for unit in UNIT_DIR.glob("toki-model@*.service"):
        unit.unlink(missing_ok=True)
    for d in UNIT_DIR.glob("toki-model@*.service.d"):
        for f in d.glob("*"):
            f.unlink(missing_ok=True)
        d.rmdir()
    _systemctl("daemon-reload")
    print("removed toki-model units")
    return 0


def status() -> int:
    proc = subprocess.run(["loginctl", "show-user", os.environ["USER"], "-p", "Linger"],
                          capture_output=True, text=True)
    linger = "on" if "yes" in proc.stdout else "off"
    print(f"linger: {linger}")
    models = boot_models()
    if not models:
        print("no boot models in the registry.")
        return 0
    print(f"{'unit':38s} {'state'}")
    for spec in models:
        unit = f"toki-model@{spec.name}.service"
        r = _systemctl("is-active", unit)
        state = r.stdout.strip() or "inactive"
        marker = "*" if Path(UNIT_DIR / unit).exists() else " "
        print(f"{marker} {unit:37s} {state}")
    return 0


def logs(name: str) -> int:
    import shutil
    if not shutil.which("journalctl"):
        print("journalctl not found", file=sys.stderr)
        return 1
    os.execvp("journalctl", ["journalctl", "--user", "-u", f"toki-model@{name}.service", "-f"])


def run(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("usage: toki service {install|uninstall|status|logs <model>}")
        return 0
    if not _systemd_available():
        print(
            "systemd user services are not available on this platform "
            "(Linux + systemctl required); nothing to do.",
            file=sys.stderr,
        )
        return 1
    cmd, *rest = argv
    if cmd == "install":
        return install()
    if cmd == "uninstall":
        return uninstall()
    if cmd == "status":
        return status()
    if cmd == "logs" and rest:
        return logs(rest[0])
    print("usage: toki service {install|uninstall|status|logs <model>}", file=sys.stderr)
    return 2
