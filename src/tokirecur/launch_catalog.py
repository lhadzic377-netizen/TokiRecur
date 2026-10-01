from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LaunchApp:
    key: str
    title: str
    description: str
    interface: str
    client: str
    model_binding: bool = True
    command: str = ""
    args: tuple[str, ...] = ()
    required_path: str | None = None
    # "openai" reads the bound daemon from the environment; "anthropic" is
    # Claude Code, which has its own settings file. Account-provider clients
    # used to be listed here and were removed; nothing sets "account" now.
    protocol: str = "openai"


LAUNCH_APPS: tuple[LaunchApp, ...] = (
    LaunchApp(
        "toki", "Chat", "Chat with models in the Toki TUI", "chat", "toki", command="toki"
    ),
    LaunchApp(
        "ornithcode",
        "Launch OrnithCode",
        "Ornith-native terminal coding agent with Flock",
        "chat",
        "ornithcode",
        command="ornithcode",
    ),
    LaunchApp(
        "opencode",
        "Launch OpenCode",
        "Anomaly's open-source coding agent",
        "chat",
        "opencode",
        command="opencode",
    ),
    LaunchApp(
        "hermes",
        "Launch Hermes Agent",
        "Self-improving AI agent built by Nous Research",
        "chat",
        "hermes",
        command="hermes",
    ),
    LaunchApp(
        "openclaw",
        "Launch OpenClaw",
        "Personal AI with 100+ skills",
        "chat",
        "openclaw",
        command="openclaw",
    ),
    LaunchApp(
        "codex",
        "Launch Codex",
        "OpenAI CLI coding agent (Responses API)",
        "chat",
        "codex",
        command="codex",
    ),
    LaunchApp(
        "mimo",
        "Launch MiMo Code",
        "Xiaomi's OpenCode-compatible coding agent",
        "chat",
        "mimo",
        command="mimo",
    ),
    LaunchApp(
        "qwen", "Launch Qwen Code", "Qwen's terminal coding agent", "chat", "qwen",
        command="qwen",
    ),
    LaunchApp(
        "droid",
        "Launch Factory Droid",
        "Factory's AI coding agent in your terminal",
        "chat",
        "droid",
        command="droid",
    ),
    LaunchApp(
        "cline",
        "Launch Cline CLI",
        "Cline's terminal coding assistant",
        "chat",
        "cline",
        command="cline",
        args=("--tui",),
    ),
    LaunchApp(
        "dsh",
        "Launch DeepSeek Harness",
        "DeepSeek Harness terminal agent",
        "chat",
        "dsh",
        command="dsh",
        args=("--profile", "tui"),
        required_path="~/.dsh/profiles/tui",
    ),
    LaunchApp(
        "grok",
        "Launch Grok Build",
        "xAI's terminal coding agent",
        "chat",
        "grok",
        command="grok",
    ),
    LaunchApp(
        "kilo",
        "Launch Kilo Code",
        "Kilo's terminal coding agent",
        "chat",
        "kilo",
        command="kilo",
    ),
    LaunchApp("pi", "Launch Pi", "Pi coding agent", "chat", "pi", command="pi"),
    LaunchApp(
        "copilot",
        "Launch GitHub Copilot",
        "GitHub Copilot CLI",
        "chat",
        "copilot",
        command="copilot",
    ),
    LaunchApp(
        "claude",
        "Launch Claude Code",
        "Anthropic's terminal coding agent (local Anthropic-compatible API)",
        "chat",
        "claude",
        model_binding=True,
        command="claude",
        protocol="anthropic",
    ),
    LaunchApp(
        "kimi",
        "Launch Kimi Code",
        "Moonshot AI's terminal coding agent",
        "chat",
        "kimi",
        command="kimi",
    ),
    LaunchApp("omp", "Launch OMP", "Oh My Pi coding agent", "chat", "omp", command="omp"),
    LaunchApp(
        "web",
        "Launch Web Dashboards",
        "browser UIs: toki · openclaw · hermes",
        "web",
        "toki",
        False,
        "toki",
    ),
)


_BY_CLIENT = {app.client: app for app in LAUNCH_APPS if app.interface != "web"}


def get_launch_app(client: str) -> LaunchApp | None:
    return _BY_CLIENT.get(client)


def configured_provider_model(client: str) -> str | None:
    if client != "claude":
        return None
    keys = (
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    )
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    try:
        settings = json.loads(
            (Path.home() / ".claude" / "settings.json").read_text()
        )
    except (OSError, TypeError, ValueError):
        return None
    env = settings.get("env") if isinstance(settings, dict) else None
    if not isinstance(env, dict):
        return None
    for key in keys:
        value = env.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def unavailable_reason(app: LaunchApp) -> str | None:
    if app.interface == "web":
        return None
    command = app.command or app.client
    if shutil.which(command) is None:
        return f"{command} not installed"
    if app.required_path:
        required = app.required_path
        if app.client == "dsh":
            required = str(
                Path(os.environ.get("DSH_HOME", "~/.dsh")).expanduser()
                / "profiles" / "tui"
            )
        path = Path(required).expanduser()
        if not path.exists():
            return "required profile not installed"
    return None
