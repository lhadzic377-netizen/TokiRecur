# Tokirecur

## Commands

```bash
.venv/bin/pytest -q --ignore=tests/integration   # unit tests (fast, no llama needed)
.venv/bin/pytest -q tests/unit/test_clichat.py   # Windows CLI chat tests
.venv/bin/ruff check src tests                   # lint (run --fix when fixing)
```

## Dev-machine gotchas

- The long-running daemon on this box is installed at `/opt/toki-recur/venv`
  and may point at a dead llama port (`--backend-url http://127.0.0.1:8088`).
  The live llama-server is on **8094** (`/home/lukehadzic/llama.cpp/build/bin/llama-server`).
- To smoke-test chat end-to-end without touching the running daemon, start a
  throwaway one from the repo:
  `nohup .venv/bin/toki serve --host 127.0.0.1 --port 8799 --backend-url http://127.0.0.1:8094 >/tmp/opencode/toki-serve-smoke.log 2>&1 &`
  then point `DaemonClient` at `http://127.0.0.1:8799` / raw `http://127.0.0.1:8094`.

## Windows fallback

- On Windows, the Textual TUIs are replaced by plain CLIs because the classic
  Windows console renders TUIs slowly and glitches:
  - bare `toki` → **CLI launcher menu** (`src/tokirecur/clilauncher.py`,
    ollama-style numbered app pick; Chat runs the CLI chat REPL, agents launch
    in new consoles, web opens dashboards). Wired in `src/tokirecur/selector.py`.
  - `toki tui` → **CLI chat** (`src/tokirecur/clichat.py`, ollama-style REPL,
    `/bye`, `"""` multiline, `TOKI_MAX_TOKENS`). Wired in
    `src/tokirecur/cli/__init__.py` (`tui` command).
- Linux/macOS keep the full Textual TUI + launcher.
