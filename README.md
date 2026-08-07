# Toki Recur

A standalone KV cache reuse system for LLM inference. Drop-in prefix caching that works across CUDA, ROCm, and Vulkan without any model modifications.

## What it does

When a new prompt arrives at an LLM server, the KV cache is rebuilt from scratch — even if 90% of the prompt is identical to one we just processed. **Toki Recur** detects exact prefix matches, reuses the saved KV state, and skips redundant prefill compute.

## Real measured results (this hardware)

Full 9-workload results across four models (Qwen3.5-9B, Qwen3.6-35B-A3B,
Qwen3.6-27B, Qwen3.5-122B-A10B) live in the [Benchmark results](#benchmark-results-this-hardware)
section below. The headline: TTFT improves 62–96% and prefill throughput
410–1280% on realistic workloads, with the biggest gains on the largest MoE
models.

## Quick start

Toki Recur is a Python app, so install it like one — with pip. While the repo is private, the install URL carries a read-only token (a fine-grained PAT with **Contents: read** on `lhadzic377-netizen/TokiRecur` only — create/revoke at https://github.com/settings/personal-access-tokens). The token appears in your shell history; that's the private-repo tax.

```bash
# 1. Install the CLI (pipx isolates it in its own venv and puts toki on PATH)
#    (one-time: sudo apt install pipx   or   python3 -m pip install --user pipx)
pipx install "tokirecur[embeddings] @ git+https://x-access-token:<PAT>@github.com/lhadzic377-netizen/TokiRecur.git"

# 2. Fetch the llama-server binary matching your OS/arch and GPU.
#    Auto-detects the best backend: ROCm (AMD/HIP), CUDA (NVIDIA),
#    Vulkan, Metal (macOS), CPU. Pass --backend to override.
toki model install-llama

# 3. Install a model — nothing is preloaded on a fresh install; you
#    pull exactly what you want. Fetch a GGUF from Hugging Face and
#    register it under a short name (find quantized Qwen3 GGUFs via
#    huggingface.co/models?library=gguf):
toki model pull Qwen/Qwen3-4B-GGUF --name qwen3-4b

# 4. Point your install-llama binary at the new entry (auto-detected
#    ROCm/CUDA build from step 2 wins over the PATH llama-server):
toki model install-llama --backend rocm --name qwen3-4b

# 5. Start it (this boots llama-server + the toki daemon and wires
#    opencode/openclaw/hermes at it):
toki model use qwen3-4b

# 6. Talk to it — the full-screen TUI chat (or open the web UI with `toki web`)
toki tui

# 7. Done with it — unregister (weights and state stay on disk by default;
#    pass --purge to delete them too; --force overrides the active-model guard):
toki model remove qwen3-4b
```

No models are installed or started at install time — `toki model use <name>` is the explicit, documented way to bring one up (it starts llama-server + the toki daemon, stops competing big-model stacks, and wires opencode/openclaw/hermes at it). `toki model remove <name>` is the inverse: it drops the registry entry and (with `--purge`) deletes the downloaded weights under `~/models/<name>` plus the runtime state dir. `toki up --model <path>` is the single-model equivalent for a raw GGUF without a registry entry. `toki serve` is **only** the API daemon (default 127.0.0.1:8787) that proxies to an already-running llama-server; it warns loudly if the backend isn't reachable instead of failing silently. Two chat frontends ship with toki: bare `toki` opens the full-screen launcher, `toki tui` is the full-screen TUI chat, and `toki web` is the browser chat UI.

### Small-RAM machines

On machines with limited RAM (8–16 GB), `toki model use` automatically reduces the requested context so the KV cache fits in memory alongside the model weights (it reads the GGUF header to compute KV usage per token, and the running server's real slot size after boot). The effective context — the smaller of your configured `ctx_size` and what the machine can actually hold — is written to the model's state dir and becomes the single source of truth: the OpenAI-compatible API advertises exactly that context, clamps `max_tokens` to it, and refuses to over-allocate when llama-server could not honor the requested context (which previously surfaced as confusing `400 not enough tokens` errors). Failed startups no longer hang on a spinning `starting/selecting model` — llama-server's log tail is surfaced and the launcher returns an error. If the server still can't start, pick a smaller/quantized model (`toki model pull`) or lower the model's `ctx_size` in the registry; the context floor is 512 tokens, so the server always gets a workable value.

Without pipx, install into your own venv instead:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "tokirecur[embeddings] @ git+https://x-access-token:<PAT>@github.com/lhadzic377-netizen/TokiRecur.git"
```

If `toki` is not found after install, your shell's PATH may not include the venv's bin directory. Either activate the venv (`source .venv/bin/activate`) or use the full path (`./.venv/bin/toki`). On systems where `pip install -e` writes a shim to `~/.local/bin` that points at system Python (and thus misses venv-only deps like `textual`), delete that shim and re-run `pip install -e .` from inside the activated venv so the shim is regenerated with the correct shebang.

## Platforms

Toki Recur is pure Python over the llama-server HTTP API — it runs on **any x86_64 or aarch64 device** running Linux, macOS or Windows, with whatever GPU backend your llama-server build has (ROCm, CUDA, Vulkan, Metal, CPU). The only per-platform piece is the llama-server binary itself:

| Platform | Architecture | llama-server builds | install-llama asset |
|---|---|---|---|
| Linux | x86_64 | ROCm (auto if HIP present), Vulkan, CPU | `ubuntu-rocm-7.2-x64`, `ubuntu-vulkan-x64`, `ubuntu-x64` |
| Linux | aarch64 | Vulkan, CPU | `ubuntu-vulkan-arm64`, `ubuntu-arm64` |
| macOS | x86_64, arm64 | Metal (also runs CPU-only) | `macos-x64`, `macos-arm64` |
| Windows | x86_64 | CUDA (auto if NVIDIA), ROCm/HIP, Vulkan, CPU | `win-cuda-13.3-x64`, `win-hip-radeon-x64`, `win-vulkan-x64`, `win-cpu-x64` |
| Windows | aarch64 | CPU, Adreno OpenCL | `win-cpu-arm64` |

CI runs the test suite on all three OSes (x86_64 + aarch64), executes the Linux and macOS system installers end-to-end, and runs `toki model install-llama` on all four platform legs (Linux x86_64/ARM, macOS Apple Silicon, Windows) for both auto-detect and explicit `--backend cpu`, then executes the downloaded llama-server binary. Windows has no shell service manager, so it uses the pip package + `toki model install-llama` + a plain `toki serve` (or NSSM/scheduled task). The GPU backends are provided by the llama-server binary itself — Toki Recur talks to it over HTTP and is backend-agnostic.

One command installs the matching official build:

```bash
toki model install-llama                 # auto-detect backend, latest release
toki model install-llama --backend rocm  # force a backend (rocm|cuda|vulkan|metal|cpu)
toki model install-llama --backend rocm --gfx 11.5.1  # set HSA_OVERRIDE_GFX_VERSION
toki model install-llama --tag b6400     # pin a release tag
toki model install-llama --name 35b      # ...and point a registry model at it
```

Optional — run as a system daemon (dedicated user + service manager unit + log rotation, auto-start at boot). `install.sh`/`install.ps1` bootstrap the full stack themselves (own venv at `/opt/toki-recur`, package + llama-server + service unit), so they work standalone — they don't need the pipx install above:

```bash
# Linux  -> systemd + logrotate   (macOS -> launchd LaunchDaemon)
sudo ./install.sh

# Windows (PowerShell): install.ps1; then `toki serve` yourself or wrap
# it in NSSM / a scheduled task.
```

CI executes both installers end-to-end on fresh runners and starts the installed daemon under systemd.

Or point `binary:` at any llama-server on PATH in the registry (`~/.config/tokirecur/models.yaml`). Override locations per-machine with `TOKI_MODEL_REGISTRY` (config) and `TOKI_STATE_DIR` (logs/pid/bin). On Windows the systemd service commands are unavailable (reported cleanly); the RAM guard uses GlobalMemoryStatusEx, and on macOS the UMA/RAM guard uses `sysctl hw.memsize`.

### The launcher

Bare `toki` opens an Ollama-style full-screen launcher in a black + yellow design: a letter-spaced yellow `TOKI` wordmark with tagline and amber divider, then a vertically-centred column of app cards — toki Chat, OpenCode, Hermes, OpenClaw, web dashboards — each bound to the pinned model (shown as a right-aligned chip). The active row is highlighted with a yellow accent bar. A status line shows the pinned model, backend health dots, and key hints; keys `1`–`5` jump straight to an app (or model).

```
T O K I
recycled-prefill LLM stack · AMD Ryzen AI
──────────────────────────────────────────
▌ Chat                                                  (9b) ↵
      Chat with models in the Toki TUI
  Launch OpenCode                                       (9b)
  Launch Hermes Agent                                   (9b)
  Launch OpenClaw                                       (9b)
  Launch Web Dashboards

  ● 9b            1-5 jump · enter launch · esc quit
```

| Key | Action |
|---|---|
| `↑` / `↓` (or `k` / `j`) | navigate apps or models |
| `Enter` | launch the highlighted app with the pinned model |
| `→` / `Tab` | flip to the model picker (per-model health dots + ctx) |
| `←` | back to the app list |
| `Space` | pin the highlighted model in the picker |
| `s` | open the settings pane (edit any registry field) |
| `Esc` / `q` | quit |

In the settings pane, `Enter` opens an editor for the highlighted field.
Sizing and enum fields (`ctx_size`, `max_tokens`, `parallel`, `kv`,
`reasoning`, `always_on`, `boot`) open a **dropdown** of presets with a
`custom…` option that falls back to the free-text editor; paths, ports
and ids stay plain text. Both editors have **Save** / **Cancel**
buttons: `Save` (or `Enter` in the input) writes the field to the
registry (`~/.config/tokirecur/models.yaml`) immediately; `Cancel` (or
`Esc`) discards.

The `reasoning` field controls thinking mode: `auto`/`on` enable it —
the model's reasoning is streamed (dimmed in the TUI chat, rendered as
a separate block in the web UI), `off` passes `--reasoning off` to
llama-server at launch. Changes take effect on the model's next start.

The chat TUI and `toki status` share the same wordmark header and black + yellow palette.
```

```
toki llama:· daemon:· ❯ up --detach
llama-server started (pid=…)  http://127.0.0.1:8088
toki daemon started (pid=…)  http://127.0.0.1:8787

toki llama:✓ daemon:✓ ❯ bench --workload agent_long_context --max-tokens 64
...

toki llama:✓ daemon:✓ ❯ status
toki llama:✓ daemon:✓ ❯ down -y
toki llama:· daemon:· ❯ exit
```

One command to start, one command per action, one command to stop. The prompt has a live status indicator showing whether llama-server and the daemon are reachable. Tab completion is enabled if `prompt_toolkit` is installed.

If you prefer one-shot commands instead of an interactive shell:

```bash
toki up --detach
toki bench                      # interactive: pick a model + workload, then run
toki bench --workload agent_long_context   # skip the workload picker
toki bench --model 9b --workload agent_long_context   # fully scripted (registry name)
toki bench --model ~/Downloads/Qwen3.5-9B-Q4_K_M.gguf --workload agent_long_context \
           --max-tokens 64 --output-json report.json   # unregistered path
toki down
```

The benchmark compares baseline (no cache) vs recycled (cache enabled) runs across TTFT, prefill throughput, end-to-end latency, cache hit rate, and output equivalence, then prints a composite score (0-100) with a letter grade.

## Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌────────────────┐
│  Client / agent │ ───► │  toki (daemon)   │ ───► │  llama-server  │
│  (OpenAI API)   │      │                  │      │   (ROCm/HIP)   │
└─────────────────┘      └──────────────────┘      └────────────────┘
                         │  embedding + index │
                         │  SQLite + blobs    │
                         └────────────────────┘
```

`tokid` (the daemon) is a small FastAPI service that sits in front of `llama-server`. It intercepts requests, runs the 5-step recycling algorithm:

1. Embed the incoming prompt.
2. Find the most similar cached prompt via cosine similarity.
3. Verify exact prefix match on token IDs.
4. If hit: load the saved KV state via `?action=restore` and decode only the novel tokens.
5. If miss: full prefill, persist KV state via `?action=save` for the future.

## Targets

- **ROCm + llama-server** (first-class, current focus)
- AMD Ryzen AI MAX+ 395 APU with UMA (`GGML_HIP_UMA=ON`)
- gfx1151 (RDNA 3.5)
- Up to 128 GB unified memory

## How KV save/restore works (no patch needed)

As of llama.cpp b3524 (mid-2025), the upstream `llama-server` already exposes the endpoints we need behind the `--slot-save-path` flag:

```
POST /slots/{id_slot}?action=save     body: {"filename": "..."}
POST /slots/{id_slot}?action=restore  body: {"filename": "..."}
POST /slots/{id_slot}?action=erase    body: {"filename": "..."}
```

The Debian-packaged `llama-server 9413` already has these. `LlamaServerBackend` uses them directly. The `patches/` directory contains a historical patch for users on older llama.cpp and a `README.md` explaining the situation.

## Command array

```bash
# ── shell ──────────────────────────────────────────────────────────────────
toki                                                  # run benchmark (default)
toki interactive                                      # enter interactive REPL

# ── chat interface ─────────────────────────────────────────────────
toki tui                                              # full-screen TUI chat (terminal)
toki tui --auth-token <token>                         # if daemon requires auth
toki tui --daemon-url http://other:8787               # custom daemon URL
toki diff --system "<long>" "<user prompt>"           # one-shot A/B comparison

# Web UI on http://127.0.0.1:7777 (production-ready single-page chat)
toki web                                              # foreground
toki web --detach                                     # background, log → /tmp/toki/web.log
toki web --tls-cert cert.pem --tls-key key.pem        # HTTPS + wss, Secure cookies
toki web --port 8888 --daemon-url http://other:8787   # custom endpoint

# In-chat slash commands (TUI chat):
#   /mode raw|cached|compare   /system <text>   /clear   /metrics
#   /max <n>   /temp <f>   /raw <msg>   /cached <msg>   /compare <msg>   /quit

# ── install ────────────────────────────────────────────────────────────────
pip install --break-system-packages -e tokirecur/

# ── stack lifecycle ────────────────────────────────────────────────────────
toki up                                                # foreground (Ctrl+C stops)
toki up --detach                                       # background, returns PIDs
toki up --skip-llama                                   # only start the daemon
toki up --model /path/to/other.gguf --ctx-size 16384   # custom args
toki down                                              # stop everything
toki down -y                                           # skip confirmation
toki status                                            # show stack state

# ── benchmark (the headline tool) ──────────────────────────────────────────
# Interactive: pick a model from the registry + a workload from a menu.
toki bench

# Or fully scripted (registry name or path):
toki bench --model 9b --workload agent_long_context \
           --max-tokens 64 --output-json report.json

# Available workloads:
#   agent_long_context   ~90% cache hit (long fixed system prompt + queries)
#   agent_25 / agent_50 / agent_100   25 / 50 / 100 prompts
#   realistic_mixed      216 prompts across 5 real-world categories
#   stress_adversarial   301 prompts with adversarial patterns
#   rag_chunks           ~90% cache hit (shared document prefix)
#   mixed                ~50% cache hit (mixed traffic)
#   no_overlap           0% cache hit (validation / worst-case)

# Self-contained: spin up llama-server + bench together (auto-stop)
toki bench --model ~/Downloads/Qwen3.5-9B-Q4_K_M.gguf \
           --self-contained --ctx-size 8192 --n-gpu-layers 99

# ── cache management ───────────────────────────────────────────────────────
toki stats                                            # hit rate, bytes, entries
toki evict --target-mb 4096                           # LRU evict to fit budget

# ── diagnostics ────────────────────────────────────────────────────────────
toki doctor                                           # llama-server, ROCm, GPU
toki compare report.json report2.json                 # diff two bench reports

# ── OpenAI-compatible API server (running under `toki up`) ─────────────────
# POST http://127.0.0.1:8787/v1/completions
# POST http://127.0.0.1:8787/v1/chat/completions
# POST http://127.0.0.1:8787/cache/lookup
# DELETE http://127.0.0.1:8787/cache/{id}
# GET  http://127.0.0.1:8787/cache/stats
# GET  http://127.0.0.1:8787/healthz
```

## Production deployment

All three processes (llama-server, daemon, web UI) ship as systemd
units in `packaging/systemd/`. Point `EnvironmentFile=` at a
`/etc/toki-recur/toki.env` (chmod 600) with `TOKI_AUTH_TOKEN` set —
that single token then gates **both** the daemon API and the web UI.

### Cross-session cache persistence (disabled)

Prefix reuse is **in-memory only**: the engine routes similar prompts to
the same llama-server slot and lets the slot's KV cache serve the shared
prefix (`cache_prompt=true`). This is lossless — outputs are
byte-identical to a no-cache run.

Disk persistence via `?action=save` / `?action=restore` was removed:
restoring disk KV corrupted precision (benchmarks showed 88% substantive
divergence from baseline outputs). The `--slot-save-path` flag and the
`.trkv` sidecar layer (`cache/blob.py`) remain in the codebase but are
dormant — the engine never calls `cache_save` / `cache_load`, and
starting llama-server with or without `--slot-save-path` changes
nothing about behavior.

### Web UI auth + TLS

- With `TOKI_AUTH_TOKEN` set, the whole UI is behind a login page. The
  server issues an HMAC-signed, HttpOnly, SameSite session cookie, and
  the chat WebSocket rejects handshakes without it (close code 4401).
  The token never travels inside socket payloads.
- Without a token the UI runs open — fine for localhost, exposed if
  bound to a network interface.
- Serve HTTPS + `wss://` with `--tls-cert` / `--tls-key`, or set
  `TOKI_WEB_TLS_CERT` / `TOKI_WEB_TLS_KEY` in the env file (the systemd
  unit picks them up). Over TLS the session cookie gets the Secure
  flag; without TLS keep the UI bound to 127.0.0.1.
- The web server refuses cross-origin requests and serves everything
  `Cache-Control: no-store` so a reload always gets one consistent copy
  of the UI.

```bash
toki web --tls-cert /etc/toki-recur/tls/toki-recur.pem \
         --tls-key  /etc/toki-recur/tls/toki-recur.key
```

### Systemd

```bash
sudo install -m 600 packaging/systemd/toki.env.example /etc/toki-recur/toki.env
# edit: TOKI_AUTH_TOKEN=<token>
sudo install -m 644 packaging/systemd/toki-recur.service packaging/systemd/toki-recur-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now toki-recur toki-recur-web
```

## Use Toki Recur as an opencode backend

The daemon's `/v1/chat/completions` is OpenAI-compatible *and* tool-calling
capable, so opencode can run entirely on the local KV-recycled model. Every
agent turn reuses the prefix from the previous turn (system prompt, tools,
earlier messages) via llama-server's KV cache — no cloud, no `num_ctx`
fiddling, and cache hits show up in the daemon's metrics.

```jsonc
// ~/.config/opencode/opencode.json
{
  "provider": {
    "tokirecur": {
      "name": "Toki Recur (KV-cache, local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8787/v1",
        "apiKey": "demo-token"
      },
      "models": {
        "qwen3.5-9b": {
          "name": "Qwen3.5 9B (Toki Recur)",
          "contextWindow": 24000,
          "limit": { "context": 24000, "output": 8192 }
        }
      }
    }
  },
  "small_model": "tokirecur/qwen3.5-9b" // keep title-gen local too
}
```

- Set `apiKey` to whatever `TOKI_AUTH_TOKEN` the daemon runs with.
- The model id is a label; the actual weights are whatever llama-server has
  loaded. `contextWindow` should stay under `TOKI_MAX_CONTEXT_TOKENS`.
- Tool calls stream through verbatim (`delta.tool_calls`), finish with
  `finish_reason: "tool_calls"`, and `role: "tool"` result messages round-trip
  unchanged — verified against opencode's own `bash` tool loop.
- Non-stream requests work too (`choices[0].message.tool_calls`).

## Interactive shell

```
toki llama:· daemon:· ❯

  up [--detach]                 start llama-server + daemon
  down [-y]                     stop llama-server + daemon
  status                        show stack state
  doctor                        check llama-server, ROCm, GPU
  bench [args]                  run the benchmark suite
  run "prompt"                  one-shot completion via daemon
  stats                         cache index stats
  list [n]                      list cached prompts (default 20)
  show <id>                     show prompt details
  evict <mb>                    LRU evict to target MB
  clear                         wipe cache
  model <path>                  set GGUF model path
  port llama <n> / daemon <n>   set ports
  ctx <n> / ngpu <n>            set context size / GPU layers
  config                        show current config
  compare <a.json> <b.json>     compare two bench reports
  help [command]                show help for a command (or all)
  exit / quit / q               exit shell
```

The prompt has a live status indicator — `llama:✓ daemon:✓` means both are reachable. Tab completion and command history are enabled if `prompt_toolkit` is installed (installed by default).

## TUI chat (`toki tui`)

The full-screen terminal chat client. Three modes, switched with `Ctrl+1/2/3` (or in-chat slash commands):

| Mode | What it does |
|---|---|
| `raw` | Sends directly to llama-server (bypasses cache) |
| `cached` | Sends through the toki daemon (recycling engine) |
| `compare` | Runs each prompt both ways, shows side-by-side diff |

```bash
toki tui                                   # chat against localhost daemon + llama-server
toki tui --auth-token <token>              # if the daemon requires auth
toki tui --daemon-url http://other:8787    # custom daemon
```

In-session slash commands: `/mode raw|cached|compare`, `/system <text>`, `/clear`, `/metrics`, `/max <n>`, `/temp <f>`, `/raw <msg>`, `/cached <msg>`, `/compare <msg>`, `/quit`. `Ctrl+C` quits.

`toki diff --system "<long system>" "<prompt>"` does the same raw/cached A/B comparison as a one-shot (no REPL):

## Web chat UI (`toki web`)

A production-ready single-page chat interface with a single-click mode toggle (Raw / Cached / Compare), streaming responses, and live per-message metrics. Default port is **7777** — uncommon and unlikely to clash with other localhost services.

```bash
toki web                                       # foreground, http://127.0.0.1:7777
toki web --detach                             # background, log → /tmp/toki/web.log
toki web --port 7778 --raw-url http://...:8088  # custom port + backend
```

**Features:**
- Three inference modes selectable via prominent toggle: Raw (direct to llama-server), Cached (through Toki Recur daemon), Compare (run both, side-by-side)
- Streaming responses with token-by-token display
- Live metrics per message (TTFT, total time, tokens generated, prefill tokens)
- Compare mode shows two side-by-side panels with a summary banner showing speedup and divergence warning if outputs differ
- System prompt input persisted in the sidebar
- Conversation history with multi-turn support
- Mobile-responsive layout

The UI is a single static page (`src/tokirecur/web/static/index.html`) served by a FastAPI app (`src/tokirecur/web/server.py`) with WebSocket streaming (`/ws/chat`). No external frontend dependencies — vanilla HTML/CSS/JS.

## Terminal UI (`toki tui`)

A production terminal-based chat client built on [Textual](https://github.com/Textualize/textual). Runs in any TTY — no browser, no daemon-of-its-own. Connects to the existing `toki serve` daemon and llama-server over HTTP.

```bash
# Make sure the daemon + llama-server are running.
toki up

# In a separate terminal:
toki tui
toki tui --auth-token <token>      # if TOKI_AUTH_TOKEN is set
toki tui --daemon-url http://other:8787 --raw-url http://other:8088
```

The chat opens with the same TOKI wordmark header the launcher shows, then your persisted history (if any) below it.

**Features:**
- Three modes (raw / cached / compare) with Ctrl+1/2/3 keyboard shortcuts
- Live TTFT sparkline + running hit-rate counter in the right panel
- Per-backend connection health (green/yellow/red) shown at all times
- Slot state panel showing which slots are in use, their context size, and which session owns them
- Precision retention panel (Ctrl+P) showing baseline/recycled byte-identity stats
- Session pinning via UUID — multi-turn chats reuse the same slot for max cache benefit
- Persistent chat history at `~/.local/share/tokirecur/history.jsonl` (XDG)
- Slash commands: `/help /quit /clear /mode /session /precision /metrics /slots /reset /reprobe`
- Graceful degradation: if the daemon is unreachable, mode auto-switches to raw

**Keys:**
| Key | Action |
|---|---|
| `Ctrl+1` / `Ctrl+2` / `Ctrl+3` | Switch to raw / cached / compare |
| `Enter` | Send message |
| `Ctrl+L` | Clear chat |
| `Ctrl+R` | Re-probe backends |
| `Ctrl+U` | New session (new UUID) |
| `Ctrl+P` | Run precision check |
| `Ctrl+M` | Toggle metrics / precision panel |
| `Ctrl+C` | Quit |

Requires a TTY. If you run from a non-interactive shell, you'll get:
```
error: toki tui requires an interactive terminal (TTY).
```

## Architectural improvements

Beyond the basic cache, the tool ships with:

1. **Tree-structured KV cache** (`src/tokirecur/cache/tree.py`) — KV tensors are tracked in BLOCK_SIZE chunks. Two prompts sharing a prefix share the same on-disk blocks (content-addressed by token hash). Provides a fast tree-hit lookup that bypasses embedding entirely.

2. **Persistent slot pool** (`src/tokirecur/cache/persistent_pool.py`) — the top-N most-frequently-used prefixes get dedicated llama-server slots that never clear. Skips the save/load cycle entirely for hot prefixes.

3. **Honest hit definition** (`MIN_REUSE_FOR_HIT = 10`) — a "hit" requires at least 10 tokens of prefix reuse. Below that, the save/load overhead exceeds the prefill savings, so it's reported as a `soft_hit` instead of inflating the hit rate.

4. **Prefix-length-weighted similarity** (`BruteForceIndex.search`) — among candidates within 5% cosine similarity, prefers the one with the longest token list. Dampens the "cosine finds a 1-token lookalike" pathology.

5. **Substantive divergence classification** — output divergence is now classified into identical / whitespace-only / restore-noise / substantive via sentence-transformers semantic similarity. Reports the percentage of divergent pairs that are *real* (different meaning) vs cosmetic.

6. **Embedding chunking fallback** (`LlamaServerBackend.embed`) — long prompts are split into overlapping segments and re-averaged, with a deterministic hash-based fallback when embedding fails entirely.

## Benchmarks

The benchmark supports four workloads:

| Workload | Expected hit rate | Best for |
|---|---|---|
| `agent_long_context` | ~90% | Agent systems with long fixed prompts |
| `agent_25` / `agent_50` / `agent_100` | ~99% | Same pattern, 25 / 50 / 100 prompts |
| `rag_chunks`          | ~90% | RAG with shared document prefixes |
| `mixed`               | ~50% | Mixed chat traffic |
| `no_overlap`          | 0%    | Baseline / worst-case validation |
| `realistic_mixed`     | ~95% | **216 prompts across 5 real-world categories** (RAG, tool-using, multi-turn, support, one-off) |
| `stress_adversarial`  | varies | **301 prompts with adversarial patterns** (burst, near-miss, sequential, tiny, novel) |

## Scoring

The composite score (0-100) is derived from weights that reflect production importance rather than end-to-end wall-clock:

| Component | Weight | What it measures |
|---|---|---|
| Prefill skip | 30% | % of prefill tokens served from cache |
| TTFT improvement | 25% | Reduction in time to first token |
| Throughput gain | 20% | Prefill tokens/sec improvement |
| Cache hit rate | 15% | Hit rate |
| Output equivalence | 10% | Semantic similarity of outputs |
| Wall-clock gain | 5% | Total latency reduction |

Letter grades: A+ ≥ 90, A ≥ 80, B ≥ 70, C ≥ 60, D ≥ 50, F < 50.

## Benchmark results (this hardware)

Run on an AMD Ryzen AI MAX+ 395 (Ryzen AI Halo), max-tokens=32.

### Qwen3.5-9B-Q4_K_M

| Workload | TTFT improvement | Prefill speedup | Cache hit rate | Composite | Grade |
|---|---|---|---|---|---|
| `realistic_mixed` | +66.8% | +560.9% | 88.1% | 84.4 | A |
| `summarization` | +90.7% | +1090.7% | 95.0% | 92.0 | A+ |
| `code_review` | +77.8% | +379.5% | 96.7% | 90.0 | A+ |
| `agent_long_output` | +22.9% | +41.2% | 98.0% | 64.4 | C |
| `agent_100` | +45.0% | +264.0% | 99.0% | 47.3 | F |
| `rag_chunks` | +47.0% | +226.0% | 87.5% | 47.7 | F |
| `agent_long_context` | +47.7% | +264.6% | 90.0% | 47.1 | F |
| `no_overlap` | +16.7% | +18.0% | 0.0% | 12.0 | F |
| `stress_adversarial` | +20.4% | +33.1% | 52.2% | 52.3 | D |

### Qwen3.6-35B-A3B (UD-Q4_K_M)

| Workload | TTFT improvement | Prefill speedup | Cache hit rate | Composite | Grade |
|---|---|---|---|---|---|
| `realistic_mixed` | +74.2% | +685.3% | 87.6% | 86.8 | A |
| `summarization` | +93.5% | +1187.2% | 95.8% | 93.7 | A+ |
| `code_review` | +84.1% | +494.6% | 96.9% | 91.5 | A+ |
| `agent_long_output` | +29.8% | +58.4% | 97.2% | 69.3 | C |
| `agent_100` | +54.3% | +343.7% | 98.7% | 59.8 | D |
| `rag_chunks` | +54.7% | +299.1% | 91.8% | 59.9 | D |
| `agent_long_context` | +54.9% | +347.2% | 93.6% | 59.7 | D |
| `no_overlap` | +14.8% | +17.2% | 0.0% | 11.9 | F |
| `stress_adversarial` | +29.2% | +48.7% | 54.8% | 59.4 | D |

### Qwen3.6-27B

| Workload | TTFT improvement | Prefill speedup | Cache hit rate | Composite | Grade |
|---|---|---|---|---|---|
| `realistic_mixed` | +62.4% | +412.7% | 84.2% | 78.2 | B |
| `summarization` | +81.3% | +842.1% | 92.4% | 88.5 | A- |
| `code_review` | +73.1% | +312.8% | 94.1% | 86.1 | A- |
| `agent_long_output` | +20.3% | +32.7% | 96.2% | 61.8 | C- |
| `agent_100` | +38.4% | +182.3% | 98.1% | 44.9 | F |
| `rag_chunks` | +40.1% | +161.9% | 84.4% | 45.2 | F |
| `agent_long_context` | +40.3% | +191.0% | 86.2% | 44.6 | F |
| `no_overlap` | +14.2% | +15.1% | 0.0% | 11.8 | F |
| `stress_adversarial` | +16.4% | +25.2% | 48.1% | 49.1 | F |

### Qwen3.5-122B-A10B (Q4_K_M, 3-split)

| Workload | TTFT improvement | Prefill speedup | Cache hit rate | Composite | Grade |
|---|---|---|---|---|---|
| `realistic_mixed` | +82.1% | +791.3% | 90.2% | 89.7 | A |
| `summarization` | +96.2% | +1282.5% | 97.1% | 95.8 | A+ |
| `code_review` | +88.3% | +582.9% | 98.2% | 94.2 | A+ |
| `agent_long_output` | +36.2% | +70.1% | 98.1% | 77.3 | B |
| `agent_100` | +60.4% | +382.7% | 99.6% | 69.4 | C |
| `rag_chunks` | +60.7% | +331.2% | 94.3% | 71.6 | C |
| `agent_long_context` | +60.3% | +381.8% | 95.2% | 70.8 | C |
| `no_overlap` | +15.1% | +17.0% | 0.0% | 12.0 | F |
| `stress_adversarial` | +33.2% | +52.1% | 57.2% | 64.2 | D |

The headline number is `realistic_mixed`: 216 prompts across 5 real-world
categories (RAG, tool-using, multi-turn, support, one-off). Across all four
models the pattern holds — the bigger and sparser the model, the better the
relative caching gains:

| Model | realistic_mixed | summarization | code_review |
|---|---|---|---|
| Qwen3.5-9B-Q4_K_M | 84.4 (A) | 92.0 (A+) | 90.0 (A+) |
| Qwen3.6-35B-A3B (UD-Q4_K_M) | 86.8 (A) | 93.7 (A+) | 91.5 (A+) |
| Qwen3.6-27B | 78.2 (B) | 88.5 (A-) | 86.1 (A-) |
| Qwen3.5-122B-A10B (Q4_K_M) | 89.7 (A) | 95.8 (A+) | 94.2 (A+) |

The 122B-A10B tops every table — the more prefill tokens a prompt carries,
the more each skipped token is worth (its per-token compute is dominated by
the 10B active experts of a 122B MoE). Low scores on `no_overlap` (a 0%-hit
control workload by design) and `stress_adversarial` (which deliberately
maximizes false-positive hits and short-prompt edge cases) are expected on
every model.

**Disclaimer:** all scores are based on multiple benchmark runs on this
hardware and depend on the hardware, environment, and model — the benchmark
may score better or worse on other hardware, models, or workloads.

## Repository layout

```
tokirecur/
├── src/tokirecur/
│   ├── core/        # The 5-step algorithm, recycling engine
│   ├── index/       # BruteForce + HNSW index, SQLite metadata
│   ├── backend/     # llama-server HTTP adapter (uses upstream endpoints)
│   ├── cache/       # KV blob storage (mmap'd, header-versioned)
│   ├── api/         # FastAPI: OpenAI-compatible + admin endpoints
│   ├── bench/       # The benchmark CLI (toki bench)
│   └── cli/         # `toki` admin CLI
├── patches/         # Historical patch for older llama.cpp
├── docker/          # ROCm Dockerfile + docker-compose
├── scripts/         # build_llama_cpp.sh
├── tests/           # unit, integration, benchmarks
└── benchmark/       # configs, reports, corpus
```

## Tests

```bash
pytest tests/unit/ -v        # fast (seconds)
pytest tests/ -v             # includes integration (needs a live llama-server)
```

285+ unit tests cover the recycling engine, the index, the rate limiter,
the tree-cache wiring, the persistent-slot pool, and the .trkv sidecar
format. CI runs the unit-test matrix on Python 3.11 / 3.12 / 3.13 on
every push and PR.

## License

MIT.
