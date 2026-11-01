<p align="center">
  <a href="https://tokirecur.com"><img src="site/img/toki.png" alt="toki the bear mascot" width="120" height="120"></a>
</p>

# Toki Recur

A standalone KV cache reuse system for LLM inference. Drop-in prefix caching that works across CUDA, ROCm, and Vulkan.

## What it does

When a new prompt arrives at an LLM server, the KV cache is rebuilt from scratch — even if 90% of the prompt is identical to one we just processed. **Toki Recur** detects exact prefix matches, reuses the saved KV state, and skips redundant prefill compute.

## Real measured results (this hardware)

Full 9-workload results across four models (Qwen3.5-9B, Qwen3.6-35B-A3B,
Qwen3.6-27B, Qwen3.5-122B-A10B) are published on [tokirecur.com](https://tokirecur.com).
The headline: TTFT improves 62–96% and prefill throughput 410–1280% on realistic
workloads, with the biggest gains on the largest MoE models.

## Quick start

Clone the repo (your Git credentials do the auth) and run the installer from the
checkout — cross-platform, no token needed:

```bash
# Linux / macOS
git clone https://github.com/lhadzic377-netizen/TokiRecur.git
cd TokiRecur
./install.sh
```

```bat
:: Windows (double-click or from cmd/PowerShell)
git clone https://github.com/lhadzic377-netizen/TokiRecur.git
cd TokiRecur
install.bat
```

Both install the `toki` CLI with pipx (auto-installed if missing, with a venv
fallback on Linux/macOS — `TOKI_VENV_DIR=<dir>` forces the venv path on both
platforms). The embeddings extra is included by default
(sentence-transformers + torch, used by the similarity index); to skip it:
`TOKI_EXTRAS= ./install.sh` (Linux/macOS) or `set TOKI_EXTRAS= && install.bat`
(Windows).

Without pipx, install into your own venv instead:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[embeddings]
```

```bash
# 1. Fetch the llama-server binary matching your OS/arch and GPU.
#    Auto-detects the best backend: ROCm (AMD/HIP), CUDA (NVIDIA),
#    SYCL (Intel Arc), Vulkan, Metal (macOS), CPU. Pass --backend
#    to override.
toki model install-llama

# 2. Install a model — nothing is preloaded on a fresh install; you
#    pull exactly what you want. Fetch a GGUF from Hugging Face and
#    register it under a short name (find quantized Qwen3 GGUFs via
#    huggingface.co/models?library=gguf):
toki model pull Qwen/Qwen3-4B-GGUF --name qwen3-4b

# 3. Point your install-llama binary at the new entry (auto-detected
#    ROCm/CUDA build from step 2 wins over the PATH llama-server):
toki model install-llama --backend rocm --name qwen3-4b

# 4. Start it (this boots llama-server + the toki daemon and wires
#    every model-capable CLI client at it):
toki model use qwen3-4b

# 5. Talk to it — the full-screen TUI chat (or open the web UI with `toki web`)
toki tui

# 6. Done with it — unregister (weights and state stay on disk by default;
#    pass --purge to delete them too; --force overrides the active-model guard):
toki model remove qwen3-4b
```

No models are installed or started at install time — `toki model use <name>` is the explicit, documented way to bring one up. `toki up --model <path>` is the single-model equivalent for a raw GGUF without a registry entry. `toki serve` is **only** the API daemon (default 127.0.0.1:8787) that proxies to an already-running llama-server. Two chat frontends ship with toki: bare `toki` opens the full-screen launcher, `toki tui` is the full-screen TUI chat, and `toki web` is the browser chat UI.

### Small-RAM machines

On machines with limited RAM (8–16 GB), `toki model use` automatically reduces the requested context so the KV cache fits in memory alongside the model weights. The effective context — the smaller of your configured `ctx_size` and what the machine can actually hold — is written to the model's state dir and becomes the single source of truth: the OpenAI-compatible API advertises exactly that context, clamps `max_tokens` to it, and refuses to over-allocate. Failed startups no longer hang on a spinning `starting/selecting model` — llama-server's log tail is surfaced and the launcher returns an error. The context floor is 512 tokens, so the server always gets a workable value.

## Platforms

Toki Recur is pure Python over the llama-server HTTP API — it runs on **any x86_64 or aarch64 device** running Linux, macOS or Windows, with whatever GPU backend your llama-server build has (ROCm, CUDA, SYCL, Vulkan, Metal, CPU). The only per-platform piece is the llama-server binary itself:

| Platform | Architecture | llama-server builds | install-llama asset | Support |
|---|---|---|---|---|
| Linux | x86_64 | ROCm (auto if HIP present), Vulkan, CPU | `ubuntu-rocm-7.14-x64`, `ubuntu-vulkan-x64`, `ubuntu-x64` | stable |
| Linux | aarch64 | Vulkan, CPU | `ubuntu-vulkan-arm64`, `ubuntu-arm64` | stable |
| macOS | x86_64, arm64 | Metal (also runs CPU-only) | `macos-x64`, `macos-arm64` | stable |
| Windows | x86_64 | CUDA (auto NVIDIA), SYCL (auto Intel Arc), Vulkan (auto AMD), ROCm/HIP (explicit AMD), CPU | `win-cuda-13.4-x64`, `win-sycl-x64`, `win-vulkan-x64`, `win-rocm-10.0-x64`, `win-cpu-x64` | beta |
| Windows | aarch64 | Vulkan, CPU | `win-vulkan-arm64`, `win-cpu-arm64` | beta |

> **macOS note:** llama.cpp ships one universal Metal build; there is no
> separate macOS CPU asset. It runs CPU-only with `-ngl 0` / `--device none`.
> `install-llama --backend cpu` on macOS installs that same build and prints
> the note; a Mac or VM without Metal falls back to CPU with an explicit
> warning, not a false "installed Metal" claim.

> **Linux ROCm note (mid-2026):** llama.cpp stopped publishing Linux ROCm
> release assets after **b10375**. `install-llama --backend rocm` still finds
> that last build, but brand-new model support lands on master first — on a
> Linux ROCm box, build the latest source instead:
> `scripts/build_llama_cpp.sh` (ROCm for the host GPU, gfx1151 on Strix
> Halo APUs). Windows ROCm keeps shipping as `win-rocm-<ver>-x64`
> (currently 10.0); its HIP runtime is bundled in the zip, so no ROCm
> toolkit install is needed.

CI runs the test suite on all three OSes (x86_64 + aarch64) and executes the
Linux and macOS system installers end-to-end. The GPU backends are provided by
the llama-server binary itself — Toki Recur talks to it over HTTP and is
backend-agnostic. Windows is **beta**: the launcher, chat, tap, and the CUDA /
SYCL / Vulkan / ROCm install paths are implemented and unit-tested, with
real-hardware verification ongoing; fixes here never touch the stable
Linux/macOS code paths.

One command installs the matching official build:

```bash
toki model install-llama                 # auto-detect backend, latest release
toki model install-llama --backend rocm  # force a backend (rocm|cuda|vulkan|metal|cpu)
toki model install-llama --backend rocm --gfx 11.5.1  # set HSA_OVERRIDE_GFX_VERSION
toki model install-llama --tag b6400     # pin a release tag
toki model install-llama --name 35b      # ...and point a registry model at it
```

Optional — run as a system daemon (dedicated user + service manager unit + log
rotation, auto-start at boot). `install.sh`/`install.ps1` bootstrap the full
stack themselves (own venv at `/opt/toki-recur`, package + llama-server +
service unit), so they work standalone:

```bash
# Linux  -> systemd + logrotate   (macOS -> launchd LaunchDaemon)
sudo ./install.sh

# Windows (PowerShell): install.ps1; then `toki serve` yourself or wrap
# it in NSSM / a scheduled task.
```

Or point `binary:` at any llama-server on PATH in the registry
(`~/.config/tokirecur/models.yaml`). Override locations per-machine with
`TOKI_MODEL_REGISTRY` (config) and `TOKI_STATE_DIR` (logs/pid/bin).

## The launcher

Bare `toki` opens an Ollama-style full-screen launcher in a black + yellow design: a letter-spaced yellow `TOKI` wordmark with tagline and amber divider, then a navigable column of app cards. The model-bound clients (Toki Chat, OpenCode, MiMo Code, Hermes, OpenClaw, Codex, Qwen Code, Factory Droid, Cline CLI, DeepSeek Harness, Grok Build, Kilo Code, Pi, GitHub Copilot CLI, Kimi Code, OMP, and Claude Code) use the pinned TokiRecur model. Each one's own config file is rewritten when it launches, so a client can never be left pointing at a model or port that is no longer running — the failure mode that made DeepSeek Harness sit there rendering nothing. OpenAI-compatible clients use the daemon's OpenAI route; Claude Code uses the daemon's local Anthropic-compatible `/anthropic/v1/messages` route. Web Dashboards remains available at the end of the list. The active row is highlighted with a yellow accent bar. Keys `1`–`9` jump to the first nine apps; use `↑`/`↓` for the full list.


| Key | Action |
|---|---|
| `1`–`9` | jump to the first nine apps (or models) |
| `↑` / `↓` (or `k` / `j`) | navigate apps or models |
| `Enter` | launch the highlighted app; model-bound clients use the pinned model |
| `→` / `Tab` | flip to the model picker (per-model health dots + ctx) |
| `←` | back to the app list |

### Claude Code and local models

Claude Code talks to the daemon's Anthropic-compatible route: a private override at
`~/.config/tokirecur/claude/settings.json` (owner-only) points `ANTHROPIC_BASE_URL`
at `http://127.0.0.1:<daemon port>/anthropic`, pins every model alias to the active
model, and sets `ANTHROPIC_API_KEY` only — a stale `ANTHROPIC_AUTH_TOKEN` in your
shell or global settings is dropped from the launch environment. The daemon accepts
either credential header, so a client that sends both still authenticates. Streams
carry Anthropic `ping` keepalives, so a slow local prompt evaluation cannot trip the
client's idle timeout and trigger a two-minute retry.

A full Claude Code setup (plugins, MCP servers, hooks, LSP) plus extended thinking
makes every prompt tens of thousands of tokens, which is slow on a local 35B. These
opt-in environment variables trim it:

| Variable | Effect |
|---|---|
| `TOKI_CLAUDE_BARE=1` | launch with `--bare` (skips hooks, LSP, plugin sync, auto-memory) |
| `TOKI_CLAUDE_MCP=none` | launch with an empty MCP config (`--strict-mcp-config`) |
| `TOKI_CLAUDE_EFFORT=low` | set the session effort level (`low`…`max`) |
| `Space` | pin the highlighted model in the picker |
| `s` | open the settings pane (edit any registry field) |
| `Esc` / `q` | quit |

In the settings pane, `Enter` opens an editor for the highlighted field. Sizing
and enum fields (`ctx_size`, `max_tokens`, `parallel`, `kv`, `reasoning`,
`always_on`, `boot`) open a **dropdown** of presets with a `custom…` option that
falls back to the free-text editor; paths, ports and ids stay plain text. The
`reasoning` field controls thinking mode: `auto`/`on` stream the model's
reasoning, `off` passes `--reasoning off` to llama-server at launch.

## Command reference

```bash
# ── stack lifecycle ────────────────────────────────────────────────────────
toki up                                                # start llama-server + daemon
toki up --detach                                       # background, returns PIDs
toki up --skip-llama                                   # only start the daemon
toki up --model /path/to/other.gguf --ctx-size 16384   # custom args
toki down                                              # stop everything
toki status                                            # show stack state

# ── chat interfaces ─────────────────────────────────────────────────────────
toki                                                  # full-screen launcher
toki tui                                              # full-screen TUI chat
toki tui --auth-token <token>                         # if daemon requires auth
toki tui --daemon-url http://other:8787               # custom daemon URL
toki web                                              # browser chat UI (127.0.0.1:7777)
toki web --detach                                     # background
toki web --tls-cert cert.pem --tls-key key.pem        # HTTPS + wss, Secure cookies
toki diff --system "<long>" "<user prompt>"           # one-shot raw/cached A/B

# ── MiMo Code ───────────────────────────────────────────────────────────────
# `toki model use` also writes MiMo Code's native mimocode.json provider
# configuration. MiMo then selects the same local model with:
#   mimo -m tokirecur-<model>/<model-id>

# ── benchmark (the headline tool) ───────────────────────────────────────────
toki bench                                            # interactive: pick model + workload
toki bench --model 9b --workload agent_long_context \
           --max-tokens 64 --output-json report.json  # fully scripted
toki bench --model ~/Downloads/Qwen3.5-9B-Q4_K_M.gguf \
           --self-contained --ctx-size 8192 --n-gpu-layers 99

# ── cache management ────────────────────────────────────────────────────────
toki stats                                            # hit rate, bytes, entries
toki evict --target-mb 4096                           # LRU evict to fit budget

# ── diagnostics ─────────────────────────────────────────────────────────────
toki doctor                                           # llama-server, ROCm, GPU
toki compare report.json report2.json                 # diff two bench reports

# ── training data (off by default) ──────────────────────────────────────────
toki tap on                                           # record every managed turn
toki tap export --format sft --out training.jsonl     # one example per turn

# ── OpenAI-compatible API server (running under `toki up`) ──────────────────
# POST http://127.0.0.1:8787/v1/completions
# POST http://127.0.0.1:8787/v1/chat/completions
# POST http://127.0.0.1:8787/cache/lookup
# DELETE http://127.0.0.1:8787/cache/{id}
# GET  http://127.0.0.1:8787/cache/stats
# GET  http://127.0.0.1:8787/healthz
```

The benchmark compares baseline (no cache) vs recycled (cache enabled) runs
across TTFT, prefill throughput, end-to-end latency, cache hit rate, and output
equivalence, then prints a composite score (0-100) with a letter grade.

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

`tokid` (the daemon) is a small FastAPI service that sits in front of
`llama-server`. It intercepts requests, runs the 5-step recycling algorithm:

1. Embed the incoming prompt.
2. Find the most similar cached prompt via cosine similarity.
3. Verify exact prefix match on token IDs.
4. If hit: load the saved KV state via `?action=restore` and decode only the novel tokens.
5. If miss: full prefill, persist KV state via `?action=save` for the future.

## How KV save/restore works (no patch needed)

As of llama.cpp b3524 (mid-2025), the upstream `llama-server` already exposes
the endpoints we need behind the `--slot-save-path` flag:

```
POST /slots/{id_slot}?action=save     body: {"filename": "..."}
POST /slots/{id_slot}?action=restore  body: {"filename": "..."}
POST /slots/{id_slot}?action=erase    body: {"filename": "..."}
```

The `patches/` directory contains a historical patch for users on older
llama.cpp and a `README.md` explaining the situation.

### Cross-session cache persistence (disabled)

Prefix reuse is **in-memory only**: the engine routes similar prompts to the
same llama-server slot and lets the slot's KV cache serve the shared prefix
(`cache_prompt=true`). This is lossless — outputs are byte-identical to a
no-cache run.

Disk persistence via `?action=save` / `?action=restore` was removed because
restoring disk KV corrupted precision (benchmarks showed 88% substantive
divergence from baseline outputs). The `--slot-save-path` flag and the `.trkv`
sidecar layer (`cache/blob.py`) remain in the codebase but are dormant — the
engine never calls `cache_save` / `cache_load`.

## Production deployment

All three processes (llama-server, daemon, web UI) ship as systemd units in
`packaging/systemd/`. Point `EnvironmentFile=` at a `/etc/toki-recur/toki.env`
(chmod 600) with `TOKI_AUTH_TOKEN` set — that single token then gates **both**
the daemon API and the web UI.

```bash
sudo install -m 600 packaging/systemd/toki.env.example /etc/toki-recur/toki.env
# edit: TOKI_AUTH_TOKEN=<token>
sudo install -m 644 packaging/systemd/toki-recur.service packaging/systemd/toki-recur-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now toki-recur toki-recur-web
```

The web UI is always open — no login page, no session cookie — which is safe on
the default 127.0.0.1 binding, but exposed if you bind it to a network
interface. `TOKI_AUTH_TOKEN` still protects the daemon: the web server
authenticates every proxied call (REST + WebSocket) with its own configured
credential, so an open UI never weakens daemon auth. Serve HTTPS + `wss://` with
`--tls-cert` / `--tls-key`, or set `TOKI_WEB_TLS_CERT` / `TOKI_WEB_TLS_KEY` in
the env file. The web server refuses cross-origin requests and serves everything
`Cache-Control: no-store`.

## Use Toki Recur as an opencode backend

The daemon's `/v1/chat/completions` is OpenAI-compatible *and* tool-calling
capable, so opencode can run entirely on the local KV-recycled model. Every
agent turn reuses the prefix from the previous turn via llama-server's KV cache
— no cloud, no `num_ctx` fiddling, and cache hits show up in the daemon's
metrics.

```jsonc
// ~/.config/opencode/opencode.json
{
  "provider": {
    "tokirecur": {
      "name": "Toki Recur (KV-cache, local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8787/v1",
        // The token the daemon runs with: `cat ~/.config/tokirecur/auth`.
        // Do not put a placeholder here — a well-known "token" protects
        // nothing, and anything that can reach the port can use the model.
        "apiKey": "<the token in ~/.config/tokirecur/auth>"
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

- Set `apiKey` to whatever `TOKI_AUTH_TOKEN` the daemon runs with —
  `cat ~/.config/tokirecur/auth`. A placeholder is not a secret: the
  daemon replaces a well-known one with a real token when it finds it.
- The model id is a label; the actual weights are whatever llama-server has
  loaded. `contextWindow` should stay under `TOKI_MAX_CONTEXT_TOKENS`.
- Tool calls stream through verbatim (`delta.tool_calls`), finish with
  `finish_reason: "tool_calls"`, and `role: "tool"` result messages round-trip
  unchanged — verified against opencode's own `bash` tool loop.

## TUI chat (`toki tui`)

The full-screen terminal chat client. Three modes, switched with `Ctrl+1/2/3` (or in-chat slash commands):

| Mode | What it does |
|---|---|
| `raw` | Sends directly to llama-server (bypasses cache) |
| `cached` | Sends through the toki daemon (recycling engine) |
| `compare` | Runs each prompt both ways, shows side-by-side diff |

```bash
toki up                       # make sure the daemon + llama-server are running
toki tui                      # in a separate terminal
toki tui --auth-token <token> # if TOKI_AUTH_TOKEN is set
toki tui --daemon-url http://other:8787 --raw-url http://other:8088
```

In-session slash commands: `/mode raw|cached|compare`, `/system <text>`,
`/clear`, `/metrics`, `/max <n>`, `/temp <f>`, `/raw <msg>`, `/cached <msg>`,
`/compare <msg>`, `/quit`. `Ctrl+C` quits.

**Features:**
- Three modes (raw / cached / compare) with Ctrl+1/2/3 keyboard shortcuts
- Live TTFT sparkline + running hit-rate counter in the right panel
- Per-backend connection health (green/yellow/red) shown at all times
- Slot state panel showing which slots are in use, their context size, and which session owns them
- Precision retention panel (Ctrl+P) showing baseline/recycled byte-identity stats
- Session pinning via UUID — multi-turn chats reuse the same slot for max cache benefit
- Persistent chat history at `~/.local/share/tokirecur/history.jsonl` (XDG)
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

`toki diff --system "<long system>" "<prompt>"` does the same raw/cached A/B
comparison as a one-shot (no REPL).

## Web chat UI (`toki web`)

A production-ready single-page chat interface with a single-click mode toggle
(Raw / Cached / Compare), streaming responses, and live per-message metrics.
Default port is **7777**. The UI is a single static page
(`src/tokirecur/web/static/index.html`) served by a FastAPI app
(`src/tokirecur/web/server.py`) with WebSocket streaming (`/ws/chat`). No
external frontend dependencies — vanilla HTML/CSS/JS.

```bash
toki web                                       # foreground, http://127.0.0.1:7777
toki web --detach                              # background, log → /tmp/toki/web.log
toki web --port 7778 --raw-url http://...:8088 # custom port + backend
```

**Features:**
- Three inference modes: Raw (direct to llama-server), Cached (through the daemon), Compare (run both, side-by-side)
- Streaming responses with token-by-token display
- Live metrics per message (TTFT, total time, tokens generated, prefill tokens)
- Compare mode shows two side-by-side panels with a summary banner showing speedup and divergence warning
- System prompt input persisted in the sidebar
- Conversation history with multi-turn support
- Mobile-responsive layout

## Recording turns for training (`toki tap`)

Off by default. With it on, the daemon appends one JSONL line per managed
turn — what the model was given, what it answered, the reasoning it produced,
and the turn's numbers — to
`~/.local/state/tokirecur/tap/YYYY-MM-DD.jsonl`. Nothing leaves the machine.

```bash
toki tap on                     # start (no restart: the daemon checks per turn)
toki tap status                 # switch, directory, files, record count
toki tap off                    # stop, keep the files
toki tap export --format sft --out training.jsonl
toki tap clear                  # delete the recordings
```

`--format sft` writes the plain `{"messages": [...]}` shape a trainer accepts,
one completed turn per line; `--with-reasoning/--no-reasoning` decides whether
the reasoning rides beside the answer. The TUI shows `● rec` while the switch
is on; the launcher's settings pane carries it as the first row ("record
turns", with the live state and how much is on the shelf) and `Ctrl+T`
toggles it from the launcher; the desktop app has the same switch in
Settings → Recording.

Honest limits: images are recorded as the captions the text-only model was
shown, not as pixels; token ids are not recorded (the engine does not return
them for chat, and re-tokenising is not guaranteed identical); and the TUI's
raw mode bypasses the daemon but writes to the same file with `mode: "raw"`.

## Architectural improvements

Beyond the basic cache, the tool ships with:

1. **Tree-structured KV cache** (`src/tokirecur/cache/tree.py`) — KV tensors are tracked in BLOCK_SIZE chunks. Two prompts sharing a prefix share the same on-disk blocks (content-addressed by token hash). Provides a fast tree-hit lookup that bypasses embedding entirely.
2. **Persistent slot pool** (`src/tokirecur/cache/persistent_pool.py`) — the top-N most-frequently-used prefixes get dedicated llama-server slots that never clear. Skips the save/load cycle entirely for hot prefixes.
3. **Honest hit definition** (`MIN_REUSE_FOR_HIT = 10`) — a "hit" requires at least 10 tokens of prefix reuse; below that, save/load overhead exceeds the prefill savings, reported as a `soft_hit` instead of inflating the hit rate.
4. **Prefix-length-weighted similarity** (`BruteForceIndex.search`) — among candidates within 5% cosine similarity, prefers the one with the longest token list.
5. **Substantive divergence classification** — output divergence is classified into identical / whitespace-only / restore-noise / substantive via sentence-transformers semantic similarity.
6. **Embedding chunking fallback** (`LlamaServerBackend.embed`) — long prompts are split into overlapping segments and re-averaged, with a deterministic hash-based fallback when embedding fails entirely.

## Benchmark workloads & scoring

| Workload | Expected hit rate | Best for |
|---|---|---|
| `realistic_mixed` | ~95% | **201 prompts across 5 real-world categories** (RAG, tool-using, multi-turn, support, one-off) |
| `stress_adversarial` | varies | **301 prompts with adversarial patterns** (burst, near-miss, sequential, tiny, novel) |
| `agent_long_context` | ~90% | Agent systems with long fixed prompts |
| `agent_25` / `agent_50` / `agent_100` | ~99% | Same pattern, 25 / 50 / 100 prompts |
| `rag_chunks` | ~90% | RAG with shared document prefixes |
| `mixed` | ~50% | Mixed chat traffic |
| `no_overlap` | 0% | Baseline / worst-case validation |

The composite score (0-100) weights production importance rather than
end-to-end wall-clock:

| Component | Weight | What it measures |
|---|---|---|
| Prefill skip | 30% | % of prefill tokens served from cache |
| TTFT improvement | 25% | Reduction in time to first token |
| Throughput gain | 20% | Prefill tokens/sec improvement |
| Cache hit rate | 15% | Hit rate |
| Output equivalence | 10% | Semantic similarity of outputs |
| Wall-clock gain | 5% | Total latency reduction |

Letter grades: A+ ≥ 90, A ≥ 80, B ≥ 70, C ≥ 60, D ≥ 50, F < 50.

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
├── scripts/         # build_llama_cpp.sh, installers
├── tests/           # unit, integration, benchmarks
└── benchmark/       # configs, reports, corpus
```

## Tests

```bash
pytest tests/unit/ -v        # fast (seconds)
pytest tests/ -v             # includes integration (needs a live llama-server)
```

285+ unit tests cover the recycling engine, the index, the rate limiter, the
tree-cache wiring, the persistent-slot pool, and the .trkv sidecar format. CI
runs the unit-test matrix on Python 3.11 / 3.12 / 3.13 on every push and PR.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Luke Hadzic.

The `patches/` directory ships a patch that modifies
[llama.cpp](https://github.com/ggerganov/llama.cpp), which is MIT licensed
(c) 2023-2026 The ggml authors — full text in [LICENSE.llama-cpp](LICENSE.llama-cpp).

## Support

Questions, security reports, and community help: <contact@tokirecur.com>.
