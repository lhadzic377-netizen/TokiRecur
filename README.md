<p align="center">
  <a href="https://tokirecur.com"><img src="site/img/mark-3d.png" alt="The Toki Recur mark" width="104" height="104"></a>
</p>

# Toki Recur

KV cache reuse for local LLM inference. Exact prefix matching in front of `llama-server`, across its CUDA, ROCm, SYCL, OpenVINO, Vulkan, Metal and CPU builds.

[![Release](https://img.shields.io/github/v/release/lhadzic377-netizen/TokiRecur?color=ffd60f&label=release)](https://github.com/lhadzic377-netizen/TokiRecur/releases)
[![Website](https://img.shields.io/badge/site-tokirecur.com-ffd60f)](https://tokirecur.com)
[![Docs](https://img.shields.io/badge/docs-tokirecur.com%2Fdocs-3a3f4b)](https://tokirecur.com/docs/)
[![License](https://img.shields.io/badge/license-MIT-4ade80)](LICENSE)

<p align="center">
  <img src="docs/site-hero.png" alt="The Toki Recur website: the headline beside a live terminal launcher" width="880">
</p>

<p align="center"><sub><a href="https://tokirecur.com">tokirecur.com</a> — benchmark table, platform matrix, install docs.</sub></p>

## What it does

When a new prompt arrives at an LLM server, the KV cache is rebuilt from scratch — even if 90% of it is identical to a prompt processed moments earlier. Toki Recur embeds the prompt, finds the closest cached prefix, verifies it token-for-token, and routes the request to the slot that already holds that state. An exact prefix match skips prefill; anything else takes the normal path.

Reuse is in memory and lossless: outputs are byte-identical to a no-cache run. Disk persistence was tried and removed — restoring disk KV corrupted precision (88% substantive divergence from baseline in the benchmark), and the `?action=save` / `?action=restore` layer is dormant in the codebase.

## Measured results

Nine workloads across four models (Qwen3.5-9B, Qwen3.6-35B-A3B, Qwen3.6-27B, Qwen3.5-122B-A10B). On the seven workloads built around a shared prefix, **TTFT fell 20–96%** and **prefill throughput rose 33–1,283%**; the two control workloads (no shared prefix, adversarial) sit at 14–33% and 15–52%, which is what a prefix cache should do. The sparse MoE models top the tables. Full table on [tokirecur.com](https://tokirecur.com).

## What reuse covers: cold vs warm

A prefix cache is only as good as the prefix it can match. Measured with one supported model, one slot, and a long system prompt:

**Setup:** Qwen3.5-2B Q4_K_M · llama-server b1099 (ROCm) · AMD Ryzen AI MAX+ 395 / Radeon 8060S · 8,192-token context · 1 slot · 1,372-token prompt (long system prompt plus a one-line user turn).

| Request | Prompt tokens | Tokens evaluated | Prefill | TTFT |
|---|---|---|---|---|
| Cold (`cache_prompt: false`) | 1,372 | 1,372 | 355 ms | 355 ms |
| Same prompt again | 1,372 | 4 | 27 ms | 27 ms |
| First line of the system prompt changed | 1,377 | 1,377 | 356 ms | 356 ms |

"Evaluated" is `timings.prompt_n` from llama-server: the prompt tokens it actually computed. A warm hit leaves 4 tokens, not zero — the template boundary around the generated answer.

What that means:

- **The match is exact and token-level, not semantic.** Change one token and everything after it is recomputed. Because the system prompt comes first, changing its opening line invalidates the entire prompt.
- **It lives in memory.** A restart, a slot spill or eviction loses it; disk restore was removed. `cache_prompt: false` forces a fresh prefill.
- **It is per model and per slot.** A different model, or a slot holding different contents, cannot serve the prefix.
- **Across harnesses only if the tokens match.** OpenCode and Claude Code carry different system prompts and tool schemas, so switching between them starts cold unless you give both the same prefix.
- **A hit needs at least 10 reused tokens.** Below that it is reported as a `soft_hit`, because the lookup and routing cost exceeds the prefill saved.

These numbers are for interpreting the ratio, not a benchmark: a 2B model at 1.4k tokens. The gain scales with how much stable prefix your harness keeps in front of the changing turn. The daemon's `cache` block in `/v1/chat/completions` is routing metadata (which slot, whether the routing key matched); the token counts above come from llama-server's own `timings`.

## Quick start

Clone and run the installer — cross-platform, no token needed. MIT licensed, public repo.

```bash
git clone https://github.com/lhadzic377-netizen/TokiRecur.git
cd TokiRecur
./install.sh          # Windows: install.bat
```

Both install the `toki` CLI with pipx (auto-installed if missing, venv fallback). The embeddings extra is included by default; skip it with `TOKI_EXTRAS= ./install.sh`. Without pipx:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[embeddings]
```

Then fetch a backend, a model, and start it:

```bash
toki model install-llama                        # auto-detect ROCm/CUDA/SYCL/Vulkan/Metal/CPU
toki model pull Qwen/Qwen3-4B-GGUF --name qwen3-4b
toki model use qwen3-4b                         # boots llama-server + daemon, wires CLI clients
toki tui                                       # full-screen chat (or `toki web` for the browser UI)
toki model remove qwen3-4b                     # --purge deletes weights; --force overrides the active-model guard
```

Nothing is installed or started at install time: `toki model use <name>` is the explicit way to bring a model up. `toki up --model <path>` is the single-model equivalent for a raw GGUF without a registry entry. `toki serve` is only the API daemon (default `127.0.0.1:8787`) and proxies to an already-running `llama-server`.

On 8–16 GB machines, `toki model use` reduces context so the KV cache fits beside the weights. The effective context is written to the model's state dir and becomes the single source of truth: the API advertises exactly that number, clamps `max_tokens` to it, and refuses to over-allocate. The floor is 512 tokens.

`toki model pull` also looks for the DSpine adapter that belongs to the target. It is not a model: the adapter reads Qwen3.8-27B's hidden states and cannot answer a prompt on its own, so it is registered as a **selectable mode** of its one target. Pulling the adapter repository directly does the same when the target is already installed; without it, the pull says which base model to fetch first. A mode is a second GGUF that changes how the base runs — today a DSpine/DFlash2 draft (`--spec-draft-model`) or a LoRA (`--lora`). The launcher's settings pane (`s`) lists `base model (no adapter)` and every installed adapter; picking one writes the mode and restarts the stack, picking the base clears it. An adapter GGUF dropped beside the model is offered in that dropdown before it is registered, so a file from the authors is plug and play. `toki model adapter add|list|use|remove` manages modes from the CLI.

The adapter is *activated* automatically only when the installed `llama-server` can load it: stock llama.cpp refuses DSpine's `dspine_*` tensors, and switching to one anyway would turn a working model into a stack that will not start. `--draft` fetches and registers the mode even on a stock build but leaves the model on its base; `--no-draft` skips it. The model row shows `[DSpine Q4_K_M]`, `[no DSpine Q4_K_M]` when the file has gone, or the LoRA's label. The fit planner and RAM gate reserve the active adapter's weights, its own 8k KV window, and the target's recurrent-state copies per drafted token. To use DSpine today, build the `dspine` branch of [Ajay9o9/llama.cpp](https://github.com/Ajay9o9/llama.cpp) (upstream `master` plus PR #31941) and point the model's `binary:` at that build.

## Platforms

Pure Python over the `llama-server` HTTP API, on any x86_64 or aarch64 device running Linux, macOS or Windows. The only per-platform piece is the `llama-server` binary itself.

| Platform | Backends | Support |
|---|---|---|
| Linux x86_64 / aarch64 | CUDA, ROCm, SYCL, OpenVINO, Vulkan, CPU | stable |
| macOS x86_64 / arm64 | Metal (runs CPU-only with `-ngl 0`) | stable |
| Windows x86_64 / aarch64 | CUDA, SYCL, Vulkan, ROCm/HIP, OpenVINO, CPU | beta |

- **macOS** ships one universal Metal build; there is no separate CPU asset, and `install-llama --backend cpu` says so rather than claiming a Metal install.
- **MLX** is a second, optional macOS backend: `toki model install-llama --backend mlx --name <model>` pip-installs Apple's `mlx-lm` and points that registry model at `mlx_lm.server`. It serves mlx-community model directories, not GGUF files. It is a serving backend only: the daemon verifies a prefix with llama-server's `/tokenize` endpoint, which `mlx_lm.server` does not serve (it answers 404), so a model on MLX gets no KV reuse.
- **CUDA** builds need `cudart`/`cublas`: the installer fetches the matching runtime and checks for a real `CUDA0` device line, warning loudly if none appears. CUDA 13.x needs driver 580.65+, the 12.8 Linux build 570.26+.
- **Linux SYCL** links oneAPI DPC++, oneMKL and Level Zero; **ROCm** uses the system runtime. The installer verifies the libraries and refuses a partial install instead of silently falling back to CPU.
- Windows is **beta**: launcher, chat, tap and every install path are implemented and unit-tested; real-hardware verification is ongoing. CI runs the suite on Linux, macOS and Windows, and executes the Linux/macOS installers end to end.

```bash
toki model install-llama                 # latest release, auto-detected backend
toki model install-llama --backend rocm  # rocm|cuda|sycl|openvino|vulkan|metal|mlx|cpu
toki model install-llama --backend rocm --gfx 11.5.1   # HSA_OVERRIDE_GFX_VERSION
toki model install-llama --tag b6400 --name 35b        # pin a release
```

Or point `binary:` at any `llama-server` on PATH in `~/.config/tokirecur/models.yaml`. `TOKI_MODEL_REGISTRY` and `TOKI_STATE_DIR` move the registry and logs/pid/bin.

## Interfaces

Three frontends over one daemon. The API is what agents actually talk to.

| Interface | Command | What it is |
|---|---|---|
| Launcher | `toki` | Full-screen app and model picker; rewrites each client's own config on launch |
| TUI | `toki tui` | Full-screen chat: raw / cached / compare, live TTFT and hit rate |
| Web | `toki web` | Single-page chat on `127.0.0.1:7777`, streaming, per-message metrics |
| API | `toki serve` | OpenAI-compatible `/v1/chat/completions`, plus Anthropic and admin routes |

The launcher starts Toki Chat, OpenCode, MiMo Code, Hermes, OpenClaw, Codex, Qwen Code, Factory Droid, Cline CLI, DeepSeek Harness, Grok Build, Kilo Code, Pi, GitHub Copilot CLI, Kimi Code, OMP and Claude Code, all pinned to the active model. OpenAI-compatible clients use the daemon's OpenAI route; Claude Code uses the local Anthropic-compatible `/anthropic/v1/messages` route with an owner-only settings override. The override carries the local token as `ANTHROPIC_AUTH_TOKEN` — not as `ANTHROPIC_API_KEY`, because Claude Code sends an API key only after it has approved it, and falls back to another credential when it has not: on a machine with a cloud provider configured, that credential belongs to that provider and the local daemon answers 401 for it.

```bash
toki tui                                  # raw / cached / compare: Ctrl+1 / 2 / 3
toki web --port 7778 --tls-cert cert.pem  # HTTPS + wss
toki diff --system "<long system>" "<prompt>"   # one-shot raw/cached A/B
```

Claude Code sends tens of thousands of tokens per prompt with its full setup. `TOKI_CLAUDE_BARE=1` launches `--bare`, `TOKI_CLAUDE_MCP=none` an empty MCP config, and `TOKI_CLAUDE_EFFORT=low` a smaller effort budget.

On a slow box — a laptop with no usable GPU, where the profile says `cpu` — the launcher also gives Claude Code a 25-minute budget (`API_TIMEOUT_MS`, `CLAUDE_STREAM_IDLE_TIMEOUT_MS`) instead of its 10-minute default, because a CPU prefill of a long prompt can spend that on its own. `TOKI_CLAUDE_TIMEOUT_MS` overrides it in milliseconds. The daemon waits longer than the client by default (`TOKI_BACKEND_TIMEOUT`, 1800 s) so it is never the side that gives up first, and it says what the wait will cost: the measured prefill rate per model becomes the estimate it puts in the first stream event and the log, so a slow turn is a number instead of a mystery.

## Command reference

```bash
# stack
toki up [--detach] [--skip-llama] [--model PATH --ctx-size N]
toki down | toki status

# models
toki model install-llama [--backend B] [--tag T] [--name N]
toki model pull <hf-repo> --name <short>     # any GGUF on Hugging Face
toki model use <name> | toki model remove <name> [--purge] [--force]

# chat
toki | toki tui | toki web [--detach] [--port N]
toki diff --system "<long>" "<prompt>"

# benchmark
toki bench                                   # interactive: model + workload
toki bench --model 9b --workload agent_long_context --max-tokens 64 --output-json report.json

# cache and diagnostics
toki stats | toki evict --target-mb 4096
toki doctor | toki hardware

# training data (off by default)
toki tap on | status | off
toki tap export --format sft --out training.jsonl
```

API, on `http://127.0.0.1:8787` while `toki up` is running:

```
POST /v1/completions          POST /v1/chat/completions
POST /cache/lookup            GET  /cache/stats
DELETE /cache/{id}            GET  /healthz
```

`toki model use` also writes MiMo Code's `mimocode.json` provider config; MiMo then selects the same model with `mimo -m tokirecur-<model>/<model-id>`.

## How it works

```
┌─────────────────┐      ┌──────────────────┐      ┌────────────────┐
│  Client / agent │ ───► │  toki (daemon)   │ ───► │  llama-server  │
│  (OpenAI API)   │      │                  │      │  (any backend) │
└─────────────────┘      └──────────────────┘      └────────────────┘
                         │  embedding + index │
                         │  SQLite metadata   │
                         └────────────────────┘
```

The daemon embeds the prompt, finds the most similar cached prompt, verifies an exact prefix match on token IDs, then routes the request to the llama-server slot that already holds that prefix (`cache_prompt=true`) and only the novel tokens are prefilled. On a miss it takes the normal path, and that slot's KV stays in memory for the next request.

Beyond the base cache:

- **Persistent slot pool** — the hottest prefixes get slots held across requests instead of being recycled after every turn; the pool evicts least-recently-used slots when it is full.
- **Tree index (dormant)** — `cache/tree.py` and the `.trkv` sidecar are still in the tree, but routing does not use them: the request path is in-memory slot routing plus the SQLite index.
- **Honest hits** — a hit requires at least 10 reused tokens; below that the lookup and routing cost exceeds the prefill saving, and it is reported as a `soft_hit` instead of inflating the rate.
- **Prefix-weighted similarity** — among candidates within 5% cosine similarity, the longest token list wins.
- **Divergence classification** — outputs are graded identical / whitespace / restore-noise (a legacy name for KV-restore artifacts) / substantive by semantic similarity.
- **Embedding chunking** — long prompts are split into overlapping segments, with a deterministic hash fallback.

`patches/` keeps a historical patch for older llama.cpp builds. Current upstream exposes the slot endpoints itself, and the engine does not call save/restore: the patch and the `.trkv` layer are both dormant.

## Production deployment

The daemon and the web UI ship as systemd units in `packaging/systemd/`; llama-server runs as its own service on `127.0.0.1:8088`. One `TOKI_AUTH_TOKEN` from a `chmod 600` env file gates the daemon API; the web UI is itself unauthenticated and proxies with that token, so protect it at the network or reverse-proxy layer.

```bash
sudo install -m 600 packaging/systemd/toki.env.example /etc/toki-recur/toki.env
# edit: TOKI_AUTH_TOKEN=<token>
sudo install -m 644 packaging/systemd/toki-recur.service packaging/systemd/toki-recur-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now toki-recur toki-recur-web
```

The web UI has no login page and no session cookie. It proxies to the daemon with the daemon's own token when the client sends none, so anyone who can reach the UI can use the model through it. That is fine on the default `127.0.0.1` binding; if you expose it, put your own authentication in front and serve HTTPS + `wss://` with `--tls-cert` / `--tls-key` (or `TOKI_WEB_TLS_CERT` / `TOKI_WEB_TLS_KEY`). It refuses cross-origin requests and sends `Cache-Control: no-store`.

## Use it as an opencode backend

The daemon's `/v1/chat/completions` is OpenAI-compatible and tool-calling capable, so opencode can run entirely on the local model, reusing the previous turn's prefix.

```jsonc
// ~/.config/opencode/opencode.json
{
  "provider": {
    "tokirecur": {
      "name": "Toki Recur (KV-cache, local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8787/v1",
        // the token the daemon runs with: cat ~/.config/tokirecur/auth
        "apiKey": "<token>"
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
  "small_model": "tokirecur/qwen3.5-9b"
}
```

`apiKey` must be the real `TOKI_AUTH_TOKEN`; a placeholder is not a secret. The model id is a label — the weights are whatever `llama-server` has loaded — and `contextWindow` should stay under `TOKI_MAX_CONTEXT_TOKENS`. Tool calls stream through verbatim, end with `finish_reason: "tool_calls"`, and `role: "tool"` messages round-trip unchanged.

## Benchmark

`toki bench` runs baseline (no cache) against recycled (cache on) on the same prompt set and scores TTFT, prefill throughput, end-to-end latency, hit rate and output equivalence. The composite is weighted toward production impact rather than wall-clock:

| Component | Weight |
|---|---|
| Prefill skip | 30% |
| TTFT improvement | 25% |
| Throughput gain | 20% |
| Cache hit rate | 15% |
| Output equivalence | 10% |
| Wall-clock gain | 5% |

Grades: A+ ≥ 90, A ≥ 80, B ≥ 70, C ≥ 60, D ≥ 50, F < 50. Workloads range from `realistic_mixed` (201 prompts across five traffic types) and `stress_adversarial` (301 adversarial prompts) to `agent_long_context`, `rag_chunks`, `mixed` and `no_overlap`.

## Recording turns (`toki tap`)

Off by default. With it on, the daemon appends one JSONL line per managed turn to `~/.local/state/tokirecur/tap/YYYY-MM-DD.jsonl`: the prompt, the answer, the reasoning and the turn's numbers. Nothing leaves the machine. `toki tap export --format sft` writes `{"messages": [...]}` lines a trainer accepts. Limits: images are recorded as the captions the model was shown, not as pixels, and token ids are not recorded.

## Repository layout

```
tokirecur/
├── src/tokirecur/
│   ├── core/        # recycling engine, the 5-step algorithm
│   ├── index/       # brute-force + HNSW index, SQLite metadata
│   ├── backend/     # llama-server HTTP adapter
│   ├── cache/       # in-memory slot pool; blob/tree modules are dormant
│   ├── api/         # FastAPI: OpenAI + Anthropic + admin routes
│   ├── bench/       # the benchmark CLI
│   ├── cli/         # `toki` admin CLI
│   ├── tui/         # full-screen launcher and chat
│   └── web/         # single-page chat UI
├── packaging/       # systemd units
├── patches/         # historical llama.cpp patch
├── docker/          # ROCm image + compose
├── scripts/         # installers and build helpers
├── tests/           # unit, integration, benchmarks
└── benchmark/       # configs, reports, corpus
```

## Tests

```bash
pytest tests/unit/ -v        # fast, no llama-server needed
pytest tests/ -v             # includes integration (needs a live llama-server)
```

1,200+ unit tests cover the recycling engine, the index, the rate limiter, the slot pool, the dormant `.trkv` format, the CLI config writers and the API. CI runs them on Python 3.11 / 3.12 / 3.13 on Linux, and on 3.13 on macOS, Windows and arm64 Linux, every push and PR.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Luke Hadzic.

`patches/` modifies [llama.cpp](https://github.com/ggerganov/llama.cpp), MIT licensed (c) 2023-2026 The ggml authors — full text in [LICENSE.llama-cpp](LICENSE.llama-cpp).

## Support

Questions and security reports: <contact@tokirecur.com>.
