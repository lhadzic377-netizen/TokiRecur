# Tokirecur

## Commands

```bash
.venv/bin/pytest -q --ignore=tests/integration   # unit tests (fast, no llama needed)
.venv/bin/pytest -q tests/unit/test_clichat.py   # Windows CLI chat tests
.venv/bin/ruff check src tests                   # lint (run --fix when fixing)
.venv/bin/mypy --ignore-missing-imports src/tokirecur   # type pass; it finds real bugs
```

`mypy` is not in the dev extra (`pip install mypy` to get it) and is not
configured for this repo (no plugin, no strictness) - treat its output as a
bug list to triage, not as a gate. It found the two argument mismatches
that were swallowing whole features: `diagnose.hint(hw, spec)` on a
one-argument function, and `snapshot_download(resume_download=...)` on a
huggingface_hub that had removed the argument.

## Dev-machine gotchas

- The long-running daemon on this box is installed at `/opt/toki-recur/venv`
  and may point at a dead llama port (`--backend-url http://127.0.0.1:8088`).
  The live llama-server is on **8094** (`/home/lukehadzic/llama.cpp/build/bin/llama-server`).
- The registry's `binary:` entries point at the toki state bin
  (`~/.local/state/tokirecur/bin/llama-server`), which is built from
  llama.cpp master for this box (gfx1151 + UMA). **Update it with**
  `LLAMA_CPP_SRC=~/llama.cpp-upstream LLAMA_CPP_UPDATE=1 LLAMA_CPP_INSTALL=1
  ./scripts/build_llama_cpp.sh` - `LLAMA_CPP_UPDATE=1` fetches and resets the
  checkout to `origin/master` (it refuses a dirty tree), `LLAMA_CPP_INSTALL=1`
  copies the result into the state bin and removes the libraries the new ABI
  version replaced. The build must run with `-DLLAMA_USE_PREBUILT_UI=OFF` (the
  script sets it): the prebuilt web UI is validated against the checkout it
  was built from and a source build has no matching assets, so the build
  fails on the asset check. The API server does not need the UI, and the
  checkout stays pristine - no local patch. Upstream ships Linux ROCm assets
  again as of ROCm 10.0, but those need a ROCm 10 runtime; this box has
  ROCm 7.x, so the release asset would not load.
- **`huggingface_hub` 1.x removed `resume_download`** from
  `snapshot_download` (resuming is the only behaviour now). Passing it
  raised TypeError and broke `toki model pull` outright. The kwarg is
  passed only when the installed hub still has it
  (`ctl._snapshot_download_kwargs`), because the old versions need it to
  resume an interrupted pull.
- **Do not put `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` on a ROCm model.** The
  HIP backend reads that CUDA name too; with it on, this box answered with
  repeated garbage tokens and an empty message, silently. It is a CUDA knob
  for the CUDA build only - 0.3.23 drops it from new ROCm installs and
  removes it from registries that older installs wrote.
- **Inkling 2 Mini needs a build that knows `inkling2_loop`** (llama.cpp
  PR #32694, merged 2026-11-20). The loop-capable build for this box is
  `~/.local/state/tokirecur/bin-inkling/llama-server` (upstream master
  29b52d3, gfx1151 + UMA); the registry entry pins it. `toki` refuses to
  start a model whose architecture the installed build does not know, and
  says which architecture the GGUF asks for. The state bin (upstream master,
  1.1.0-dev) also knows it, so a model that needs the loop can point there.
- **Inkling on this iGPU is slow by nature, not by misconfiguration:** it is
  a looped transformer (24 physical layers, the 12-layer core runs 3x per
  token = 48 layer evaluations per token, 128 experts with 8 active).
  Measured on the 8060S: prefill ~260 tok/s at 7k tokens falling to
  ~100 tok/s at 69k, decode ~7 tok/s, GPU pegged at 100%. codex sends
  ~63k tokens per turn, so the first turn after a restart costs 8-11
  minutes; the KV prefix is cached after that (follow-ups evaluate only
  the new tokens), so do not restart the stack between turns. Bigger
  batches (`-b`/`-ub`) and a smaller context do not move it.
- To smoke-test chat end-to-end without touching the running daemon, start a
  throwaway one from the repo:
  `nohup .venv/bin/toki serve --host 127.0.0.1 --port 8799 --backend-url http://127.0.0.1:8094 >/tmp/opencode/toki-serve-smoke.log 2>&1 &`
  then point `DaemonClient` at `http://127.0.0.1:8799` / raw `http://127.0.0.1:8094`.

## Windows

- The full-screen Textual launcher and TUI chat run on Windows (Windows
  Terminal or the Windows 10+ console). Bare `toki` → `selector.run()` →
  `tui.launcher`; `toki tui` → `tui.app`.
- `TOKI_CLI=1` selects the plain fallbacks: `clilauncher.py` (numbered
  menu) for bare `toki`, `clichat.py` (ollama-style REPL) for `toki tui`.
  They are also used automatically when Textual cannot be imported.
- All process/launch differences live in `src/tokirecur/winproc.py`:
  `pid_alive` (OpenProcess, never `os.kill(pid, 0)`), `detach_kwargs`
  (CREATE_NEW_PROCESS_GROUP + CREATE_NO_WINDOW), `resolve_argv`
  (self-launch via `python -m tokirecur.cli`, `.cmd`/`.bat` through
  `cmd.exe /c`), and `format_cmdline`. Do not reintroduce raw
  `start_new_session=True`, `os.execvpe`, or `os.kill(pid, 0)` in
  cross-platform code.
- Status polling in the launcher collects in a Textual worker thread
  (`_poll_statuses_bg`); keep blocking probes out of the event loop.
- Windows package: `scripts/package-windows.sh` builds
  `dist/toki-windows-<version>.zip` (wheel + `scripts/windows/install.ps1`
  + wrappers). `--offline 3.12` adds a win_amd64 wheelhouse; the installer
  refuses a wheelhouse built for a different Python.

## MemT native memory

- MemT models carry memory weights (`blk.*.memt_*` tensors) that
  **upstream llama.cpp refuses to load**. The memory lives in the
  `memt-native-memory` branch of Brishen/llama.cpp (upstream + one
  commit; the memory is CPU and CUDA). Build it with
  `toki model build-memt` (into `~/.local/state/tokirecur/bin-memt`) and
  convert a checkpoint with `toki model convert-memt` (the branch's own
  converter, then `llama-quantize --tensor-type memt_=q8_0` - the memory
  weights stay at Q8_0).
- The branch's server takes three extra request fields on
  `/v1/chat/completions` and `/completion`: `memory_id` (the memory
  state to read), `memory_commit` (write this turn's messages into it
  when the reply ends normally), `memory_documents` (documents to write
  into it before generating). The daemon carries them through the
  OpenAI, Anthropic and Gemini routes, and **only when the client used
  them** - a build without the branch rejects an unknown field. The
  memory states are in RAM by id: `GET /memory`,
  `POST /memory/{id}?action=save|restore|erase` (save and restore need
  `--slot-save-path`).
- `src/tokirecur/memt.py` holds the knowledge (which weights are MemT
  weights, the build steps, the conversion steps);
  `arch.binary_supports_memt` tells a branch build from an upstream one
  by the `memt_q8_0` type name in the binary and its libraries.

## GPU / diagnosis

- `toki doctor` (module `src/tokirecur/diagnose.py`): GPU hardware detection
  (nvidia-smi / ROCm / Vulkan / Metal), per-binary build backend
  (`binary_backend`: CUDA/ROCm/Vulkan DLLs on Windows, `ldd` on POSIX), and
  per-model offload from llama.log (`ctl.gpu_state` — parses both classic
  "offloaded N/M layers to GPU" and modern "- CUDA0 : device (sm_xx, MiB)"
  device_info lines; skips system_info lines).
- `_llama_args` (ctl) passes `-ngl` from `TOKI_NGL` or auto-detect
  (`_default_ngl`: GPU present → 99, else 0); registry `extra_args` pinning
  `-ngl`/`--n-gpu-layers` wins and is never duplicated.
- Log quirk: some llama.cpp builds (e.g. custom Clang b10008) suppress
  device_info at `-lv 3`; the launcher then falls back to the build-backend
  label ("log terse (build: ROCm)").

## Streaming / slot lifecycle

- llama-server reads **`id_slot`**, not `slot_id`. Sending `slot_id` is
  silently ignored and the server auto-assigns a slot. `backend/__init__.py`
  uses `id_slot` in every completion payload; do not rename it back.
- llama-server cancels a generation when its HTTP client's socket closes,
  which is how an abandoned harness turn is stopped. The streaming path
  therefore uses a **dedicated `httpx.AsyncClient` per request**: a shared
  pooled client kept the socket open after producer cancellation and the
  task ran to `max_tokens` with nobody reading it. Closing that client is
  the cancel primitive; `POST /slots/{id}?action=erase` (`clear_slot`) is
  only a KV reset and is queued behind any task still running on that slot
  (`?action=clear` is invalid, HTTP 400).
- Starlette's `StreamingResponse` cancels the body task on ASGI spec < 2.4
  disconnect but does not `aclose()` the body generator. The daemon uses
  `ClosingStreamingResponse`, which closes it; each SSE route then closes
  its relay in a `finally`, `_aiter_with_heartbeat` cancels its pending
  `__anext__`, and `serve_chat_stream` releases the slot and closes the
  upstream stream. All four layers must stay intact, or a disconnect
  leaves a task generating into a queue nobody consumes.
- The three-abort smoke test (from the repo `.venv`, daemon on 8848 with a
  backend on 8149) is in `/tmp/opencode/abort-test.py`: abort a streaming
  chat after the first chunk, poll `/slots`, expect every slot free within
  ~0.25s, then send a normal follow-up. Measured 0.25s on 2026-11-22.
