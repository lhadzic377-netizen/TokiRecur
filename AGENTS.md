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
- The registry's `binary:` entries point at the toki state bin
  (`~/.local/state/tokirecur/bin/llama-server`), which was rebuilt from
  llama.cpp master (commit b94041a98) for this box. llama.cpp ships Linux
  ROCm assets again as of ROCm 10.0 (b12024); rebuild with
  `./scripts/build_llama_cpp.sh` (LLAMA_CPP_SRC + LLAMA_CPP_BLD overridable).
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
  says which architecture the GGUF asks for.
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
