# Changelog

All notable changes to TokiRecur are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.13] - 2026-11-08

### Added
- **A slow turn is a number, not a mystery.** The daemon now measures
  how fast the machine prefills per model, projects the cost of the
  next prompt from that rate, and says so before the wait: the estimate
  rides in the first event of the Anthropic stream, the measured
  prefill is in the non-streaming response, and a prefill that will
  take more than 20 seconds is logged with its numbers.

### Fixed
- **A slow client is no longer abandoned.** Claude Code gives a local
  endpoint ten minutes and treats 300 seconds of silence as a dead
  stream; a CPU-only laptop can spend that on a prefill. The daemon
  waited 600 s for llama-server, the same number as the client's
  budget, so whichever gave up first turned a slow turn into a 500 or
  a timeout. The upstream timeout is now 1800 s
  (`TOKI_BACKEND_TIMEOUT`), and on a box the profile calls CPU-only the
  launcher gives Claude Code a 25-minute budget
  (`TOKI_CLAUDE_TIMEOUT_MS` overrides it).
- **An install no longer dies over a GPU build it did not ask for.** The
  hardware profile recommended CPU on a laptop whose llama-server was a
  working Vulkan build, and the install refused to overwrite it — which
  is right, and also aborted the whole install, so the agent-config
  wiring and the verification never ran. An auto-detected CPU backend
  now keeps the existing GPU build and lets the install finish; an
  explicit `--backend cpu` still refuses, and both messages name the
  exact command (`toki model install-llama --backend vulkan`) instead
  of a bare flag.
- **The hardware profile read RAM in the wrong unit.** `/proc/meminfo`
  reports kibibytes and the profile divided by 1024**3, so every Linux
  box reported a thousand times too little: an 8 GB machine showed
  `ram=0GB`, and that number feeds the KV and thread sizing. It also
  now never reports zero, and a profile cached with no RAM in it is
  re-profiled instead of trusted.
- **An Intel iGPU is no longer called a discrete GPU.** A laptop with an
  integrated Arc part (no dedicated VRAM) was described as
  `vram=0.0GB use=no (discrete GPU)`: the VRAM clause was skipped when
  the number was zero. Intel parts with no dedicated VRAM are now
  reported as integrated, Lunar Lake (0x7d) is in the iGPU list, and
  the reason line says why the CPU build is recommended.
- **The installers agree about the extras.** The README, `install.bat`
  and the curl/irm one-liners install the embeddings extra by default;
  `install.sh` did not, so the benchmark's semantic similarity quietly
  degraded to its proxy on Linux. It does now, and `TOKI_EXTRAS=` still
  means no extras.
- **Claude Code talks to the daemon again.** Claude Code sends the
  `ANTHROPIC_API_KEY` from the environment only when that key is on its
  approved list in `~/.claude.json`; a key it has never seen gets a
  prompt, and a key that was declined is never sent again. Claude Code
  then fell back to `ANTHROPIC_AUTH_TOKEN` — on a machine with another
  provider configured, that token was not ours — and the daemon answered
  `401 invalid token` for every request, while Claude Code showed
  "Please run /login" and retried ten times against a server that was
  working. The local route now travels as `ANTHROPIC_AUTH_TOKEN`, which
  Claude Code sends without asking, a stale rejection is cleared, the
  401 names the fix (`toki model use <name>`), and `toki doctor` reports
  a key Claude Code will not send.

## [0.3.12] - 2026-11-07

### Added
- **Windows runs the real launcher and TUI.** Bare `toki` opens the Textual
  launcher and `toki tui` the full-screen chat, instead of falling back to the
  numbered menu and REPL. All process differences live in `winproc.py`
  (`OpenProcess` instead of `os.kill`, detached process groups, `.cmd`/`.bat`
  through `cmd.exe`), and status polling runs off the event loop.
  `TOKI_CLI=1` still selects the plain fallbacks on any platform.
- **The OS/arch/backend matrix is complete.** Linux x86_64 and aarch64:
  CUDA 12.8/13.4, ROCm 10.0, SYCL, OpenVINO 2026.4.1, Vulkan, CPU. Windows
  x86_64: CUDA, SYCL, Vulkan, ROCm/HIP, OpenVINO, CPU; Windows ARM64: CUDA,
  Vulkan, CPU. macOS: one universal Metal build that runs CPU-only with
  `-ngl 0`.
- **Installs prove the runtime they claim.** CUDA fetches the matching
  `cudart`/`cublas`, checks the library major and requires a real `CUDA0`
  device line; Linux SYCL verifies DPC++, oneMKL and Level Zero; ROCm verifies
  `libamdhip64.so`. A partial install is refused rather than left to fall back
  to CPU while reporting success.
- **A public site and docs at [tokirecur.com](https://tokirecur.com)**, built
  and deployed by the repo, with an installer that owns only marked regions of
  `/etc/caddy/Caddyfile` so other sites on the same machine are untouched.
- **CI gates for the site**: axe against WCAG 2.2 A/AA at three widths, link
  crawling, contrast, wire weight, LCP and CLS.

### Fixed
- CPU mode is honest: a registry-pinned `-ngl 0` is a true CPU launch,
  `toki up` CPU mode no longer asks for a GPU, and an explicit CPU launch
  does not warn about an unused GPU.
- Assistant messages are normalized and tool schemas flattened before they
  reach llama-server, so clients that send either shape work.
- Slow prefills keep the stream alive, and a busy backend is no longer
  reported as wedged.
- The harness advertises the model id clients actually use.
- Loading a second model beside a running one warns instead of silently
  competing for the GPU.
- Windows: Store Python aliases are no longer mistaken for a real
  installation, the batch installer has CRLF endings and an ASCII-safe path,
  and the suite no longer asserts POSIX or macOS behaviour on Windows.
- Unit tests no longer read the developer's own model registry.

### Changed
- The repository is public, so the install path no longer asks for a token.
- The README was rewritten around install, platforms, interfaces and
  commands; the docs code blocks lost their separate black header strip.

## [0.3.6] - 2026-10-01

### Added
- **A fresh-install test that walks the whole chain** in a temporary home:
  register a model, sync every client config, build every launch plan, check
  that every plan carries the daemon token, and authenticate against a real
  daemon with the token that ends up in those configs.

  This exists because every fault found while wiring up DeepSeek Harness had
  the same shape - correct when written, silently wrong later - and none of
  them was *guarded*. A config naming a dead port, a profile that was never
  bound, a token the daemon does not accept: all three surface as an error
  inside somebody else's application, so neither the installer job nor the
  unit tests could ever see them. It runs on all three CI platforms, because
  the config paths are not the same everywhere.

  It found nothing new in the product, which is the point - it locks in the
  chain that took five releases to get right.

## [0.3.5] - 2026-10-01

### Added
- **The DeepSeek Harness TUI and web UI now open together.** Selecting dsh
  starts the browser UI in the background and opens it, then hands the
  terminal to the TUI. Both surfaces are the same model on the same daemon,
  so there was no reason to make them two separate selections.

  The companion is started *before* the foreground launch on purpose: a
  foreground launch `exec`s and replaces the process, so anything spawned
  afterwards never exists. A companion that fails to start is reported and
  the TUI launches anyway - a browser that will not open must not cost you
  the terminal.

  dsh-web still works as its own entry for when you only want the browser.

## [0.3.4] - 2026-10-01

### Fixed
- **DeepSeek Harness Web opens even when one is already running.** A second
  `dsh web` dies with `EADDRINUSE: address already in use 127.0.0.1:3080` and
  never prints a URL, so the launcher had nothing to open and reported a
  failure the user could not act on - while a perfectly good instance was
  serving on 3080. It now reuses the running instance's URL instead.
- **A launch no longer picks up the previous run's token.** The log is opened
  in append mode and the URL scan started at offset 0, so the first match was
  last run's URL: the browser opened a token for a session that no longer
  existed. Scanning now starts at the file size recorded when the process was
  spawned.

## [0.3.3] - 2026-10-01

### Added
- **DeepSeek Harness Web is in the launcher.** `dsh web` takes over the
  terminal and only prints its URL, so from tokirecur it was unreachable:
  the one web row is a hardcoded toki entry and `launch_cmd` had no web
  branch for dsh. There is now a "Launch DeepSeek Harness Web" entry that
  starts it detached, reads the tokenised URL it prints on boot, and opens
  it in the browser. A hardcoded `http://127.0.0.1:3080/` would have 401'd -
  the fence accepts the URL-plus-token it prints, nothing else.

## [0.3.2] - 2026-10-01

### Fixed
- **The DeepSeek Harness web UI works.** `dsh web` served its UI, showed the
  model selector, and then failed every turn with `API key is invalid`.
- **Every dsh profile is now bound, not just `tui`.** `write_dsh_config`
  hardcoded the `tui` profile, so `web` and `headless` kept the empty patch
  layer and fell through to `deepseek-official/deepseek-flash`, which demands
  a `DEEPSEEK_API_KEY` this machine does not have. It now walks the profiles
  that exist and writes each one. A profile directory with no
  `cordis.patch.yml` is skipped rather than created — inventing a user patch
  layer is not ours to do.
- **`auth_token()` now reads `~/.config/tokirecur/auth`.** It read the env var
  and otherwise invented a fresh random token, so it never agreed with
  `api.app._get_auth_token`, which reads env then that file. With no env var
  set, two calls returned two different tokens and neither matched the
  daemon — so a written client config could not work, and a stray
  `TOKI_AUTH_TOKEN` in the environment silently overrode the configured token
  on the daemon side. The two resolvers now have the same order, and
  `test_security.py` is hermetic so it cannot pass by reading the machine's
  real auth file again.

Verified in a browser: `http://127.0.0.1:3080/?token=…` loads the Harness UI,
lists the local `Qwen3.5-0.8B-Q4_K_M`, and a turn completes in ~2s.

## [0.3.1] - 2026-10-01

### Fixed
- **The DeepSeek Harness TUI actually starts.** Two independent faults, and
  the first one was hiding the second.

  0.3.0 fixed a stale `~/.dsh/settings.yaml` and dsh stopped reporting
  `TRANSPORT: Connection error` — which turned out to be only the first
  layer. What remained was
  `ERR_PACKAGE_PATH_NOT_EXPORTED` on
  `@deepseek-ai/dsh-tool-subagent/model-selection-settings`, thrown by
  `@tomowang/dsh-tui@0.8.1`: that bundle is built against
  `@deepseek-ai/dsh-*` `^0.1.2-rc.1` and the installed dsh was `0.1.1-rc.2`.
  Aligning to dsh `0.1.7-rc.2` with `@tomowang/dsh-tui` `0.9.0` (an exact peer
  match) boots the TUI. `dsh --profile headless` still answers on both.

  dsh 0.1.7 migrates `settings.yaml` aside to `settings.yaml.imported` and
  expects a named profile to be bound by that profile's own
  `cordis.patch.yml`. `settings.yaml` is the config for `headless` only:
  `dsh --profile tui --dump-config` and `--dump-default-config` came back
  identical, so the profile ignored it and fell through to its bundle default
  of `deepseek-official/deepseek-flash`, then asked for a
  `DEEPSEEK_API_KEY` this machine does not have. `write_dsh_config` now
  writes that profile layer too, so both layers are set by one call and
  cannot drift.

## [0.3.0] - 2026-10-01

### Fixed
- **DeepSeek Harness now starts.** `dsh` was pinned in `~/.dsh/settings.yaml`
  to a model id and a daemon port that no longer existed, so it failed with a
  bare `TRANSPORT: Connection error` and its TUI profile rendered nothing. The
  file was only ever written by `toki model use`, so switching the active
  model by any other path — or adding and removing models — left every
  config-owning client pointing at a dead endpoint.
- Client configs are now rewritten at launch time, not just on model switch.
  Launching is the last moment the answer is known and cannot go stale
  afterwards. A failed write warns and still launches rather than blocking.

### Removed
- **Meta Code, Cursor Agent and Devin are no longer offered in the launcher.**
  All three authenticate against their own account rather than a local stack,
  so none of them could use a model bound here. `protocol="account"` existed
  only for them and is gone with it; `openai` and `anthropic` remain.

## [0.2.33] - 2026-08-13

### Changed
- Hardware profiling drives backend detection and CPU tuning (`toki doctor`,
  `--profile`/`--no-profile`, 30-day profile TTL).
- Installers no longer download a model at install time; model selection is
  a separate, explicit step.
