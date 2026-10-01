# Changelog

All notable changes to TokiRecur are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
