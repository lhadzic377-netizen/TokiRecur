# Changelog

All notable changes to TokiRecur are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
