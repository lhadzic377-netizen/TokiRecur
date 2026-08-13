# Contributing to TokiRecur

TokiRecur is open source. The code is small, the seams are deliberate, and
fixes ship through the normal git flow. This document captures the conventions
so a quick edit doesn't have to rediscover them.

## Contribution model

TokiRecur is maintained by a single maintainer, and every contribution is
reviewed and approved by them before it lands. The practical rule:

**Nothing merges without an explicit maintainer review.** Direct pushes to
`main` are restricted. All changes arrive as pull requests from a fork or a
feature branch, and the maintainer merges them.

## Local setup

```bash
git clone https://github.com/lhadzic377-netizen/TokiRecur
cd TokiRecur
./install.sh
```

`install.sh` runs `pipx install -e .` so any edit to `src/tokirecur/**` is
picked up the moment the daemon restarts. No rebuild step.

## Dev loop

```bash
.venv/bin/pytest -q --ignore=tests/integration   # unit tests, ~12s
.venv/bin/pytest -q tests/unit/test_foo.py        # one file
.venv/bin/pytest -k "test_name"                   # one test
.venv/bin/ruff check src tests                    # lint
.venv/bin/ruff check --fix src tests              # auto-fix
```

The test suite is fast and the loop is short: edit, run pytest, run ruff, commit.

## Conventions

- **Bump the version in `src/tokirecur/__init__.py` only.** `pyproject.toml`
  reads it dynamically via `tool.setuptools.dynamic.version`. Do not edit
  `pyproject.toml`'s `version` field — there isn't one any more.
- **One PR per change.** Small, surgical fixes land faster than bundled ones.
- **A fix without a test is incomplete.** Add a test under
  `tests/unit/test_<module>.py` next to the code it covers. Test names
  describe the behaviour, not the implementation: `test_download_retries_on_remote_protocol_error`,
  not `test_helper_a`.
- **User-facing errors live in the command that raises them** for now. The
  install-llama error vocabulary (`_raise_download_error`) is the existing
  pattern; copy its shape (label + cause + URL + likely causes + fixes).
- **Constants at the top of the file.** `_DOWNLOAD_ATTEMPTS`,
  `_RETRYABLE_STREAM_ERRORS`, `_RESTART_REQUIRED_FIELDS` are right above
  their consumers. Add new dials in the same style.
- **Local imports are only for POSIX-only / Windows-only code paths**
  (e.g. `import signal` inside the SIGCHLD handler). Otherwise lift them
  to module level.

## Adding a CLI command

1. Decorate with `@cli.command()` (or `@<group>.command()` for a subcommand).
2. Click options go above the function body; help text in the docstring.
3. The handler returns `None`; errors go through `click.ClickException` or
   `click.echo(..., err=True) + raise SystemExit(1)`.
4. Add a unit test under `tests/unit/test_<group>.py` using CliRunner.

## Adding a TUI screen

Textual only on Linux/macOS. On Windows the TUI is replaced by a plain CLI
in `src/tokirecur/clilauncher.py` / `clichat.py` — keep that in sync.

## Filing a PR

- Branch from `main`.
- Title: `<area>: <what>` (e.g. `install-llama: ETXTBSY-safe binary replace`).
- Body: the symptom, the root cause, the fix. One paragraph each.
- All checks green: unit tests, ruff, smoke test (if behaviour changed).
- **Wait for maintainer review.** The maintainer approves and merges every PR;
  do not merge your own unless you are the maintainer.

## Review process

1. Open the PR with the checklist above satisfied.
2. The maintainer reviews the diff, runs the checks, and either requests
   changes or approves.
3. On approval the maintainer merges. First-time contributors are thanked in
   `AUTHORS` at merge time.

## Where things live

| Path | Purpose |
| --- | --- |
| `src/tokirecur/cli/` | Click commands, argument parsing, output formatting. |
| `src/tokirecur/ctl/` | Registry, model state, daemon lifecycle. |
| `src/tokirecur/api/` | FastAPI app the daemon serves. |
| `src/tokirecur/backend/` | llama-server HTTP client, tool-call sanitisation. |
| `src/tokirecur/tui/` | Textual launcher + chat UI (POSIX). |
| `src/tokirecur/clilauncher.py`, `clichat.py` | Plain-CLI replacements (Windows). |
| `src/tokirecur/diagnose.py` | `toki doctor` — GPU, binary, model state. |
| `tests/unit/` | Fast tests, no llama-server required. |
| `tests/integration/` | End-to-end tests, need a running llama-server. |
| `AGENTS.md` | Dev-machine gotchas (this box). |
| `install.sh` / `install.ps1` | One-shot installer; both pick up `main` automatically. |
