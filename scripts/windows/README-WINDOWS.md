# TokiRecur on Windows

This folder is a self-contained Windows installer: the TokiRecur wheel,
a PowerShell installer, and `toki` wrappers. No repository access, no
token, no admin rights.

## Requirements

- Windows 10 (1809 or newer) or Windows 11, 64-bit.
- Python 3.10 – 3.13, 64-bit, from
  <https://www.python.org/downloads/windows/>. Tick **Add python.exe to
  PATH** during setup. The Microsoft Store build works but is harder to
  find from scripts; the python.org build is the tested one.
- Windows Terminal is recommended (it is the default terminal on
  Windows 11). The launcher also runs in the classic console, but
  Windows Terminal is faster and renders the box drawing cleanly.
- Network access for the Python dependencies. If this folder contains a
  `wheelhouse\` directory, the installer uses it and does not need the
  network.

## Install

Unzip, open PowerShell in the unzipped folder, then:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

That creates `%LOCALAPPDATA%\TokiRecur\venv`, installs the wheel there,
writes `toki` into `%LOCALAPPDATA%\TokiRecur\bin`, and adds that folder
to your user PATH. Open a **new** terminal afterwards.

Options:

```powershell
.\install.ps1 -Python "py -3.12"      # pick the interpreter
.\install.ps1 -InstallDir D:\Toki     # install elsewhere
.\install.ps1 -NoPath                 # leave PATH alone
.\install.ps1 -NoDoctor               # skip the doctor at the end
.\install.ps1 -Wheelhouse .\wheels    # explicit offline dependencies
.\install.ps1 -Online                 # ignore the bundled wheelhouse
```

The offline package is built for one Python version (the file name says
which, e.g. `-py3.12-offline`). If your Python is a different version,
either install the matching Python, use the online package, or pass
`-Online`.

If PowerShell refuses to run the script, the command above already
bypasses the policy for that one invocation. To allow local scripts
permanently:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## First model

```powershell
toki model install-llama                  # llama-server build for this machine
toki model pull <hf-repo> --name <name>   # download a GGUF
toki model use <name>                     # start llama-server + the daemon
toki doctor                               # GPU, backend, ports, context
```

`toki model pull` accepts any GGUF repo on Hugging Face, including
gated ones (`hf auth login` first). `toki model use` starts the stack
the registry describes and rewrites the client configs so the same
model is selected in every client.

On an NVIDIA box, `install-llama` picks the CUDA build (13.x on a
580.65+ driver, 12.4 on 551.61–580.64) and downloads the matching
cudart runtime archive (~400 MB) with it — the CUDA build does not load
without those DLLs. If the driver is older than 551.61, update it
first; the installer says so and `toki doctor` prints the verdict.
Installing a GPU build while a model is running fails with a "stop the
model stack" message; `toki model stop --force` then re-run.

## The launcher

```powershell
toki
```

is the full-screen launcher: apps on the left, the pinned model and its
live settings on the right, `→` to configure, `Enter` to launch. The
same TUI runs on Linux, macOS and Windows.

```powershell
toki tui        # full-screen chat
toki chat       # plain CLI chat (no full-screen)
```

If you are on a remote shell, a screen reader, or a console that cannot
host a full-screen app:

```powershell
$env:TOKI_CLI = "1"
toki            # numbered menu instead of the full-screen launcher
```

`TOKI_CLI=1` also makes `toki tui` use the plain CLI chat. The
fallback is automatic when Textual is not importable.

## Where things live

| What | Path |
| --- | --- |
| Registry | `%USERPROFILE%\.config\tokirecur\models.yaml` |
| State, logs, pid files | `%USERPROFILE%\.local\state\tokirecur\` |
| Installed llama-server | `%USERPROFILE%\.local\state\tokirecur\bin\llama-server.exe` |
| Venv | `%LOCALAPPDATA%\TokiRecur\venv` |
| `toki` command | `%LOCALAPPDATA%\TokiRecur\bin\toki.cmd` |

Every path can be overridden: `TOKI_MODEL_REGISTRY`, `TOKI_STATE_DIR`,
`TOKI_ACTIVE_MODEL`.

## Troubleshooting

- **`toki` is not recognised** — the installer added the bin folder to
  the *user* PATH; open a new terminal (or sign out and back in).
- **`toki doctor` says CPU only** — install the GPU driver (NVIDIA CUDA
  or Vulkan). `toki model install-llama` picks the matching build; the
  doctor names the build's backend.
- **The launcher looks garbled** — run it in Windows Terminal
  (`winget install Microsoft.WindowsTerminal`), or use `TOKI_CLI=1`.
- **A model will not start** — read
  `%USERPROFILE%\.local\state\tokirecur\<model>\llama.log`; `toki
  doctor` prints the same tail and the RAM estimate.
- **Antivirus flags the downloaded llama-server** — it is an unsigned
  upstream llama.cpp release; allow the file or build from source.
- **Very long model paths** — enable long paths
  (`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled=1`)
  or keep models in a short folder such as `C:\models`.

## Notes

- The launcher, model lifecycle and packaging are covered by the unit
  suite on every platform; the Windows-specific branches (process
  probing, detached spawns, `.cmd` shims) are exercised by tests that
  patch the platform flag. If something misbehaves on real Windows,
  `toki doctor` output plus the log above is the useful report.
- Uninstall: delete `%LOCALAPPDATA%\TokiRecur` and remove that folder
  from your user PATH. Models and the registry live under
  `%USERPROFILE%` and are left alone.
