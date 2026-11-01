"""`toki up` starts the stack the rest of the product talks to.

Two things were wrong with it:

- Its default model was a hard-coded path to a starter download
  (``~/Downloads/Qwen3.5-9B-Q4_K_M.gguf``). On any machine that had not
  kept that exact file the command failed before it did anything, and the
  hint it gave was "set TOKI_MODEL or pass --model" — a pointer to a file
  that was never the active model anyway. It now starts the active model
  from the registry, the way the launcher and `toki model use` do.
- It started llama-server with no ``--alias``, so the stack answered to
  the model's *file path*. A registry-started stack answers to the model's
  name and to every alias the clients are configured with, and a client
  that lists models was shown a path.
"""

import pytest

from tokirecur import ctl, lifecycle


def test_the_alias_reaches_llama_server():
    """The names a client may ask for have to be on the command line."""
    args = lifecycle.llama_default_args(
        "/models/m/m.gguf", 8088, 4096, 99, alias="m,default,tokirecur/m"
    )
    assert "--alias" in args
    assert args[args.index("--alias") + 1] == "m,default,tokirecur/m"


def test_no_alias_means_no_alias_flag():
    args = lifecycle.llama_default_args("/models/m/m.gguf", 8088, 4096, 99)
    assert "--alias" not in args


def test_the_registry_names_are_used_as_the_aliases():
    """The union the clients are configured with."""
    spec = ctl.ModelSpec(
        name="m", label="M", gguf="/models/m/m-Q4_K_M.gguf",
        llama_port=1, daemon_port=2, ctx_size=10, max_tokens=10,
        model_id="m-q4_k_m", parallel=1, kv="f16",
    )
    aliases = ctl.model_aliases(spec)
    assert "m-q4_k_m" in aliases
    assert "m-Q4_K_M" in aliases
    assert "default" in aliases
    assert "tokirecur/m" in aliases
    assert "tokirecur-m/m-q4_k_m" in aliases


@pytest.mark.parametrize(
    "option", ["model", "llama_port", "daemon_port", "ctx_size"]
)
def test_nothing_is_hard_coded_any_more(option):
    """The old default was a path from a previous install."""
    from tokirecur.cli import up

    defaults = {p.name: p.default for p in up.params}
    assert defaults[option] is None, (
        f"--{option.replace('_', '-')} still defaults to "
        f"{defaults[option]!r} instead of the active model"
    )


def test_the_unified_cache_and_registry_parallel_reach_the_command_line():
    """`-c` is the window the model gets, not a budget split by slots.

    Without ``-kvu`` llama.cpp divides ``-c`` across ``--parallel``
    slots: a registry saying 131072 was served as 4x32768, and the
    stack advertised a context it did not have. The slot count is the
    registry's, not a hard-coded 4.
    """
    args = lifecycle.llama_default_args(
        "/models/m/m.gguf", 8088, 131072, 99, parallel=2
    )
    assert "-kvu" in args
    assert args[args.index("--parallel") + 1] == "2"


def test_cpu_mode_is_a_real_cpu_launch(monkeypatch):
    import tokirecur.hardware as hardware

    monkeypatch.setattr(lifecycle, "find_llama_server", lambda: "llama-server")
    monkeypatch.setattr(
        lifecycle, "_help_text", lambda binary: "--device --load-mode"
    )
    monkeypatch.setattr(hardware, "cpu_topology", lambda: {})
    monkeypatch.setattr(hardware, "effective_cpu_threads", lambda topology: 8)
    args = lifecycle.llama_default_args("/models/m/m.gguf", 8088, 4096, 0)
    assert args[args.index("-ngl") + 1] == "0"
    assert args[args.index("--device") + 1] == "none"
    assert args[args.index("-t") + 1] == "8"
    assert args[args.index("--load-mode") + 1] == "mmap+mlock"
    assert "-b" in args
    assert "-ub" in args


def test_gpu_mode_does_not_add_cpu_flags(monkeypatch):
    monkeypatch.setattr(lifecycle, "find_llama_server", lambda: "llama-server")
    monkeypatch.setattr(
        lifecycle, "_help_text", lambda binary: "--device --load-mode"
    )
    args = lifecycle.llama_default_args("/models/m/m.gguf", 8088, 4096, 99)
    assert "--device" not in args
    assert "-t" not in args
    assert "--load-mode" not in args
    assert "-b" not in args
    assert "-ub" not in args
