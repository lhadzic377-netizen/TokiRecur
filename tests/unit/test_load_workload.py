"""Tests for the JSON workload loader and the bench interactive launcher."""

import json
from unittest.mock import patch

import pytest

from tokirecur.bench.cli import (
    _interactive_pick_workload,
    _resolve_bench_target,
    build_argparser,
    load_workload,
)


class TestLoadWorkload:
    def test_load_simple_workload(self, tmp_path):
        path = tmp_path / "test.json"
        path.write_text(json.dumps({
            "name": "test_wl",
            "description": "Test workload",
            "prompts": ["p1", "p2", "p3"],
            "categories": ["a", "b", "c"],
        }))
        w = load_workload(path)
        assert w.name == "test_wl"
        assert w.description == "Test workload"
        assert w.prompts == ["p1", "p2", "p3"]
        assert w.categories == ["a", "b", "c"]

    def test_load_workload_without_categories(self, tmp_path):
        path = tmp_path / "test.json"
        path.write_text(json.dumps({
            "name": "no_cat",
            "prompts": ["p1", "p2"],
        }))
        w = load_workload(path)
        assert w.name == "no_cat"
        assert w.categories is None

    def test_load_workload_uses_filename_as_name(self, tmp_path):
        path = tmp_path / "auto_name.json"
        path.write_text(json.dumps({"prompts": ["x"]}))
        w = load_workload(path)
        assert w.name == "auto_name"

    def test_categories_length_mismatch_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({
            "name": "bad",
            "prompts": ["a", "b"],
            "categories": ["x"],  # wrong length
        }))
        with pytest.raises(ValueError, match="must match"):
            load_workload(path)

    def test_prompts_must_be_list(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"prompts": "not a list"}))
        with pytest.raises(ValueError, match="must be a list"):
            load_workload(path)

    def test_non_dict_top_level_raises_valueerror(self, tmp_path):
        """A top-level list/scalar used to hit AttributeError on data.get;
        it must be a clean ValueError the CLI boundary can catch."""
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(["not", "an", "object"]))
        with pytest.raises(ValueError, match="must be an object"):
            load_workload(path)

    def test_malformed_json_raises_valueerror(self, tmp_path):
        """JSONDecodeError subclasses ValueError, so the CLI's
        except (ValueError, OSError) turns it into exit 2, not a traceback."""
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        with pytest.raises(ValueError):
            load_workload(path)


class TestResolveBenchTarget:
    def test_path_passthrough(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        path, url = _resolve_bench_target(str(gguf))
        assert path == gguf
        assert url is None  # path-on-disk leaves --base-url untouched

    def test_none_signals_picker(self):
        path, url = _resolve_bench_target(None)
        assert path is None and url is None

    def test_bad_string_returns_none(self):
        """A non-path / non-registry string yields ``(None, None)`` so
        the caller can print a useful error."""
        path, url = _resolve_bench_target("/no/such/file.gguf")
        assert path is None and url is None


class TestArgparserInteractive:
    def test_model_is_now_optional(self):
        ns = build_argparser().parse_args([])
        assert ns.model is None

    def test_workload_default_present(self):
        ns = build_argparser().parse_args([])
        assert ns.workload == "agent_100"

    def test_explicit_workload_preserved(self):
        ns = build_argparser().parse_args(["--workload", "realistic_mixed"])
        assert ns.workload == "realistic_mixed"


class TestInteractivePickers:
    def test_pick_workload_returns_name(self):
        with patch("rich.prompt.Prompt.ask", return_value="1"):
            name = _interactive_pick_workload()
        # First workload in WORKLOADS dict
        assert name == list(__import__("tokirecur.bench.cli", fromlist=["WORKLOADS"]).WORKLOADS.keys())[0]

    def test_pick_workload_retry_on_bad_input(self):
        with patch("rich.prompt.Prompt.ask", side_effect=["abc", "0", "1"]):
            name = _interactive_pick_workload()
        assert name is not None

    def test_pick_workload_quit_raises(self):
        with patch("rich.prompt.Prompt.ask", return_value="q"):
            with pytest.raises(KeyboardInterrupt):
                _interactive_pick_workload()


class TestBaseUrlAutoRouting:
    """A registry-resolved or interactively-picked model should auto-
    route ``--base-url`` to its own llama-server port. Without this,
    the bench always hits :8080 even when the active model lives on
    :8089 (the bug the user hit)."""

    def test_registry_name_resolves_with_url(self):
        """`_resolve_bench_target('35b')` returns the spec's gguf
        path AND its llama_url, so main() can auto-route."""
        from tokirecur.bench.cli import _resolve_bench_target
        path, url = _resolve_bench_target("35b")
        if path is None:
            pytest.skip("no registry entry for '35b' in this env")
        assert url is not None, "registry lookup must produce a URL"
        assert "8089" in url

    def test_user_explicit_base_url_wins(self):
        """If the user passes ``--base-url``, argparse keeps it."""
        from tokirecur.bench.cli import build_argparser
        ns = build_argparser().parse_args(["--base-url", "http://h:1234"])
        assert ns.base_url == "http://h:1234"
        ns = build_argparser().parse_args([])
        assert ns.base_url == "http://127.0.0.1:8080"  # default
