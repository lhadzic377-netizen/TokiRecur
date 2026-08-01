"""Tests for the JSON workload loader."""

import json
import pytest

from tokirecur.bench.cli import load_workload


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
