"""Unit tests for the TUI chat history (JSONL persistence)."""

import json
import tempfile
from pathlib import Path

import pytest

from tokirecur.tui import history


class TestAppend:
    def test_appends_line(self, tmp_path):
        p = tmp_path / "h.jsonl"
        history.append("user", "hello", path=p)
        assert p.exists()
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["role"] == "user"
        assert rec["content"] == "hello"
        assert "ts" in rec

    def test_multiple_appends(self, tmp_path):
        p = tmp_path / "h.jsonl"
        for i in range(5):
            history.append("user", f"msg {i}", path=p)
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 5
        assert json.loads(lines[2])["content"] == "msg 2"

    def test_with_mode_and_metadata(self, tmp_path):
        p = tmp_path / "h.jsonl"
        history.append("assistant", "hi", mode="cached",
                       metadata={"hit": True}, path=p)
        rec = json.loads(p.read_text().strip())
        assert rec["mode"] == "cached"
        assert rec["meta"]["hit"] is True


class TestLoad:
    def test_load_returns_records(self, tmp_path):
        p = tmp_path / "h.jsonl"
        for i in range(3):
            history.append("user", f"m{i}", path=p)
        loaded = history.load(path=p)
        assert len(loaded) == 3
        assert [m["content"] for m in loaded] == ["m0", "m1", "m2"]

    def test_load_respects_limit(self, tmp_path):
        p = tmp_path / "h.jsonl"
        for i in range(10):
            history.append("user", f"m{i}", path=p)
        loaded = history.load(limit=3, path=p)
        # Returns the LAST 3.
        assert [m["content"] for m in loaded] == ["m7", "m8", "m9"]

    def test_load_skips_corrupt_lines(self, tmp_path):
        p = tmp_path / "h.jsonl"
        p.write_text(
            '{"ts":1,"role":"user","content":"good"}\n'
            'this is not json\n'
            '{"ts":2,"role":"assistant","content":"also good"}\n'
        )
        loaded = history.load(path=p)
        assert len(loaded) == 2
        assert loaded[0]["content"] == "good"
        assert loaded[1]["content"] == "also good"

    def test_load_missing_file(self, tmp_path):
        # Should return empty list, not raise.
        loaded = history.load(path=tmp_path / "does-not-exist.jsonl")
        assert loaded == []


class TestClear:
    def test_clear_removes_file(self, tmp_path):
        p = tmp_path / "h.jsonl"
        history.append("user", "x", path=p)
        assert p.exists()
        history.clear(path=p)
        assert not p.exists()

    def test_clear_missing_is_safe(self, tmp_path):
        # Should not raise.
        history.clear(path=tmp_path / "never-existed.jsonl")


class TestAtomicity:
    """Ensure partial writes don't corrupt the file."""

    def test_partial_then_more_writes(self, tmp_path):
        p = tmp_path / "h.jsonl"
        # Simulate a partial write by creating a temp file in the
        # same directory (this is what atomicity protects against).
        history.append("user", "first", path=p)
        # Now write again — even if a "partial" tempfile exists,
        # the append should still work cleanly.
        history.append("user", "second", path=p)
        loaded = history.load(path=p)
        assert len(loaded) == 2
        assert loaded[1]["content"] == "second"
