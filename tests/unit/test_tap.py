"""The tap: opt-in recording of turns for training data."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from tokirecur import tap
from tokirecur.cli import cli


@pytest.fixture(autouse=True)
def _tap_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKI_TAP_DIR", str(tmp_path / "tap"))
    monkeypatch.setenv("TOKI_TAP_FLAG", str(tmp_path / "config" / "tap"))
    monkeypatch.delenv("TOKI_TAP", raising=False)
    monkeypatch.delenv("TOKI_TAP_MAX_MB", raising=False)
    monkeypatch.delenv("TOKI_TAP_FILE_MB", raising=False)
    tap._warned.clear()


def _result(**over):
    base = dict(
        output_text="answer",
        reasoning="thinking",
        finish_reason="stop",
        reuse_depth=100,
        prefilled_tokens=20,
        generated_tokens=7,
        hit=True,
        soft_hit=False,
        embedding_similarity=0.98,
        slot_id=0,
        spilled=False,
        ttft_ms=12.5,
        total_ms=99.25,
        generated_tokens_per_sec=3.5,
        tool_calls=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _lines() -> list[dict]:
    files = sorted(Path(tap.tap_dir()).glob("*.jsonl"))
    return [json.loads(line) for f in files for line in f.read_text().splitlines()]


def test_off_by_default_writes_nothing():
    assert tap.is_enabled() is False
    assert tap.record(endpoint="chat", model="m", result=_result()) is False
    assert _lines() == []


def test_switch_is_a_file_so_no_restart_is_needed():
    path = tap.set_enabled(True)
    assert path.is_file()
    assert tap.is_enabled() is True
    assert tap.record(endpoint="chat", model="m", result=_result()) is True
    tap.set_enabled(False)
    assert not path.exists()
    assert tap.is_enabled() is False
    assert len(_lines()) == 1  # switching off keeps what was recorded


def test_env_wins_in_both_directions(monkeypatch):
    tap.set_enabled(True)
    monkeypatch.setenv("TOKI_TAP", "0")
    assert tap.is_enabled() is False
    tap.set_enabled(False)
    monkeypatch.setenv("TOKI_TAP", "1")
    assert tap.is_enabled() is True


def test_record_shape_is_trainable():
    tap.set_enabled(True)
    messages = [{"role": "user", "content": "hi"}]
    tap.record(
        endpoint="chat.completions",
        model="qwen",
        result=_result(),
        messages=messages,
        effort="high",
        session_id="s1",
        params={"temperature": 0.7},
    )
    [row] = _lines()
    assert row["v"] == tap.RECORD_VERSION
    assert row["messages"] == messages
    assert row["completion"] == "answer"
    assert row["reasoning"] == "thinking"
    assert row["effort"] == "high"
    assert row["session_id"] == "s1"
    assert row["mode"] == "cached"
    # The whole prompt, not just the part the backend decoded.
    assert row["usage"]["prompt_tokens"] == 120
    assert row["cache"]["reused_tokens"] == 100
    assert row["cache"]["prefilled_tokens"] == 20
    assert row["cache"]["hit"] is True
    assert row["timings"]["ttft_ms"] == 12.5
    assert json.loads(json.dumps(row)) == row  # no unserialisable values


def test_native_prompt_and_errors_are_kept():
    tap.set_enabled(True)
    tap.record(endpoint="completions", model="m", prompt="once upon a time")
    tap.record(
        endpoint="chat.completions", model="m",
        messages=[{"role": "user", "content": "x"}],
        error="RuntimeError: boom",
    )
    rows = _lines()
    assert rows[0]["prompt"] == "once upon a time"
    assert rows[0]["usage"]["prompt_tokens"] == 0
    assert rows[1]["error"] == "RuntimeError: boom"
    assert rows[1]["completion"] == ""


def test_answer_is_not_optional_so_a_half_written_turn_is_not_silently_empty():
    tap.set_enabled(True)
    tap.record(endpoint="chat", model="m", result=_result(output_text=""))
    [row] = _lines()
    assert row["completion"] == ""
    assert row["reasoning"] == "thinking"


def test_file_rolls_before_it_grows_without_bound(monkeypatch):
    monkeypatch.setenv("TOKI_TAP_FILE_MB", "0.001")  # ~1 KiB
    tap.set_enabled(True)
    for _ in range(40):
        tap.record(endpoint="chat", model="m", result=_result(output_text="x" * 60))
    files = sorted(Path(tap.tap_dir()).glob("*.jsonl"))
    assert len(files) > 1
    assert files[0].name.endswith(".jsonl")
    assert len(_lines()) == 40  # rolling never loses a turn


def test_cap_drops_the_oldest_but_never_the_active_file(monkeypatch):
    monkeypatch.setenv("TOKI_TAP_DIR", str(Path(tap.tap_dir())))
    monkeypatch.setenv("TOKI_TAP_FILE_MB", "0.001")
    monkeypatch.setenv("TOKI_TAP_MAX_MB", "0.002")
    tap.set_enabled(True)
    for _ in range(30):
        tap.record(endpoint="chat", model="m", result=_result(output_text="y" * 60))
    files = sorted(Path(tap.tap_dir()).glob("*.jsonl"))
    active = tap._base_path()
    total = sum(f.stat().st_size for f in files)
    assert active.exists()  # today's file is never the one dropped
    assert total <= 2 * 1024 or len(files) == 1


def test_clear_removes_files_but_leaves_the_switch():
    tap.set_enabled(True)
    tap.record(endpoint="chat", model="m", result=_result())
    removed, freed = tap.clear()
    assert removed == 1 and freed > 0
    assert tap.is_enabled() is True
    assert _lines() == []


def test_a_failed_write_never_raises_into_a_request(monkeypatch):
    tap.set_enabled(True)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(tap.os, "open", boom)
    assert tap.record(endpoint="chat", model="m", result=_result()) is False


def test_pytest_without_explicit_tap_dir_does_not_write(monkeypatch, tmp_path):
    monkeypatch.delenv("TOKI_TAP_DIR", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")
    tap.set_enabled(True)
    assert tap.record(endpoint="chat", model="m", result=_result()) is False
    assert not list((tmp_path / "tokirecur" / "tap").glob("*.jsonl"))


def test_sft_export_is_plain_messages():
    tap.set_enabled(True)
    tap.record(
        endpoint="chat.completions", model="m", result=_result(),
        messages=[{"role": "user", "content": "q"}], effort="low",
    )
    [example] = list(tap.sft_records(tap.iter_records()))
    assert example["messages"] == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "answer"},
    ]
    assert example["reasoning"] == "thinking"
    assert example["effort"] == "low"
    [plain] = list(tap.sft_records(tap.iter_records(), with_reasoning=False))
    assert "reasoning" not in plain


def test_sft_skips_failed_turns():
    tap.set_enabled(True)
    tap.record(
        endpoint="chat", model="m", result=_result(),
        messages=[{"role": "user", "content": "q"}],
    )
    tap.record(endpoint="chat", model="m", error="boom")
    assert len(list(tap.sft_records(tap.iter_records()))) == 1


def test_raw_turn_is_marked_raw():
    tap.set_enabled(True)
    tap.raw_turn(
        model="m", messages=[{"role": "user", "content": "q"}],
        completion="a", reasoning="r", effort="medium",
    )
    [row] = _lines()
    assert row["mode"] == "raw"
    assert row["completion"] == "a"


def test_cli_round_trip(tmp_path):
    runner = CliRunner()
    out = runner.invoke(cli, ["tap", "status"])
    assert out.exit_code == 0
    assert "off" in out.output

    out = runner.invoke(cli, ["tap", "on"])
    assert out.exit_code == 0
    assert tap.is_enabled() is True

    tap.record(
        endpoint="chat.completions", model="m", result=_result(),
        messages=[{"role": "user", "content": "q"}],
    )

    target = tmp_path / "sft.jsonl"
    out = runner.invoke(
        cli, ["tap", "export", "--format", "sft", "--out", str(target)]
    )
    assert out.exit_code == 0, out.output
    [row] = [json.loads(line) for line in target.read_text().splitlines()]
    assert row["messages"][-1] == {"role": "assistant", "content": "answer"}

    out = runner.invoke(cli, ["tap", "clear", "--yes"])
    assert out.exit_code == 0
    assert "removed 1 file" in out.output

    out = runner.invoke(cli, ["tap", "off"])
    assert out.exit_code == 0
    assert tap.is_enabled() is False
