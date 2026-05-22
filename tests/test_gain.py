"""Tests for the token savings tracker (gain module)."""

import json
import time

import pytest

from honeycomb.gain import GainEntry, GainTracker


def test_record_and_summary(tmp_path):
    """Record events and verify summary shows data."""
    tracker = GainTracker(data_dir=tmp_path)

    entry = tracker.record(
        command="pytest",
        raw_tokens=2000,
        compressed_tokens=200,
        label="distill",
    )
    assert isinstance(entry, GainEntry)
    assert entry.command == "pytest"
    assert entry.raw_tokens == 2000
    assert entry.compressed_tokens == 200

    summary = tracker.summary()
    assert "Total events: 1" in summary
    assert "2,000" in summary  # raw tokens formatted
    assert "pytest" in summary


def test_daily_breakdown(tmp_path):
    """Record events and verify daily breakdown."""
    tracker = GainTracker(data_dir=tmp_path)

    tracker.record(command="pytest", raw_tokens=1000, compressed_tokens=100)
    tracker.record(command="cargo test", raw_tokens=500, compressed_tokens=50)

    daily = tracker.daily()
    assert "Daily Token Savings" in daily
    # Should have at least one date line
    assert "events" in daily
    assert "saved" in daily


def test_history(tmp_path):
    """Shows recent compression events."""
    tracker = GainTracker(data_dir=tmp_path)

    tracker.record(command="pytest", raw_tokens=1000, compressed_tokens=200)
    tracker.record(command="cargo build", raw_tokens=3000, compressed_tokens=300)

    hist = tracker.history(limit=10)
    assert "Compression History" in hist
    assert "pytest" in hist
    assert "cargo build" in hist


def test_history_limit(tmp_path):
    """History respects the limit parameter."""
    tracker = GainTracker(data_dir=tmp_path)

    for i in range(10):
        tracker.record(command=f"cmd{i}", raw_tokens=100, compressed_tokens=10)

    hist = tracker.history(limit=3)
    # Should show only the last 3 entries
    lines = [l for l in hist.split("\n") if "cmd" in l]
    assert len(lines) == 3
    # Should be the most recent 3
    assert "cmd7" in hist
    assert "cmd8" in hist
    assert "cmd9" in hist


def test_graph(tmp_path):
    """Produces an ASCII bar graph."""
    tracker = GainTracker(data_dir=tmp_path)

    tracker.record(command="pytest", raw_tokens=1000, compressed_tokens=200)
    tracker.record(command="cargo", raw_tokens=2000, compressed_tokens=400)

    graph = tracker.graph()
    assert "Token Savings" in graph
    assert "█" in graph or "░" in graph
    assert "|" in graph


def test_discover_finds_weak_commands(tmp_path):
    """Identifies commands with low savings (< 30%)."""
    tracker = GainTracker(data_dir=tmp_path)

    # Good compression: 90% savings
    tracker.record(command="pytest", raw_tokens=1000, compressed_tokens=100)

    # Bad compression: only 10% savings
    tracker.record(command="git diff", raw_tokens=1000, compressed_tokens=900)

    # Zero savings (passthrough)
    tracker.record(command="echo hello", raw_tokens=500, compressed_tokens=500)

    discovery = tracker.discover()
    assert "Compression Discovery" in discovery
    assert "git diff" in discovery
    assert "echo hello" in discovery


def test_discover_all_good(tmp_path):
    """When all commands have good savings, says so."""
    tracker = GainTracker(data_dir=tmp_path)

    tracker.record(command="pytest", raw_tokens=1000, compressed_tokens=100)
    tracker.record(command="cargo", raw_tokens=2000, compressed_tokens=200)

    discovery = tracker.discover()
    assert "good compression" in discovery or "All commands" in discovery.lower() or "> 30%" in discovery


def test_persistence(tmp_path):
    """Data survives tracker recreation."""
    tracker1 = GainTracker(data_dir=tmp_path)
    tracker1.record(command="pytest", raw_tokens=1000, compressed_tokens=200)
    tracker1.record(command="cargo", raw_tokens=500, compressed_tokens=50)

    # Create a new tracker pointing to the same directory
    tracker2 = GainTracker(data_dir=tmp_path)
    summary = tracker2.summary()
    assert "Total events: 2" in summary
    assert "pytest" in summary
    assert "cargo" in summary


def test_persistence_jsonl_format(tmp_path):
    """Data is persisted as JSONL."""
    tracker = GainTracker(data_dir=tmp_path)
    tracker.record(command="pytest", raw_tokens=1000, compressed_tokens=200)

    jsonl_path = tmp_path / "gain.jsonl"
    assert jsonl_path.exists()

    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 1

    data = json.loads(lines[0])
    assert data["cmd"] == "pytest"
    assert data["raw"] == 1000
    assert data["compressed"] == 200


def test_clear(tmp_path):
    """Clear removes all data and the file."""
    tracker = GainTracker(data_dir=tmp_path)
    tracker.record(command="pytest", raw_tokens=1000, compressed_tokens=200)

    assert "Total events: 1" in tracker.summary()

    tracker.clear()

    assert tracker.summary() == "No compression data recorded."
    assert not (tmp_path / "gain.jsonl").exists()


def test_to_json(tmp_path):
    """Exports correct structure."""
    tracker = GainTracker(data_dir=tmp_path)
    tracker.record(command="pytest", raw_tokens=1000, compressed_tokens=200, label="distill")
    tracker.record(command="cargo", raw_tokens=500, compressed_tokens=100)

    data = tracker.to_json()
    assert data["version"] == 1
    assert len(data["entries"]) == 2
    assert data["summary"]["total_events"] == 2
    assert data["summary"]["total_raw"] == 1500
    assert data["summary"]["total_compressed"] == 300
    assert data["summary"]["total_saved"] == 1200


def test_to_json_entry_structure(tmp_path):
    """Each entry in to_json has the expected fields."""
    tracker = GainTracker(data_dir=tmp_path)
    tracker.record(command="pytest", raw_tokens=1000, compressed_tokens=200, label="distill")

    data = tracker.to_json()
    entry = data["entries"][0]
    assert "timestamp" in entry
    assert "command" in entry
    assert "raw_tokens" in entry
    assert "compressed_tokens" in entry
    assert "label" in entry


def test_empty_tracker(tmp_path):
    """Summary for no data returns appropriate message."""
    tracker = GainTracker(data_dir=tmp_path)

    assert tracker.summary() == "No compression data recorded."
    assert tracker.daily() == "No compression data recorded."
    assert tracker.history() == "No compression data recorded."
    assert tracker.graph() == "No compression data recorded."
    assert tracker.discover() == "No compression data recorded."


def test_savings_calculation(tmp_path):
    """Verifies saved_tokens and savings_pct on GainEntry."""
    entry = GainEntry(
        timestamp=time.time(),
        command="pytest",
        raw_tokens=1000,
        compressed_tokens=250,
    )
    assert entry.saved_tokens == 750
    assert entry.savings_pct == 75.0

    # Zero raw tokens
    zero_entry = GainEntry(
        timestamp=time.time(),
        command="echo",
        raw_tokens=0,
        compressed_tokens=0,
    )
    assert zero_entry.saved_tokens == 0
    assert zero_entry.savings_pct == 0.0

    # Compressed > raw (shouldn't happen but handle gracefully)
    inverted = GainEntry(
        timestamp=time.time(),
        command="weird",
        raw_tokens=100,
        compressed_tokens=200,
    )
    assert inverted.saved_tokens == 0  # max(0, ...) clamps


def test_gain_entry_date(tmp_path):
    """GainEntry.date returns formatted date string."""
    entry = GainEntry(
        timestamp=1700000000.0,  # 2023-11-14
        command="test",
        raw_tokens=100,
        compressed_tokens=10,
    )
    assert entry.date == "2023-11-14"


def test_gain_entry_to_dict_roundtrip():
    """GainEntry serializes and deserializes correctly."""
    entry = GainEntry(
        timestamp=1700000000.0,
        command="pytest",
        raw_tokens=1000,
        compressed_tokens=200,
        label="distill",
    )
    d = entry.to_dict()
    restored = GainEntry.from_dict(d)

    assert restored.timestamp == entry.timestamp
    assert restored.command == entry.command
    assert restored.raw_tokens == entry.raw_tokens
    assert restored.compressed_tokens == entry.compressed_tokens
    assert restored.label == entry.label
