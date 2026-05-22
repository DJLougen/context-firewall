"""Tests for the failure tee module."""

import os
import time

import pytest

from honeycomb.tee import FailureTee, TeeResult, _sanitize_command


def test_maybe_save_on_failure(tmp_path):
    """Saves content when is_failure=True and mode='failures'."""
    tee = FailureTee(tee_dir=tmp_path, mode="failures")
    result = tee.maybe_save(
        content="error: build failed\nlots of stderr",
        command="cargo test",
        is_failure=True,
    )
    assert result is not None
    assert isinstance(result, TeeResult)
    assert result.bytes_saved > 0
    assert os.path.isfile(result.tee_path)


def test_no_save_on_success(tmp_path):
    """Does not save when is_failure=False and mode='failures'."""
    tee = FailureTee(tee_dir=tmp_path, mode="failures")
    result = tee.maybe_save(
        content="all tests passed",
        command="cargo test",
        is_failure=False,
    )
    assert result is None
    assert tee.list_files() == []


def test_always_mode(tmp_path):
    """Saves regardless of failure status when mode='always'."""
    tee = FailureTee(tee_dir=tmp_path, mode="always")

    result_fail = tee.maybe_save(
        content="error output", command="cmd1", is_failure=True
    )
    result_ok = tee.maybe_save(
        content="success output", command="cmd2", is_failure=False
    )

    assert result_fail is not None
    assert result_ok is not None
    assert len(tee.list_files()) == 2


def test_never_mode(tmp_path):
    """Never saves when mode='never'."""
    tee = FailureTee(tee_dir=tmp_path, mode="never")

    result = tee.maybe_save(
        content="some output", command="cmd", is_failure=True
    )
    assert result is None
    assert not tee.enabled


def test_disabled(tmp_path):
    """Does not save when enabled=False."""
    tee = FailureTee(tee_dir=tmp_path, enabled=False)
    result = tee.maybe_save(
        content="error output", command="cmd", is_failure=True
    )
    assert result is None
    assert not tee.enabled


def test_read_back(tmp_path):
    """Save then read returns original content."""
    tee = FailureTee(tee_dir=tmp_path, mode="always")
    original = "line 1\nline 2\nline 3\nspecial chars: <>&\"'"

    result = tee.maybe_save(content=original, command="echo")
    assert result is not None

    read_back = tee.read(result.tee_path)
    assert read_back == original


def test_read_nonexistent(tmp_path):
    """Reading a nonexistent path returns None."""
    tee = FailureTee(tee_dir=tmp_path)
    assert tee.read(str(tmp_path / "nonexistent.log")) is None


def test_reference_line_format(tmp_path):
    """Reference line contains the file path."""
    tee = FailureTee(tee_dir=tmp_path, mode="always")
    result = tee.maybe_save(content="some output", command="pytest")
    assert result is not None
    assert result.tee_path in result.reference_line
    assert "[full output:" in result.reference_line


def test_cleanup_removes_old_files(tmp_path):
    """Cleanup removes files older than max_age_seconds."""
    tee = FailureTee(tee_dir=tmp_path, mode="always")

    # Create a file
    result = tee.maybe_save(content="old output", command="old_cmd")
    assert result is not None
    old_path = result.tee_path

    # Artificially age the file by setting mtime to 2 hours ago
    old_time = time.time() - 7200
    os.utime(old_path, (old_time, old_time))

    # Create a new file
    result2 = tee.maybe_save(content="new output", command="new_cmd")
    assert result2 is not None

    # Cleanup files older than 1 hour
    removed = tee.cleanup(max_age_seconds=3600)
    assert removed == 1
    assert not os.path.exists(old_path)

    # New file should still exist
    remaining = tee.list_files()
    assert len(remaining) == 1


def test_list_files(tmp_path):
    """Lists all saved .log files."""
    tee = FailureTee(tee_dir=tmp_path, mode="always")

    assert tee.list_files() == []

    tee.maybe_save(content="output1", command="cmd1")
    tee.maybe_save(content="output2", command="cmd2")
    tee.maybe_save(content="output3", command="cmd3")

    files = tee.list_files()
    assert len(files) == 3
    assert all(f.endswith(".log") for f in files)


def test_empty_content_not_saved(tmp_path):
    """Empty or whitespace-only content is not saved."""
    tee = FailureTee(tee_dir=tmp_path, mode="always")

    assert tee.maybe_save(content="", command="cmd") is None
    assert tee.maybe_save(content="   ", command="cmd") is None
    assert tee.maybe_save(content="\n\t\n", command="cmd") is None
    assert tee.list_files() == []


def test_sanitize_command():
    """Command names are sanitized for use in filenames."""
    # Normal command
    assert _sanitize_command("cargo test") == "cargo_test"

    # Special characters replaced
    safe = _sanitize_command("git diff --stat | head -20")
    assert "|" not in safe
    assert " " not in safe

    # Truncated to 40 chars
    long_cmd = "a" * 100
    assert len(_sanitize_command(long_cmd)) <= 40

    # Empty falls back to 'cmd'
    assert _sanitize_command("") == "cmd"

    # Only first 3 words used
    assert _sanitize_command("one two three four five") == "one_two_three"


def test_list_files_empty_dir(tmp_path):
    """list_files returns empty list when dir doesn't exist."""
    tee = FailureTee(tee_dir=tmp_path / "nonexistent")
    assert tee.list_files() == []


def test_cleanup_empty_dir(tmp_path):
    """cleanup returns 0 when dir doesn't exist."""
    tee = FailureTee(tee_dir=tmp_path / "nonexistent")
    assert tee.cleanup(max_age_seconds=60) == 0
