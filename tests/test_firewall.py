"""Integration tests for the full firewall pipeline."""

import json
from pathlib import Path

from context_firewall.budget import BudgetConfig
from context_firewall.firewall import (
    CompressedMessage,
    ContextFirewall,
    Message,
    _infer_content_type,
)
from context_firewall.io import load_training_data, make_row, read_jsonl, write_jsonl
from context_firewall.labels import ContentType, Label


# ---------------------------------------------------------------------------
# Content type inference
# ---------------------------------------------------------------------------

def test_infer_system_message():
    """System messages should be inferred as SYSTEM."""
    msg = Message(role="system", content="You are a helpful assistant.")
    assert _infer_content_type(msg) == ContentType.SYSTEM


def test_infer_user_goal():
    """User messages should be inferred as USER_GOAL."""
    msg = Message(role="user", content="Fix the bug in foo.py")
    assert _infer_content_type(msg) == ContentType.USER_GOAL


def test_infer_test_output():
    """Test output should be inferred as TOOL_RESULT_TEST."""
    msg = Message(role="tool", content="94 passed, 2 failed in 3.5s")
    assert _infer_content_type(msg) == ContentType.TOOL_RESULT_TEST


def test_infer_error_trace():
    """Error traces should be inferred as TOOL_RESULT_ERROR."""
    msg = Message(role="tool", content="Traceback (most recent call last):\nValueError: bad")
    assert _infer_content_type(msg) == ContentType.TOOL_RESULT_ERROR


def test_infer_file_content():
    """File content should be inferred as TOOL_RESULT_FILE."""
    msg = Message(role="tool", content="class Foo:\n    def bar(self):\n        pass")
    assert _infer_content_type(msg) == ContentType.TOOL_RESULT_FILE


def test_infer_agent_patch():
    """Agent patches should be inferred as AGENT_PATCH."""
    msg = Message(role="assistant", content="diff --git a/foo.py b/foo.py\n+new line")
    assert _infer_content_type(msg) == ContentType.AGENT_PATCH


# ---------------------------------------------------------------------------
# Hot loop
# ---------------------------------------------------------------------------

def test_process_system_message():
    """System messages should be kept as CORE."""
    fw = ContextFirewall()
    result = fw.process(Message(role="system", content="You are a helpful assistant."))
    
    assert result.label == Label.CORE
    assert result.content_type == ContentType.SYSTEM
    assert "helpful assistant" in result.content


def test_process_user_goal():
    """User goals should be kept as CORE."""
    fw = ContextFirewall()
    result = fw.process(Message(role="user", content="Fix the bug in foo.py"))
    
    assert result.label == Label.CORE
    assert result.content_type == ContentType.USER_GOAL
    assert "Fix the bug" in result.content


def test_process_test_output():
    """Test output should be DISTILLed."""
    fw = ContextFirewall()
    result = fw.process(Message(
        role="tool",
        content="94 passed, 2 failed in 3.5s\nlots of other output\nmore output",
    ))
    
    assert result.label == Label.DISTILL
    assert result.content_type == ContentType.TOOL_RESULT_TEST
    assert "94 passed, 2 failed" in result.content
    assert result.compressed_tokens < result.original_tokens


def test_process_file_content():
    """File content should be COMPACTed or DISTILLed."""
    fw = ContextFirewall()
    
    # Small file → DISTILL
    result = fw.process(Message(
        role="tool",
        content="class Foo:\n    def bar(self):\n        return 42",
    ))
    assert result.label in (Label.DISTILL, Label.COMPACT)
    assert result.content_type == ContentType.TOOL_RESULT_FILE


def test_process_error_trace():
    """Error traces should be DISTILLed."""
    fw = ContextFirewall()
    result = fw.process(Message(
        role="tool",
        content='Traceback (most recent call last):\n  File "foo.py", line 42\nValueError: bad input',
    ))
    
    assert result.label in (Label.CORE, Label.DISTILL)
    assert result.content_type == ContentType.TOOL_RESULT_ERROR


def test_process_returns_compressed_message():
    """Process should return a CompressedMessage with all fields."""
    fw = ContextFirewall()
    result = fw.process(Message(role="user", content="Fix the bug"))
    
    assert isinstance(result, CompressedMessage)
    assert result.role == "user"
    assert result.label is not None
    assert result.content_type is not None
    assert result.original_tokens > 0
    assert result.compressed_tokens > 0


# ---------------------------------------------------------------------------
# Cool loop
# ---------------------------------------------------------------------------

def test_cool_loop_drops_stale_file_reads():
    """Cool loop should drop stale file reads."""
    fw = ContextFirewall(cool_interval=2)
    
    # Turn 1: read foo.py
    fw.process(Message(
        role="tool",
        content="Contents of src/foo.py:\nclass Foo:\n    pass",
    ))
    
    # Turn 2: edit foo.py (triggers cool loop at interval=2)
    fw.process(Message(
        role="assistant",
        content="diff --git a/src/foo.py b/src/foo.py\n+new line",
    ))
    
    # The file read should be dropped
    active = fw.session.get_active_entries()
    file_reads = [e for e in active if e.content_type == ContentType.TOOL_RESULT_FILE]
    assert len(file_reads) == 0


def test_cool_loop_enforces_budget():
    """Cool loop should enforce budget when over."""
    fw = ContextFirewall(
        budget_config=BudgetConfig(target_tokens=20),
        cool_interval=2,
    )
    
    # Add messages that exceed budget
    for i in range(5):
        fw.process(Message(
            role="tool",
            content=f"Command output line {i} " * 50,
            content_type=ContentType.TOOL_RESULT_COMMAND,
        ))
    
    # Should be under or near budget after cool passes
    stats = fw.get_stats()
    assert stats["active_entries"] < 5  # Some should be dropped


# ---------------------------------------------------------------------------
# Context window
# ---------------------------------------------------------------------------

def test_get_context_window():
    """Should return compressed context as message dicts."""
    fw = ContextFirewall()
    
    fw.process(Message(role="system", content="You are helpful."))
    fw.process(Message(role="user", content="Fix the bug."))
    
    window = fw.get_context_window()
    assert len(window) == 2
    assert window[0]["role"] == "system"
    assert window[1]["role"] == "user"


def test_get_context_window_excludes_dropped():
    """Context window should exclude dropped entries."""
    fw = ContextFirewall(cool_interval=2)
    
    # Turn 1: read foo.py
    fw.process(Message(
        role="tool",
        content="Contents of src/foo.py:\nclass Foo",
    ))
    
    # Turn 2: edit foo.py (triggers cool loop)
    fw.process(Message(
        role="assistant",
        content="diff --git a/src/foo.py b/src/foo.py\n+new",
    ))
    
    window = fw.get_context_window()
    # The stale file read should be excluded
    assert all("src/foo.py" not in m.get("content", "") or "Edited" in m.get("content", "") 
               for m in window)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_get_stats():
    """Should return session statistics."""
    fw = ContextFirewall()
    
    fw.process(Message(role="system", content="You are helpful."))
    fw.process(Message(role="user", content="Fix the bug in foo.py"))
    
    stats = fw.get_stats()
    assert stats["turn_count"] == 2
    assert stats["total_entries"] == 2
    assert stats["active_entries"] == 2
    assert stats["total_tokens"] > 0
    assert stats["compression_ratio"] >= 1.0


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def test_write_and_read_jsonl(tmp_path: Path):
    """Should round-trip JSONL data."""
    path = tmp_path / "test.jsonl"
    
    rows = [
        make_row("system", "You are helpful.", ContentType.SYSTEM, Label.CORE, turn=1),
        make_row("user", "Fix the bug.", ContentType.USER_GOAL, Label.CORE, turn=2),
    ]
    
    write_jsonl(path, rows)
    loaded = read_jsonl(path)
    
    assert len(loaded) == 2
    assert loaded[0]["role"] == "system"
    assert loaded[1]["content"] == "Fix the bug."


def test_load_training_data(tmp_path: Path):
    """Should load training data in classifier-ready format."""
    path = tmp_path / "train.jsonl"
    
    rows = [
        make_row("system", "You are helpful.", ContentType.SYSTEM, Label.CORE),
        make_row("tool", "94 passed, 2 failed", ContentType.TOOL_RESULT_TEST, Label.DISTILL),
    ]
    
    write_jsonl(path, rows)
    feature_texts, labels = load_training_data(path)
    
    assert len(feature_texts) == 2
    assert len(labels) == 2
    assert labels[0] == "core"
    assert labels[1] == "distill"


# ---------------------------------------------------------------------------
# End-to-end session simulation
# ---------------------------------------------------------------------------

def test_full_session_simulation():
    """Simulate a full agent session and verify compression."""
    fw = ContextFirewall(cool_interval=3)
    
    # Turn 1: System prompt
    r1 = fw.process(Message(role="system", content="You are a coding assistant."))
    assert r1.label == Label.CORE
    
    # Turn 2: User goal
    r2 = fw.process(Message(role="user", content="Fix the bug in src/foo.py where bar() returns None."))
    assert r2.label == Label.CORE
    
    # Turn 3: Read file
    r3 = fw.process(Message(
        role="tool",
        content="# src/foo.py\ndef bar():\n    return None\n\ndef baz():\n    return 42",
    ))
    assert r3.content_type == ContentType.TOOL_RESULT_FILE
    
    # Turn 4: Reasoning
    r4 = fw.process(Message(
        role="assistant",
        content="The issue is that bar() returns None instead of 42. I should change it to return 42.",
    ))
    assert r4.content_type == ContentType.AGENT_REASONING
    
    # Turn 5: Apply patch
    r5 = fw.process(Message(
        role="assistant",
        content="diff --git a/src/foo.py b/src/foo.py\n--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1,3 +1,3 @@\n def bar():\n-    return None\n+    return 42",
    ))
    assert r5.content_type == ContentType.AGENT_PATCH
    
    # Turn 6: Run tests (triggers cool loop at interval=3)
    r6 = fw.process(Message(
        role="tool",
        content="pytest -v\ntest_foo.py::test_bar PASSED\n94 passed in 3.5s",
    ))
    assert r6.content_type == ContentType.TOOL_RESULT_TEST
    
    # Verify stats
    stats = fw.get_stats()
    assert stats["turn_count"] == 6
    assert stats["compression_ratio"] >= 1.0
    
    # Verify context window is clean
    window = fw.get_context_window()
    assert len(window) > 0
    
    # All entries should have content
    for msg in window:
        assert len(msg["content"]) > 0
