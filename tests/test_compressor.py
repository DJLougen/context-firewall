"""Tests for deterministic compression rules."""

from context_firewall.compressor import (
    compress,
    compress_command_output,
    compress_error_trace,
    compress_file_content,
    compress_reasoning,
    compress_search_result,
    compress_test_output,
    compress_tool_call,
)
from context_firewall.labels import ContentType, Label


# ---------------------------------------------------------------------------
# Test output compression
# ---------------------------------------------------------------------------

def test_compress_test_output_distill():
    """DISTILL should extract summary and failures."""
    content = """pytest -v
test_foo.py::test_bar PASSED
test_foo.py::test_baz FAILED
test_qux.py::test_quux PASSED
94 passed, 2 failed in 3.5s
FAILURES
test_foo.py::test_baz
AssertionError: expected 5 got 3
"""
    result = compress_test_output(content, Label.DISTILL)
    assert "94 passed, 2 failed" in result
    assert "AssertionError" in result


def test_compress_test_output_compact():
    """COMPACT should extract just the summary."""
    content = "94 passed, 2 failed in 3.5s\nlots of other output"
    result = compress_test_output(content, Label.COMPACT)
    assert "94 passed, 2 failed" in result
    assert len(result) < 100


def test_compress_test_output_drop():
    """DROP should return empty string."""
    content = "94 passed, 2 failed"
    result = compress_test_output(content, Label.DROP)
    assert result == ""


# ---------------------------------------------------------------------------
# File content compression
# ---------------------------------------------------------------------------

def test_compress_file_content_distill():
    """DISTILL should extract path and structure."""
    content = """# src/foo.py
class Foo:
    def __init__(self):
        pass
    
    def bar(self):
        return 42

def baz():
    pass
"""
    result = compress_file_content(content, Label.DISTILL)
    assert "class Foo" in result
    assert "def bar" in result
    assert "def baz" in result


def test_compress_file_content_compact():
    """COMPACT should extract path and line count only."""
    content = "line1\nline2\nline3\nline4\nline5"
    result = compress_file_content(content, Label.COMPACT)
    assert "5 lines" in result


def test_compress_file_content_drop():
    """DROP should return empty string."""
    content = "class Foo:\n    pass"
    result = compress_file_content(content, Label.DROP)
    assert result == ""


# ---------------------------------------------------------------------------
# Command output compression
# ---------------------------------------------------------------------------

def test_compress_command_output_distill():
    """DISTILL should extract exit code and last lines."""
    content = """Running build...
Compiling foo.c
Compiling bar.c
Linking...
Build complete
exit=0
"""
    result = compress_command_output(content, Label.DISTILL)
    assert "exit=0" in result
    assert "Build complete" in result


def test_compress_command_output_compact():
    """COMPACT should extract just the exit code."""
    content = "lots of output\nexit=1"
    result = compress_command_output(content, Label.COMPACT)
    assert "exit=1" in result
    assert len(result) < 100


def test_compress_command_output_drop():
    """DROP should return empty string."""
    content = "output"
    result = compress_command_output(content, Label.DROP)
    assert result == ""


# ---------------------------------------------------------------------------
# Error trace compression
# ---------------------------------------------------------------------------

def test_compress_error_trace_distill():
    """DISTILL should extract error type, message, and top frame."""
    content = """Traceback (most recent call last):
  File "foo.py", line 42, in bar
    result = baz()
  File "foo.py", line 50, in baz
    return int("not a number")
ValueError: invalid literal for int() with base 10: 'not a number'
"""
    result = compress_error_trace(content, Label.DISTILL)
    assert "ValueError" in result
    assert "foo.py:42" in result or "foo.py:50" in result


def test_compress_error_trace_compact():
    """COMPACT should extract just error type and message."""
    content = "ValueError: invalid literal for int() with base 10"
    result = compress_error_trace(content, Label.COMPACT)
    assert "ValueError" in result
    assert "invalid literal" in result


def test_compress_error_trace_drop():
    """DROP should return empty string."""
    content = "Error: something went wrong"
    result = compress_error_trace(content, Label.DROP)
    assert result == ""


# ---------------------------------------------------------------------------
# Search result compression
# ---------------------------------------------------------------------------

def test_compress_search_result_distill():
    """DISTILL should extract matched files with line numbers."""
    content = """foo.py:10: def bar():
foo.py:20: def baz():
qux.py:5: class Qux:
"""
    result = compress_search_result(content, Label.DISTILL)
    assert "foo.py:10" in result
    assert "qux.py:5" in result


def test_compress_search_result_compact():
    """COMPACT should extract just the match count."""
    content = "foo.py:10: match\nbar.py:20: match\nbaz.py:30: match"
    result = compress_search_result(content, Label.COMPACT)
    assert "3 matches" in result


def test_compress_search_result_drop():
    """DROP should return empty string."""
    content = "search results"
    result = compress_search_result(content, Label.DROP)
    assert result == ""


# ---------------------------------------------------------------------------
# Tool call compression
# ---------------------------------------------------------------------------

def test_compress_tool_call_drop():
    """DROP should return empty string."""
    content = "tool_name: read_file\npath: foo.py"
    result = compress_tool_call(content, Label.DROP)
    assert result == ""


def test_compress_tool_call_distill():
    """DISTILL should extract just the tool name."""
    content = 'tool_name: "read_file"\nargs: {path: "foo.py"}'
    result = compress_tool_call(content, Label.DISTILL)
    assert "read_file" in result


# ---------------------------------------------------------------------------
# Reasoning compression
# ---------------------------------------------------------------------------

def test_compress_reasoning_distill():
     """DISTILL should extract conclusion/decision."""
     content = "Let me analyze this step by step. The issue is that the database connection times out after 30 seconds."
     result = compress_reasoning(content, Label.DISTILL)
     assert "database" in result or "timeout" in result


def test_compress_reasoning_compact():
    """COMPACT should extract first 50 chars."""
    content = "This is a long reasoning chain about many things"
    result = compress_reasoning(content, Label.COMPACT)
    assert result.startswith("Reasoning:")
    assert len(result) < 100


def test_compress_reasoning_drop():
    """DROP should return empty string."""
    content = "reasoning"
    result = compress_reasoning(content, Label.DROP)
    assert result == ""


# ---------------------------------------------------------------------------
# Main compress() function
# ---------------------------------------------------------------------------

def test_compress_escalate_passes_through():
    """ESCALATE should pass content through unchanged."""
    content = "Complex content that needs LLM judgment"
    result = compress(content, ContentType.UNKNOWN, Label.ESCALATE)
    assert result == content


def test_compress_stale_compresses_as_compact():
    """STALE should compress aggressively (will be dropped on next cool pass)."""
    content = "class Foo:\n    def bar(self):\n        return 42\n" * 20
    result = compress(content, ContentType.TOOL_RESULT_FILE, Label.STALE)
    assert len(result) < len(content)


def test_compress_unknown_content_type():
    """Should handle unknown content types with generic compression."""
    content = "Some unknown content that is very long" * 10
    result = compress(content, ContentType.UNKNOWN, Label.DISTILL)
    assert len(result) < len(content)
    assert result.endswith("...")


def test_compress_dispatches_to_correct_compressor():
    """Should dispatch to the correct type-specific compressor."""
    content = "94 passed, 2 failed"
    result = compress(content, ContentType.TOOL_RESULT_TEST, Label.COMPACT)
    assert "94 passed, 2 failed" in result
