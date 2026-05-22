"""Tests for production-readiness fixes.

Covers:
- JSON tool call parsing
- Node.js/Rust error trace support
- Search result false positive filtering
- Empty entry filtering in get_context_window
- Model loading fallback
- Token estimation for empty strings
- __repr__ methods
- Model versioning
"""

import warnings
from pathlib import Path

from context_firewall.compressor import (
    compress,
    compress_error_trace,
    compress_search_result,
    compress_tool_call,
)
from context_firewall.firewall import (
    CompressedMessage,
    ContextFirewall,
    Message,
    _infer_content_type,
)
from context_firewall.labels import ContentType, Label
from context_firewall.session import _estimate_tokens


# ---------------------------------------------------------------------------
# JSON tool call parsing
# ---------------------------------------------------------------------------

def test_compress_tool_call_json_format():
    """Should extract tool name from JSON format."""
    content = '{"name": "read_file", "arguments": {"path": "src/foo.py"}}'
    result = compress_tool_call(content, Label.DISTILL)
    assert "read_file" in result


def test_infer_json_tool_call():
    """Should detect JSON tool calls in assistant messages."""
    msg = Message(
        role="assistant",
        content='{"name": "read_file", "arguments": {"path": "src/foo.py"}}',
    )
    assert _infer_content_type(msg) == ContentType.TOOL_CALL


# ---------------------------------------------------------------------------
# Node.js / Rust error traces
# ---------------------------------------------------------------------------

def test_compress_error_trace_nodejs():
    """Should extract frame from Node.js error traces."""
    content = 'TypeError: Cannot read property "foo" of undefined\n    at processItem (src/utils.js:23:15)\n    at Array.map (<anonymous>)'
    result = compress_error_trace(content, Label.DISTILL)
    assert "TypeError" in result
    assert "src/utils.js:23" in result


def test_compress_error_trace_rust():
    """Should extract frame from Rust error traces."""
    content = "thread 'main' panicked at 'index out of bounds'\nStackError: panicked at src/main.rs:42"
    result = compress_error_trace(content, Label.DISTILL)
    assert "src/main.rs:42" in result


# ---------------------------------------------------------------------------
# Search result false positives
# ---------------------------------------------------------------------------

def test_compress_search_result_ignores_urls():
    """Should not match URL patterns as file:line."""
    content = "http://example.com:8080/api/v1\nhttps://docs.rs:443/crate"
    result = compress_search_result(content, Label.DISTILL)
    # Should not extract these as file:line matches
    assert "example.com" not in result or "matches" in result


def test_compress_search_result_matches_real_files():
    """Should match real file:line patterns."""
    content = "src/foo.py:42: def bar():\ntests/test_foo.py:10: test_bar"
    result = compress_search_result(content, Label.DISTILL)
    assert "src/foo.py:42" in result


# ---------------------------------------------------------------------------
# Empty entry filtering
# ---------------------------------------------------------------------------

def test_get_context_window_filters_empty_entries():
    """Context window should not include empty (DROP'd) entries."""
    fw = ContextFirewall()
    
    # Process a tool call (will be DROP'd)
    fw.process(Message(
        role="assistant",
        content='tool_name: read_file\nargs: {path: "foo.py"}',
        content_type=ContentType.TOOL_CALL,
    ))
    
    # Process a system message (will be kept)
    fw.process(Message(role="system", content="You are helpful."))
    
    window = fw.get_context_window()
    
    # Should only have the system message, not the empty tool call
    assert len(window) == 1
    assert window[0]["role"] == "system"
    assert all(m["content"] for m in window)  # No empty content


# ---------------------------------------------------------------------------
# Token estimation for empty strings
# ---------------------------------------------------------------------------

def test_estimate_tokens_empty_string():
    """Empty strings should have 0 tokens."""
    assert _estimate_tokens("") == 0


def test_estimate_tokens_non_empty():
    """Non-empty strings should have at least 1 token."""
    assert _estimate_tokens("hello") >= 1
    assert _estimate_tokens("x" * 100) > _estimate_tokens("x" * 10)


# ---------------------------------------------------------------------------
# Model loading fallback
# ---------------------------------------------------------------------------

def test_model_loading_fallback(tmp_path: Path):
    """Should fall back to rules when model loading fails."""
    # Create a corrupt model file
    corrupt_path = tmp_path / "corrupt.joblib"
    corrupt_path.write_text("not a valid joblib file")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fw = ContextFirewall(model_path=corrupt_path)
        
        # Should have warned
        assert len(w) == 1
        assert "Failed to load" in str(w[0].message)
        assert "Falling back" in str(w[0].message)
    
    # Should still work with rule-based classification
    result = fw.process(Message(role="system", content="You are helpful."))
    assert result.label == Label.CORE


# ---------------------------------------------------------------------------
# __repr__ methods
# ---------------------------------------------------------------------------

def test_message_repr():
    """Message should have a useful repr."""
    msg = Message(role="user", content="Fix the bug in foo.py")
    r = repr(msg)
    assert "Message" in r
    assert "user" in r
    assert "Fix the bug" in r


def test_message_repr_truncates_long_content():
    """Message repr should truncate long content."""
    msg = Message(role="user", content="x" * 200)
    r = repr(msg)
    assert "..." in r
    assert len(r) < 200


def test_compressed_message_repr():
    """CompressedMessage should have a useful repr."""
    msg = CompressedMessage(
        role="tool",
        content="94 passed",
        label=Label.DISTILL,
        content_type=ContentType.TOOL_RESULT_TEST,
        original_tokens=100,
        compressed_tokens=10,
    )
    r = repr(msg)
    assert "CompressedMessage" in r
    assert "distill" in r
    assert "100->10" in r


# ---------------------------------------------------------------------------
# Model versioning
# ---------------------------------------------------------------------------

def test_model_versioning(tmp_path: Path):
    """Trained models should include version metadata."""
    import joblib
    from context_firewall.classifier import _MODEL_VERSION, train
    from context_firewall.io import make_row, write_jsonl
    
    # Generate minimal training data
    rows = []
    for i in range(30):
        rows.append(make_row(
            role="system",
            content=f"You are helpful. {i}",
            content_type=ContentType.SYSTEM,
            label=Label.CORE,
        ))
        rows.append(make_row(
            role="tool",
            content=f"94 passed in test {i}",
            content_type=ContentType.TOOL_RESULT_TEST,
            label=Label.DISTILL,
        ))
    
    train_path = tmp_path / "train.jsonl"
    model_path = tmp_path / "model.joblib"
    
    write_jsonl(train_path, rows)
    train(train_path, output_path=model_path)
    
    # Load and verify version
    loaded = joblib.load(model_path)
    assert isinstance(loaded, dict)
    assert "pipeline" in loaded
    assert "version" in loaded
    assert loaded["version"] == _MODEL_VERSION


def test_load_model_legacy_format(tmp_path: Path):
    """Should handle legacy (bare pipeline) model format."""
    import joblib
    from context_firewall.classifier import _load_model
    from sklearn.pipeline import Pipeline
    
    # Create a legacy-format model (bare pipeline)
    legacy_path = tmp_path / "legacy.joblib"
    dummy_pipeline = Pipeline([])
    joblib.dump(dummy_pipeline, legacy_path)
    
    pipeline, version = _load_model(legacy_path)
    assert version is None


# ---------------------------------------------------------------------------
# Content type inference edge cases
# ---------------------------------------------------------------------------

def test_infer_unknown_tool_output():
    """Unrecognized tool output should be UNKNOWN, not COMMAND."""
    msg = Message(role="tool", content="Some random output with no patterns")
    assert _infer_content_type(msg) == ContentType.UNKNOWN


def test_infer_command_with_exit_code():
    """Tool output with exit code should be COMMAND."""
    msg = Message(role="tool", content="$ npm install\nadded 123 packages\nexit=0")
    assert _infer_content_type(msg) == ContentType.TOOL_RESULT_COMMAND


# ---------------------------------------------------------------------------
# STALE compression
# ---------------------------------------------------------------------------

def test_stale_compresses_not_passes_through():
    """STALE entries should be compressed, not passed through verbatim."""
    content = "class Foo:\n    def bar(self):\n        return 42\n" * 50
    result = compress(content, ContentType.TOOL_RESULT_FILE, Label.STALE)
    # Should be much shorter than original
    assert len(result) < len(content) / 2


# ---------------------------------------------------------------------------
# Budget token estimation for dropped entries
# ---------------------------------------------------------------------------

def test_budget_handles_dropped_entries():
    """Budget should handle entries with 0 tokens (dropped)."""
    from context_firewall.budget import BudgetConfig, BudgetManager
    from context_firewall.session import SessionState
    
    config = BudgetConfig(target_tokens=100)
    manager = BudgetManager(config)
    
    session = SessionState()
    session.advance_turn()
    
    # Add an entry with empty compressed content
    session.record(
        role="tool",
        content_type=ContentType.TOOL_CALL,
        label=Label.DROP,
        original="tool call content",
        compressed="",
    )
    
    # Should not be over budget (0 tokens)
    assert not manager.is_over_budget(session)
