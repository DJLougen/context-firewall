"""Tests for session state tracking."""

from context_firewall.labels import ContentType, Label
from context_firewall.session import SessionState, _estimate_tokens, _extract_file_paths


def test_session_tracks_turn_count():
    """Session should track turn numbers."""
    session = SessionState()
    assert session.turn_count == 0
    session.advance_turn()
    assert session.turn_count == 1
    session.advance_turn()
    assert session.turn_count == 2


def test_session_records_entries():
    """Session should record context entries."""
    session = SessionState()
    session.advance_turn()
    
    entry = session.record(
        role="tool",
        content_type=ContentType.TOOL_RESULT_FILE,
        label=Label.DISTILL,
        original="file content here",
        compressed="compressed",
    )
    
    assert entry.turn == 1
    assert entry.role == "tool"
    assert entry.content_type == ContentType.TOOL_RESULT_FILE
    assert len(session.entries) == 1


def test_session_detects_duplicates():
    """Session should detect duplicate content."""
    session = SessionState()
    session.advance_turn()
    
    content = "identical content"
    session.record("tool", ContentType.TOOL_RESULT_COMMAND, Label.DISTILL, content, "c")
    
    assert session.is_duplicate(content) is True
    assert session.is_duplicate("different content") is False


def test_session_tracks_file_reads():
    """Session should track file reads."""
    session = SessionState()
    session.advance_turn()
    
    session.record(
        role="tool",
        content_type=ContentType.TOOL_RESULT_FILE,
        label=Label.DISTILL,
        original="Contents of src/foo.py:\nclass Foo:\n    pass",
        compressed="src/foo.py (3 lines): class Foo",
    )
    
    assert "src/foo.py" in session._file_reads


def test_session_detects_stale_file_reads():
    """File reads should become stale when the file is edited."""
    session = SessionState()
    
    # Turn 1: read foo.py
    session.advance_turn()
    read_entry = session.record(
        role="tool",
        content_type=ContentType.TOOL_RESULT_FILE,
        label=Label.DISTILL,
        original="Contents of src/foo.py:\nclass Foo:\n    pass",
        compressed="src/foo.py (3 lines): class Foo",
    )
    
    # Turn 2: edit foo.py
    session.advance_turn()
    session.record(
        role="assistant",
        content_type=ContentType.AGENT_PATCH,
        label=Label.DISTILL,
        original="diff --git a/src/foo.py b/src/foo.py\n+new line",
        compressed="Edited src/foo.py: +1/-0",
    )
    
    # The read should now be stale
    assert session.is_file_stale(read_entry) is True


def test_session_detects_superseded_file_reads():
    """File reads should be superseded by later reads of the same file."""
    session = SessionState()
    
    # Turn 1: read foo.py
    session.advance_turn()
    read1 = session.record(
        role="tool",
        content_type=ContentType.TOOL_RESULT_FILE,
        label=Label.DISTILL,
        original="Contents of src/foo.py:\nline 1",
        compressed="src/foo.py (1 lines)",
    )
    
    # Turn 2: read foo.py again
    session.advance_turn()
    session.record(
        role="tool",
        content_type=ContentType.TOOL_RESULT_FILE,
        label=Label.DISTILL,
        original="Contents of src/foo.py:\nline 1\nline 2",
        compressed="src/foo.py (2 lines)",
    )
    
    assert session.is_superseded(read1) is True


def test_session_cool_pass_drops_stale_entries():
    """Cool pass should drop stale and superseded entries."""
    session = SessionState()
    
    # Turn 1: read foo.py
    session.advance_turn()
    session.record(
        role="tool",
        content_type=ContentType.TOOL_RESULT_FILE,
        label=Label.DISTILL,
        original="Contents of src/foo.py:\nclass Foo",
        compressed="src/foo.py (1 lines)",
    )
    
    # Turn 2: edit foo.py
    session.advance_turn()
    session.record(
        role="assistant",
        content_type=ContentType.AGENT_PATCH,
        label=Label.DISTILL,
        original="diff --git a/src/foo.py b/src/foo.py\n+new",
        compressed="Edited src/foo.py",
    )
    
    # Cool pass should drop the stale read
    dropped = session.cool_pass()
    assert dropped == 1
    assert session.entries[0].dropped is True


def test_session_cool_pass_drops_stale_labeled_entries():
    """Cool pass should drop entries labeled STALE."""
    session = SessionState()
    session.advance_turn()
    
    session.record(
        role="assistant",
        content_type=ContentType.AGENT_REASONING,
        label=Label.STALE,
        original="old reasoning",
        compressed="old reasoning",
    )
    
    dropped = session.cool_pass()
    assert dropped == 1


def test_session_token_tracking():
    """Session should track token counts."""
    session = SessionState()
    session.advance_turn()
    
    session.record(
        role="tool",
        content_type=ContentType.TOOL_RESULT_FILE,
        label=Label.DISTILL,
        original="x" * 1000,  # ~250 tokens
        compressed="x" * 100,   # ~25 tokens
    )
    
    assert session.get_total_tokens() > 0
    assert session.get_total_original_tokens() > session.get_total_tokens()


def test_session_compression_ratio():
    """Session should compute compression ratio."""
    session = SessionState()
    session.advance_turn()
    
    session.record(
        role="tool",
        content_type=ContentType.TOOL_RESULT_FILE,
        label=Label.DISTILL,
        original="x" * 4000,  # ~1000 tokens
        compressed="x" * 400,   # ~100 tokens
    )
    
    ratio = session.get_compression_ratio()
    assert ratio > 1.0  # Compressed is smaller


def test_session_active_entries():
    """Session should return only non-dropped entries."""
    session = SessionState()
    session.advance_turn()
    
    session.record("tool", ContentType.TOOL_RESULT_FILE, Label.DISTILL, "a" * 100, "a" * 10)
    session.record("tool", ContentType.TOOL_RESULT_FILE, Label.DISTILL, "b" * 100, "b" * 10)
    
    session.entries[0].mark_dropped()
    
    active = session.get_active_entries()
    assert len(active) == 1


def test_extract_file_paths_windows():
    """Should extract Windows paths."""
    paths = _extract_file_paths("Error in C:/Users/foo/bar.py at line 10")
    assert "C:/Users/foo/bar.py" in paths


def test_extract_file_paths_unix():
    """Should extract Unix paths."""
    paths = _extract_file_paths("Error in /home/user/code.py at line 10")
    assert "/home/user/code.py" in paths


def test_extract_file_paths_relative():
    """Should extract relative paths with known directories."""
    paths = _extract_file_paths("Modified src/utils/helpers.py")
    assert "src/utils/helpers.py" in paths


def test_estimate_tokens():
    """Token estimation should be reasonable."""
    assert _estimate_tokens("hello world") >= 1
    assert _estimate_tokens("x" * 100) > _estimate_tokens("x" * 10)


def test_session_get_turn_age():
    """Should compute turn age for duplicate content."""
    session = SessionState()
    
    session.advance_turn()
    content = "some content"
    session.record("tool", ContentType.TOOL_RESULT_COMMAND, Label.DISTILL, content, "c")
    
    session.advance_turn()
    session.advance_turn()
    
    # Content was added 2 turns ago
    assert session.get_turn_age(content) == 2
    assert session.get_turn_age("different") == 0
