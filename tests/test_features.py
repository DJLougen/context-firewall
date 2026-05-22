"""Tests for feature extraction."""

from context_firewall.features import extract_features, extract_text_features, features_to_text


def test_extract_text_features_detects_code_blocks():
    """Should detect code blocks."""
    content = "Here's some code:\n```python\ndef foo():\n    pass\n```"
    features = extract_text_features(content)
    assert features["has_code_block"] is True


def test_extract_text_features_detects_paths():
    """Should detect file paths."""
    content = "The file is at /home/user/code.py or C:/Users/test.txt"
    features = extract_text_features(content)
    assert features["has_paths"] is True


def test_extract_text_features_detects_tracebacks():
    """Should detect error traces."""
    content = "Traceback (most recent call last):\n  File 'foo.py', line 10\nValueError: invalid"
    features = extract_text_features(content)
    assert features["has_traceback"] is True


def test_extract_text_features_detects_test_output():
    """Should detect test output patterns."""
    content = "94 passed, 2 failed in 3.5s"
    features = extract_text_features(content)
    assert features["has_test_output"] is True


def test_extract_text_features_detects_exit_codes():
    """Should detect exit codes."""
    content = "Command finished with exit=0"
    features = extract_text_features(content)
    assert features["has_exit_code"] is True


def test_extract_text_features_detects_diffs():
    """Should detect diff/patch content."""
    content = "diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n@@ -1,3 +1,4 @@"
    features = extract_text_features(content)
    assert features["has_diff"] is True


def test_extract_text_features_counts_errors():
    """Should count error occurrences."""
    content = "Error: something went wrong\nError: another issue\nException: third problem"
    features = extract_text_features(content)
    assert features["error_count"] == 3


def test_extract_text_features_counts_lines():
    """Should count lines correctly."""
    content = "line1\nline2\nline3\nline4"
    features = extract_text_features(content)
    assert features["line_count"] == 4


def test_extract_features_includes_role():
    """Should include message role in features."""
    features = extract_features("content", "user")
    assert features["role"] == "user"


def test_features_to_text_buckets_line_count():
    """Should bucket line counts to avoid overfitting."""
    features_few = {"role": "assistant", "line_count": 5}
    features_some = {"role": "assistant", "line_count": 30}
    features_many = {"role": "assistant", "line_count": 100}
    features_huge = {"role": "assistant", "line_count": 500}
    
    assert "lines_few" in features_to_text(features_few)
    assert "lines_some" in features_to_text(features_some)
    assert "lines_many" in features_to_text(features_many)
    assert "lines_huge" in features_to_text(features_huge)


def test_features_to_text_includes_boolean_flags():
    """Should include boolean flags as tokens."""
    features = {"role": "tool", "has_traceback": True, "has_paths": False}
    text = features_to_text(features)
    assert "has_traceback" in text
    assert "has_paths" not in text


def test_features_to_text_buckets_error_count():
    """Should bucket error counts."""
    features_none = {"role": "assistant", "error_count": 0}
    features_few = {"role": "assistant", "error_count": 2}
    features_many = {"role": "assistant", "error_count": 5}
    
    assert "errors_none" in features_to_text(features_none)
    assert "errors_few" in features_to_text(features_few)
    assert "errors_many" in features_to_text(features_many)
