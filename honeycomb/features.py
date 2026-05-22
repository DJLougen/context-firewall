"""Feature extraction for message classification."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from honeycomb.session import SessionState


# Max content size to analyze for features (10KB)
MAX_CONTENT_SIZE = 10_000


# Cache compiled regex patterns
_RE_PATHS = re.compile(r"[A-Za-z]:[/\\][^\s]+|/[^\s]+")
_RE_TRACEBACK = re.compile(r"Traceback|Error|Exception", re.I)
_RE_TEST_OUTPUT = re.compile(r"\d+\s*(passed|failed|error)", re.I)
_RE_EXIT_CODE = re.compile(r"exit[= ]+\d+|returned \d+", re.I)
_RE_DIFF = re.compile(r"^[-+]{3} |^@@ |^diff --git", re.M)
_RE_ERROR_COUNT = re.compile(r"Error|Exception|Failed|Traceback", re.I)
# Content-type signal
_RE_TOOL_CALL = re.compile(r'^tool_name\s*:\s*\w+|"name"\s*:\s*"[^"]+"', re.M)
_RE_CODE_STRUCTURE = re.compile(r"^(class|def|import|from|export|function|pub |fn |struct |enum |impl )", re.M)
_RE_NUMBERED_LIST = re.compile(r"^\s*\d+[.)]\s+", re.M)
def extract_text_features(content: str) -> dict[str, float | int | bool]:
    """Extract text-based features from message content.
    
    These are cheap signals that correlate with content type and label.
    Caps content to MAX_CONTENT_SIZE for performance.
    """
    # Cap content size for analysis
    analysis_content = content[:MAX_CONTENT_SIZE]
    
    lines = analysis_content.split("\n")
    words = analysis_content.split()
    
    # Structural signals
    has_code_block = "```" in content
    has_paths = bool(_RE_PATHS.search(analysis_content))
    has_traceback = bool(_RE_TRACEBACK.search(analysis_content))
    has_test_output = bool(_RE_TEST_OUTPUT.search(analysis_content))
    has_exit_code = bool(_RE_EXIT_CODE.search(analysis_content))
    has_diff = bool(_RE_DIFF.search(analysis_content))
    has_tool_call = bool(_RE_TOOL_CALL.search(analysis_content))
    has_code_structure = bool(_RE_CODE_STRUCTURE.search(analysis_content))
    has_numbered_list = bool(_RE_NUMBERED_LIST.search(analysis_content))
    
    # Content signals
    error_count = len(_RE_ERROR_COUNT.findall(analysis_content))
    path_count = len(_RE_PATHS.findall(analysis_content))
    line_count = len(lines)
    word_count = len(words)
    char_count = len(content)
    
    # Age-related signals (will be combined with session state)
    is_short = char_count < 100
    is_long = char_count > 5000
    
    return {
        "has_code_block": has_code_block,
        "has_paths": has_paths,
        "has_traceback": has_traceback,
        "has_test_output": has_test_output,
        "has_exit_code": has_exit_code,
        "has_diff": has_diff,
        "has_tool_call": has_tool_call,
        "has_code_structure": has_code_structure,
        "has_numbered_list": has_numbered_list,
        "error_count": error_count,
        "path_count": path_count,
        "line_count": line_count,
        "word_count": word_count,
        "char_count": char_count,
        "is_short": is_short,
        "is_long": is_long,
    }


def extract_session_features(content: str, session: "SessionState | None" = None, file_paths: list[str] | None = None) -> dict[str, float | int]:
    """Extract session-context features.
    
    These depend on what's already in the session (turn age, seen-before, etc.).
    """
    if session is None:
        return {
            "turn_age": 0,
            "is_duplicate": False,
            "file_age": 0,
        }
    
    # Turn age: how many turns ago was similar content?
    turn_age = session.get_turn_age(content)
    
    # Duplicate detection: is this content already in context?
    is_duplicate = session.is_duplicate(content)
    
    # File age: if this is about a file, how long since last reference?
    file_age = session.get_file_age(content, file_paths)
    
    return {
        "turn_age": turn_age,
        "is_duplicate": is_duplicate,
        "file_age": file_age,
    }


def extract_features(content: str, role: str, session: "SessionState | None" = None, file_paths: list[str] | None = None, content_type: str | None = None) -> dict[str, float | int | bool]:
    """Extract all features for classification.

    Combines text features with session-context features.
    """
    features = extract_text_features(content)
    features.update(extract_session_features(content, session, file_paths))
    features["role"] = role
    if content_type:
        features["content_type"] = content_type
    return features


def features_to_text(features: dict[str, float | int | bool]) -> str:
    """Convert features dict to text representation for TF-IDF.

    This is used by the classifier to build feature vectors.
    """
    parts = []

    # Add role
    parts.append(f"role={features.get('role', 'unknown')}")

    # Add content type (strongest signal for label prediction)
    ct = features.get("content_type")
    if ct:
        parts.append(f"ct={ct}")

    # Add boolean flags
    for key in ["has_code_block", "has_paths", "has_traceback", "has_test_output",
                "has_exit_code", "has_diff", "has_tool_call", "has_code_structure",
                "has_numbered_list", "is_short", "is_long", "is_duplicate"]:
        if features.get(key):
            parts.append(key)

    # Add numeric ranges (bucketed to avoid overfitting to exact values)
    line_count = features.get("line_count", 0)
    if line_count < 10:
        parts.append("lines_few")
    elif line_count < 50:
        parts.append("lines_some")
    elif line_count < 200:
        parts.append("lines_many")
    else:
        parts.append("lines_huge")

    error_count = features.get("error_count", 0)
    if error_count == 0:
        parts.append("errors_none")
    elif error_count < 3:
        parts.append("errors_few")
    else:
        parts.append("errors_many")

    # Add session-context features (turn age, file age)
    turn_age = features.get("turn_age", 0)
    if turn_age == 0:
        parts.append("age_new")
    elif turn_age < 5:
        parts.append("age_recent")
    elif turn_age < 15:
        parts.append("age_medium")
    else:
        parts.append("age_old")

    file_age = features.get("file_age", 0)
    if file_age == 0:
        parts.append("fileage_none")
    elif file_age < 5:
        parts.append("fileage_recent")
    else:
        parts.append("fileage_old")

    return " ".join(parts)