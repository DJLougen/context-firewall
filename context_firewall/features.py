"""Feature extraction for message classification.

Extracts signals from messages that help the classifier assign labels.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from context_firewall.session import SessionState


def extract_text_features(content: str) -> dict[str, float | int | bool]:
    """Extract text-based features from message content.
    
    These are cheap signals that correlate with content type and label.
    """
    lines = content.split("\n")
    words = content.split()
    
    # Structural signals
    has_code_block = "```" in content
    has_paths = bool(re.search(r"[A-Za-z]:[/\\][^\s]+|/[^\s]+", content))
    has_traceback = bool(re.search(r"Traceback|Error|Exception", content, re.I))
    has_test_output = bool(re.search(r"\d+\s*(passed|failed|error)", content, re.I))
    has_exit_code = bool(re.search(r"exit[= ]+\d+|returned \d+", content, re.I))
    has_diff = bool(re.search(r"^[-+]{3} |^@@ |^diff --git", content, re.M))
    
    # Content signals
    error_count = len(re.findall(r"Error|Exception|Failed|Traceback", content, re.I))
    path_count = len(re.findall(r"[A-Za-z]:[/\\][^\s]+|/[^\s]+", content))
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
        "error_count": error_count,
        "path_count": path_count,
        "line_count": line_count,
        "word_count": word_count,
        "char_count": char_count,
        "is_short": is_short,
        "is_long": is_long,
    }


def extract_session_features(content: str, session: "SessionState | None") -> dict[str, float | int]:
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
    file_age = session.get_file_age(content)
    
    return {
        "turn_age": turn_age,
        "is_duplicate": is_duplicate,
        "file_age": file_age,
    }


def extract_features(content: str, role: str, session: "SessionState | None" = None) -> dict[str, float | int | bool]:
    """Extract all features for classification.
    
    Combines text features with session-context features.
    """
    features = extract_text_features(content)
    features.update(extract_session_features(content, session))
    features["role"] = role
    return features


def features_to_text(features: dict[str, float | int | bool]) -> str:
    """Convert features dict to text representation for TF-IDF.
    
    This is used by the classifier to build feature vectors.
    """
    parts = []
    
    # Add role
    parts.append(f"role={features.get('role', 'unknown')}")
    
    # Add boolean flags
    for key in ["has_code_block", "has_paths", "has_traceback", "has_test_output", 
                "has_exit_code", "has_diff", "is_short", "is_long", "is_duplicate"]:
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
    
    return " ".join(parts)
