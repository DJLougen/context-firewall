"""JSONL I/O for training and evaluation data.

Each line in a JSONL file is a JSON object with:
- role: message role (system, user, assistant, tool)
- content: raw message content
- content_type: inferred content type
- label: target compression label
- turn: turn number in session
- session_id: session identifier (for grouping)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from honeycomb.labels import ContentType, Label


def write_jsonl(
    path: str | Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write rows to a JSONL file.
    
    Each row should have: role, content, content_type, label, turn, session_id.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read rows from a JSONL file."""
    path = Path(path)
    rows = []
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Iterate over rows in a JSONL file (memory-efficient)."""
    path = Path(path)
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def make_row(
    role: str,
    content: str,
    content_type: ContentType,
    label: Label,
    turn: int = 0,
    session_id: str = "default",
) -> dict[str, Any]:
    """Create a training row from components."""
    return {
        "role": role,
        "content": content,
        "content_type": content_type.value,
        "label": label.value,
        "turn": turn,
        "session_id": session_id,
    }


def validate_row(row: dict[str, Any]) -> bool:
    """Validate that a row has all required fields."""
    required = {"role", "content", "content_type", "label"}
    if not required.issubset(row.keys()):
        return False
    
    # Validate content_type
    try:
        ContentType(row["content_type"])
    except ValueError:
        return False
    
    # Validate label
    try:
        Label(row["label"])
    except ValueError:
        return False
    
    return True


def load_training_data(path: str | Path) -> tuple[list[str], list[str]]:
    """Load training data and return (feature_texts, labels).
    
    This is the format expected by the classifier training pipeline.
    """
    from honeycomb.features import extract_features, features_to_text
    
    rows = read_jsonl(path)
    feature_texts = []
    labels = []
    
    skipped = 0
    for row in rows:
        if not validate_row(row):
            skipped += 1
            continue
        
        features = extract_features(row["content"], row["role"])
        feature_text = features_to_text(features)
        
        feature_texts.append(feature_text)
        labels.append(row["label"])

    if skipped > 0:
        import warnings
        warnings.warn(
            f"Skipped {skipped} invalid rows in {path}. "
            f"Each row needs: role, content, content_type, label.",
            stacklevel=2,
        )
    
    return feature_texts, labels
