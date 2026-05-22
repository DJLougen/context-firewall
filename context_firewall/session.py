"""Session state tracking for staleness detection and redundancy elimination.

The session tracker maintains a record of what's in the context window:
- Which files have been read, edited, or created (with turn numbers)
- Which tools have been called (with turn numbers)
- Content hashes for duplicate detection
- Turn age for each entry

This powers the cool loop's staleness checks and the classifier's
session-context features.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from context_firewall.labels import ContentType, Label


@dataclass
class ContextEntry:
    """A single entry in the compressed context window."""
    
    turn: int
    """Turn number when this entry was added."""
    
    role: str
    """Message role: system, user, assistant, tool."""
    
    content_type: ContentType
    """Classified content type."""
    
    label: Label
    """Assigned compression label."""
    
    original_content: str
    """The raw content before compression."""
    
    compressed_content: str
    """The compressed content (what the LLM sees)."""
    
    content_hash: str
    """SHA-256 hash of original content for dedup."""
    
    original_tokens: int
    """Approximate token count of original content."""
    
    compressed_tokens: int
    """Approximate token count of compressed content."""
    
    file_paths: list[str] = field(default_factory=list)
    """File paths referenced in this entry."""
    
    dropped: bool = False
    """Whether this entry has been dropped by the cool loop."""
    
    def mark_dropped(self) -> None:
        """Mark this entry as dropped."""
        self.dropped = True
def _estimate_tokens(text: str) -> int:
    """Estimate token count (~4 chars per token for English/code).

    Returns 0 for empty strings (dropped entries).
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def _extract_file_paths(content: str) -> list[str]:
    """Extract file paths mentioned in content.

    Caps content to 10KB for performance on large messages.
    """
    # Cap content size for analysis
    analysis_content = content[:10_000]

    # Match common path patterns
    patterns = [
        r"[A-Za-z]:[/\\][\w./\\-]+\.\w+",     # Windows absolute
        r"/[\w./-]+\.\w+",                      # Unix absolute
        r"(?:src|lib|tests?|scripts?)/[\w./-]+\.\w+",  # Relative with known dirs
    ]
    paths = set()
    for pattern in patterns:
        paths.update(re.findall(pattern, analysis_content))
    return sorted(paths)


def _content_hash(content: str) -> str:
    """Compute content hash for dedup."""
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]


class SessionState:
    """Tracks the state of an agent session for context management.
    
    This is the stateful component that the firewall updates on each
    message and queries during feature extraction and cool-loop passes.
    """
    
    def __init__(self) -> None:
        self.entries: list[ContextEntry] = []
        self.turn_count: int = 0
        
        # File tracking: path -> (last_read_turn, last_edit_turn)
        self._file_reads: dict[str, int] = {}
        self._file_edits: dict[str, int] = {}
        
        # Content hash tracking for dedup
        self._content_hashes: set[str] = set()
        
        # Tool call tracking
        self._tool_calls: dict[str, int] = {}  # tool_name -> last_call_turn
        
        # Optimization: track last entry index checked in cool_pass
        self._last_cool_pass_index: int = 0
        
        # Optimization: cached token total (updated incrementally)
        self._cached_total_tokens: int = 0
    
    def advance_turn(self) -> int:
        """Advance to the next turn. Returns the new turn number."""
        self.turn_count += 1
        return self.turn_count
    
    def record(
        self,
        role: str,
        content_type: ContentType,
        label: Label,
        original: str,
        compressed: str,
        file_paths: list[str] | None = None,
    ) -> ContextEntry:
        """Record a new context entry.
        
        Updates internal tracking (file reads, hashes, etc.) as a side effect.
        """
        content_hash = _content_hash(original)
        if file_paths is None:
            file_paths = _extract_file_paths(original)
        
        entry = ContextEntry(
            turn=self.turn_count,
            role=role,
            content_type=content_type,
            label=label,
            original_content=original,
            compressed_content=compressed,
            content_hash=content_hash,
            original_tokens=_estimate_tokens(original),
            compressed_tokens=_estimate_tokens(compressed),
            file_paths=file_paths,
        )
        
        self.entries.append(entry)
        self._content_hashes.add(content_hash)
        self._cached_total_tokens += entry.compressed_tokens
        
        # Update file tracking
        if content_type == ContentType.TOOL_RESULT_FILE:
            for path in file_paths:
                self._file_reads[path] = self.turn_count
        elif content_type == ContentType.AGENT_PATCH:
            for path in file_paths:
                self._file_edits[path] = self.turn_count
        
        # Update tool call tracking
        if content_type == ContentType.TOOL_CALL:
            tool_match = re.search(r"tool[_ ]?name[\"']?\s*[:=]\s*[\"']?(\w+)", original, re.I)
            if tool_match:
                self._tool_calls[tool_match.group(1)] = self.turn_count
        
        return entry
    
    def is_duplicate(self, content: str) -> bool:
        """Check if this content (or very similar) is already in context."""
        h = _content_hash(content)
        return h in self._content_hashes
    
    def get_turn_age(self, content: str) -> int:
        """Get the turn age of similar content, or 0 if not found."""
        h = _content_hash(content)
        for entry in reversed(self.entries):
            if entry.content_hash == h:
                return self.turn_count - entry.turn
        return 0
    
    def get_file_age(self, content: str, file_paths: list[str] | None = None) -> int:
        """Get the turn age since a file referenced in content was last accessed."""
        if file_paths is None:
            file_paths = _extract_file_paths(content)
        if not file_paths:
            return 0
        
        min_age = float("inf")
        for path in file_paths:
            last_access = max(
                self._file_reads.get(path, 0),
                self._file_edits.get(path, 0),
            )
            if last_access > 0:
                age = self.turn_count - last_access
                min_age = min(min_age, age)
        
        return int(min_age) if min_age != float("inf") else 0
    
    def is_file_stale(self, entry: ContextEntry) -> bool:
        """Check if a file-read entry is stale (file was edited since).
        
        A file read is stale if the file was edited in a later turn.
        """
        if entry.content_type != ContentType.TOOL_RESULT_FILE:
            return False
        
        for path in entry.file_paths:
            edit_turn = self._file_edits.get(path, 0)
            if edit_turn > entry.turn:
                return True
        
        return False
    
    def is_superseded(self, entry: ContextEntry) -> bool:
        """Check if an entry is superseded by a newer version.
        
        A file read is superseded if the same file was read again later.
        A tool call result is superseded if the same tool was called again.
        """
        if entry.dropped:
            return True
        
        # File read superseded by later read of same file
        if entry.content_type == ContentType.TOOL_RESULT_FILE:
            for path in entry.file_paths:
                read_turn = self._file_reads.get(path, 0)
                if read_turn > entry.turn:
                    return True
        
        return False
    
    def get_active_entries(self) -> list[ContextEntry]:
        """Get all non-dropped entries."""
        return [e for e in self.entries if not e.dropped]
    
    def get_total_tokens(self) -> int:
        """Get total token count of active entries.
        
        Uses cached value for O(1) performance.
        """
        return self._cached_total_tokens
    
    def get_total_original_tokens(self) -> int:
        """Get total token count of original (uncompressed) entries."""
        return sum(e.original_tokens for e in self.entries)
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio (original / compressed)."""
        original = self.get_total_original_tokens()
        compressed = self.get_total_tokens()
        if compressed == 0:
            return 0.0
        return original / compressed
    
    def cool_pass(self) -> int:
        """Run the cool loop staleness pass.
        
        Marks stale and superseded entries as dropped.
        Returns the number of entries dropped.
        
        Optimization: Only check file reads and STALE entries.
        Other content types can't become stale/superseded.
        """
        dropped = 0
        
        for entry in self.entries:
            if entry.dropped:
                continue
            
            # Only check entries that can become stale/superseded
            if entry.content_type == ContentType.TOOL_RESULT_FILE:
                if self.is_file_stale(entry) or self.is_superseded(entry):
                    self._cached_total_tokens -= entry.compressed_tokens
                    entry.mark_dropped()
                    dropped += 1
            elif entry.label == Label.STALE:
                self._cached_total_tokens -= entry.compressed_tokens
                entry.mark_dropped()
                dropped += 1
        
        return dropped
