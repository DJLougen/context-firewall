"""Failure tee: save raw command output on failure for later re-read.

When a command fails, the compressed output may not contain enough context
for the LLM to diagnose the issue. The tee saves the full raw output to
a file so the LLM can reference it without re-executing the command.

Inspired by rtk's tee system: when a command fails, rtk saves the full
unfiltered output and appends a reference path to the compressed output.

Usage:
    from honeycomb.tee import FailureTee

    tee = FailureTee()
    result = tee.maybe_save(
        content=raw_output,
        command="cargo test",
        is_failure=True,
    )
    if result:
        # result.tee_path contains the saved file path
        # result.reference_line is "[full output: /path/to/file]"
        compressed += "\\n" + result.reference_line
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TeeResult:
    """Result of saving raw output via the failure tee."""

    tee_path: str
    """Absolute path where raw output was saved."""

    reference_line: str
    """Human-readable reference line to append to compressed output."""

    bytes_saved: int
    """Number of bytes saved to disk."""


class FailureTee:
    """Saves raw command output on failure for later re-read.

    Args:
        tee_dir: Directory to save tee files. Defaults to
                 ``~/.local/share/honeycomb/tee``.
        enabled: Whether the tee is active. When False, ``maybe_save``
                 returns None.
        mode: When to save. One of:
              - ``"failures"`` (default): only save on command failure
              - ``"always"``: save every command output
              - ``"never"``: equivalent to ``enabled=False``
    """

    def __init__(
        self,
        tee_dir: Optional[str | Path] = None,
        enabled: bool = True,
        mode: str = "failures",
    ) -> None:
        if tee_dir is None:
            tee_dir = Path.home() / ".local" / "share" / "honeycomb" / "tee"
        self._tee_dir = Path(tee_dir)
        self._enabled = enabled and mode != "never"
        self._mode = mode

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def tee_dir(self) -> Path:
        return self._tee_dir

    def maybe_save(
        self,
        content: str,
        command: str = "unknown",
        is_failure: bool = False,
    ) -> Optional[TeeResult]:
        """Save raw output if conditions are met.

        Args:
            content: Raw command output to potentially save.
            command: The command that produced this output (for filenames).
            is_failure: Whether the command failed.

        Returns:
            TeeResult if the output was saved, None otherwise.
        """
        if not self._enabled:
            return None

        if self._mode == "failures" and not is_failure:
            return None

        if not content or not content.strip():
            return None

        # Ensure directory exists
        self._tee_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename: timestamp_command_hash
        ts = int(time.time())
        safe_cmd = _sanitize_command(command)
        content_hash = hashlib.sha256(
            content.encode("utf-8", errors="replace")
        ).hexdigest()[:8]
        filename = f"{ts}_{safe_cmd}_{content_hash}.log"
        tee_path = self._tee_dir / filename

        # Write raw output
        content_bytes = content.encode("utf-8", errors="replace")
        tee_path.write_bytes(content_bytes)

        reference = f"[full output: {tee_path}]"

        return TeeResult(
            tee_path=str(tee_path),
            reference_line=reference,
            bytes_saved=len(content_bytes),
        )

    def read(self, tee_path: str) -> Optional[str]:
        """Read back a previously saved tee file.

        Args:
            tee_path: Path returned by a previous ``maybe_save`` call.

        Returns:
            Raw content, or None if the file doesn't exist.
        """
        path = Path(tee_path)
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
        return None

    def cleanup(self, max_age_seconds: float = 86400 * 7) -> int:
        """Remove tee files older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age in seconds (default: 7 days).

        Returns:
            Number of files removed.
        """
        if not self._tee_dir.exists():
            return 0

        now = time.time()
        removed = 0
        for entry in self._tee_dir.iterdir():
            if entry.is_file() and entry.suffix == ".log":
                age = now - entry.stat().st_mtime
                if age > max_age_seconds:
                    entry.unlink(missing_ok=True)
                    removed += 1

        return removed

    def list_files(self) -> list[str]:
        """List all saved tee files."""
        if not self._tee_dir.exists():
            return []
        return sorted(
            str(p) for p in self._tee_dir.iterdir()
            if p.is_file() and p.suffix == ".log"
        )


def _sanitize_command(command: str) -> str:
    """Convert a command string into a safe filename component."""
    # Take first 3 words, replace unsafe chars
    words = command.split()[:3]
    safe = "_".join(words)
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in safe)
    return safe[:40] or "cmd"


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_tee: Optional[FailureTee] = None


def get_tee() -> FailureTee:
    """Get or create the default FailureTee instance."""
    global _default_tee
    if _default_tee is None:
        _default_tee = FailureTee()
    return _default_tee


def reset_tee() -> None:
    """Reset the default tee instance (for testing)."""
    global _default_tee
    _default_tee = None
