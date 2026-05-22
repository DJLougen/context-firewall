"""Token savings analytics (rtk gain-style).

Tracks compression savings over time, persisted to a local JSON file.
Provides a CLI (``honeycomb-gain``) and programmatic API for viewing
savings history, daily breakdowns, and discovering missed opportunities.

Usage:
    from honeycomb.gain import GainTracker

    tracker = GainTracker()
    tracker.record(
        command="pytest",
        raw_tokens=2000,
        compressed_tokens=200,
    )

    # View summary
    print(tracker.summary())

    # View daily breakdown
    print(tracker.daily())

    # View recent history
    print(tracker.history(limit=10))
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class GainEntry:
    """A single compression event."""

    timestamp: float
    command: str
    raw_tokens: int
    compressed_tokens: int
    label: str = ""

    @property
    def saved_tokens(self) -> int:
        return max(0, self.raw_tokens - self.compressed_tokens)

    @property
    def savings_pct(self) -> float:
        if self.raw_tokens == 0:
            return 0.0
        return (self.saved_tokens / self.raw_tokens) * 100.0

    @property
    def date(self) -> str:
        return datetime.fromtimestamp(
            self.timestamp, tz=timezone.utc
        ).strftime("%Y-%m-%d")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.timestamp,
            "cmd": self.command,
            "raw": self.raw_tokens,
            "compressed": self.compressed_tokens,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GainEntry:
        return cls(
            timestamp=d["ts"],
            command=d["cmd"],
            raw_tokens=d["raw"],
            compressed_tokens=d["compressed"],
            label=d.get("label", ""),
        )


class GainTracker:
    """Persistent token savings tracker.

    Args:
        data_dir: Directory to store the gain log. Defaults to
                  ``~/.local/share/honeycomb``.
    """

    def __init__(self, data_dir: Optional[str | Path] = None) -> None:
        if data_dir is None:
            data_dir = Path.home() / ".local" / "share" / "honeycomb"
        self._data_dir = Path(data_dir)
        self._entries: list[GainEntry] = []
        self._loaded = False
        self._buffer: list[str] = []
        self._dir_ensured = False
        self._file_handle = None

    @property
    def _path(self) -> Path:
        return self._data_dir / "gain.jsonl"

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self._path.exists():
            try:
                with open(self._path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._entries.append(
                                GainEntry.from_dict(json.loads(line))
                            )
            except (json.JSONDecodeError, KeyError):
                pass

    def record(
        self,
        command: str,
        raw_tokens: int,
        compressed_tokens: int,
        label: str = "",
    ) -> GainEntry:
        """Record a compression event.

        Args:
            command: Command or content type that was compressed.
            raw_tokens: Token count before compression.
            compressed_tokens: Token count after compression.
            label: Compression label applied (e.g. "distill", "compact").

        Returns:
            The recorded GainEntry.
        """
        entry = GainEntry(
            timestamp=time.time(),
            command=command,
            raw_tokens=raw_tokens,
            compressed_tokens=compressed_tokens,
            label=label,
        )

        self._ensure_loaded()
        self._entries.append(entry)

        # Write immediately via persistent file handle (avoids open/close overhead)
        if not self._dir_ensured:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._dir_ensured = True
        if self._file_handle is None:
            self._file_handle = open(self._path, "a")
        self._file_handle.write(json.dumps(entry.to_dict()) + "\n")
        self._file_handle.flush()

        return entry

    def _flush_buffer(self) -> None:
        """No-op (writes are immediate via persistent file handle)."""
        pass

    def __del__(self) -> None:
        """Close persistent file handle on destruction."""
        try:
            if self._file_handle is not None:
                self._file_handle.close()
        except Exception:
            pass

    def summary(self) -> str:
        """Generate a summary of total token savings."""
        self._flush_buffer()
        self._ensure_loaded()
        if not self._entries:
            return "No compression data recorded."

        total_raw = sum(e.raw_tokens for e in self._entries)
        total_compressed = sum(e.compressed_tokens for e in self._entries)
        total_saved = total_raw - total_compressed
        savings_pct = (total_saved / total_raw * 100) if total_raw else 0

        # By command
        by_cmd: dict[str, list[GainEntry]] = defaultdict(list)
        for e in self._entries:
            by_cmd[e.command].append(e)

        lines = [
            "=== Honey-Comb Token Savings ===",
            f"Total events: {len(self._entries)}",
            f"Raw tokens: {total_raw:,}",
            f"Compressed tokens: {total_compressed:,}",
            f"Saved: {total_saved:,} ({savings_pct:.1f}%)",
            "",
            "By command:",
        ]

        for cmd, entries in sorted(by_cmd.items(), key=lambda x: -sum(e.saved_tokens for e in x[1])):
            cmd_raw = sum(e.raw_tokens for e in entries)
            cmd_saved = sum(e.saved_tokens for e in entries)
            cmd_pct = (cmd_saved / cmd_raw * 100) if cmd_raw else 0
            lines.append(
                f"  {cmd}: {len(entries)}x, saved {cmd_saved:,} tokens ({cmd_pct:.0f}%)"
            )

        return "\n".join(lines)

    def daily(self, days: int = 30) -> str:
        """Generate a day-by-day breakdown.

        Args:
            days: Number of days to show (default 30).
        """
        self._flush_buffer()
        self._ensure_loaded()
        if not self._entries:
            return "No compression data recorded."

        by_date: dict[str, list[GainEntry]] = defaultdict(list)
        for e in self._entries:
            by_date[e.date].append(e)

        lines = ["=== Daily Token Savings ==="]
        dates = sorted(by_date.keys(), reverse=True)[:days]

        for date in dates:
            entries = by_date[date]
            raw = sum(e.raw_tokens for e in entries)
            saved = sum(e.saved_tokens for e in entries)
            pct = (saved / raw * 100) if raw else 0
            lines.append(f"  {date}: {len(entries)} events, saved {saved:,} ({pct:.0f}%)")

        return "\n".join(lines)

    def history(self, limit: int = 20) -> str:
        """Show recent compression events.

        Args:
            limit: Number of events to show (default 20).
        """
        self._ensure_loaded()
        if not self._entries:
            return "No compression data recorded."

        lines = ["=== Recent Compression History ==="]
        for e in self._entries[-limit:]:
            ts = datetime.fromtimestamp(
                e.timestamp, tz=timezone.utc
            ).strftime("%H:%M:%S")
            lines.append(
                f"  {ts} {e.command}: {e.raw_tokens} → {e.compressed_tokens} "
                f"(-{e.saved_tokens}, {e.savings_pct:.0f}%)"
            )

        return "\n".join(lines)

    def graph(self, days: int = 30) -> str:
        """Generate an ASCII bar graph of daily savings.

        Args:
            days: Number of days to show (default 30).
        """
        self._ensure_loaded()
        if not self._entries:
            return "No compression data recorded."

        by_date: dict[str, int] = defaultdict(int)
        for e in self._entries:
            by_date[e.date] += e.saved_tokens

        dates = sorted(by_date.keys(), reverse=True)[:days]
        if not dates:
            return "No compression data recorded."

        max_saved = max(by_date.values()) if by_date else 1
        bar_width = 40

        lines = ["=== Token Savings (last 30 days) ==="]
        for date in reversed(dates):
            saved = by_date[date]
            bar_len = int((saved / max_saved) * bar_width) if max_saved else 0
            bar = "█" * bar_len + "░" * (bar_width - bar_len)
            lines.append(f"  {date} |{bar}| {saved:,}")

        return "\n".join(lines)

    def discover(self) -> str:
        """Find missed compression opportunities.

        Identifies commands that had low savings (< 30%), suggesting
        the command filters may need improvement.
        """
        self._ensure_loaded()
        if not self._entries:
            return "No compression data recorded."

        by_cmd: dict[str, list[GainEntry]] = defaultdict(list)
        for e in self._entries:
            by_cmd[e.command].append(e)

        weak = []
        passthrough = []
        for cmd, entries in by_cmd.items():
            raw = sum(e.raw_tokens for e in entries)
            compressed = sum(e.compressed_tokens for e in entries)
            if raw == 0:
                continue
            pct = ((raw - compressed) / raw) * 100
            if pct < 30:
                weak.append((cmd, len(entries), pct))
            if pct == 0:
                passthrough.append((cmd, len(entries)))

        lines = ["=== Compression Discovery ==="]

        if weak:
            lines.append("Low savings (< 30%):")
            for cmd, count, pct in sorted(weak, key=lambda x: x[2]):
                lines.append(f"  {cmd}: {count}x, only {pct:.0f}% savings")

        if passthrough:
            lines.append("Passthrough (0% savings):")
            for cmd, count in passthrough:
                lines.append(f"  {cmd}: {count}x")

        if not weak and not passthrough:
            lines.append("All commands have good compression (> 30% savings).")

        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Export all data as JSON."""
        self._ensure_loaded()
        return {
            "version": 1,
            "entries": [asdict(e) for e in self._entries],
            "summary": {
                "total_events": len(self._entries),
                "total_raw": sum(e.raw_tokens for e in self._entries),
                "total_compressed": sum(e.compressed_tokens for e in self._entries),
                "total_saved": sum(e.saved_tokens for e in self._entries),
            },
        }

    def clear(self) -> None:
        """Clear all recorded data."""
        self._entries.clear()
        # Close persistent file handle before deleting (Windows requires this)
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
        if self._path.exists():
            self._path.unlink()


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_tracker: Optional[GainTracker] = None


def get_tracker() -> GainTracker:
    """Get or create the default GainTracker instance."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = GainTracker()
    return _default_tracker


def reset_tracker() -> None:
    """Reset the default tracker (for testing)."""
    global _default_tracker
    _default_tracker = None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    """CLI entry point for ``honeycomb-gain``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Honey-Comb token savings analytics (rtk gain-style)."
    )
    parser.add_argument(
        "--graph", action="store_true",
        help="Show ASCII bar graph of daily savings.",
    )
    parser.add_argument(
        "--history", action="store_true",
        help="Show recent compression events.",
    )
    parser.add_argument(
        "--daily", action="store_true",
        help="Show day-by-day breakdown.",
    )
    parser.add_argument(
        "--discover", action="store_true",
        help="Find missed compression opportunities.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Export all data as JSON.",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Clear all recorded data.",
    )
    parser.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format (default: text).",
    )
    args = parser.parse_args()

    tracker = get_tracker()

    if args.clear:
        tracker.clear()
        print("Cleared all gain data.")
        return

    if args.all:
        data = tracker.to_json()
        print(json.dumps(data, indent=2, default=str))
        return

    if args.graph:
        print(tracker.graph())
    elif args.history:
        print(tracker.history())
    elif args.daily:
        print(tracker.daily())
    elif args.discover:
        print(tracker.discover())
    else:
        print(tracker.summary())
