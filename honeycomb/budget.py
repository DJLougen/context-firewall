"""Token budget management for context window compression.

The budget manager enforces a target token budget on the context window.
When the budget is exceeded, it force-downgrades the lowest-priority
entries to more aggressive compression levels.

Priority order (highest retention):
  CORE (100) > ESCALATE (90) > DISTILL (80) > COMPACT (60) > STALE (20) > DROP (0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from honeycomb.compressor import compress
from honeycomb.labels import Label

if TYPE_CHECKING:
    from honeycomb.session import ContextEntry, SessionState


# Downgrade chain: when under pressure, labels get more aggressive
_DOWNGRADE: dict[Label, Label] = {
    Label.CORE: Label.DISTILL,
    Label.DISTILL: Label.COMPACT,
    Label.COMPACT: Label.DROP,
    Label.STALE: Label.DROP,
    Label.ESCALATE: Label.DISTILL,
    Label.DROP: Label.DROP,  # Already dropped
}


@dataclass
class BudgetConfig:
    """Configuration for the budget manager."""
    
    target_tokens: int = 10_000
    """Target token budget for the context window."""
    
    headroom: float = 0.1
    """Fraction of target to keep as headroom (default 10%)."""
    
    @property
    def hard_limit(self) -> int:
        """Hard token limit (target + headroom)."""
        return int(self.target_tokens * (1.0 + self.headroom))


class BudgetManager:
    """Enforces token budget on the context window.
    
    When the budget is exceeded, the manager force-downgrades entries
    starting from the lowest priority labels.
    """
    
    def __init__(self, config: BudgetConfig | None = None) -> None:
        self.config = config or BudgetConfig()
    
    def is_over_budget(self, session: "SessionState") -> bool:
        """Check if the session is over budget."""
        return session.get_total_tokens() > self.config.target_tokens
    
    def enforce(self, session: "SessionState") -> int:
        """Enforce the token budget by downgrading entries.

        Iteratively downgrades the lowest-priority entries to more
        aggressive compression until the budget is met.

        Returns the number of entries downgraded.
        """
        downgraded = 0

        # Sort once by priority (lowest first), then by turn age (oldest first)
        active = session.get_active_entries()
        candidates = sorted(active, key=lambda e: (e.label.priority(), e.turn))
        candidate_idx = 0

        while session.get_total_tokens() > self.config.target_tokens:
            if candidate_idx >= len(candidates):
                break

            # Find the next entry that can be downgraded
            entry = None
            while candidate_idx < len(candidates):
                candidate = candidates[candidate_idx]
                candidate_idx += 1
                if _DOWNGRADE.get(candidate.label, candidate.label) != candidate.label:
                    entry = candidate
                    break

            if entry is None:
                # No more downgrades possible
                break

            # Apply downgrade
            new_label = _DOWNGRADE[entry.label]
            old_tokens = entry.compressed_tokens

            if new_label == Label.DROP:
                session._cached_total_tokens -= old_tokens
                entry.mark_dropped()
            else:
                # Re-compress with new label
                entry.compressed_content = compress(
                    entry.original_content,
                    entry.content_type,
                    new_label,
                )
                new_tokens = max(0, len(entry.compressed_content) // 4)
                session._cached_total_tokens += (new_tokens - old_tokens)
                entry.compressed_tokens = new_tokens
                entry.label = new_label

            downgraded += 1

            # Safety: if no tokens were freed, break to avoid infinite loop
            if new_label != Label.DROP:
                new_tokens = entry.compressed_tokens
                if new_tokens >= old_tokens:
                    break

        return downgraded
