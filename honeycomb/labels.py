"""Label taxonomy and content type definitions for context depollution."""

from __future__ import annotations

from enum import Enum


class Label(str, Enum):
    """Depollution strategy labels for context messages.

    The classifier assigns one of these to each incoming message.
    Deterministic regex extractors then apply the strategy — no model
    reads or understands the text.
    """
    
    CORE = "core"
    """Keep verbatim. Active goal, current error, system instructions."""
    
    DISTILL = "distill"
    """Extract key info, discard rest. Tool outputs, reasoning conclusions."""
    
    COMPACT = "compact"
    """Structural summary. File contents → path + line count + key symbols."""
    
    DROP = "drop"
    """Remove entirely. Superseded content, completed tool calls."""
    
    STALE = "stale"
    """Was relevant, now superseded by newer version. Dropped on next cool pass."""
    
    ESCALATE = "escalate"
    """Ambiguous content, needs LLM judgment. Rare."""
    
    def priority(self) -> int:
        """Retention priority (higher = keep longer under budget pressure).
        
        Used by budget manager when forced to drop content.
        """
        priorities = {
            Label.CORE: 100,
            Label.DISTILL: 80,
            Label.COMPACT: 60,
            Label.STALE: 20,
            Label.DROP: 0,
            Label.ESCALATE: 90,  # Keep until LLM decides
        }
        return priorities[self]


class ContentType(str, Enum):
    """Content types for messages in an agent session.

    The depolluter uses these to apply type-specific extraction rules.
    """
    
    SYSTEM = "system"
    """System prompt or instructions."""
    
    USER_GOAL = "user_goal"
    """User-provided goal or task description."""
    
    AGENT_REASONING = "reasoning"
    """Agent's internal reasoning chain."""
    
    TOOL_CALL = "tool_call"
    """Agent's request to invoke a tool."""
    
    TOOL_RESULT_FILE = "tool_result_file"
    """File read operation result."""
    
    TOOL_RESULT_TEST = "tool_result_test"
    """Test execution result."""
    
    TOOL_RESULT_COMMAND = "tool_result_command"
    """Shell command execution result."""
    
    TOOL_RESULT_SEARCH = "tool_result_search"
    """Search/query result."""
    
    TOOL_RESULT_ERROR = "tool_result_error"
    """Error trace or exception."""
    
    AGENT_PATCH = "agent_patch"
    """Agent's code edit or patch."""

    TOOL_RESULT_GIT = "tool_result_git"
    """Git command output (status, log, diff, push, pull, commit)."""

    TOOL_RESULT_BUILD = "tool_result_build"
    """Build/compile output (cargo build, tsc, next build, etc.)."""

    TOOL_RESULT_LINT = "tool_result_lint"
    """Lint output (ruff, eslint, clippy, golangci-lint, etc.)."""

    TOOL_RESULT_CONTAINER = "tool_result_container"
    """Container output (docker ps/images/logs, kubectl)."""

    TOOL_RESULT_DIRECTORY = "tool_result_directory"
    """Directory listing output (ls, tree, find)."""
    
    UNKNOWN = "unknown"
    """Unrecognized content type."""
    
    def is_tool_result(self) -> bool:
        """Check if this is a tool result type."""
        return self.value.startswith("tool_result_")
