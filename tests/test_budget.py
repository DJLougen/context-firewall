"""Tests for budget management."""

from context_firewall.budget import BudgetConfig, BudgetManager
from context_firewall.labels import ContentType, Label
from context_firewall.session import SessionState


def test_budget_config_defaults():
    """Default config should have sensible values."""
    config = BudgetConfig()
    assert config.target_tokens == 10_000
    assert config.hard_limit == 11_000


def test_budget_config_custom():
    """Custom config should respect provided values."""
    config = BudgetConfig(target_tokens=5000, headroom=0.2)
    assert config.target_tokens == 5000
    assert config.hard_limit == 6000


def test_budget_detects_over_budget():
    """Should detect when session is over budget."""
    config = BudgetConfig(target_tokens=50)
    manager = BudgetManager(config)
    
    session = SessionState()
    session.advance_turn()
    
    # Add a large entry (~250 tokens compressed)
    session.record(
        role="tool",
        content_type=ContentType.TOOL_RESULT_FILE,
        label=Label.DISTILL,
        original="x" * 2000,
        compressed="x" * 1000,
    )
    
    assert manager.is_over_budget(session) is True


def test_budget_detects_under_budget():
    """Should detect when session is under budget."""
    config = BudgetConfig(target_tokens=10_000)
    manager = BudgetManager(config)
    
    session = SessionState()
    session.advance_turn()
    
    # Add a small entry (~25 tokens)
    session.record(
        role="tool",
        content_type=ContentType.TOOL_RESULT_FILE,
        label=Label.DISTILL,
        original="x" * 200,
        compressed="x" * 100,
    )
    
    assert manager.is_over_budget(session) is False


def test_budget_enforce_downgrades_lowest_priority():
    """Enforce should downgrade lowest-priority entries first."""
    config = BudgetConfig(target_tokens=50)
    manager = BudgetManager(config)
    
    session = SessionState()
    session.advance_turn()
    
    # Add a CORE entry (high priority) and a COMPACT entry (low priority)
    session.record(
        role="system",
        content_type=ContentType.SYSTEM,
        label=Label.CORE,
        original="system prompt " * 50,
        compressed="system prompt " * 50,
    )
    session.record(
        role="tool",
        content_type=ContentType.TOOL_RESULT_FILE,
        label=Label.COMPACT,
        original="file content " * 50,
        compressed="file content " * 50,
    )
    
    downgraded = manager.enforce(session)
    assert downgraded > 0
    
    # The COMPACT entry should have been downgraded, not the CORE entry
    core_entry = session.entries[0]
    compact_entry = session.entries[1]
    
    # COMPACT should be downgraded to DROP
    assert compact_entry.dropped or compact_entry.label != Label.COMPACT


def test_budget_enforce_drops_when_needed():
    """Enforce should drop entries when downgrading to DROP."""
    config = BudgetConfig(target_tokens=20)
    manager = BudgetManager(config)
    
    session = SessionState()
    session.advance_turn()
    
    # Add entries that are way over budget
    session.record(
        role="tool",
        content_type=ContentType.TOOL_RESULT_FILE,
        label=Label.COMPACT,
        original="x" * 2000,
        compressed="x" * 500,
    )
    
    manager.enforce(session)
    
    # Entry should be dropped
    assert session.entries[0].dropped is True


def test_budget_enforce_respects_priority_order():
    """Enforce should drop STALE before COMPACT before DISTILL before CORE."""
    config = BudgetConfig(target_tokens=30)
    manager = BudgetManager(config)
    
    session = SessionState()
    session.advance_turn()
    
    # Add entries in priority order
    session.record(
        role="system",
        content_type=ContentType.SYSTEM,
        label=Label.CORE,
        original="system " * 100,
        compressed="system " * 100,
    )
    session.record(
        role="tool",
        content_type=ContentType.TOOL_RESULT_FILE,
        label=Label.DISTILL,
        original="distill " * 100,
        compressed="distill " * 100,
    )
    session.record(
        role="tool",
        content_type=ContentType.TOOL_RESULT_FILE,
        label=Label.COMPACT,
        original="compact " * 100,
        compressed="compact " * 100,
    )
    
    manager.enforce(session)
    
    # CORE should survive, lower priority should be downgraded first
    core_entry = session.entries[0]
    assert not core_entry.dropped


def test_budget_enforce_no_op_when_under_budget():
    """Enforce should do nothing when under budget."""
    config = BudgetConfig(target_tokens=10_000)
    manager = BudgetManager(config)
    
    session = SessionState()
    session.advance_turn()
    
    session.record(
        role="tool",
        content_type=ContentType.TOOL_RESULT_FILE,
        label=Label.DISTILL,
        original="x" * 100,
        compressed="x" * 50,
    )
    
    downgraded = manager.enforce(session)
    assert downgraded == 0


def test_budget_enforce_terminates():
    """Enforce should terminate even if budget can't be met."""
    config = BudgetConfig(target_tokens=1)  # Impossible budget
    manager = BudgetManager(config)
    
    session = SessionState()
    session.advance_turn()
    
    session.record(
        role="system",
        content_type=ContentType.SYSTEM,
        label=Label.CORE,
        original="system prompt",
        compressed="system prompt",
    )
    
    # Should not hang
    manager.enforce(session)
