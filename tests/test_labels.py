"""Tests for label taxonomy and content types."""

from honeycomb.labels import ContentType, Label


def test_label_priority_ordering():
    """Labels should have a clear priority ordering for budget management."""
    assert Label.CORE.priority() > Label.DISTILL.priority()
    assert Label.DISTILL.priority() > Label.COMPACT.priority()
    assert Label.COMPACT.priority() > Label.STALE.priority()
    assert Label.STALE.priority() > Label.DROP.priority()
    assert Label.ESCALATE.priority() > Label.STALE.priority()


def test_label_values():
    """Labels should have stable string values for serialization."""
    assert Label.CORE.value == "core"
    assert Label.DISTILL.value == "distill"
    assert Label.COMPACT.value == "compact"
    assert Label.DROP.value == "drop"
    assert Label.STALE.value == "stale"
    assert Label.ESCALATE.value == "escalate"


def test_content_type_tool_result_detection():
    """is_tool_result() should correctly identify tool result types."""
    assert ContentType.TOOL_RESULT_FILE.is_tool_result()
    assert ContentType.TOOL_RESULT_TEST.is_tool_result()
    assert ContentType.TOOL_RESULT_COMMAND.is_tool_result()
    assert ContentType.TOOL_RESULT_SEARCH.is_tool_result()
    assert ContentType.TOOL_RESULT_ERROR.is_tool_result()
    
    assert not ContentType.SYSTEM.is_tool_result()
    assert not ContentType.USER_GOAL.is_tool_result()
    assert not ContentType.AGENT_REASONING.is_tool_result()
    assert not ContentType.TOOL_CALL.is_tool_result()
    assert not ContentType.AGENT_PATCH.is_tool_result()


def test_label_is_string_enum():
    """Labels should be usable as strings."""
    label = Label.CORE
    assert label == "core"
    assert label.value == "core"


def test_content_type_is_string_enum():
    """Content types should be usable as strings."""
    ct = ContentType.TOOL_RESULT_FILE
    assert ct == "tool_result_file"
    assert ct.value == "tool_result_file"
