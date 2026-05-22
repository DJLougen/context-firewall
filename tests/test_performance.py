"""Performance tests for production readiness.

Verifies:
- Latency bounds (p50, p95, p99)
- Memory usage
- Throughput under load
- Edge cases (large messages, empty messages)
- Stability over sustained load
"""

import time
from pathlib import Path

import pytest

from honeycomb.budget import BudgetConfig
from honeycomb.compressor import compress
from honeycomb.firewall import HoneyComb, Message
from honeycomb.labels import ContentType, Label
from honeycomb.session import SessionState


# ---------------------------------------------------------------------------
# Latency tests
# ---------------------------------------------------------------------------

def test_rule_based_latency_p99():
    """Rule-based classification should complete in <1ms per message (p99)."""
    fw = HoneyComb()
    
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Fix the bug in src/foo.py"),
        Message(role="tool", content="94 passed, 2 failed in 3.5s"),
        Message(role="tool", content="class Foo:\n    def bar(self):\n        return 42"),
        Message(role="assistant", content="diff --git a/foo.py b/foo.py\n+new line"),
    ]
    
    # Warmup
    for msg in messages:
        fw.process(msg)
    
    # Measure
    fw2 = HoneyComb()
    latencies = []
    for _ in range(1000):
        msg = messages[len(latencies) % len(messages)]
        start = time.perf_counter()
        fw2.process(msg)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)
    
    latencies.sort()
    p50 = latencies[500]
    p95 = latencies[950]
    p99 = latencies[990]
    
    # Production requirements
    assert p50 < 0.5, f"p50 latency {p50:.3f}ms exceeds 0.5ms"
    assert p95 < 1.0, f"p95 latency {p95:.3f}ms exceeds 1.0ms"
    assert p99 < 1.0, f"p99 latency {p99:.3f}ms exceeds 1.0ms"


def test_ml_classifier_latency_p99():
    """ML classification should complete in <5ms per message (p99)."""
    model_path = Path("models/context_firewall.joblib")
    if not model_path.exists():
        pytest.skip("Model not trained")
    
    fw = HoneyComb(model_path=model_path)
    
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Fix the bug in src/foo.py"),
        Message(role="tool", content="94 passed, 2 failed in 3.5s"),
    ]
    
    # Warmup
    for msg in messages:
        fw.process(msg)
    
    # Measure
    fw2 = HoneyComb(model_path=model_path)
    latencies = []
    for _ in range(1000):
        msg = messages[len(latencies) % len(messages)]
        start = time.perf_counter()
        fw2.process(msg)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)
    
    latencies.sort()
    p50 = latencies[500]
    p95 = latencies[950]
    p99 = latencies[990]
    
    # Production requirements
    assert p50 < 2.0, f"p50 latency {p50:.3f}ms exceeds 2.0ms"
    assert p95 < 5.0, f"p95 latency {p95:.3f}ms exceeds 5.0ms"
    assert p99 < 5.0, f"p99 latency {p99:.3f}ms exceeds 5.0ms"


def test_compression_latency():
    """Compression should complete in <0.5ms per message."""
    content = "class Foo:\n    def bar(self):\n        return 42\n" * 100
    
    latencies = []
    for _ in range(1000):
        start = time.perf_counter()
        compress(content, ContentType.TOOL_RESULT_FILE, Label.COMPACT)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)
    
    latencies.sort()
    p99 = latencies[990]
    
    assert p99 < 0.5, f"Compression p99 latency {p99:.3f}ms exceeds 0.5ms"


def test_cool_loop_latency():
    """Cool loop should complete in <10ms for typical sessions."""
    fw = HoneyComb(cool_interval=100)  # Don't auto-trigger
    
    # Build up a session
    for i in range(50):
        fw.process(Message(
            role="tool",
            content=f"Command output {i}\n" + "output line\n" * 20,
            content_type=ContentType.TOOL_RESULT_COMMAND,
        ))
    
    # Measure cool loop
    start = time.perf_counter()
    fw._cool_pass()
    elapsed = (time.perf_counter() - start) * 1000  # ms
    
    assert elapsed < 10, f"Cool loop took {elapsed:.2f}ms, exceeds 10ms"


# ---------------------------------------------------------------------------
# Memory tests
# ---------------------------------------------------------------------------

def test_memory_usage_reasonable():
    """Session memory should not exceed reasonable bounds."""
    import sys
    
    fw = HoneyComb()
    
    # Process 100 messages
    for i in range(100):
        fw.process(Message(
            role="tool",
            content=f"Message {i}\n" + "content " * 100,
            content_type=ContentType.TOOL_RESULT_COMMAND,
        ))
    
    # Estimate memory usage
    session_size = sys.getsizeof(fw.session.entries)
    for entry in fw.session.entries:
        session_size += sys.getsizeof(entry)
        session_size += sys.getsizeof(entry.original_content)
        session_size += sys.getsizeof(entry.compressed_content)
    
    # Should be under 10MB for 100 messages
    assert session_size < 10_000_000, f"Session uses {session_size} bytes, exceeds 10MB"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_very_large_message():
    """Should handle very large messages (>1MB) without crashing."""
    content = "x" * 1_500_000  # 1.5MB
    
    fw = HoneyComb()
    start = time.perf_counter()
    result = fw.process(Message(role="tool", content=content, content_type=ContentType.TOOL_RESULT_COMMAND))
    elapsed = (time.perf_counter() - start) * 1000  # ms
    
    # Should complete in reasonable time (<100ms)
    assert elapsed < 100, f"Large message took {elapsed:.2f}ms"
    assert result.content is not None


def test_empty_message():
    """Should handle empty messages gracefully."""
    fw = HoneyComb()
    result = fw.process(Message(role="tool", content="", content_type=ContentType.TOOL_RESULT_COMMAND))
    
    assert result.content == ""
    assert result.compressed_tokens == 0


def test_unicode_content():
    """Should handle Unicode content without errors."""
    content = "Hello 世界 🌍 Привет мир"
    
    fw = HoneyComb()
    result = fw.process(Message(role="user", content=content))
    
    assert result.content is not None


def test_binary_like_content():
    """Should handle binary-like content (null bytes, etc.)."""
    content = "output\x00with\x00nulls\x00and\x00binary"
    
    fw = HoneyComb()
    result = fw.process(Message(role="tool", content=content))
    
    assert result.content is not None


# ---------------------------------------------------------------------------
# Throughput tests
# ---------------------------------------------------------------------------

def test_sustained_throughput():
    """Should sustain >1000 msg/s over extended period."""
    fw = HoneyComb(thread_safe=False, metrics_enabled=False)  # Max performance

    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Fix the bug"),
        Message(role="tool", content="94 passed"),
    ]
    
    # Process 10,000 messages
    start = time.perf_counter()
    for i in range(10_000):
        msg = messages[i % len(messages)]
        fw.process(msg)
    elapsed = time.perf_counter() - start
    
    throughput = 10_000 / elapsed
    
    assert throughput > 1000, f"Throughput {throughput:.0f} msg/s is below 1000 msg/s"


def test_no_memory_leak():
    """Memory should not grow unbounded over time."""
    import gc
    
    fw = HoneyComb()
    
    # Process messages in batches
    initial_count = len(gc.get_objects())
    
    for batch in range(10):
        for i in range(100):
            fw.process(Message(
                role="tool",
                content=f"Batch {batch} message {i}",
                content_type=ContentType.TOOL_RESULT_COMMAND,
            ))
        
        gc.collect()
    
    final_count = len(gc.get_objects())
    
    # Allow some growth, but not unbounded
    growth_ratio = final_count / initial_count
    assert growth_ratio < 2.0, f"Object count grew by {growth_ratio:.2f}x"


# ---------------------------------------------------------------------------
# Compression quality
# ---------------------------------------------------------------------------

def test_compression_ratio_realistic():
    """Compression should achieve >5x ratio on realistic workloads."""
    fw = HoneyComb()
    
    # Simulate realistic session
    fw.process(Message(role="system", content="You are a helpful coding assistant."))
    fw.process(Message(role="user", content="Fix the bug in src/foo.py"))
    
    for i in range(20):
        fw.process(Message(
            role="tool",
            content="94 passed, 2 failed in 3.5s\n" + "test output line\n" * 100,
        ))
        fw.process(Message(
            role="tool",
            content="class Foo:\n    def bar(self):\n        pass\n" * 50,
        ))
    
    stats = fw.get_stats()
    ratio = stats["compression_ratio"]
    
    assert ratio > 5, f"Compression ratio {ratio:.1f}x is below 5x"


def test_compression_preserves_important_info():
    """Compression should preserve error messages and test results."""
    fw = HoneyComb()
    
    # Error trace
    error_result = fw.process(Message(
        role="tool",
        content='Traceback (most recent call last):\n  File "foo.py", line 42\nValueError: invalid input',
    ))
    
    assert "ValueError" in error_result.content
    assert "foo.py" in error_result.content
    
    # Test output
    test_result = fw.process(Message(
        role="tool",
        content="94 passed, 2 failed in 3.5s\nAssertionError: expected 5 got 3",
    ))
    
    assert "passed" in test_result.content
    assert "failed" in test_result.content


# ---------------------------------------------------------------------------
# Stability tests
# ---------------------------------------------------------------------------

def test_repeated_processing_consistent():
    """Processing the same message multiple times should be consistent."""
    fw = HoneyComb()
    
    msg = Message(role="tool", content="94 passed, 2 failed")
    
    result1 = fw.process(msg)
    result2 = fw.process(msg)
    
    # Should produce same label
    assert result1.label == result2.label
    assert result1.content_type == result2.content_type


def test_session_isolation():
    """Different firewall instances should have isolated sessions."""
    fw1 = HoneyComb()
    fw2 = HoneyComb()
    
    fw1.process(Message(role="system", content="Session 1"))
    fw2.process(Message(role="system", content="Session 2"))
    
    assert fw1.session.turn_count == 1
    assert fw2.session.turn_count == 1
    
    window1 = fw1.get_context_window()
    window2 = fw2.get_context_window()
    
    assert window1[0]["content"] == "Session 1"
    assert window2[0]["content"] == "Session 2"
