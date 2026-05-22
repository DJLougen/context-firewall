"""Benchmark the context firewall's latency and compression ratio."""

from __future__ import annotations

import time
from pathlib import Path

from honeycomb.firewall import HoneyComb, Message
from honeycomb.labels import ContentType


def benchmark_hot_loop(num_messages: int = 1000) -> dict:
    """Benchmark the hot loop (per-message processing)."""
    fw = HoneyComb()
    
    messages = [
        Message(role="system", content="You are a helpful coding assistant."),
        Message(role="user", content="Fix the bug in src/foo.py"),
        Message(role="tool", content="94 passed, 2 failed in 3.5s"),
        Message(role="tool", content="class Foo:\n    def bar(self):\n        return 42"),
        Message(role="tool", content='Traceback:\n  File "foo.py", line 42\nValueError: bad'),
        Message(role="assistant", content="The issue is that bar() returns None."),
        Message(role="assistant", content="diff --git a/foo.py b/foo.py\n+return 42"),
        Message(role="tool", content="$ npm install\nadded 123 packages\nexit=0"),
    ]
    
    # Warmup
    for msg in messages:
        fw.process(msg)
    
    # Benchmark
    fw2 = HoneyComb()
    start = time.perf_counter()
    for i in range(num_messages):
        msg = messages[i % len(messages)]
        fw2.process(msg)
    elapsed = time.perf_counter() - start
    
    per_message_ms = (elapsed / num_messages) * 1000
    
    return {
        "num_messages": num_messages,
        "total_ms": elapsed * 1000,
        "per_message_ms": per_message_ms,
        "messages_per_second": num_messages / elapsed,
    }


def benchmark_ml_classifier(num_messages: int = 1000) -> dict:
    """Benchmark with ML classifier loaded."""
    model_path = Path("models/context_firewall.joblib")
    if not model_path.exists():
        return {"error": "Model not trained yet. Run: cf-train examples/train.jsonl"}
    
    fw = HoneyComb(model_path=model_path)
    
    messages = [
        Message(role="system", content="You are a helpful coding assistant."),
        Message(role="user", content="Fix the bug in src/foo.py"),
        Message(role="tool", content="94 passed, 2 failed in 3.5s"),
        Message(role="tool", content="class Foo:\n    def bar(self):\n        return 42"),
    ]
    
    # Warmup
    for msg in messages:
        fw.process(msg)
    
    # Benchmark
    fw2 = HoneyComb(model_path=model_path)
    start = time.perf_counter()
    for i in range(num_messages):
        msg = messages[i % len(messages)]
        fw2.process(msg)
    elapsed = time.perf_counter() - start
    
    per_message_ms = (elapsed / num_messages) * 1000
    
    return {
        "num_messages": num_messages,
        "total_ms": elapsed * 1000,
        "per_message_ms": per_message_ms,
        "messages_per_second": num_messages / elapsed,
    }


def benchmark_compression_ratio() -> dict:
    """Benchmark compression ratio on a realistic session."""
    fw = HoneyComb()
    
    # Simulate a 20-turn session
    fw.process(Message(role="system", content="You are a helpful coding assistant."))
    fw.process(Message(role="user", content="Fix the bug in src/foo.py where bar() returns None."))
    
    for i in range(18):
        if i % 4 == 0:
            fw.process(Message(
                role="tool",
                content="94 passed, 2 failed in 3.5s\n" + "test output line\n" * 100,
            ))
        elif i % 4 == 1:
            fw.process(Message(
                role="tool",
                content="# src/foo.py\n" + "def func():\n    pass\n" * 50,
            ))
        elif i % 4 == 2:
            fw.process(Message(
                role="assistant",
                content=f"Reasoning about the bug in iteration {i}. " * 20,
            ))
        else:
            fw.process(Message(
                role="tool",
                content=f"$ command {i}\n" + "output line\n" * 30 + f"exit=0",
            ))
    
    stats = fw.get_stats()
    
    return {
        "turns": stats["turn_count"],
        "original_tokens": stats["original_tokens"],
        "compressed_tokens": stats["total_tokens"],
        "compression_ratio": stats["compression_ratio"],
        "active_entries": stats["active_entries"],
        "total_entries": stats["total_entries"],
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Context Firewall Benchmark")
    print("=" * 60)
    
    print("\n--- Hot Loop (rule-based) ---")
    result = benchmark_hot_loop()
    print(f"  Messages: {result['num_messages']}")
    print(f"  Per message: {result['per_message_ms']:.3f} ms")
    print(f"  Throughput: {result['messages_per_second']:.0f} msg/s")
    
    print("\n--- Hot Loop (ML classifier) ---")
    result = benchmark_ml_classifier()
    if "error" in result:
        print(f"  {result['error']}")
    else:
        print(f"  Messages: {result['num_messages']}")
        print(f"  Per message: {result['per_message_ms']:.3f} ms")
        print(f"  Throughput: {result['messages_per_second']:.0f} msg/s")
    
    print("\n--- Compression Ratio ---")
    result = benchmark_compression_ratio()
    print(f"  Turns: {result['turns']}")
    print(f"  Original: {result['original_tokens']} tokens")
    print(f"  Compressed: {result['compressed_tokens']} tokens")
    print(f"  Ratio: {result['compression_ratio']:.1f}x")
    print(f"  Active entries: {result['active_entries']}/{result['total_entries']}")
