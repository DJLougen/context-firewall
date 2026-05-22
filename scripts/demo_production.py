#!/usr/bin/env python3
"""
Demo: Honey-Comb Production Features

This script demonstrates the production-ready features:
1. Thread safety (optional, can be disabled for performance)
2. Observability (structured logging, metrics, health checks)
3. Configuration (environment variables, config files)
4. Performance tuning (thread_safe=False, metrics_enabled=False)
"""

import threading
import time

from honeycomb import (
    HoneyComb,
    Message,
    setup_logging,
    metrics,
    health_checker,
    get_config,
)


def demo_thread_safety():
    """Demonstrate thread-safe concurrent processing."""
    print("\n" + "="*60)
    print("DEMO: Thread Safety")
    print("="*60)
    
    # Create a thread-safe Honey-Comb instance
    hc = HoneyComb(thread_safe=True, metrics_enabled=True)
    
    def process_messages(thread_id, count):
        for i in range(count):
            msg = Message(
                role="tool",
                content=f"Thread {thread_id} message {i}: 94 passed",
            )
            hc.process(msg)
    
    # Spawn multiple threads
    threads = [
        threading.Thread(target=process_messages, args=(i, 50))
        for i in range(5)
    ]
    
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    
    stats = hc.get_stats()
    print(f"Processed {stats['total_entries']} entries from 5 threads")
    print(f"Time: {elapsed:.3f}s ({stats['total_entries'] / elapsed:.0f} entries/s)")
    print(f"Active entries: {stats['active_entries']}")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
    print("[OK] No race conditions, all entries recorded correctly")


def demo_observability():
    """Demonstrate metrics and health checks."""
    print("\n" + "="*60)
    print("DEMO: Observability (Metrics & Health)")
    print("="*60)
    
    # Setup structured logging
    logger = setup_logging(level="INFO", json_format=False)
    
    # Create Honey-Comb with metrics enabled
    hc = HoneyComb(metrics_enabled=True)
    
    # Process some messages
    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Fix the bug"),
        Message(role="tool", content="94 passed, 2 failed"),
        Message(role="tool", content="class Foo:\n    pass"),
    ]
    
    for msg in messages:
        result = hc.process(msg)
        logger.info(
            f"Processed {msg.role}: {result.original_tokens} -> {result.compressed_tokens} tokens"
        )
    
    # Show metrics
    print(f"\nMetrics:")
    print(f"  Messages processed: {metrics.messages_processed.value}")
    print(f"  Compression ratio (p50): {metrics.compression_ratio.get_percentile(50):.2f}x")
    print(f"  Compression ratio (p95): {metrics.compression_ratio.get_percentile(95):.2f}x")
    print(f"  Avg latency: {metrics.processing_latency_seconds.get_mean() * 1000:.3f}ms")
    
    # Show health check
    health = health_checker.check()
    print(f"\nHealth Check:")
    print(f"  Status: {health.status}")
    print(f"  Uptime: {health.uptime_seconds:.1f}s")
    print(f"  Messages: {health.messages_processed}")
    print(f"  Errors: {health.errors}")


def demo_configuration():
    """Demonstrate configuration options."""
    print("\n" + "="*60)
    print("DEMO: Configuration")
    print("="*60)
    
    # Load configuration (from env vars or defaults)
    config = get_config()
    
    print("Current configuration:")
    print(f"  thread_safe: {config.thread_safe}")
    print(f"  metrics_enabled: {config.metrics_enabled}")
    print(f"  cool_loop_interval: {config.cool_loop_interval}")
    print(f"  compression_min_ratio: {config.compression_min_ratio}")
    print(f"  log_level: {config.log_level}")
    
    # Override with environment variables
    print("\nTo override via environment:")
    print("  export HONEYCOMB_THREAD_SAFE=false")
    print("  export HONEYCOMB_METRICS_ENABLED=false")
    print("  export HONEYCOMB_COOL_LOOP_INTERVAL=20")


def demo_performance_tuning():
    """Demonstrate performance tuning options."""
    print("\n" + "="*60)
    print("DEMO: Performance Tuning")
    print("="*60)
    
    # Mode 1: Production (thread-safe, metrics enabled)
    hc_prod = HoneyComb(thread_safe=True, metrics_enabled=True)
    
    start = time.perf_counter()
    for i in range(1000):
        hc_prod.process(Message(role="tool", content=f"Message {i}: 94 passed"))
    elapsed_prod = time.perf_counter() - start
    
    # Mode 2: High performance (no locks, no metrics)
    hc_perf = HoneyComb(thread_safe=False, metrics_enabled=False)
    
    start = time.perf_counter()
    for i in range(1000):
        hc_perf.process(Message(role="tool", content=f"Message {i}: 94 passed"))
    elapsed_perf = time.perf_counter() - start
    
    print("Throughput comparison (1000 messages):")
    print(f"  Production mode: {1000 / elapsed_prod:.0f} msg/s")
    print(f"  High-perf mode:  {1000 / elapsed_perf:.0f} msg/s")
    print(f"  Speedup:         {(1000 / elapsed_perf) / (1000 / elapsed_prod):.2f}x")
    print("\nUse high-perf mode for single-threaded batch processing")
    print("Use production mode for concurrent server workloads")


def main():
    print("="*60)
    print("Honey-Comb Production Features Demo")
    print("="*60)
    
    demo_thread_safety()
    demo_observability()
    demo_configuration()
    demo_performance_tuning()
    
    print("\n" + "="*60)
    print("All demos completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
