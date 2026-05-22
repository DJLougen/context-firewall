"""Thread safety tests for SessionState.

These tests verify that concurrent access to SessionState is safe
when multiple threads are reading and writing simultaneously.
"""

import threading
import time

from honeycomb.labels import ContentType, Label
from honeycomb.session import SessionState


def test_concurrent_record():
    """Multiple threads recording entries simultaneously."""
    session = SessionState()
    num_threads = 10
    entries_per_thread = 100
    
    def record_entries(thread_id):
        for i in range(entries_per_thread):
            session.record(
                role="user",
                content_type=ContentType.USER_GOAL,
                label=Label.CORE,
                original=f"Thread {thread_id} message {i}",
                compressed=f"T{thread_id}M{i}",
            )
    
    threads = [
        threading.Thread(target=record_entries, args=(i,))
        for i in range(num_threads)
    ]
    
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    
    # Verify all entries were recorded
    assert len(session.entries) == num_threads * entries_per_thread
    assert session.get_total_tokens() > 0
    
    # Verify no data corruption
    assert len(session._content_hashes) == num_threads * entries_per_thread
    
    print(f"[OK] {num_threads} threads x {entries_per_thread} entries = {len(session.entries)} total")
    print(f"  Time: {elapsed:.3f}s ({num_threads * entries_per_thread / elapsed:.0f} entries/s)")


def test_concurrent_read_write():
    """Concurrent reads and writes don't cause race conditions."""
    session = SessionState()
    num_writers = 5
    num_readers = 5
    ops_per_thread = 50
    
    stop_flag = threading.Event()
    
    def writer(thread_id):
        for i in range(ops_per_thread):
            session.advance_turn()
            session.record(
                role="tool",
                content_type=ContentType.TOOL_RESULT_TEST,
                label=Label.DISTILL,
                original=f"Test output {thread_id}-{i}",
                compressed=f"Pass {i}",
            )
    
    def reader():
        count = 0
        while not stop_flag.is_set() and count < ops_per_thread:
            # Read operations
            _ = session.get_total_tokens()
            _ = session.get_active_entries()
            _ = session.get_compression_ratio()
            _ = session.is_duplicate(f"Test output 0-{count % 10}")
            count += 1
    
    writers = [
        threading.Thread(target=writer, args=(i,))
        for i in range(num_writers)
    ]
    readers = [
        threading.Thread(target=reader)
        for _ in range(num_readers)
    ]
    
    for t in writers + readers:
        t.start()
    
    for t in writers:
        t.join()
    
    stop_flag.set()
    
    for t in readers:
        t.join()
    
    # Verify consistency
    assert len(session.entries) == num_writers * ops_per_thread
    print(f"[OK] Concurrent read/write: {len(session.entries)} entries, no race conditions")


def test_concurrent_cool_pass():
    """Cool pass can run concurrently with record operations."""
    session = SessionState()
    
    # Pre-populate with file reads
    for i in range(10):
        session.advance_turn()
        session.record(
            role="tool",
            content_type=ContentType.TOOL_RESULT_FILE,
            label=Label.COMPACT,
            original=f"File {i}",
            compressed=f"F{i}",
            file_paths=[f"src/file{i}.py"],
        )
    
    # Now edit those files (makes them stale)
    for i in range(10):
        session.advance_turn()
        session.record(
            role="tool",
            content_type=ContentType.AGENT_PATCH,
            label=Label.DISTILL,
            original=f"Edit {i}",
            compressed=f"E{i}",
            file_paths=[f"src/file{i}.py"],
        )
    
    initial_count = len(session.get_active_entries())
    
    def run_cool_pass():
        return session.cool_pass()
    
    def add_more_entries():
        for i in range(20):
            session.advance_turn()
            session.record(
                role="user",
                content_type=ContentType.USER_GOAL,
                label=Label.CORE,
                original=f"New goal {i}",
                compressed=f"G{i}",
            )
    
    t1 = threading.Thread(target=run_cool_pass)
    t2 = threading.Thread(target=add_more_entries)
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    # Verify file reads were dropped
    final_count = len(session.get_active_entries())
    assert final_count < initial_count + 20  # Some were dropped
    print(f"[OK] Cool pass concurrent: {initial_count} -> {final_count} active entries")


def test_reentrant_lock():
    """Verify RLock allows reentrant calls from same thread."""
    session = SessionState()
    
    # get_compression_ratio calls get_total_original_tokens and get_total_tokens
    # Both acquire the lock, so it must be reentrant
    session.advance_turn()
    session.record(
        role="user",
        content_type=ContentType.USER_GOAL,
        label=Label.CORE,
        original="Test message",
        compressed="Test",
    )
    
    # This should not deadlock
    ratio = session.get_compression_ratio()
    assert ratio > 0
    print(f"[OK] Reentrant lock: compression ratio = {ratio:.2f}x")


if __name__ == "__main__":
    test_concurrent_record()
    test_concurrent_read_write()
    test_concurrent_cool_pass()
    test_reentrant_lock()
    print("\n✅ All threading tests passed")
