# Honey-Comb Architecture

## Overview

Honey-Comb is a CPU-only inline context compression system for agent harnesses. It applies the same principle as busyBee-cpu to context management: **most compression decisions are mechanical**.

## Two-Loop Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HOT LOOP (per message)                     │
│                     ~0.035ms (rules)                          │
│                     ~0.8ms (ML)                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │         Raw Agent Message                │
        │  role: "tool"                           │
        │  content: "94 passed, 2 failed..."      │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      Feature Extraction                 │
        │  - Role                                 │
        │  - Content type signals                 │
        │  - Length metrics                       │
        │  - Duplicate detection                  │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      Classification                     │
        │  Rule-based OR ML Classifier            │
        │  → Label: CORE/DISTILL/COMPACT/         │
        │           DROP/STALE/ESCALATE           │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      Compression                        │
        │  Deterministic rules per label          │
        │  - Test output → summary + failures     │
        │  - File → structure skeleton            │
        │  - Reasoning → key conclusions          │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      Compressed Context Entry           │
        │  role: "tool"                           │
        │  content: "94 passed, 2 failed\n        │
        │           Failures:\n                   │
        │           test_auth.py::test_jwt"       │
        └─────────────────────────────────────────┘
                              │
                              ▼
                    [Context Window]


┌─────────────────────────────────────────────────────────────┐
│                   COOL LOOP (every N turns)                   │
│                     ~10-50ms                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      Staleness Detection                │
        │  - File read before later edit? → DROP  │
        │  - File read again later? → DROP        │
        │  - Error before successful test? → DROP │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      Budget Enforcement                 │
        │  - Over token budget?                   │
        │  - Force-downgrade lowest priority      │
        │  - COMPACT → DROP                       │
        └─────────────────────────────────────────┘
                              │
                              ▼
                    [Clean Context Window]
```

## Label Taxonomy

| Label    | Strategy          | Example                                              |
|----------|-------------------|------------------------------------------------------|
| CORE     | Keep verbatim     | System prompt, user goal, current error              |
| DISTILL  | Extract key info  | Test output → "94 passed, 2 failed" + failure details|
| COMPACT  | Structural summary| File → "src/foo.py (200 lines): class Foo, def bar()"|
| DROP     | Remove entirely   | Completed tool calls (result is what matters)        |
| STALE    | Mark for deletion | File read before a later edit of the same file       |
| ESCALATE | Defer to LLM      | Ambiguous content (rare)                             |

## Compression Examples

### Test Output (DISTILL)
```
BEFORE (500 lines):
============================= test session starts ==============================
platform linux -- Python 3.12.0
rootdir: /home/user/project
collected 94 items

tests/test_auth.py::test_create_token PASSED                             [  1%]
tests/test_auth.py::test_validate_token PASSED                           [  2%]
... (494 more lines) ...
tests/test_api.py::test_endpoint_42 FAILED                              [100%]

================================== FAILURES ===================================
__________________________ test_endpoint_42 ___________________________
def test_endpoint_42():
>       assert response.status_code == 200
E       AssertionError: assert 500 == 200
=========================== short test summary info ============================
FAILED tests/test_api.py::test_endpoint_42 - AssertionError: assert 500 == 200
========================= 2 failed, 92 passed in 3.5s =========================

AFTER (3 lines):
92 passed, 2 failed in 3.5s
Failures:
tests/test_api.py::test_endpoint_42
AssertionError: assert 500 == 200
```

### File Content (COMPACT)
```
BEFORE (200 lines):
import jwt
import time
from datetime import datetime, timedelta

SECRET_KEY = "super-secret-key"

class TokenExpiredError(Exception):
    pass

class TokenInvalidError(Exception):
    pass

def create_token(user_id: int, expires_in: int = 3600) -> str:
    """Create a JWT token for a user."""
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(seconds=expires_in),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def validate_token(token: str) -> dict:
    """Validate a JWT token and return the payload."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return True  # BUG
    except jwt.InvalidTokenError as e:
        raise TokenInvalidError(f"Invalid token: {e}")

# ... 180 more lines ...

AFTER (5 lines):
src/auth.py (200 lines):
  imports: jwt, time, datetime, timedelta
  constants: SECRET_KEY
  classes: TokenExpiredError, TokenInvalidError
  functions: create_token(), validate_token(), refresh_token(), revoke_token()
```

### Reasoning (DISTILL)
```
BEFORE (15 lines):
Looking at the test failures, I can see that test_endpoint_42 is failing
because the response status code is 500 instead of 200. This suggests
there's an internal server error in the endpoint. Let me read the endpoint
code to understand what's happening. The endpoint is defined in
src/api/endpoints.py around line 42. I should also check if there are any
recent changes to this file that might have introduced the bug. Actually,
looking at the git log, I can see that this endpoint was modified 2 hours
ago to add input validation. That might be the culprit - perhaps the
validation is too strict and rejecting valid requests.

AFTER (3 lines):
Test failure: test_endpoint_42 (500 vs 200)
Root cause: Recent validation changes in src/api/endpoints.py (2 hours ago)
Action: Review input validation logic
```

## Integration Points

### Hot Loop Integration
```python
from honeycomb import HoneyComb, Message

hc = HoneyComb()

# In your agent loop
for raw_message in agent_messages:
    compressed = hc.process(Message(
        role=raw_message["role"],
        content=raw_message["content"],
    ))
    # Send compressed to LLM
    llm_response = llm.complete(compressed)
```

### Cool Loop Integration
```python
from honeycomb import HoneyComb, Message
from honeycomb.budget import BudgetConfig

hc = HoneyComb(
    budget_config=BudgetConfig(target_tokens=10_000),
    cool_interval=5,  # Run every 5 turns
)

# Cool loop runs automatically during process()
# Or trigger manually:
hc.run_cool_loop()
```

## Thread Safety

All session state operations are thread-safe by default:

```python
import threading
from honeycomb import HoneyComb, Message

hc = HoneyComb(thread_safe=True)  # Default

# Safe to call from multiple threads
def worker(thread_id):
    for i in range(100):
        hc.process(Message(role="tool", content=f"Thread {thread_id}: {i}"))

threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# All 500 messages processed correctly, no race conditions
assert len(hc.session.entries) == 500
```

For single-threaded workloads, disable locks for 1.45x performance:

```python
hc = HoneyComb(thread_safe=False)  # 24,000 msg/s vs 17,000 msg/s
```

## Observability

### Structured Logging
```python
from honeycomb import setup_logging

logger = setup_logging(level="INFO", json_format=True)
# 2026-05-22 10:30:45,123 [INFO] honeycomb.firewall: Processed message (role=tool, label=DISTILL, compression=6.3x)
```

### Metrics
```python
from honeycomb import metrics

# Automatic metrics
print(f"Messages: {metrics.messages_processed.value}")
print(f"Compression p50: {metrics.compression_ratio.get_percentile(50):.2f}x")
print(f"Compression p95: {metrics.compression_ratio.get_percentile(95):.2f}x")
print(f"Avg latency: {metrics.processing_latency_seconds.get_mean() * 1000:.3f}ms")

# Export for Prometheus
prometheus_text = metrics.export_prometheus()
# # HELP honeycomb_messages_processed_total Total messages processed
# # TYPE honeycomb_messages_processed_total counter
# honeycomb_messages_processed_total 1234.0
```

### Health Checks
```python
from honeycomb import health_checker

health = health_checker.check()
print(f"Status: {health.status}")  # "healthy" / "degraded" / "unhealthy"
print(f"Uptime: {health.uptime_seconds:.1f}s")
print(f"Messages: {health.messages_processed}")
print(f"Errors: {health.errors}")
```

## Performance Characteristics

### Latency
- **Rule-based classification**: 0.035ms per message
- **ML classification**: 0.8ms per message
- **Compression**: 0.1-5ms depending on content size
- **Total hot loop**: 0.1-6ms per message

### Throughput
- **Production mode** (thread-safe + metrics): ~17,000 msg/s
- **High-perf mode** (single-threaded, no metrics): ~24,000 msg/s

### Compression Ratio
- **Realistic agent sessions**: 6.3x average
- **Test output**: 50-100x (500 lines → 3 lines)
- **File content**: 20-40x (200 lines → 5 lines)
- **Reasoning**: 3-5x (15 lines → 3 lines)

### Memory
- **Session state**: ~1KB per entry
- **ML model**: ~2MB (TF-IDF + VotingClassifier)
- **Total overhead**: <10MB for typical sessions

## Comparison: busyBee-cpu vs Honey-Comb

| Aspect | busyBee-cpu | Honey-Comb |
|--------|-------------|------------|
| **Problem** | Tool selection | Context compression |
| **Principle** | Most tool decisions are mechanical | Most compression is mechanical |
| **Classifier** | Which of 4 actions to take | Which compression strategy to apply |
| **Resolver** | Fill arguments from state | Execute compression per content type |
| **Escalation** | Defer to LLM for reasoning | Defer to LLM for ambiguous content |
| **Latency** | ~1ms | ~0.035ms (rules) / ~0.8ms (ML) |
| **Accuracy** | 96.4% on held-out SWE-bench | 94.5% on held-out eval |

Both use the same architecture: **TF-IDF + VotingClassifier on CPU**, with deterministic resolvers/compressors, escalating to the LLM only when uncertain.
