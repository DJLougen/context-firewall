# context-firewall

<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-22c55e?style=flat-square" alt="version"/>
  <img src="https://img.shields.io/badge/accuracy-95.8%25-22c55e?style=flat-square" alt="accuracy"/>
  <img src="https://img.shields.io/badge/training-1317%20examples-8b5cf6?style=flat-square" alt="training"/>
  <img src="https://img.shields.io/badge/tests-91%20passed-3b82f6?style=flat-square" alt="tests"/>
  <img src="https://img.shields.io/badge/license-MIT-64748b?style=flat-square" alt="license"/>
</p>

**CPU-only inline context compression for agent harnesses.**

## What It Does

Agent context windows fill up with noise: 500-line test outputs where everything passed, file contents from 10 turns ago, reasoning chains about bugs that are already fixed. Today's approach is reactive — call an LLM to summarize when the window gets too long.

context-firewall takes a different approach: **compress every message on the way in**, before it ever enters the context window. A CPU classifier (~1-5ms per message) labels each message with a compression strategy, and deterministic rules execute it. The LLM only sees clean, compressed context.

```
Every message enters the agent loop:
  raw → classify(2ms) → compress → context window
  raw → classify(2ms) → compress → context window
  raw → classify(2ms) → compress → context window
  
LLM sees: clean, compressed, non-polluted context
```

No batch summarization. No "when do I compress?" threshold. Every message, every time.

## The Two Loops

```
HOT LOOP (per message, ~1-5ms):
  raw message → classifier → label → compressor → compressed context entry

COOL LOOP (every N turns, ~10-50ms):
  walk compressed context → drop stale/superseded entries
  budget check → force-downgrade if over budget
```

Both loops are CPU-only. The LLM only ever sees clean, compressed, non-polluted context.

## Label Taxonomy

| Label | Strategy | Example |
|-------|----------|---------|
| `CORE` | Keep verbatim | Active goal, current error, system prompt |
| `DISTILL` | Extract key info | Test output → "94 passed, 2 failed" + failure details |
| `COMPACT` | Structural summary | File → "src/foo.py (200 lines): class Foo, def bar()" |
| `DROP` | Remove entirely | Completed tool calls (the result is what matters) |
| `STALE` | Mark for deletion | File read before a later edit of the same file |
| `ESCALATE` | Defer to LLM | Ambiguous content (rare) |

## Quick Start

```bash
git clone https://github.com/DJLougen/context-firewall.git
cd context-firewall
pip install -e ".[dev]"

# Run tests
pytest tests/

# Generate training data
python scripts/generate_synthetic.py

# Train classifier
cf-train examples/train.jsonl --eval examples/eval.jsonl
```

## Usage

### Rule-based (no training needed)

```python
from context_firewall import ContextFirewall, Message

fw = ContextFirewall()  # Uses rule-based classification

# Process every message inline
for raw_message in agent_messages:
    compressed = fw.process(Message(
        role=raw_message["role"],
        content=raw_message["content"],
    ))
    # Send compressed.content to your LLM
    send_to_llm({"role": compressed.role, "content": compressed.content})

# Get the full compressed context window
window = fw.get_context_window()
```

### ML classifier (trained)

```python
from context_firewall import ContextFirewall, Message

fw = ContextFirewall(model_path="models/context_firewall.joblib")

# Same API — just with ML classification instead of rules
compressed = fw.process(Message(role="tool", content="94 passed, 2 failed..."))
```

### With budget management

```python
from context_firewall import ContextFirewall, Message
from context_firewall.budget import BudgetConfig

fw = ContextFirewall(
    budget_config=BudgetConfig(target_tokens=10_000),
    cool_interval=5,  # Run cool loop every 5 turns
)
```

## Architecture

```
context_firewall/
  labels.py          Label taxonomy (CORE/DISTILL/COMPACT/DROP/STALE/ESCALATE)
  features.py        Message-level feature extraction
  compressor.py      Deterministic per-label compression rules
  session.py         Turn tracking, staleness detection, supersession
  budget.py          Token budget management
  classifier.py      TF-IDF + VotingClassifier (SGD + NB + LR)
  firewall.py        Main orchestrator (hot loop + cool loop)
  io.py              JSONL I/O for training data
  cli_train.py       Training CLI entry point
scripts/
  generate_synthetic.py  Synthetic training data generator
examples/
  train.jsonl        Training examples (1317 rows)
  eval.jsonl         Evaluation examples (262 rows)
tests/
  test_labels.py     6 tests
  test_features.py   12 tests
  test_compressor.py 25 tests
  test_session.py    17 tests
  test_budget.py     8 tests
  test_firewall.py   18 tests
  test_classifier.py 5 tests
```

## Performance

| Metric | Value |
|--------|-------|
| Classification accuracy | 95.8% (262 held-out examples) |
| Training examples | 1,317 |
| Per-message latency | ~1-5ms (CPU) |
| Cool loop latency | ~10-50ms (every N turns) |
| Compression ratio | 5-10x typical |
| Tests | 91 passed |

## How It Works

### Hot Loop (per message)

1. **Extract features** from the message: role, content type signals (paths, errors, code blocks), turn age, duplicate detection.
2. **Classify** the message into a compression label using either rules or the ML classifier.
3. **Compress** using deterministic rules specific to the content type and label.
4. **Record** in the session state for staleness tracking.

### Cool Loop (every N turns)

1. **Staleness check**: Walk the compressed context and drop entries that are stale (file read before a later edit) or superseded (file read again later).
2. **Budget enforcement**: If over the token budget, force-downgrade the lowest-priority entries to more aggressive compression.

### Compression Examples

**Test output** (DISTILL):
```
Before: 500 lines of pytest output
After:  "94 passed, 2 failed in 3.5s\nFailures:\ntest_foo.py::test_bar\nAssertionError: expected 5 got 3"
```

**File content** (COMPACT):
```
Before: 200 lines of source code
After:  "src/foo.py (200 lines):\nclass Foo\ndef bar()\ndef baz()"
```

**Command output** (DISTILL):
```
Before: 100 lines of build output
After:  "exit=0\nBuild complete\nSuccessfully built abc123"
```

**Error trace** (DISTILL):
```
Before: 50-line traceback
After:  "ValueError: invalid literal for int()\nat foo.py:42"
```

## Relationship to busyBee-cpu

context-firewall applies the same principle as [busyBee-cpu](https://github.com/DJLougen/busyBee-cpu) to a different problem:

| | busyBee-cpu | context-firewall |
|--|-------------|-----------------|
| **Problem** | Tool selection in agent loops | Context compression in agent loops |
| **Principle** | Most decisions are mechanical | Most compression is mechanical |
| **Classifier** | Which of 4 actions to take | Which compression strategy to apply |
| **Resolver** | Fill arguments from state | Execute compression per content type |
| **Escalation** | Defer to LLM for reasoning | Defer to LLM for ambiguous content |

Both use the same architecture: TF-IDF + VotingClassifier on CPU, with deterministic resolvers/compressors, escalating to the LLM only when uncertain.

## Dependencies

- `scikit-learn >= 1.4` — classifiers and pipelines
- `joblib >= 1.4` — model serialization
- `numpy >= 1.24` — numeric operations

## License

MIT
