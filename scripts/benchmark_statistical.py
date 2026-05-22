"""Statistically rigorous benchmark with confidence intervals."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
from scipy import stats

from honeycomb.firewall import HoneyComb, Message


def run_trial_hot_loop(num_messages: int = 1000) -> float:
    """Run one trial and return per-message latency in ms."""
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
    
    start = time.perf_counter()
    for i in range(num_messages):
        msg = messages[i % len(messages)]
        fw.process(msg)
    elapsed = time.perf_counter() - start
    
    return (elapsed / num_messages) * 1000


def run_trial_ml_classifier(num_messages: int = 1000, model_path: Path | None = None) -> float:
    """Run one ML trial and return per-message latency in ms."""
    if model_path is None:
        model_path = Path("models/honeycomb.joblib")
    
    fw = HoneyComb(model_path=model_path)
    messages = [
        Message(role="system", content="You are a helpful coding assistant."),
        Message(role="user", content="Fix the bug in src/foo.py"),
        Message(role="tool", content="94 passed, 2 failed in 3.5s"),
        Message(role="tool", content="class Foo:\n    def bar(self):\n        return 42"),
    ]
    
    start = time.perf_counter()
    for i in range(num_messages):
        msg = messages[i % len(messages)]
        fw.process(msg)
    elapsed = time.perf_counter() - start
    
    return (elapsed / num_messages) * 1000


def run_trial_compression() -> dict:
    """Run one compression trial and return stats."""
    fw = HoneyComb()
    
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
    
    return fw.get_stats()


def compute_stats(values: np.ndarray, name: str) -> dict:
    """Compute statistical summary with 95% confidence interval."""
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    se = std / np.sqrt(n)
    ci95 = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
    
    return {
        "name": name,
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_lower": ci95[0],
        "ci95_upper": ci95[1],
        "min": np.min(values),
        "max": np.max(values),
        "p50": np.percentile(values, 50),
        "p95": np.percentile(values, 95),
        "p99": np.percentile(values, 99),
    }


def benchmark_accuracy_significance():
    """Test if classification accuracy is significantly better than chance."""
    from honeycomb.classifier import RuleBasedClassifier
    from honeycomb.labels import Label
    import json
    
    # Load eval data
    eval_path = Path("examples/eval.jsonl")
    if not eval_path.exists():
        return {"error": "eval.jsonl not found"}
    
    examples = []
    with open(eval_path) as f:
        for line in f:
            examples.append(json.loads(line))
    
    if not examples:
        return {"error": "No eval examples"}
    
    # Test with rule-based classifier
    clf = RuleBasedClassifier()
    correct = 0
    total = len(examples)
    
    for ex in examples:
        features = ex["features"]
        true_label = Label(ex["label"])
        pred_label = clf.predict(features)
        if pred_label == true_label:
            correct += 1
    
    accuracy = correct / total
    
    # Binomial test: is accuracy significantly > 0.25 (chance for 4 classes)?
    p_value = stats.binom_test(correct, total, 0.25, alternative='greater')
    
    # Wilson score interval for accuracy
    ci95 = stats.binom.interval(0.95, total, accuracy)
    ci95_lower = ci95[0] / total
    ci95_upper = ci95[1] / total
    
    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "p_value_vs_chance": p_value,
        "ci95_lower": ci95_lower,
        "ci95_upper": ci95_upper,
        "significantly_better_than_chance": p_value < 0.05,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Statistical Benchmark (30 trials each)")
    print("=" * 70)
    
    num_trials = 30
    
    # Hot loop benchmark
    print(f"\n--- Rule-Based Latency ({num_trials} trials × 1000 messages) ---")
    latencies = []
    for i in range(num_trials):
        lat = run_trial_hot_loop()
        latencies.append(lat)
        gc.collect()
    
    stats_hot = compute_stats(np.array(latencies), "Rule-based latency (ms)")
    print(f"  Mean: {stats_hot['mean']:.4f} ± {stats_hot['std']:.4f} ms")
    print(f"  95% CI: [{stats_hot['ci95_lower']:.4f}, {stats_hot['ci95_upper']:.4f}] ms")
    print(f"  Range: [{stats_hot['min']:.4f}, {stats_hot['max']:.4f}] ms")
    print(f"  Percentiles: p50={stats_hot['p50']:.4f}, p95={stats_hot['p95']:.4f}, p99={stats_hot['p99']:.4f} ms")
    print(f"  Throughput: {1000/stats_hot['mean']:.0f} msg/s (95% CI: [{1000/stats_hot['ci95_upper']:.0f}, {1000/stats_hot['ci95_lower']:.0f}])")
    
    # ML classifier benchmark
    model_path = Path("models/honeycomb.joblib")
    if model_path.exists():
        print(f"\n--- ML Classifier Latency ({num_trials} trials × 1000 messages) ---")
        latencies_ml = []
        for i in range(num_trials):
            lat = run_trial_ml_classifier(model_path=model_path)
            latencies_ml.append(lat)
            gc.collect()
        
        stats_ml = compute_stats(np.array(latencies_ml), "ML latency (ms)")
        print(f"  Mean: {stats_ml['mean']:.4f} ± {stats_ml['std']:.4f} ms")
        print(f"  95% CI: [{stats_ml['ci95_lower']:.4f}, {stats_ml['ci95_upper']:.4f}] ms")
        print(f"  Range: [{stats_ml['min']:.4f}, {stats_ml['max']:.4f}] ms")
        print(f"  Percentiles: p50={stats_ml['p50']:.4f}, p95={stats_ml['p95']:.4f}, p99={stats_ml['p99']:.4f} ms")
        print(f"  Throughput: {1000/stats_ml['mean']:.0f} msg/s (95% CI: [{1000/stats_ml['ci95_upper']:.0f}, {1000/stats_ml['ci95_lower']:.0f}])")
    else:
        print("\n--- ML Classifier: Model not trained ---")
    
    # Compression ratio benchmark
    print(f"\n--- Compression Ratio ({num_trials} trials) ---")
    ratios = []
    originals = []
    compressed = []
    for i in range(num_trials):
        result = run_trial_compression()
        ratios.append(result["compression_ratio"])
        originals.append(result["original_tokens"])
        compressed.append(result["total_tokens"])
        gc.collect()
    
    stats_ratio = compute_stats(np.array(ratios), "Compression ratio")
    print(f"  Mean: {stats_ratio['mean']:.2f}x ± {stats_ratio['std']:.2f}x")
    print(f"  95% CI: [{stats_ratio['ci95_lower']:.2f}, {stats_ratio['ci95_upper']:.2f}]x")
    print(f"  Range: [{stats_ratio['min']:.2f}, {stats_ratio['max']:.2f}]x")
    print(f"  Tokens: {np.mean(originals):.0f} → {np.mean(compressed):.0f}")
    
    # Accuracy significance test
    print(f"\n--- Classification Accuracy Significance ---")
    acc_stats = benchmark_accuracy_significance()
    if "error" not in acc_stats:
        print(f"  Accuracy: {acc_stats['accuracy']*100:.1f}% ({acc_stats['correct']}/{acc_stats['total']})")
        print(f"  95% CI: [{acc_stats['ci95_lower']*100:.1f}%, {acc_stats['ci95_upper']*100:.1f}%]")
        print(f"  p-value vs 25% chance: {acc_stats['p_value_vs_chance']:.2e}")
        print(f"  Statistically significant: {'YES' if acc_stats['significantly_better_than_chance'] else 'NO'}")
    else:
        print(f"  {acc_stats['error']}")
    
    print("\n" + "=" * 70)
