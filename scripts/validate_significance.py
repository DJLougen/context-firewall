"""
Statistical Significance Validation for Honey-Comb

Validates all key claims with proper statistical methods:
1. Classification accuracy with bootstrap confidence intervals
2. Compression ratio across multiple sessions with significance testing
3. Throughput with proper statistics (mean, CI, variance)
4. Token savings (absolute reduction) across sessions

Generates publication-ready charts and saves results to JSON.
"""

from __future__ import annotations
import sys
import io
import json
import time
import random
from pathlib import Path
from dataclasses import dataclass, asdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from honeycomb.firewall import HoneyComb, Message
from honeycomb.io import read_jsonl

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150

OUTPUT_DIR = project_root / "docs" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class StatisticalResult:
    metric: str
    n_samples: int
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    p_value: float | None = None
    baseline: float | None = None
    effect_size: float | None = None
    significant: bool | None = None


def bootstrap_ci(
    data: np.ndarray, n_bootstrap: int = 10000, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Bootstrap confidence interval. Returns (mean, ci_lower, ci_upper)."""
    bootstrap_means = np.empty(n_bootstrap)
    n = len(data)
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)
    mean = float(np.mean(data))
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    return mean, ci_lower, ci_upper


def bootstrap_dist(
    data: np.ndarray, n_bootstrap: int = 10000
) -> np.ndarray:
    """Return full bootstrap distribution of the mean."""
    n = len(data)
    bootstrap_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)
    return bootstrap_means


def t_test_one_sample(data: np.ndarray, mu: float) -> float:
    """One-sample t-test p-value."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    if std == 0 or n < 2:
        return 1.0 if mean == mu else 0.0
    t_stat = (mean - mu) / (std / np.sqrt(n))
    from math import erf, sqrt
    z = abs(t_stat)
    return 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((mean1 - mean2) / pooled_std)


# ---------------------------------------------------------------------------
# 1. Classification Accuracy
# ---------------------------------------------------------------------------

def validate_accuracy(n_bootstrap: int = 10000) -> tuple[StatisticalResult, np.ndarray]:
    """Validate ML classification accuracy with bootstrap CI."""
    print("\n[1/4] Validating Classification Accuracy (ML classifier)...")

    eval_path = project_root / "examples" / "eval.jsonl"
    model_path = project_root / "models" / "honeycomb.joblib"

    if not eval_path.exists():
        print("  ERROR: eval.jsonl not found")
        return StatisticalResult("accuracy", 0, 0, 0, 0, 0), np.array([])
    if not model_path.exists():
        print("  ERROR: ML model not found at models/honeycomb.joblib")
        return StatisticalResult("accuracy", 0, 0, 0, 0, 0), np.array([])

    eval_data = read_jsonl(eval_path)
    n_samples = len(eval_data)
    print(f"  Loaded {n_samples} evaluation examples")

    hc = HoneyComb(model_path=str(model_path))

    correct = np.array([
        1 if hc.process(Message(role=row["role"], content=row["content"])).label.value == row["label"] else 0
        for row in eval_data
    ])

    mean, ci_lower, ci_upper = bootstrap_ci(correct, n_bootstrap)
    std = float(np.std(correct, ddof=1))

    random_chance = 1.0 / 4  # 4 labels actually present in eval
    p_value = t_test_one_sample(correct, random_chance)

    effect_size = cohens_d(
        correct,
        np.random.choice([0, 1], size=n_samples, p=[1 - random_chance, random_chance]),
    )

    result = StatisticalResult(
        metric="Classification Accuracy",
        n_samples=n_samples,
        mean=mean * 100,
        std=std * 100,
        ci_lower=ci_lower * 100,
        ci_upper=ci_upper * 100,
        p_value=p_value,
        baseline=random_chance * 100,
        effect_size=effect_size,
        significant=p_value < 0.05,
    )

    print(f"  Accuracy: {result.mean:.1f}% (95% CI: [{result.ci_lower:.1f}%, {result.ci_upper:.1f}%])")
    print(f"  Baseline (random): {result.baseline:.1f}%")
    print(f"  p-value: {result.p_value:.2e} (significant: {result.significant})")
    print(f"  Effect size (Cohen's d): {result.effect_size:.2f}")

    boot_dist = bootstrap_dist(correct, n_bootstrap) * 100
    return result, boot_dist


# ---------------------------------------------------------------------------
# 2. Compression Ratio
# ---------------------------------------------------------------------------

def _make_session_content(seed: int) -> list[tuple[str, str]]:
    """Generate a realistic synthetic session. Returns list of (role, content)."""
    rng = random.Random(seed)
    messages = []

    # System prompt
    messages.append(("system", "You are an expert software engineer."))

    # User goal
    messages.append(("user", f"Fix bug in module_{seed}.py related to token validation."))

    # Mix of tool outputs
    for i in range(rng.randint(5, 15)):
        kind = rng.choice(["test", "file", "reasoning", "command", "error"])

        if kind == "test":
            n = rng.randint(20, 100)
            lines = [f"{'=' * 70}"]
            for j in range(n):
                status = "PASSED" if rng.random() > 0.05 else "FAILED"
                lines.append(f"tests/test_mod{seed}.py::test_{j} {status}")
            n_failed = sum(1 for line in lines if "FAILED" in line)
            lines.append(f"{'=' * 70}")
            lines.append(f"{n - n_failed} passed, {n_failed} failed in {rng.uniform(1, 10):.2f}s")
            messages.append(("tool", "\n".join(lines)))

        elif kind == "file":
            n_lines = rng.randint(50, 200)
            lines = [f"# src/module_{seed}.py"]
            for j in range(n_lines):
                lines.append(f"def func_{j}(): return {j}")
            messages.append(("tool", "\n".join(lines)))

        elif kind == "reasoning":
            messages.append(("assistant",
                f"Looking at the issue, I think the problem is in function {i}. "
                f"Let me check the implementation on line {i * 10}. "
                f"Actually, I see the bug now - the validation returns True on expired tokens. "
                f"I need to raise TokenExpiredError instead."))

        elif kind == "command":
            lines = [f"$ make build module_{seed}"]
            for j in range(rng.randint(10, 50)):
                lines.append(f"Compiling src/part_{j}.c... OK")
            lines.append("Build complete.")
            messages.append(("tool", "\n".join(lines)))

        elif kind == "error":
            messages.append(("tool",
                f"Traceback (most recent call last):\n"
                f"  File 'src/module_{seed}.py', line {rng.randint(1, 100)}\n"
                f"ValueError: invalid value {rng.randint(0, 999)}"))

    return messages


def validate_compression(n_sessions: int = 100, n_bootstrap: int = 10000) -> tuple[StatisticalResult, np.ndarray]:
    """Validate compression ratio across diverse sessions."""
    print(f"\n[2/4] Validating Compression Ratio ({n_sessions} sessions)...")

    ratios = []
    for seed in range(n_sessions):
        hc = HoneyComb()
        for role, content in _make_session_content(seed):
            hc.process(Message(role=role, content=content))
        stats = hc.get_stats()
        if stats["original_tokens"] > 0:
            ratios.append(stats["compression_ratio"])

    ratios = np.array(ratios)
    mean, ci_lower, ci_upper = bootstrap_ci(ratios, n_bootstrap)
    std = float(np.std(ratios, ddof=1))

    p_value = t_test_one_sample(ratios, 1.0)
    no_compression = np.ones_like(ratios)
    effect_size = cohens_d(ratios, no_compression) if std > 0 else 0.0

    result = StatisticalResult(
        metric="Compression Ratio",
        n_samples=len(ratios),
        mean=mean,
        std=std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        baseline=1.0,
        effect_size=effect_size,
        significant=p_value < 0.05,
    )

    print(f"  Compression Ratio: {result.mean:.2f}x (95% CI: [{result.ci_lower:.2f}x, {result.ci_upper:.2f}x])")
    print(f"  Std dev: {result.std:.2f}x")
    print(f"  vs Baseline (1.0x): p={result.p_value:.2e} (significant: {result.significant})")
    print(f"  Effect size (Cohen's d): {result.effect_size:.2f}")

    return result, ratios


# ---------------------------------------------------------------------------
# 3. Throughput
# ---------------------------------------------------------------------------

def validate_throughput(n_trials: int = 100, n_bootstrap: int = 10000) -> dict[str, StatisticalResult]:
    """Validate throughput with proper statistics."""
    print(f"\n[3/4] Validating Throughput ({n_trials} trials)...")

    # Rule-based
    rule_throughputs = np.empty(n_trials)
    for trial in range(n_trials):
        hc = HoneyComb()
        start = time.perf_counter()
        for i in range(1000):
            hc.process(Message(role="tool", content=f"Test {i}: 94 passed, 2 failed in 3.5s"))
        elapsed = time.perf_counter() - start
        rule_throughputs[trial] = 1000 / elapsed

    r_mean, r_ci_lo, r_ci_hi = bootstrap_ci(rule_throughputs, n_bootstrap)
    r_std = float(np.std(rule_throughputs, ddof=1))

    rule_result = StatisticalResult(
        metric="Rule-based Throughput", n_samples=n_trials,
        mean=r_mean, std=r_std, ci_lower=r_ci_lo, ci_upper=r_ci_hi,
    )
    print(f"  Rule-based: {r_mean:.0f} msg/s (95% CI: [{r_ci_lo:.0f}, {r_ci_hi:.0f}])")

    results = {"rule": rule_result, "rule_raw": rule_throughputs}

    # ML-based
    model_path = project_root / "models" / "honeycomb.joblib"
    if model_path.exists():
        ml_throughputs = np.empty(n_trials)
        for trial in range(n_trials):
            hc = HoneyComb(model_path=str(model_path))
            start = time.perf_counter()
            for i in range(1000):
                hc.process(Message(role="tool", content=f"Test {i}: 94 passed, 2 failed in 3.5s"))
            elapsed = time.perf_counter() - start
            ml_throughputs[trial] = 1000 / elapsed

        m_mean, m_ci_lo, m_ci_hi = bootstrap_ci(ml_throughputs, n_bootstrap)
        m_std = float(np.std(ml_throughputs, ddof=1))

        ml_result = StatisticalResult(
            metric="ML-based Throughput", n_samples=n_trials,
            mean=m_mean, std=m_std, ci_lower=m_ci_lo, ci_upper=m_ci_hi,
        )
        print(f"  ML-based: {m_mean:.0f} msg/s (95% CI: [{m_ci_lo:.0f}, {m_ci_hi:.0f}])")
        results["ml"] = ml_result
        results["ml_raw"] = ml_throughputs
    else:
        print("  ML-based: SKIPPED (model not found)")

    return results


# ---------------------------------------------------------------------------
# 4. Token Savings
# ---------------------------------------------------------------------------

def validate_token_savings(n_sessions: int = 100, n_bootstrap: int = 10000) -> tuple[StatisticalResult, np.ndarray]:
    """Validate absolute token savings across sessions."""
    print(f"\n[4/4] Validating Token Savings ({n_sessions} sessions)...")

    savings = []
    for seed in range(n_sessions):
        hc = HoneyComb()
        for role, content in _make_session_content(seed):
            hc.process(Message(role=role, content=content))
        stats = hc.get_stats()
        original = stats["original_tokens"]
        compressed = stats["total_tokens"]
        savings.append(original - compressed)

    savings = np.array(savings, dtype=float)
    mean, ci_lower, ci_upper = bootstrap_ci(savings, n_bootstrap)
    std = float(np.std(savings, ddof=1))
    p_value = t_test_one_sample(savings, 0)

    result = StatisticalResult(
        metric="Token Savings",
        n_samples=len(savings),
        mean=mean,
        std=std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        baseline=0,
        significant=p_value < 0.05,
    )

    print(f"  Token Savings: {result.mean:.0f} tokens/session (95% CI: [{result.ci_lower:.0f}, {result.ci_upper:.0f}])")
    print(f"  p-value: {result.p_value:.2e} (significant: {result.significant})")

    return result, savings


# ---------------------------------------------------------------------------
# Chart Generation
# ---------------------------------------------------------------------------

def plot_accuracy(result: StatisticalResult, boot_dist: np.ndarray):
    """Histogram of bootstrap accuracy distribution with CI."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(boot_dist, bins=50, color="#3b82f6", edgecolor="black", alpha=0.7, density=True)
    ax.axvline(result.mean, color="#1e40af", linewidth=2, linestyle="-", label=f"Mean: {result.mean:.1f}%")
    ax.axvline(result.ci_lower, color="#ef4444", linewidth=2, linestyle="--", label=f"95% CI: [{result.ci_lower:.1f}%, {result.ci_upper:.1f}%]")
    ax.axvline(result.ci_upper, color="#ef4444", linewidth=2, linestyle="--")
    ax.axvline(result.baseline, color="#f59e0b", linewidth=2, linestyle=":", label=f"Random baseline: {result.baseline:.1f}%")
    ax.set_xlabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Classification Accuracy Distribution\n"
        f"(n={result.n_samples}, p={result.p_value:.2e}, Cohen's d={result.effect_size:.2f})",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stat_accuracy.png", bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved: {OUTPUT_DIR / 'stat_accuracy.png'}")


def plot_compression(ratios: np.ndarray, result: StatisticalResult):
    """Histogram of compression ratios across sessions."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ratios, bins=30, color="#10b981", edgecolor="black", alpha=0.7)
    ax.axvline(result.mean, color="#065f46", linewidth=2, linestyle="-", label=f"Mean: {result.mean:.2f}x")
    ax.axvline(result.ci_lower, color="#ef4444", linewidth=2, linestyle="--", label=f"95% CI: [{result.ci_lower:.2f}x, {result.ci_upper:.2f}x]")
    ax.axvline(result.ci_upper, color="#ef4444", linewidth=2, linestyle="--")
    ax.axvline(1.0, color="#f59e0b", linewidth=2, linestyle=":", label="No compression baseline: 1.0x")
    ax.set_xlabel("Compression Ratio", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Compression Ratio Distribution Across {result.n_samples} Sessions\n"
        f"(p={result.p_value:.2e}, Cohen's d={result.effect_size:.2f})",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stat_compression.png", bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved: {OUTPUT_DIR / 'stat_compression.png'}")


def plot_throughput(throughput_results: dict):
    """Box plot of throughput across modes."""
    fig, ax = plt.subplots(figsize=(8, 5))

    data = [throughput_results["rule_raw"]]
    labels = ["Rule-based"]
    colors = ["#3b82f6"]

    if "ml_raw" in throughput_results:
        data.append(throughput_results["ml_raw"])
        labels.append("ML-based")
        colors.append("#f59e0b")

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add mean markers
    for i, d in enumerate(data):
        ax.plot(i + 1, np.mean(d), "D", color="red", markersize=8, zorder=5, label="Mean" if i == 0 else "")

    # Annotate
    rule_r = throughput_results["rule"]
    ax.text(1, rule_r.mean * 1.05, f"{rule_r.mean:.0f} msg/s\n[{rule_r.ci_lower:.0f}, {rule_r.ci_upper:.0f}]",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
    if "ml" in throughput_results:
        ml_r = throughput_results["ml"]
        ax.text(2, ml_r.mean * 1.05, f"{ml_r.mean:.0f} msg/s\n[{ml_r.ci_lower:.0f}, {ml_r.ci_upper:.0f}]",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Throughput (msg/s)", fontsize=12, fontweight="bold")
    ax.set_title("Throughput Distribution (100 trials, 1000 messages each)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stat_throughput.png", bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved: {OUTPUT_DIR / 'stat_throughput.png'}")


def plot_token_savings(savings: np.ndarray, result: StatisticalResult):
    """Histogram of token savings across sessions."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(savings, bins=30, color="#8b5cf6", edgecolor="black", alpha=0.7)
    ax.axvline(result.mean, color="#4c1d95", linewidth=2, linestyle="-", label=f"Mean: {result.mean:.0f} tokens saved")
    ax.axvline(result.ci_lower, color="#ef4444", linewidth=2, linestyle="--", label=f"95% CI: [{result.ci_lower:.0f}, {result.ci_upper:.0f}]")
    ax.axvline(result.ci_upper, color="#ef4444", linewidth=2, linestyle="--")
    ax.set_xlabel("Tokens Saved Per Session", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Token Savings Distribution Across {result.n_samples} Sessions\n"
        f"(p={result.p_value:.2e})",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stat_savings.png", bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved: {OUTPUT_DIR / 'stat_savings.png'}")


def plot_summary_dashboard(
    acc_result: StatisticalResult,
    comp_ratios: np.ndarray,
    comp_result: StatisticalResult,
    throughput_results: dict,
    savings: np.ndarray,
    sav_result: StatisticalResult,
):
    """4-panel summary dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Honey-Comb Statistical Significance Dashboard", fontsize=16, fontweight="bold", y=0.98)

    # Panel 1: Accuracy
    ax = axes[0, 0]
    ax.bar(["Accuracy"], [acc_result.mean], yerr=[[acc_result.mean - acc_result.ci_lower], [acc_result.ci_upper - acc_result.mean]],
           color="#3b82f6", edgecolor="black", capsize=10, linewidth=1.5)
    ax.axhline(acc_result.baseline, color="#f59e0b", linewidth=2, linestyle=":", label=f"Random: {acc_result.baseline:.1f}%")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title(f"Classification Accuracy\np={acc_result.p_value:.1e}, d={acc_result.effect_size:.2f}", fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Compression
    ax = axes[0, 1]
    ax.hist(comp_ratios, bins=25, color="#10b981", edgecolor="black", alpha=0.7)
    ax.axvline(comp_result.mean, color="#065f46", linewidth=2, label=f"Mean: {comp_result.mean:.1f}x")
    ax.axvline(1.0, color="#f59e0b", linewidth=2, linestyle=":", label="No compression")
    ax.set_xlabel("Compression Ratio", fontweight="bold")
    ax.set_title(f"Compression Ratio (n={comp_result.n_samples})\np={comp_result.p_value:.1e}, d={comp_result.effect_size:.1f}", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Throughput
    ax = axes[1, 0]
    tp_data = [throughput_results["rule_raw"]]
    tp_labels = ["Rule"]
    tp_colors = ["#3b82f6"]
    if "ml_raw" in throughput_results:
        tp_data.append(throughput_results["ml_raw"])
        tp_labels.append("ML")
        tp_colors.append("#f59e0b")
    bp = ax.boxplot(tp_data, tick_labels=tp_labels, patch_artist=True, widths=0.4)
    for patch, color in zip(bp["boxes"], tp_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("msg/s", fontweight="bold")
    ax.set_title("Throughput (100 trials)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Panel 4: Token savings
    ax = axes[1, 1]
    ax.hist(savings, bins=25, color="#8b5cf6", edgecolor="black", alpha=0.7)
    ax.axvline(sav_result.mean, color="#4c1d95", linewidth=2, label=f"Mean: {sav_result.mean:.0f}")
    ax.set_xlabel("Tokens Saved", fontweight="bold")
    ax.set_title(f"Token Savings (n={sav_result.n_samples})\np={sav_result.p_value:.1e}", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_DIR / "stat_summary.png", bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved: {OUTPUT_DIR / 'stat_summary.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Honey-Comb Statistical Significance Validation")
    print("=" * 70)

    acc_result, acc_boot = validate_accuracy()
    comp_result, comp_ratios = validate_compression()
    throughput_results = validate_throughput()
    sav_result, savings = validate_token_savings()

    # Report
    print("\n" + "=" * 70)
    print("STATISTICAL VALIDATION REPORT")
    print("=" * 70)

    print(f"\n1. CLASSIFICATION ACCURACY (n={acc_result.n_samples})")
    print(f"   Mean: {acc_result.mean:.1f}%")
    print(f"   95% CI: [{acc_result.ci_lower:.1f}%, {acc_result.ci_upper:.1f}%]")
    print(f"   vs Baseline ({acc_result.baseline:.1f}%): p={acc_result.p_value:.2e}")
    print(f"   Statistically Significant: {'YES' if acc_result.significant else 'NO'}")

    print(f"\n2. COMPRESSION RATIO (n={comp_result.n_samples})")
    print(f"   Mean: {comp_result.mean:.2f}x")
    print(f"   95% CI: [{comp_result.ci_lower:.2f}x, {comp_result.ci_upper:.2f}x]")
    print(f"   vs Baseline (1.0x): p={comp_result.p_value:.2e}")
    print(f"   Statistically Significant: {'YES' if comp_result.significant else 'NO'}")

    rule_r = throughput_results["rule"]
    print(f"\n3. THROUGHPUT (n={rule_r.n_samples} trials)")
    print(f"   Rule-based: {rule_r.mean:.0f} msg/s (95% CI: [{rule_r.ci_lower:.0f}, {rule_r.ci_upper:.0f}])")
    if "ml" in throughput_results:
        ml_r = throughput_results["ml"]
        print(f"   ML-based: {ml_r.mean:.0f} msg/s (95% CI: [{ml_r.ci_lower:.0f}, {ml_r.ci_upper:.0f}])")

    print(f"\n4. TOKEN SAVINGS (n={sav_result.n_samples})")
    print(f"   Mean: {sav_result.mean:.0f} tokens/session")
    print(f"   95% CI: [{sav_result.ci_lower:.0f}, {sav_result.ci_upper:.0f}]")
    print(f"   vs Baseline (0): p={sav_result.p_value:.2e}")
    print(f"   Statistically Significant: {'YES' if sav_result.significant else 'NO'}")

    all_sig = all([acc_result.significant, comp_result.significant, sav_result.significant])
    print("\n" + "=" * 70)
    if all_sig:
        print("ALL KEY METRICS ARE STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        print("WARNING: Some metrics did not reach statistical significance.")
    print("=" * 70)

    # Generate charts
    print("\nGenerating charts...")
    plot_accuracy(acc_result, acc_boot)
    plot_compression(comp_ratios, comp_result)
    plot_throughput(throughput_results)
    plot_token_savings(savings, sav_result)
    plot_summary_dashboard(acc_result, comp_ratios, comp_result, throughput_results, savings, sav_result)

    # Save JSON
    output = {
        "accuracy": {k: v for k, v in asdict(acc_result).items()},
        "compression": {k: v for k, v in asdict(comp_result).items()},
        "throughput_rule": {k: v for k, v in asdict(throughput_results["rule"]).items()},
        "token_savings": {k: v for k, v in asdict(sav_result).items()},
    }
    if "ml" in throughput_results:
        output["throughput_ml"] = {k: v for k, v in asdict(throughput_results["ml"]).items()}

    json_path = project_root / "docs" / "statistical_validation.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  [OK] Saved: {json_path}")


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    main()
