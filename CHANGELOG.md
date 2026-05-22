# Changelog

All notable changes to Honey-Comb will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-05-22

### Added
- **Statistical Validation**: All performance claims now backed by rigorous statistical methods
  - Bootstrap confidence intervals (10,000 resamples) for all key metrics
  - One-sample t-tests vs appropriate baselines
  - Cohen's d effect size calculations
  - Automated validation script: `scripts/validate_significance.py`
  - Results saved to `docs/statistical_validation.json`
  - Publication-ready charts generated automatically

- **Statistical Significance Section**: New README section with validated results
  - Classification accuracy: 84.2% end-to-end [79.9%, 88.3%] vs 25% random baseline (p < 0.001, d = 1.42)
  - Compression ratio: 13.7x [12.2x, 15.3x] vs 1.0x baseline (p < 0.001, d = 2.33)
  - Token savings: 3,103 tokens/session [2,815, 3,398] (p < 0.001)
  - Throughput: 13,635 msg/s rules [13,374, 13,867], 1,028 msg/s ML [995, 1,057]

- **Visual Gallery**: Statistical validation dashboard with 5 charts
  - `stat_summary.png` — 4-panel overview of all metrics
  - `stat_accuracy.png` — Bootstrap distribution of classification accuracy
  - `stat_compression.png` — Distribution of compression ratios across 100 sessions
  - `stat_throughput.png` — Throughput box plots for rule-based and ML modes
  - `stat_savings.png` — Token savings distribution

### Changed
- **Honest Accuracy Reporting**: Distinguishes end-to-end pipeline accuracy (84.2%) from isolated ML classifier accuracy (94.5%)
  - End-to-end reflects full pipeline including content-type detection
  - Isolated classifier reports 94.5% on the same held-out eval set
  - Both significantly better than 25% random baseline

- **Performance Table**: Updated with statistically validated numbers and 95% CIs

- **Documentation**: Added `AGENTS.md` with setup instructions for AI agents

- **Repo Rename**: GitHub repo renamed to `DJLougen/honey-comb` with all references updated

### Fixed
- Fixed matplotlib deprecation warnings (`labels` → `tick_labels` in boxplot)
- Fixed import errors in validation scripts (`load_jsonl` → `read_jsonl`)
- Fixed synthetic session generation (flattened nested list comprehensions)


## [0.2.0] - 2026-05-22

### Added
- **Thread Safety**: All session state operations are now thread-safe by default
  - `SessionState` uses `threading.RLock()` for concurrent access
  - Optional `thread_safe=False` parameter for single-threaded performance (1.45x faster)
  - Comprehensive threading test suite (4 new tests)
  
- **Observability**: Production-grade monitoring and diagnostics
  - Structured JSON logging with configurable log levels
  - Prometheus-compatible metrics (messages processed, compression ratios, latency percentiles)
  - Health check endpoint with status, uptime, and error counts
  - Metrics export in Prometheus text format
  
- **Configuration**: Flexible configuration management
  - Environment variable support (`HONEYCOMB_*` prefix)
  - Config file loading (JSON, YAML)
  - Runtime configuration validation
  - Configuration dataclass with sensible defaults
  
- **Performance Tuning**: Fine-grained control over performance vs features
  - `metrics_enabled` parameter to disable metrics overhead
  - `thread_safe` parameter to disable locking overhead
  - Production mode: ~17,000 msg/s (thread-safe + metrics)
  - High-perf mode: ~24,000 msg/s (single-threaded, no metrics)

### Changed
- **Rename**: Project renamed from "Context Firewall" to "Honey-Comb"
  - Package: `context_firewall` → `honeycomb`
  - Class: `ContextFirewall` → `HoneyComb`
  - CLI: `cf-train` → `honeycomb-train`
  - Tagline: "Keep the honey, drop the wax"
  
- **Session Management**: Enhanced thread-safe session state
  - All state mutations protected by reentrant locks
  - Configurable lock mode (thread-safe vs high-performance)
  - Improved documentation of thread safety guarantees

### Performance
- Throughput benchmarks updated with thread safety overhead
- Production mode: 17,000 msg/s (safe for concurrent use)
- High-perf mode: 24,000 msg/s (single-threaded batch processing)
- Compression ratio: 6.3x on realistic agent sessions

### Tests
- Added `test_threading.py` with 4 comprehensive threading tests
- Total test count: 129 tests (up from 125)
- All tests passing with thread safety enabled

### Documentation
- Added "Production Features" section to README
- Thread safety usage examples
- Observability and metrics documentation
- Configuration guide with environment variables
- Performance tuning comparison table
- Added `scripts/demo_production.py` demonstrating all production features

## [0.1.0] - 2026-05-20

### Added
- Initial release of Context Firewall (now Honey-Comb)
- CPU-only inline context compression for agent harnesses
- Hot loop: per-message classification and compression (~1ms)
- Cool loop: periodic staleness detection and budget enforcement
- Label taxonomy: CORE, DISTILL, COMPACT, DROP, STALE, ESCALATE
- Rule-based classifier (no training required)
- ML classifier support (TF-IDF + VotingClassifier)
- Budget management with automatic downgrades
- 6.3x compression ratio on realistic sessions
- 125 passing tests
- Demo scripts showing raw vs compressed context
