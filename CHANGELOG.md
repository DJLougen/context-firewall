# Changelog

All notable changes to Honey-Comb will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
