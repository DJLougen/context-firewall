"""Observability: structured logging and metrics for production deployment.

This module provides:
- Structured JSON logging for Honey-Comb operations
- Prometheus-compatible metrics (if prometheus_client is available)
- Health check endpoint for monitoring

Usage:
    from honeycomb.observability import setup_logging, metrics
    
    # Setup logging
    setup_logging(level="INFO", json_format=True)
    
    # Record metrics
    metrics.messages_processed.inc()
    metrics.compression_ratio.observe(12.5)
"""

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Structured Logging
# ---------------------------------------------------------------------------

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    stream: Any = None,
) -> logging.Logger:
    """Configure structured logging for Honey-Comb.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format (True) or human-readable (False)
        stream: Output stream (default: sys.stderr)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("honeycomb")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create handler
    handler = logging.StreamHandler(stream or sys.stderr)
    
    # Set formatter
    if json_format:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Counter:
    """Simple counter metric."""
    name: str
    help_text: str
    value: int = 0
    
    def inc(self, amount: int = 1) -> None:
        """Increment counter."""
        self.value += amount
    
    def get(self) -> int:
        """Get current value."""
        return self.value
    
    def reset(self) -> None:
        """Reset to zero."""
        self.value = 0


@dataclass
class Histogram:
    """Simple histogram metric with buckets."""
    name: str
    help_text: str
    buckets: list[float] = field(default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    values: list[float] = field(default_factory=list)
    
    def observe(self, value: float) -> None:
        """Record an observation."""
        self.values.append(value)
    
    def get_count(self) -> int:
        """Get number of observations."""
        return len(self.values)
    
    def get_sum(self) -> float:
        """Get sum of all observations."""
        return sum(self.values)
    
    def get_mean(self) -> float:
        """Get mean value."""
        return sum(self.values) / len(self.values) if self.values else 0.0
    
    def get_percentile(self, p: float) -> float:
        """Get percentile (0-100)."""
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        k = (len(sorted_values) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_values) else f
        d = k - f
        return sorted_values[f] + d * (sorted_values[c] - sorted_values[f])
    
    def reset(self) -> None:
        """Clear all observations."""
        self.values.clear()


@dataclass
class Gauge:
    """Simple gauge metric."""
    name: str
    help_text: str
    value: float = 0.0
    
    def set(self, value: float) -> None:
        """Set gauge value."""
        self.value = value
    
    def inc(self, amount: float = 1.0) -> None:
        """Increment gauge."""
        self.value += amount
    
    def dec(self, amount: float = 1.0) -> None:
        """Decrement gauge."""
        self.value -= amount
    
    def get(self) -> float:
        """Get current value."""
        return self.value


@dataclass
class HoneyCombMetrics:
    """Collection of Honey-Comb metrics."""
    
    # Counters
    messages_processed: Counter = field(
        default_factory=lambda: Counter(
            "honeycomb_messages_processed_total",
            "Total number of messages processed"
        )
    )
    messages_by_label: Dict[str, Counter] = field(default_factory=dict)
    errors_total: Counter = field(
        default_factory=lambda: Counter(
            "honeycomb_errors_total",
            "Total number of errors"
        )
    )
    
    # Histograms
    compression_ratio: Histogram = field(
        default_factory=lambda: Histogram(
            "honeycomb_compression_ratio",
            "Compression ratio (original / compressed)"
        )
    )
    processing_latency_seconds: Histogram = field(
        default_factory=lambda: Histogram(
            "honeycomb_processing_latency_seconds",
            "Time to process a single message"
        )
    )
    
    # Gauges
    active_entries: Gauge = field(
        default_factory=lambda: Gauge(
            "honeycomb_active_entries",
            "Number of active (non-dropped) context entries"
        )
    )
    total_tokens: Gauge = field(
        default_factory=lambda: Gauge(
            "honeycomb_total_tokens",
            "Total tokens in compressed context"
        )
    )
    session_turns: Gauge = field(
        default_factory=lambda: Gauge(
            "honeycomb_session_turns",
            "Number of turns in current session"
        )
    )
    
    def record_message(self, label: str, compression_ratio: float) -> None:
        """Record metrics for a processed message."""
        self.messages_processed.inc()

        if label not in self.messages_by_label:
            self.messages_by_label[label] = Counter(
                f"honeycomb_messages_{label}_total",
                f"Messages with label {label}"
            )
        self.messages_by_label[label].inc()

        self.compression_ratio.observe(compression_ratio)
    
    def update_session_state(self, active: int, tokens: int, turns: int) -> None:
        """Update session state gauges."""
        self.active_entries.set(active)
        self.total_tokens.set(tokens)
        self.session_turns.set(turns)
    
    def record_error(self) -> None:
        """Record an error."""
        self.errors_total.inc()
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        
        # Counters
        lines.append(f"# HELP {self.messages_processed.name} {self.messages_processed.help_text}")
        lines.append(f"# TYPE {self.messages_processed.name} counter")
        lines.append(f"{self.messages_processed.name} {self.messages_processed.value}")
        
        for label, counter in self.messages_by_label.items():
            lines.append(f"# HELP {counter.name} {counter.help_text}")
            lines.append(f"# TYPE {counter.name} counter")
            lines.append(f"{counter.name} {counter.value}")
        
        lines.append(f"# HELP {self.errors_total.name} {self.errors_total.help_text}")
        lines.append(f"# TYPE {self.errors_total.name} counter")
        lines.append(f"{self.errors_total.name} {self.errors_total.value}")
        
        # Histograms
        for hist in [self.compression_ratio, self.processing_latency_seconds]:
            lines.append(f"# HELP {hist.name} {hist.help_text}")
            lines.append(f"# TYPE {hist.name} histogram")
            lines.append(f"{hist.name}_count {hist.get_count()}")
            lines.append(f"{hist.name}_sum {hist.get_sum():.6f}")
            for bucket in hist.buckets:
                count = sum(1 for v in hist.values if v <= bucket)
                lines.append(f"{hist.name}_bucket{{le=\"{bucket}\"}} {count}")
            lines.append(f"{hist.name}_bucket{{le=\"+Inf\"}} {hist.get_count()}")
        
        # Gauges
        for gauge in [self.active_entries, self.total_tokens, self.session_turns]:
            lines.append(f"# HELP {gauge.name} {gauge.help_text}")
            lines.append(f"# TYPE {gauge.name} gauge")
            lines.append(f"{gauge.name} {gauge.value}")
        
        return "\n".join(lines)


# Global metrics instance
metrics = HoneyCombMetrics()


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@dataclass
class HealthStatus:
    """Health check status."""
    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    uptime_seconds: float
    messages_processed: int
    active_sessions: int
    errors: int
    compression_ratio_p50: float
    compression_ratio_p95: float
    avg_latency_ms: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "status": self.status,
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "messages_processed": self.messages_processed,
            "active_sessions": self.active_sessions,
            "errors": self.errors,
            "compression_ratio": {
                "p50": round(self.compression_ratio_p50, 2),
                "p95": round(self.compression_ratio_p95, 2),
            },
            "avg_latency_ms": round(self.avg_latency_ms, 3),
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class HealthChecker:
    """Health check endpoint for monitoring."""
    
    def __init__(self, version: str = "0.1.0"):
        self.version = version
        self.start_time = time.time()
        self.active_sessions = 0
    
    def check(self) -> HealthStatus:
        """Perform health check."""
        uptime = time.time() - self.start_time
        
        # Determine status based on metrics
        status = "healthy"
        if metrics.errors_total.value > 100:
            status = "unhealthy"
        elif metrics.errors_total.value > 10:
            status = "degraded"
        
        return HealthStatus(
            status=status,
            version=self.version,
            uptime_seconds=uptime,
            messages_processed=metrics.messages_processed.value,
            active_sessions=self.active_sessions,
            errors=metrics.errors_total.value,
            compression_ratio_p50=metrics.compression_ratio.get_percentile(50),
            compression_ratio_p95=metrics.compression_ratio.get_percentile(95),
            avg_latency_ms=metrics.processing_latency_seconds.get_mean() * 1000,
        )


# Global health checker
health_checker = HealthChecker()


# ---------------------------------------------------------------------------
# Context Manager for Timing
# ---------------------------------------------------------------------------

@contextmanager
def timer(metric_name: str):
    """Context manager to time operations and record metrics.
    
    Usage:
        with timer("processing"):
            process_message()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if metric_name == "processing":
            metrics.processing_latency_seconds.observe(elapsed)


# ---------------------------------------------------------------------------
# Prometheus Integration (Optional)
# ---------------------------------------------------------------------------

def try_import_prometheus():
    """Try to import prometheus_client for native Prometheus support."""
    try:
        from prometheus_client import Counter as PromCounter
        from prometheus_client import Histogram as PromHistogram
        from prometheus_client import Gauge as PromGauge
        from prometheus_client import start_http_server
        
        return True, {
            "Counter": PromCounter,
            "Histogram": PromHistogram,
            "Gauge": PromGauge,
            "start_http_server": start_http_server,
        }
    except ImportError:
        return False, {}


def start_metrics_server(port: int = 8000) -> bool:
    """Start Prometheus metrics HTTP server (if prometheus_client available).
    
    Args:
        port: Port to serve metrics on (default: 8000)
    
    Returns:
        True if server started, False if prometheus_client not available
    """
    available, prom = try_import_prometheus()
    if not available:
        logging.warning(
            "prometheus_client not installed. Install with: pip install prometheus-client"
        )
        return False
    
    prom["start_http_server"](port)
    logging.info(f"Prometheus metrics server started on port {port}")
    return True
