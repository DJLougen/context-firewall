"""Honey-Comb: CPU-only inline context compression for agent harnesses. Keep the honey, drop the wax."""

__version__ = "0.1.0"

from honeycomb.labels import Label, ContentType
from honeycomb.firewall import HoneyComb, Message, CompressedMessage
from honeycomb.config import get_config, load_config, HoneyCombConfig
from honeycomb.observability import metrics, setup_logging, health_checker

__all__ = [
    "Label",
    "ContentType",
    "HoneyComb",
    "Message",
    "CompressedMessage",
    "get_config",
    "load_config",
    "HoneyCombConfig",
    "metrics",
    "setup_logging",
    "health_checker",
]
