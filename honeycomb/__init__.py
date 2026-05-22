"""Honey-Comb: CPU-only inline context compression for agent harnesses. Keep the honey, drop the wax."""

__version__ = "0.3.0"

from honeycomb.labels import Label, ContentType
from honeycomb.firewall import HoneyComb, Message, CompressedMessage
from honeycomb.config import get_config, load_config, HoneyCombConfig
from honeycomb.observability import metrics, setup_logging, health_checker
from honeycomb.tee import FailureTee, get_tee, reset_tee
from honeycomb.gain import GainTracker, get_tracker, reset_tracker
from honeycomb.command_filters import (
    detect_and_filter,
    list_supported_commands,
    FilterResult,
)

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
    # rtk-style features
    "FailureTee",
    "get_tee",
    "reset_tee",
    "GainTracker",
    "get_tracker",
    "reset_tracker",
    "detect_and_filter",
    "list_supported_commands",
    "FilterResult",
]
