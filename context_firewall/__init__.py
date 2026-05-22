"""Context Firewall: CPU-only inline context compression for agent harnesses."""

__version__ = "0.1.0"

from context_firewall.labels import Label, ContentType
from context_firewall.firewall import ContextFirewall, Message, CompressedMessage

__all__ = [
    "Label",
    "ContentType",
    "ContextFirewall",
    "Message",
    "CompressedMessage",
]
