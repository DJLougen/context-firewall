"""Context Firewall: main orchestrator for inline context compression.

The firewall processes every message through two loops:

HOT LOOP (per message, ~1-5ms):
  raw message → classify → compress → record in session

COOL LOOP (every N turns, ~10-50ms):
  walk compressed context → drop stale/superseded entries
  budget check → force-downgrade if over budget

Both loops are CPU-only. The LLM only sees clean, compressed context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from context_firewall.budget import BudgetConfig, BudgetManager
from context_firewall.compressor import compress
from context_firewall.features import extract_features, features_to_text
from context_firewall.labels import ContentType, Label
from context_firewall.session import SessionState


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """Input message to the firewall."""
    
    role: str
    """Message role: system, user, assistant, tool."""
    
    content: str
    """Raw message content."""
    
    content_type: ContentType | None = None
    """Optional content type hint. If None, the firewall infers it."""


@dataclass
class CompressedMessage:
    """Output message from the firewall."""
    
    role: str
    """Message role (preserved from input)."""
    
    content: str
    """Compressed content (what the LLM sees)."""
    
    label: Label
    """Compression label that was applied."""
    
    content_type: ContentType
    """Inferred or provided content type."""
    
    original_tokens: int
    """Approximate token count of original content."""
    
    compressed_tokens: int
    """Approximate token count of compressed content."""
    
    @property
    def compression_ratio(self) -> float:
        """Compression ratio (original / compressed)."""
        if self.compressed_tokens == 0:
            return 0.0
        return self.original_tokens / self.compressed_tokens


# ---------------------------------------------------------------------------
# Content type inference
# ---------------------------------------------------------------------------

def _infer_content_type(message: Message) -> ContentType:
    """Infer content type from role and content signals."""
    role = message.role.lower()
    content = message.content
    
    # System messages
    if role == "system":
        return ContentType.SYSTEM
    
    # User messages are usually goals
    if role == "user":
        return ContentType.USER_GOAL
    
    # Assistant messages: reasoning or patches
    if role == "assistant":
        if re.search(r"^[-+]{3} |^diff --git|^@@ ", content, re.M):
            return ContentType.AGENT_PATCH
        return ContentType.AGENT_REASONING
    
    # Tool messages: infer from content
    if role == "tool":
        # Test output
        if re.search(r"\d+\s*(passed|failed|error)", content, re.I):
            return ContentType.TOOL_RESULT_TEST
        
        # Error traces
        if re.search(r"Traceback|Error:|Exception:", content):
            return ContentType.TOOL_RESULT_ERROR
        
        # File content (has code structure)
        if re.search(r"^(class|def|import|from|export|function) ", content, re.M):
            return ContentType.TOOL_RESULT_FILE
        
        # Search results (file:line patterns)
        if re.search(r"[^\s:]+:\d+:", content):
            return ContentType.TOOL_RESULT_SEARCH
        
        # Command output (has exit code or generic output)
        if re.search(r"exit[= ]+\d+|\$\s+\w+", content, re.I):
            return ContentType.TOOL_RESULT_COMMAND
        
        return ContentType.TOOL_RESULT_COMMAND
    
    return ContentType.UNKNOWN


# ---------------------------------------------------------------------------
# Rule-based classifier (used when no ML model is loaded)
# ---------------------------------------------------------------------------

def _classify_rules(message: Message, content_type: ContentType, session: SessionState) -> Label:
    """Rule-based label assignment.
    
    This is the fallback when no ML classifier is loaded.
    It handles the obvious cases mechanically.
    """
    role = message.role.lower()
    content = message.content
    
    # System prompts are always CORE
    if content_type == ContentType.SYSTEM:
        return Label.CORE
    
    # Active user goals are CORE
    if content_type == ContentType.USER_GOAL:
        return Label.CORE
    
    # Tool calls are DROP (the result is what matters)
    if content_type == ContentType.TOOL_CALL:
        return Label.DROP
    
    # Error traces: CORE if recent, DISTILL if older
    if content_type == ContentType.TOOL_RESULT_ERROR:
        if session.turn_count <= 2:
            return Label.CORE
        return Label.DISTILL
    
    # Test output: DISTILL (extract summary)
    if content_type == ContentType.TOOL_RESULT_TEST:
        return Label.DISTILL
    
    # File content: COMPACT if large, DISTILL if small
    if content_type == ContentType.TOOL_RESULT_FILE:
        if len(content) > 2000:
            return Label.COMPACT
        return Label.DISTILL
    
    # Command output: DISTILL
    if content_type == ContentType.TOOL_RESULT_COMMAND:
        return Label.DISTILL
    
    # Search results: DISTILL
    if content_type == ContentType.TOOL_RESULT_SEARCH:
        return Label.DISTILL
    
    # Agent patches: DISTILL (keep summary of what changed)
    if content_type == ContentType.AGENT_PATCH:
        return Label.DISTILL
    
    # Agent reasoning: DISTILL if short, COMPACT if long
    if content_type == ContentType.AGENT_REASONING:
        if len(content) < 200:
            return Label.DISTILL
        return Label.COMPACT
    
    # Default: DISTILL
    return Label.DISTILL


# ---------------------------------------------------------------------------
# ML classifier wrapper
# ---------------------------------------------------------------------------

class MLClassifier:
    """Wraps a trained scikit-learn classifier for label prediction."""
    
    def __init__(self, model: Any) -> None:
        self.model = model
    
    def predict(self, features_text: str) -> Label:
        """Predict label from feature text."""
        prediction = self.model.predict([features_text])[0]
        try:
            return Label(prediction)
        except ValueError:
            return Label.DISTILL  # Fallback
    
    def predict_proba(self, features_text: str) -> dict[Label, float]:
        """Predict label probabilities."""
        if not hasattr(self.model, "predict_proba"):
            return {}
        
        probas = self.model.predict_proba([features_text])[0]
        classes = self.model.classes_
        
        result = {}
        for cls, prob in zip(classes, probas):
            try:
                label = Label(cls)
                result[label] = float(prob)
            except ValueError:
                pass
        
        return result


# ---------------------------------------------------------------------------
# Context Firewall
# ---------------------------------------------------------------------------

class ContextFirewall:
    """Main entry point for inline context compression.
    
    Usage:
        firewall = ContextFirewall()
        
        for message in agent_messages:
            compressed = firewall.process(message)
            # Send compressed.content to LLM
    
    The firewall maintains session state across calls. Create a new
    instance for each agent session.
    """
    
    def __init__(
        self,
        model_path: str | Path | None = None,
        budget_config: BudgetConfig | None = None,
        cool_interval: int = 5,
    ) -> None:
        """Initialize the firewall.
        
        Args:
            model_path: Path to a trained classifier model. If None, uses
                       rule-based classification.
            budget_config: Token budget configuration. If None, uses defaults.
            cool_interval: Run cool loop every N turns (default 5).
        """
        self.session = SessionState()
        self.budget = BudgetManager(budget_config)
        self.cool_interval = cool_interval
        
        # Load ML classifier if provided
        self._classifier: MLClassifier | None = None
        if model_path is not None:
            import joblib
            model = joblib.load(model_path)
            self._classifier = MLClassifier(model)
    
    def process(self, message: Message) -> CompressedMessage:
        """Process a message through the hot loop.
        
        This is the main entry point. Call this for every message
        that enters the agent's context window.
        
        Returns the compressed message (what the LLM should see).
        """
        # Advance turn
        self.session.advance_turn()
        
        # Infer content type if not provided
        content_type = message.content_type or _infer_content_type(message)
        
        # Extract features
        features = extract_features(message.content, message.role, self.session)
        features_text = features_to_text(features)
        
        # Classify (hot loop: ~1-3ms)
        if self._classifier is not None:
            label = self._classifier.predict(features_text)
        else:
            label = _classify_rules(message, content_type, self.session)
        
        # Compress (deterministic, ~0.1ms)
        compressed_content = compress(message.content, content_type, label)
        
        # Record in session
        entry = self.session.record(
            role=message.role,
            content_type=content_type,
            label=label,
            original=message.content,
            compressed=compressed_content,
        )
        
        # Cool loop: periodic staleness pass
        if self.session.turn_count % self.cool_interval == 0:
            self._cool_pass()
        
        return CompressedMessage(
            role=message.role,
            content=compressed_content,
            label=label,
            content_type=content_type,
            original_tokens=entry.original_tokens,
            compressed_tokens=entry.compressed_tokens,
        )
    
    def _cool_pass(self) -> None:
        """Run the cool loop: staleness check + budget enforcement."""
        # Drop stale and superseded entries
        self.session.cool_pass()
        
        # Enforce budget if needed
        if self.budget.is_over_budget(self.session):
            self.budget.enforce(self.session)
    
    def get_context_window(self) -> list[dict[str, str]]:
        """Get the current compressed context window.
        
        Returns a list of message dicts suitable for sending to an LLM.
        """
        return [
            {"role": entry.role, "content": entry.compressed_content}
            for entry in self.session.get_active_entries()
        ]
    
    def get_stats(self) -> dict[str, Any]:
        """Get session statistics."""
        return {
            "turn_count": self.session.turn_count,
            "total_entries": len(self.session.entries),
            "active_entries": len(self.session.get_active_entries()),
            "total_tokens": self.session.get_total_tokens(),
            "original_tokens": self.session.get_total_original_tokens(),
            "compression_ratio": self.session.get_compression_ratio(),
            "budget_target": self.budget.config.target_tokens,
            "over_budget": self.budget.is_over_budget(self.session),
        }
