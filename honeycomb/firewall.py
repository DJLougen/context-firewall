"""Honey-Comb: main orchestrator for inline context depollution.

Keep the honey, drop the wax.

The comb processes every message through two loops:

HOT LOOP (per message, ~1-5ms):
  raw message → classify → depollute → record in session

COOL LOOP (every N turns, ~10-50ms):
  walk context → drop stale/superseded entries
  budget check → force-downgrade if over budget

Both loops are CPU-only. The LLM only sees clean, depolluted context.
"""


from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from honeycomb.budget import BudgetConfig, BudgetManager
from honeycomb.compressor import compress
from honeycomb.features import extract_features, features_to_text
from honeycomb.labels import ContentType, Label
from honeycomb.observability import metrics, timer, health_checker
from honeycomb.tee import FailureTee, get_tee
from honeycomb.gain import GainTracker, get_tracker
from honeycomb.command_filters import detect_and_filter
from honeycomb.session import SessionState, _extract_file_paths, _estimate_tokens

import re
_RE_FAILURE = re.compile(r"exit[= ]*[1-9]|error|failed|traceback", re.I)

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
    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Message(role={self.role!r}, content={content_preview!r})"


@dataclass
class CompressedMessage:
    """Output message from the firewall — depolluted, not compressed."""
    
    role: str
    """Message role (preserved from input)."""
    
    content: str
    """Depolluted content (what the LLM sees)."""
    
    label: Label
    """Depollution label that was applied."""
    
    content_type: ContentType
    """Inferred or provided content type."""
    
    original_tokens: int
    """Approximate token count of original content."""
    
    compressed_tokens: int
    """Approximate token count of compressed content."""

    tee_path: str | None = None
    """Path to raw output saved by failure tee (None if not saved)."""
    
    @property
    def compression_ratio(self) -> float:
        """Compression ratio (original / compressed)."""
        if self.compressed_tokens == 0:
            return 0.0
        return self.original_tokens / self.compressed_tokens
    def __repr__(self) -> str:
        return (f"CompressedMessage(role={self.role!r}, label={self.label.value!r}, "
                f"tokens={self.original_tokens}->{self.compressed_tokens})")


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

    # Assistant messages: reasoning, patches, or tool calls
    if role == "assistant":
        if re.search(r"^[-+]{3} |^diff --git|^@@ ", content, re.M):
            return ContentType.AGENT_PATCH
        # JSON tool call: {"name": "read_file", ...}
        if re.search(r'"name"\s*:\s*"(?:read_file|run_tests|apply_patch|search|run_command)"', content):
            return ContentType.TOOL_CALL
        # YAML-style tool call: tool_name: run_command\nargs: {...}
        if re.search(r"^tool_name\s*:\s*\w+", content, re.M):
            return ContentType.TOOL_CALL
        # Generic JSON tool call: {"name": "...", ...} or {"tool": "..."}
        if re.search(r'"(?:name|tool|function)"\s*:\s*"[^"]+"', content) and re.search(r'"(?:args|parameters|arguments|params)"\s*:', content):
            return ContentType.TOOL_CALL
        return ContentType.AGENT_REASONING

    # Tool messages: infer from content
    if role == "tool":
        # Git diff / patch output (check early — diffs contain code-like lines)
        if re.search(r"^diff --git|^[-+]{3} [ab]/", content, re.M):
            return ContentType.AGENT_PATCH

        # Command output with $ prompt — check BEFORE git/test patterns
        # because `$ git status` is command output, and `$ python -m pytest` is command not test
        if re.search(r"^\$\s+\w+", content, re.M):
            return ContentType.TOOL_RESULT_COMMAND

        # Git status/log/push/pull output
        if re.search(
            r"On branch |HEAD detached|Changes (not |to be )?staged|"
            r"nothing to commit|Untracked files:|Your branch is|"
            r"^\*?\s+[0-9a-f]{7,}\s+.+$|"  # git log one-line (hex hash required)
            r"^commit [0-9a-f]{7,40}|"
            r"\[detached HEAD|Enumerating objects|Counting objects|"
            r"Writing objects|remote:|To github\.com|"
            r"Already up to date|fast-forward|merge conflict|"
            r"^\[\S+ [0-9a-f]{7,}\]|\d+ files? changed.*insertions|"
            r"create mode \d+ |files? changed, \d+ insertions",
            content, re.M,
        ):
            return ContentType.TOOL_RESULT_GIT

        # Test output (multiple formats)
        if re.search(
            r"\d+\s*(passed|failed|error|passing|failing)|"  # pytest / mocha / jest
            r"test result:\s*\w+|"  # cargo test
            r"(passing|failing)\s+\d+|"  # mocha/jest (word-first)
            r"[✓✗✔✘]\s+|"  # mocha/jest symbols
            r"^ok\s+\S+.*\d+\.\d+s$|"  # go test ok
            r"^FAIL\s+\S+",  # go test fail
            content, re.M | re.I,
        ):
            return ContentType.TOOL_RESULT_TEST

        # Build output
        if re.search(
            r"Compiling\s+\S+|"  # cargo build
            r"Finished\s+(dev|release)\s+\[|"  # cargo build
            r"error\[E\d+\]:|"  # rustc error
            r"^tsc\s|TS\d{4,5}:|"  # typescript
            r"Build (complete|succeeded|failed)|"
            r"Successfully built\s+\S+",
            content, re.M,
        ):
            return ContentType.TOOL_RESULT_BUILD

        # Lint output
        if re.search(
            r"Found\s+\d+\s+errors?|"
            r"\d+ problems? \(\d+ errors?, \d+ warnings?\)|"  # eslint
            r"(warning|error)\[|"  # clippy / ruff
            r"^src/\S+:\d+:\d+: [A-Z]\d{2,4}\s|"  # ruff line
            r"(rubocop|golangci-lint|pylint)\s",
            content, re.M,
        ):
            return ContentType.TOOL_RESULT_LINT

        # Container output
        if re.search(
            r"^CONTAINER\s+ID\s|^IMAGE\s+|"  # docker ps
            r"REPOSITORY\s+TAG\s+IMAGE|"  # docker images
            r"^NAME\s+READY\s+STATUS\s|"  # kubectl pods
            r"^(?:service|deployment|pod)/\S+\s+",  # kubectl
            content, re.M,
        ):
            return ContentType.TOOL_RESULT_CONTAINER

        # File content (has code structure) — check BEFORE error traces
        # because files may define Error classes
        if re.search(r"^(class|def|import|from|export|function|pub |fn |struct |enum |impl )", content, re.M):
            return ContentType.TOOL_RESULT_FILE

        # Error traces — check BEFORE search results because JS error traces
        # contain file:line: patterns (e.g., "at processItem (src/utils.js:42:5)")
        if re.search(
            r"Traceback \(most recent call last\)|"
            r"^\w+Error: |^\w+Exception: |"
            r"^Error: |"  # JS errors
            r"panic: |^fatal error: ",  # Go panics
            content, re.M,
        ):
            return ContentType.TOOL_RESULT_ERROR

        # Directory listing
        if re.search(
            r"^total\s+\d+$|"  # ls -l
            r"^[d-][rwx-]{9}\s|"  # ls -l perms
            r"^[├│└].*[├│└]|[─]{3,}",  # tree output
            content, re.M,
        ):
            return ContentType.TOOL_RESULT_DIRECTORY

        # Search results (file:line patterns)
        if re.search(r"[^\s:]+:\d+:", content):
            return ContentType.TOOL_RESULT_SEARCH

        # Command output (has exit code but no $ prompt)
        if re.search(r"exit[= ]+\d+", content, re.M):
            return ContentType.TOOL_RESULT_COMMAND

        # Default: unknown
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

    # File content: COMPACT if large (>200 chars), DISTILL if small
    if content_type == ContentType.TOOL_RESULT_FILE:
        if len(content) > 200:
            return Label.COMPACT
        return Label.DISTILL

    # Command output: DISTILL
    if content_type == ContentType.TOOL_RESULT_COMMAND:
        return Label.DISTILL

    # Search results: DISTILL
    if content_type == ContentType.TOOL_RESULT_SEARCH:
        return Label.DISTILL

    # Git output: DISTILL (extract key changes/status)
    if content_type == ContentType.TOOL_RESULT_GIT:
        return Label.DISTILL

    # Build output: DISTILL (extract errors or success)
    if content_type == ContentType.TOOL_RESULT_BUILD:
        return Label.DISTILL

    # Lint output: DISTILL (extract violations)
    if content_type == ContentType.TOOL_RESULT_LINT:
        return Label.DISTILL

    # Container output: DISTILL
    if content_type == ContentType.TOOL_RESULT_CONTAINER:
        return Label.DISTILL

    # Directory listing: COMPACT (structural summary)
    if content_type == ContentType.TOOL_RESULT_DIRECTORY:
        return Label.COMPACT

    # Agent patches: DISTILL (keep summary of what changed)
    if content_type == ContentType.AGENT_PATCH:
        return Label.DISTILL

    # Agent reasoning: COMPACT for long multi-step (>=400 chars), DISTILL for shorter
    if content_type == ContentType.AGENT_REASONING:
        if len(content) >= 400:
            return Label.COMPACT
        return Label.DISTILL

    # Default: DISTILL
    return Label.DISTILL


# ---------------------------------------------------------------------------
# ML classifier wrapper
# ---------------------------------------------------------------------------

class MLClassifier:
    """Wraps a trained scikit-learn classifier for label prediction.

    When prediction confidence is below the threshold, returns ESCALATE
    instead of guessing — the compressor will defer to the LLM for
    ambiguous content rather than risk data loss from misclassification.
    """

    def __init__(self, model: Any, confidence_threshold: float = 0.0) -> None:
        self.model = model
        self.confidence_threshold = confidence_threshold

    def predict(self, features_text: str) -> Label:
        """Predict label from feature text.

        Returns ESCALATE if confidence is below threshold.
        Uses a single predict_proba call to get both prediction and confidence.
        When threshold is 0 (default), uses the fast predict path.
        """
        if self.confidence_threshold > 0 and hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba([features_text])[0]
            max_idx = max(range(len(probas)), key=lambda i: probas[i])
            max_prob = probas[max_idx]
            if max_prob < self.confidence_threshold:
                return Label.ESCALATE
            try:
                return Label(self.model.classes_[max_idx])
            except (ValueError, IndexError):
                return Label.DISTILL

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
# Honey-Comb
# ---------------------------------------------------------------------------

class HoneyComb:
    """Main entry point for inline context depollution.

    Usage:
        comb = HoneyComb()

        for message in agent_messages:
            compressed = comb.process(message)
            # Send compressed.content to LLM

    The comb maintains session state across calls. Create a new
    instance for each agent session.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        budget_config: BudgetConfig | None = None,
        cool_interval: int = 5,
        thread_safe: bool = True,
        metrics_enabled: bool = True,
        tee_enabled: bool = True,
        tee_mode: str = "failures",
        gain_enabled: bool = True,
    ) -> None:
        """Initialize the firewall.

        Args:
            model_path: Path to a trained classifier model. If None, uses
                       rule-based classification.
            budget_config: Token budget configuration. If None, uses defaults.
            cool_interval: Run cool loop every N turns (default 5).
            thread_safe: Enable thread-safe operations (default True).
            metrics_enabled: Enable metrics recording (default True).
            tee_enabled: Enable failure tee (default True). Saves raw output
                        on command failure for later re-read.
            tee_mode: When to save tee files: "failures" (default), "always", "never".
            gain_enabled: Enable gain tracking (default True). Records token savings.
        """
        self.session = SessionState(thread_safe=thread_safe)
        self.budget = BudgetManager(budget_config)
        self.cool_interval = cool_interval
        self._metrics_enabled = metrics_enabled
        self._tee = FailureTee(enabled=tee_enabled, mode=tee_mode)
        self._gain = GainTracker() if gain_enabled else None

        # Load ML classifier if provided
        self._classifier: MLClassifier | None = None
        if model_path is not None:
            try:
                from honeycomb.classifier import _load_model
                pipeline, _ = _load_model(model_path)
                self._classifier = MLClassifier(pipeline)
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Failed to load ML classifier from {model_path}: {e}. "
                    f"Falling back to rule-based classification.",
                    stacklevel=2,
                )
                self._classifier = None
    
    def process(self, message: Message) -> CompressedMessage:
        """Process a message through the hot loop.
        
        This is the main entry point. Call this for every message
        that enters the agent's context window.
        
        Returns the compressed message (what the LLM should see).
        """
        if self._metrics_enabled:
            with timer("processing"):
                return self._process_impl(message)
        else:
            return self._process_impl(message)

    def _process_impl(self, message: Message) -> CompressedMessage:
        """Internal implementation of process()."""
        try:
            # Advance turn before classification (classifier uses turn_count)
            self.session.advance_turn()

            # Infer content type if not provided
            content_type = message.content_type or _infer_content_type(message)

            # Extract file paths for messages that reference files
            # (tool results, patches, and tool calls — needed for staleness tracking)
            if (message.role.lower() == "tool"
                    or content_type.is_tool_result()
                    or content_type == ContentType.AGENT_PATCH
                    or content_type == ContentType.TOOL_CALL):
                file_paths = _extract_file_paths(message.content)
            else:
                file_paths = []

            # Extract features (pass content_type for ML classification)
            features = extract_features(message.content, message.role, self.session, file_paths, content_type=content_type.value)
            features_text = features_to_text(features)

            # Classify (hot loop: ~1-3ms)
            if self._classifier is not None:
                label = self._classifier.predict(features_text)
            else:
                label = _classify_rules(message, content_type, self.session)

            # Command filter: call once, share result with compress + tee
            filter_result = None
            if content_type in (ContentType.TOOL_RESULT_COMMAND, ContentType.TOOL_RESULT_TEST) and len(message.content) > 80:
                filter_result = detect_and_filter(message.content)

            # Compress (deterministic, ~0.1ms)
            compressed_content = compress(message.content, content_type, label, filter_result=filter_result)

            # Failure tee: save raw output on command failure (rtk-style)
            tee_path = None
            if self._tee.enabled and content_type.is_tool_result() and len(message.content) > 500:
                is_failure = (
                    filter_result.is_failure if filter_result
                    else bool(_RE_FAILURE.search(message.content[:6144]))
                )
                tee_result = self._tee.maybe_save(
                    content=message.content,
                    command=filter_result.command if filter_result else content_type.value,
                    is_failure=is_failure,
                )
                if tee_result:
                    tee_path = tee_result.tee_path
                    compressed_content += "\n" + tee_result.reference_line

            # Gain tracking: record token savings (rtk-style analytics)
            if self._gain is not None:
                cmd_name = (
                    filter_result.command if filter_result
                    else content_type.value
                )
                self._gain.record(
                    command=cmd_name,
                    raw_tokens=_estimate_tokens(message.content),
                    compressed_tokens=_estimate_tokens(compressed_content),
                    label=label.value,
                )

            # Record in session
            entry = self.session.record(
                role=message.role,
                content_type=content_type,
                label=label,
                original=message.content,
                compressed=compressed_content,
                file_paths=file_paths,
            )

            # Cool loop: periodic staleness pass
            if self.session.turn_count % self.cool_interval == 0:
                self._cool_pass()

            # Record metrics
            if self._metrics_enabled:
                compression_ratio = entry.original_tokens / max(entry.compressed_tokens, 1)
                metrics.record_message(
                    label=label.value,
                    compression_ratio=compression_ratio,
                )
                active = len(self.session.get_active_entries())
                tokens = self.session.get_total_tokens()
                metrics.update_session_state(
                    active=active, tokens=tokens, turns=self.session.turn_count,
                )

            return CompressedMessage(
                role=message.role,
                content=compressed_content,
                label=label,
                content_type=content_type,
                original_tokens=entry.original_tokens,
                compressed_tokens=entry.compressed_tokens,
                tee_path=tee_path,
            )
        except Exception as e:
            metrics.record_error()
            raise
    
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
        Filters out dropped entries and entries with empty content.
        """
        return [
            {"role": entry.role, "content": entry.compressed_content}
            for entry in self.session.get_active_entries()
            if entry.compressed_content  # Skip empty (DROP'd) entries
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
