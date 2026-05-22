"""Deterministic depollution rules per content type and label.

Each label maps to a strategy:
  - CORE: keep verbatim (system prompts, active goals, current errors)
  - DISTILL: extract key information (test results, reasoning, command output)
  - COMPACT: structural summary only (file contents, diffs)
  - DROP: remove entirely (completed tool calls, acknowledged errors)
  - STALE: mark for deletion (superseded file reads)
  - ESCALATE: defer to LLM summarization (ambiguous content)

The actual content removal is done here with regex extractors —
no model reads or understands the text.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from honeycomb.labels import ContentType, Label
from honeycomb.command_filters import detect_and_filter, FilterResult


# ---------------------------------------------------------------------------
# Pre-compiled regex patterns (hot loop optimization)
# ---------------------------------------------------------------------------
_RE_TEST_SUMMARY = re.compile(r"=*\s*\d+\s*(passed|failed|error)", re.I)
_RE_TEST_FAILED = re.compile(r"(\S+::\S+)")
_RE_TEST_ERROR = re.compile(r"(Error|Exception|assert)", re.I)
_RE_FILE_PATH = re.compile(
    r"(?:^(?:File: |# |// )?)"
    r"([A-Za-z]:[/\\][\w./\\-]+\.\w+"  # Windows absolute
    r"|/[\w./-]+\.\w+"                  # Unix absolute
    r"|(?:src|lib|tests?|scripts?|pkg|cmd|internal)/[\w./-]+\.\w+)",  # Relative
    re.I | re.M,
)
_RE_PY_SYMBOL = re.compile(r"^\s*(class|def|async def) ")
_RE_TS_SYMBOL = re.compile(r"^\s*(export )?(class|function|const|let) ")
_RE_RS_SYMBOL = re.compile(r"^\s*(pub )?(fn|struct|enum|impl) ")
_RE_EXIT_CODE = re.compile(r"exit[= ]+(\d+)|returned (\d+)", re.I)
_RE_ERROR_TYPE = re.compile(r"(\w+Error|\w+Exception|TypeError|ReferenceError|SyntaxError):\s*(.+)")
_RE_PY_FRAME = re.compile(r'File "([^"]+)", line (\d+)')
_RE_NODE_FRAME = re.compile(r"at\s+\S+\s+\(([^)]+):(\d+):\d+\)")
_RE_RS_FRAME = re.compile(r"at\s+([\w/]+\.rs):(\d+)")
_RE_SEARCH_MATCH = re.compile(r"([\w./\\-]+\.\w+):(\d+):")
_RE_TOOL_JSON = re.compile(r'"name"\s*:\s*"([^"]+)"')
_RE_TOOL_KV = re.compile(r"tool[_ ]?name[\"']?\s*[:=]\s*[\"']?(\w+)", re.I)
_RE_DIFF_FILES = re.compile(r"(?:diff --git a/|^\+\+\+ b/)([^\s]+)", re.M)
_RE_DIFF_ADD = re.compile(r"^\+[^+]", re.M)
_RE_DIFF_DEL = re.compile(r"^-[^-]", re.M)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _scan_window(content: str, head: int = 4096, tail: int = 2048) -> str:
    """Return head+tail window for regex scanning on large content.

    For content > head+tail, returns first `head` + last `tail` chars.
    Otherwise returns the full content unchanged.
    """
    if len(content) <= head + tail:
        return content
    return content[:head] + content[-tail:]

def _last_n_lines(content: str, n: int) -> list[str]:
    """Extract last n non-empty lines efficiently without splitting the entire content."""
    if len(content) <= 4096:
        # Small content: just split
        lines = content.split("\n")
        return [line.strip() for line in lines[-n:] if line.strip()]
    # Large content: scan backwards from the end
    # Find the position n newlines from the end
    pos = len(content)
    for _ in range(n + 1):
        pos = content.rfind("\n", 0, pos)
        if pos < 0:
            # Fewer than n newlines in content — fall back to full split
            lines = content.split("\n")
            return [line.strip() for line in lines[-n:] if line.strip()]
    tail = content[pos + 1:]
    lines = tail.split("\n")
    return [line.strip() for line in lines[-n:] if line.strip()]


# ---------------------------------------------------------------------------
# Tool result compressors
# ---------------------------------------------------------------------------

def compress_test_output(content: str, label: Label, filter_result: Optional[FilterResult] = None) -> str:
    """Compress test execution output.

    First attempts command-specific filtering (pytest, cargo test, go test,
    jest, rspec), then falls back to generic test output compression.

    DISTILL: Extract pass/fail counts and failure details.
    COMPACT: Just the summary line.
    DROP: Empty string.
    """
    if label == Label.DROP:
        return ""

    # Try command-specific filter first (rtk-style, skip for tiny content)
    if filter_result is None and label in (Label.DISTILL, Label.COMPACT) and len(content) > 80:
        filter_result = detect_and_filter(content)
    if filter_result is not None:
        return filter_result.compressed

    # Generic fallback — use scan window for large content
    window = _scan_window(content)
    lines = window.split("\n")
    summary = None
    failed_tests = []
    error_messages = []
    in_failures = False

    for i, line in enumerate(lines):
        if _RE_TEST_SUMMARY.search(line):
            summary = line.strip()

        if "FAILURES" in line or ("FAILED" in line and "::" not in line):
            in_failures = True

        if "FAILED" in line and "::" in line:
            match = _RE_TEST_FAILED.search(line)
            if match and match.group(1) not in failed_tests:
                failed_tests.append(match.group(1))

        if in_failures and _RE_TEST_ERROR.search(line):
            error_msg = line.strip()
            if error_msg and error_msg not in error_messages and "FAILED" not in error_msg:
                error_messages.append(error_msg)
                if len(error_messages) >= 3:
                    break

    if label == Label.COMPACT:
        return summary or f"Test output: {len(lines)} lines"

    result = summary or f"Test output: {len(lines)} lines"
    if failed_tests:
        result += "\n" + "\n".join(f"FAILED {t}" for t in failed_tests)
    if error_messages:
        result += "\n" + "\n".join(error_messages)
    return result

def compress_file_content(content: str, label: Label) -> str:
    """Compress file read results.

    DISTILL: Path + line count + function/class signatures.
    COMPACT: Path + line count only.
    DROP: Empty string.
    """
    if label == Label.DROP:
        return ""

    lines = content.split("\n")
    line_count = len(lines)

    # Try to extract file path from first line or content
    path_match = _RE_FILE_PATH.search(content)
    path = path_match.group(1) if path_match else "file"

    if label == Label.COMPACT:
        return f"{path} ({line_count} lines)"

    # DISTILL: extract structure
    symbols = []
    for line in lines:
        # Python (including indented methods)
        if _RE_PY_SYMBOL.match(line):
            symbols.append(line.strip())
        # TypeScript/JS
        elif _RE_TS_SYMBOL.match(line):
            symbols.append(line.strip()[:80])
        # Rust
        elif _RE_RS_SYMBOL.match(line):
            symbols.append(line.strip()[:80])

    if symbols:
        return f"{path} ({line_count} lines):\n" + "\n".join(symbols[:10])
    else:
        return f"{path} ({line_count} lines)"


def compress_command_output(content: str, label: Label, filter_result: Optional[FilterResult] = None) -> str:
    """Compress shell command output.

    First attempts command-specific filtering (rtk-style), then falls back
    to generic compression.

    DISTILL: Exit code + last 3 lines (or command-specific filter).
    COMPACT: Exit code only.
    DROP: Empty string.
    """
    if label == Label.DROP:
        return ""

    if not content.strip():
        return ""

    # Try command-specific filter first (rtk-style, skip for tiny content)
    if filter_result is None and label in (Label.DISTILL, Label.COMPACT) and len(content) > 80:
        filter_result = detect_and_filter(content)
    if filter_result is not None:
        return filter_result.compressed

    # Generic fallback — use scan window for regex on large content
    window = _scan_window(content)
    line_count = content.count("\n") + 1

    # Look for exit code in the windowed content
    exit_match = _RE_EXIT_CODE.search(window)
    exit_code = exit_match.group(1) or exit_match.group(2) if exit_match else "?"

    if label == Label.COMPACT:
        return f"Command output: exit={exit_code}, {line_count} lines"

    # DISTILL: exit code + last 3 lines (efficient for large content)
    last_lines = _last_n_lines(content, 3)
    return f"exit={exit_code}\n" + "\n".join(last_lines)


def compress_error_trace(content: str, label: Label) -> str:
    """Compress error/exception traces.
    
    DISTILL: Error type + message + top frame.
    COMPACT: Error type + message only.
    DROP: Empty string.
    """
    if label == Label.DROP:
        return ""
    
    # Use scan window for large content
    window = _scan_window(content)
    lines = window.split("\n")

    # Extract error type and message
    error_match = _RE_ERROR_TYPE.search(window)
    if error_match:
        error_type = error_match.group(1)
        error_msg = error_match.group(2).strip()[:200]
        error_line = f"{error_type}: {error_msg}"
    else:
        error_line = "Error: " + (lines[0][:200] if lines else "unknown")

    if label == Label.COMPACT:
        return error_line

    # DISTILL: error + top frame
    # Python: File "foo.py", line 42
    frame_match = _RE_PY_FRAME.search(window)
    if frame_match:
        frame = f"at {frame_match.group(1)}:{frame_match.group(2)}"
        return f"{error_line}\n{frame}"

    # Node.js: at functionName (file.js:10:5)
    node_match = _RE_NODE_FRAME.search(window)
    if node_match:
        frame = f"at {node_match.group(1)}:{node_match.group(2)}"
        return f"{error_line}\n{frame}"

    # Rust: at src/foo.rs:42
    rust_match = _RE_RS_FRAME.search(window)
    if rust_match:
        frame = f"at {rust_match.group(1)}:{rust_match.group(2)}"
        return f"{error_line}\n{frame}"

    return error_line


def compress_search_result(content: str, label: Label) -> str:
    """Compress search/query results.
    
    DISTILL: Matched files/paths with line numbers.
    COMPACT: Count of matches.
    DROP: Empty string.
    """
    if label == Label.DROP:
        return ""
    
    lines = content.split("\n")
    
    # Extract file:line patterns (require file extension to avoid URL/time false positives)
    matches = _RE_SEARCH_MATCH.findall(content)

    if label == Label.COMPACT:
        return f"Search: {len(matches)} matches"

    # DISTILL: unique files with line numbers
    if matches:
        unique = list(dict.fromkeys((f, l) for f, l in matches[:10]))
        return "Search results:\n" + "\n".join(f"{f}:{l}" for f, l in unique)

    return f"Search: {len(lines)} lines"


def compress_tool_call(content: str, label: Label) -> str:
    """Compress tool call requests.
    
    Usually DROP (the result is what matters).
    """
    if label == Label.DROP:
        return ""
    
    # DISTILL: just the tool name
    # Try JSON format: {"name": "read_file", ...}
    json_match = _RE_TOOL_JSON.search(content)
    if json_match:
        return f"Called: {json_match.group(1)}"

    # Try key=value format: tool_name: read_file
    tool_match = _RE_TOOL_KV.search(content)
    if tool_match:
        return f"Called: {tool_match.group(1)}"

    return "Tool call"


def compress_agent_patch(content: str, label: Label) -> str:
    """Compress code edits/patches.
    
    DISTILL: Files changed + summary.
    COMPACT: Files changed only.
    DROP: Empty string.
    """
    if label == Label.DROP:
        return ""
    
    # Extract changed files
    files = _RE_DIFF_FILES.findall(content)
    files = list(dict.fromkeys(files))  # Dedupe

    if label == Label.COMPACT:
        return f"Edited: {', '.join(files) if files else 'unknown'}"

    # DISTILL: files + change summary
    additions = len(_RE_DIFF_ADD.findall(content))
    deletions = len(_RE_DIFF_DEL.findall(content))
    
    summary = f"Edited {len(files)} file(s): +{additions}/-{deletions}"
    if files:
        summary += "\n" + "\n".join(f"  {f}" for f in files[:5])
    
    return summary


def compress_reasoning(content: str, label: Label) -> str:
    """Compress agent reasoning.

    CORE: Keep verbatim (short reasoning is valuable).
    DISTILL: Extract conclusion/decision.
    COMPACT: "Reasoning: [first 200 chars]"
    DROP: Empty string.
    """
    if label == Label.CORE:
        return content

    if label == Label.DROP:
        return ""

    if label == Label.COMPACT:
        preview = content[:200].replace("\n", " ").strip()
        return f"Reasoning: {preview}..."

    # DISTILL: extract the key decision/action
    lines = content.split("\n")

    # Look for structured content first (headers, bullet points, numbered lists)
    key_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("- ") or stripped.startswith("* "):
            key_lines.append(stripped)
        elif re.match(r"^\d+\.\s+", stripped):  # Numbered list
            key_lines.append(stripped)

    if key_lines:
        return "Decision:\n" + "\n".join(key_lines[:10])

    # Look for conclusion markers
    conclusion_markers = [
        r"(?:therefore|thus|hence|conclusion)[:\s]+(.{10,}?)(?:\.|$)",
        r"(?:decided to|going to)\s+(.{10,}?)(?:\.|$)",
        r"(?:the fix is|the issue is|the problem is)[:\s]+(.{10,}?)(?:\.|$)",
    ]

    for pattern in conclusion_markers:
        match = re.search(pattern, content, re.I | re.M)
        if match:
            result = match.group(1).strip()
            return f"Decision: {result[:300]}"

    # Fallback: first paragraph (usually contains the main point)
    paragraphs = content.split("\n\n")
    if paragraphs and paragraphs[0].strip():
        return f"Reasoning: {paragraphs[0].strip()[:300]}"

    return f"Reasoning: {content[:300]}..."

def compress_user_goal(content: str, label: Label) -> str:
    """Compress user goals.
    
    CORE: Keep verbatim (always relevant while active).
    DISTILL: First 100 chars.
    """
    if label == Label.CORE:
        return content
    
    # DISTILL: truncate
    return content[:100] + ("..." if len(content) > 100 else "")


def compress_system(content: str, label: Label) -> str:
    """Compress system prompts.
    
    Usually CORE (keep verbatim).
    """
    if label == Label.CORE:
        return content
    
    return content[:200] + "..."


# ---------------------------------------------------------------------------
# Compressor registry
# ---------------------------------------------------------------------------

CompressorFn = Callable[[str, Label], str]

_COMPRESSORS: dict[ContentType, CompressorFn] = {
    ContentType.TOOL_RESULT_TEST: compress_test_output,
    ContentType.TOOL_RESULT_FILE: compress_file_content,
    ContentType.TOOL_RESULT_COMMAND: compress_command_output,
    ContentType.TOOL_RESULT_ERROR: compress_error_trace,
    ContentType.TOOL_RESULT_SEARCH: compress_search_result,
    ContentType.TOOL_CALL: compress_tool_call,
    ContentType.AGENT_PATCH: compress_agent_patch,
    ContentType.AGENT_REASONING: compress_reasoning,
    ContentType.USER_GOAL: compress_user_goal,
    ContentType.SYSTEM: compress_system,
}


def compress(content: str, content_type: ContentType, label: Label, filter_result=None) -> str:
    """Apply compression based on content type and label.

    This is the main entry point for the deterministic compressor.
    """
    # ESCALATE: pass through unchanged (LLM will decide)
    if label == Label.ESCALATE:
        return content

    # STALE: compress aggressively (will be dropped on next cool pass)
    if label == Label.STALE:
        # Treat as COMPACT for compression purposes
        label = Label.COMPACT

    # Get the compressor for this content type
    compressor_fn = _COMPRESSORS.get(content_type)

    if compressor_fn is None:
        # Unknown content type: apply generic compression
        if label == Label.DROP:
            return ""
        elif label == Label.COMPACT:
            return content[:100] + "..."
        elif label == Label.DISTILL:
            return content[:200] + "..."
        else:
            return content

    # Pass filter_result to compressors that support it
    if content_type in (ContentType.TOOL_RESULT_TEST, ContentType.TOOL_RESULT_COMMAND):
        return compressor_fn(content, label, filter_result=filter_result)
    else:
        return compressor_fn(content, label)
