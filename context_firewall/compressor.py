"""Deterministic compression rules per content type and label.

Each compressor takes raw content and produces compressed content based on
the assigned label. These are pure functions with no learned parameters.
"""

from __future__ import annotations

import re
from typing import Callable

from context_firewall.labels import ContentType, Label


# ---------------------------------------------------------------------------
# Tool result compressors
# ---------------------------------------------------------------------------

def compress_test_output(content: str, label: Label) -> str:
    """Compress test execution output.
    
    DISTILL: Extract pass/fail counts and failure details.
    COMPACT: Just the summary line.
    DROP: Empty string.
    """
    if label == Label.DROP:
        return ""
    
    lines = content.split("\n")
    
    # Look for summary line (pytest-style: "94 passed, 2 failed")
    summary = None
    failures = []
    in_failure_section = False
    
    for i, line in enumerate(lines):
        # Pytest summary
        if re.search(r"\d+ passed|\d+ failed|\d+ error", line, re.I):
            summary = line.strip()
        
        # Capture failure details - look for actual error messages
        if re.search(r"Error|Exception|AssertionError|assert", line, re.I):
            failures.append(line.strip())
            if len(failures) >= 5:  # Cap at 5 failures
                break
    
    if label == Label.COMPACT:
        return summary or f"Test output: {len(lines)} lines"
    
    # DISTILL
    if summary and failures:
        return f"{summary}\nFailures:\n" + "\n".join(failures[:3])
    elif summary:
        return summary
    else:
        # Fallback: last 5 lines
        return "\n".join(lines[-5:])


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
    # Match: "File: path", "# path", "// path", or bare path on first line
    path_match = re.search(
        r"(?:^(?:File: |# |// )?)"
        r"([A-Za-z]:[/\\][\w./\\-]+\.\w+"  # Windows absolute
        r"|/[\w./-]+\.\w+"                  # Unix absolute
        r"|(?:src|lib|tests?|scripts?|pkg|cmd|internal)/[\w./-]+\.\w+)",  # Relative
        content, re.I | re.M,
    )
    path = path_match.group(1) if path_match else "file"
    
    if label == Label.COMPACT:
        return f"{path} ({line_count} lines)"
    
    # DISTILL: extract structure
    symbols = []
    for line in lines:
        # Python (including indented methods)
        if re.match(r"^\s*(class|def|async def) ", line):
            symbols.append(line.strip())
        # TypeScript/JS
        elif re.match(r"^\s*(export )?(class|function|const|let) ", line):
            symbols.append(line.strip()[:80])
        # Rust
        elif re.match(r"^\s*(pub )?(fn|struct|enum|impl) ", line):
            symbols.append(line.strip()[:80])
    
    if symbols:
        return f"{path} ({line_count} lines):\n" + "\n".join(symbols[:10])
    else:
        return f"{path} ({line_count} lines)"


def compress_command_output(content: str, label: Label) -> str:
    """Compress shell command output.
    
    DISTILL: Exit code + last 3 lines.
    COMPACT: Exit code only.
    DROP: Empty string.
    """
    if label == Label.DROP:
        return ""

    if not content.strip():
        return ""

    lines = content.split("\n")
    
    # Look for exit code
    exit_match = re.search(r"exit[= ]+(\d+)|returned (\d+)", content, re.I)
    exit_code = exit_match.group(1) or exit_match.group(2) if exit_match else "?"
    
    if label == Label.COMPACT:
        return f"Command output: exit={exit_code}, {len(lines)} lines"
    
    # DISTILL: exit code + last 3 lines
    last_lines = [line.strip() for line in lines[-3:] if line.strip()]
    return f"exit={exit_code}\n" + "\n".join(last_lines)


def compress_error_trace(content: str, label: Label) -> str:
    """Compress error/exception traces.
    
    DISTILL: Error type + message + top frame.
    COMPACT: Error type + message only.
    DROP: Empty string.
    """
    if label == Label.DROP:
        return ""
    
    lines = content.split("\n")
    
    # Extract error type and message
    error_match = re.search(r"(\w+Error|\w+Exception|TypeError|ReferenceError|SyntaxError):\s*(.+)", content)
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
    frame_match = re.search(r'File "([^"]+)", line (\d+)', content)
    if frame_match:
        frame = f"at {frame_match.group(1)}:{frame_match.group(2)}"
        return f"{error_line}\n{frame}"

    # Node.js: at functionName (file.js:10:5)
    node_match = re.search(r"at\s+\S+\s+\(([^)]+):(\d+):\d+\)", content)
    if node_match:
        frame = f"at {node_match.group(1)}:{node_match.group(2)}"
        return f"{error_line}\n{frame}"

    # Rust: at src/foo.rs:42
    rust_match = re.search(r"at\s+([\w/]+\.rs):(\d+)", content)
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
    matches = re.findall(r"([\w./\\-]+\.\w+):(\d+):", content)
    
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
    json_match = re.search(r'"name"\s*:\s*"([^"]+)"', content)
    if json_match:
        return f"Called: {json_match.group(1)}"

    # Try key=value format: tool_name: read_file
    tool_match = re.search(r"tool[_ ]?name[\"']?\s*[:=]\s*[\"']?(\w+)", content, re.I)
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
    files = re.findall(r"(?:diff --git a/|^\+\+\+ b/)([^\s]+)", content, re.M)
    files = list(dict.fromkeys(files))  # Dedupe
    
    if label == Label.COMPACT:
        return f"Edited: {', '.join(files) if files else 'unknown'}"
    
    # DISTILL: files + change summary
    additions = len(re.findall(r"^\+[^+]", content, re.M))
    deletions = len(re.findall(r"^-[^-]", content, re.M))
    
    summary = f"Edited {len(files)} file(s): +{additions}/-{deletions}"
    if files:
        summary += "\n" + "\n".join(f"  {f}" for f in files[:5])
    
    return summary


def compress_reasoning(content: str, label: Label) -> str:
    """Compress agent reasoning.

    DISTILL: Extract conclusion/decision.
    COMPACT: "Reasoning: [first 50 chars]"
    DROP: Empty string.
    """
    if label == Label.DROP:
        return ""

    if label == Label.COMPACT:
        first_line = content.split("\n")[0][:50]
        return f"Reasoning: {first_line}..."

    # DISTILL: look for conclusion markers
    # Avoid "so" — too common in English, causes false positives
    conclusion_markers = [
        r"(?:therefore|thus|hence|conclusion)[:\s]+(.{10,}?)(?:\.|$)",
        r"(?:decided to|will|should|going to)\s+(.{10,}?)(?:\.|$)",
        r"(?:the fix is|the issue is|the problem is)[:\s]+(.{10,}?)(?:\.|$)",
    ]

    for pattern in conclusion_markers:
        match = re.search(pattern, content, re.I | re.M)
        if match:
            result = match.group(1).strip()
            return f"Decision: {result[:200]}"

    # Fallback: last non-empty sentence
    sentences = re.split(r"[.!?]\s+", content)
    last = next((s.strip() for s in reversed(sentences) if s.strip()), "")
    if last:
        return f"Decision: {last[:200]}"

    return f"Reasoning: {content[:200]}..."

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


def compress(content: str, content_type: ContentType, label: Label) -> str:
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
    
    return compressor_fn(content, label)
