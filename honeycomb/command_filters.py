"""Command-specific output filters (rtk-style).

Detects which command produced the output and applies hand-tuned compression
that generic rules cannot match. Each filter extracts the signal that an LLM
actually needs, discarding boilerplate, progress bars, and decoration.

Supported command families:
  - git: status, log, diff, add, commit, push, pull
  - test runners: pytest, cargo test, go test, jest/vitest, rspec
  - build: cargo build, cargo clippy
  - linters: ruff, eslint, golangci-lint, rubocop
  - containers: docker ps/images/logs, kubectl pods/services
  - directory listing: ls/tree-style output
  - package managers: pip list, pnpm list, npm list
  - AWS CLI: sts, ec2, lambda, s3, logs, cloudformation, dynamodb, iam

Usage:
    from honeycomb.command_filters import detect_and_filter

    result = detect_and_filter(raw_output)
    if result is not None:
        compressed = result  # Command was recognized and filtered
    else:
        compressed = generic_compress(raw_output)  # Fall through
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Detection + dispatch
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FilterResult:
    """Result of a command-specific filter."""

    command: str
    """Detected command family (e.g. 'git', 'pytest', 'cargo_test')."""

    compressed: str
    """Compressed output."""

    raw_tokens: int
    """Approximate token count of raw output."""

    compressed_tokens: int
    """Approximate token count of compressed output."""

    is_failure: bool
    """Whether the command output indicates failure (non-zero exit)."""


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Git filters
# ---------------------------------------------------------------------------

def _detect_git(content: str) -> bool:
    """Detect git output from characteristic markers."""
    return bool(re.search(
        r"On branch |HEAD detached|Changes (not |to be )?staged|"
        r"nothing to commit|Untracked files:|Your branch is|"
        r"^\* |^\*?\s+\w+\s+\w{7,}\s+.+$|"  # git log one-line
        r"^diff --git |^commit [0-9a-f]{7,40}|"
        r"\[detached HEAD|Enumerating objects|Counting objects|"
        r"Writing objects|remote:|To github\.com|fatal:|"
        r"Already up to date|fast-forward|merge conflict|"
        r"^\[\S+ [0-9a-f]{7,}\]|\d+ files? changed.*insertions|"
        r"create mode \d+ |files? changed, \d+ insertions",
        content, re.M,
    ))


def _filter_git_status(content: str) -> str:
    """Compress git status output."""
    lines = content.split("\n")

    # Already clean
    if re.search(r"nothing to commit.*working tree clean", content):
        branch = _extract_branch(content)
        return f"git status: clean ({branch})"

    branch = _extract_branch(content)
    staged = re.findall(r"(?:modified|new file|deleted|renamed):\s+(\S+)", content)
    unstaged = []
    untracked = re.findall(r"^\t(\S+)", content, re.M)

    # Parse sections more carefully
    section = None
    for line in lines:
        stripped = line.strip()
        if "Changes to be committed" in stripped:
            section = "staged"
        elif "Changes not staged" in stripped:
            section = "unstaged"
        elif "Untracked files" in stripped:
            section = "untracked"
        elif section and re.match(r"(?:modified|new file|deleted|renamed):\s+(\S+)", stripped):
            path = re.match(r"(?:modified|new file|deleted|renamed):\s+(\S+)", stripped)
            if path:
                if section == "unstaged":
                    unstaged.append(path.group(1))
        elif section == "untracked" and stripped and not stripped.startswith("#"):
            if stripped not in untracked:
                untracked.append(stripped)

    parts = [f"git status ({branch})"]
    if staged:
        parts.append(f"  staged: {', '.join(staged[:10])}")
    if unstaged:
        parts.append(f"  modified: {', '.join(unstaged[:10])}")
    if untracked:
        parts.append(f"  untracked: {len(untracked)} file(s)")
    ahead = re.search(r"Your branch is ahead of '\S+' by (\d+)", content)
    behind = re.search(r"Your branch is behind '\S+' by (\d+)", content)
    if ahead:
        parts.append(f"  ahead {ahead.group(1)}")
    if behind:
        parts.append(f"  behind {behind.group(1)}")
    return "\n".join(parts)


def _extract_branch(content: str) -> str:
    match = re.search(r"On branch (\S+)|HEAD detached (?:from|at) (\S+)", content)
    if match:
        return match.group(1) or match.group(2)
    return "?"


def _filter_git_log(content: str) -> str:
    """Compress git log output to one-line-per-commit."""
    lines = content.split("\n")
    commits = []
    current_hash = None
    current_msg = None

    for line in lines:
        # Standard git log format
        hash_match = re.match(r"^commit ([0-9a-f]{7,40})", line)
        if hash_match:
            if current_hash and current_msg:
                commits.append(f"{current_hash[:7]} {current_msg}")
            current_hash = hash_match.group(1)
            current_msg = None
            continue

        # One-line format (already compact): abc1234 message (date)
        oneline = re.match(r"^([0-9a-f]{7,40})\s+(.+?)(?:\s+\(.+\))?$", line)
        if oneline and not line.startswith("commit"):
            commits.append(f"{oneline.group(1)[:7]} {oneline.group(2)}")
            continue

        if line.strip().startswith("Date:") or line.strip().startswith("Author:"):
            continue
        if line.strip() and current_hash and current_msg is None:
            current_msg = line.strip()

    if current_hash and current_msg:
        commits.append(f"{current_hash[:7]} {current_msg}")

    if not commits:
        return f"git log: {len(lines)} lines"

    # Limit to 10 most recent
    shown = commits[:10]
    suffix = f" (+{len(commits) - 10} more)" if len(commits) > 10 else ""
    return f"git log ({len(commits)} commits):\n" + "\n".join(shown) + suffix


def _filter_git_diff(content: str) -> str:
    """Compress git diff output."""
    files = re.findall(r"^diff --git a/(\S+) b/(\S+)", content, re.M)
    additions = len(re.findall(r"^\+[^+]", content, re.M))
    deletions = len(re.findall(r"^-[^-]", content, re.M))
    file_list = list(dict.fromkeys(b for _, b in files))

    if not file_list:
        # Maybe it's a unified diff without the header
        hunks = re.findall(r"^@@ .+ @@", content, re.M)
        return f"diff: {len(hunks)} hunk(s), +{additions}/-{deletions}"

    summary = f"diff: {len(file_list)} file(s) +{additions}/-{deletions}"
    if file_list:
        summary += "\n" + "\n".join(f"  {f}" for f in file_list[:10])
    return summary


def _filter_git_push_pull(content: str) -> str:
    """Compress git push/pull output."""
    # Success patterns
    if re.search(r"Everything up[- ]to[- ]date", content):
        return "ok (up to date)"

    # Push success
    push_match = re.search(
        r"To (\S+)\s*\n\s*\S+\.\.\S+\s+(\S+)\s+->\s+(\S+)", content
    )
    if push_match:
        return f"ok {push_match.group(3)}"

    # Pull success
    pull_match = re.search(r"(\d+) files? changed.*?(\d+) insertions?.*?(\d+) deletions?", content)
    if pull_match:
        return f"ok {pull_match.group(1)} files +{pull_match.group(2)} -{pull_match.group(3)}"

    fast_forward = re.search(r"Fast-forward", content)
    if fast_forward:
        files_changed = re.search(r"(\d+) files? changed", content)
        if files_changed:
            return f"ok fast-forward ({files_changed.group(1)} files)"
        return "ok fast-forward"

    # Already up to date
    if re.search(r"Already up to date", content):
        return "ok (already up to date)"

    # Enumerating/counting without error = success
    if re.search(r"Enumerating objects|Counting objects|Writing objects", content):
        if not re.search(r"error|fatal|rejected", content, re.I):
            branch = re.search(r"-> (\S+)", content)
            return f"ok {branch.group(1)}" if branch else "ok"

    # Error
    error = re.search(r"fatal: (.+)|error: (.+)|rejected \((.+)\)", content, re.I)
    if error:
        msg = error.group(1) or error.group(2) or error.group(3)
        return f"FAILED: {msg.strip()[:200]}"

    return content[:200]


def _filter_git_add_commit(content: str) -> str:
    """Compress git add/commit output."""
    if re.search(r"nothing to commit", content):
        return "nothing to commit"
    if re.search(r"nothing added to commit", content):
        return "nothing added"

    # Commit success
    commit_match = re.search(
        r"\[(\S+)\s+([0-9a-f]+)\]\s+(.+)", content
    )
    if commit_match:
        branch = commit_match.group(1)
        sha = commit_match.group(2)
        msg = commit_match.group(3).strip()
        return f"ok {sha} on {branch}: {msg[:100]}"

    # Detached HEAD commit
    detached = re.search(r"\[detached HEAD ([0-9a-f]+)\]\s+(.+)", content)
    if detached:
        return f"ok {detached.group(1)}: {detached.group(2).strip()[:100]}"

    # Files staged
    files = re.findall(r"(?:modified|new file|deleted):\s+(\S+)", content)
    if files:
        return f"staged: {', '.join(files[:10])}"

    return content[:200]


def _filter_git(content: str) -> FilterResult:
    """Route git output to the appropriate sub-filter."""
    is_log = bool(re.search(r"^commit [0-9a-f]{7}|^Author:|^Date:", content, re.M))
    is_diff = bool(re.search(r"^diff --git |^--- a/|^\+\+\+ b/|^@@ .+ @@", content, re.M))
    is_push_pull = bool(re.search(
        r"Enumerating objects|Counting objects|Writing objects|"
        r"remote:|To github\.com|Already up.to.date|"
        r"Fast-forward|files? changed.*insertions|rejected|fatal:",
        content, re.I,
    ))
    is_status = bool(re.search(
        r"On branch |HEAD detached|Changes (not |to be )?staged|"
        r"nothing to commit.*working tree clean|Untracked files:",
        content, re.M,
    ))
    is_add_commit = bool(re.search(
        r"nothing to commit|nothing added|\[\S+\s+[0-9a-f]+\]|\[detached HEAD",
        content, re.M,
    ))

    if is_log and not is_diff:
        compressed = _filter_git_log(content)
        cmd = "git_log"
    elif is_diff:
        compressed = _filter_git_diff(content)
        cmd = "git_diff"
    elif is_status:
        compressed = _filter_git_status(content)
        cmd = "git_status"
    elif is_add_commit:
        compressed = _filter_git_add_commit(content)
        cmd = "git_add_commit"
    elif is_push_pull:
        compressed = _filter_git_push_pull(content)
        cmd = "git_push_pull"
    else:
        compressed = _filter_git_status(content)
        cmd = "git_status"

    is_failure = bool(re.search(r"fatal:|error:|rejected|conflict", content, re.I))
    return FilterResult(
        command=cmd,
        compressed=compressed,
        raw_tokens=_estimate_tokens(content),
        compressed_tokens=_estimate_tokens(compressed),
        is_failure=is_failure,
    )


# ---------------------------------------------------------------------------
# Test runner filters
# ---------------------------------------------------------------------------

def _detect_pytest(content: str) -> bool:
    return bool(re.search(
        r"={3,}\s*(test session starts|FAILURES|ERRORS|short test summary|PASSED|FAILED)|"
        r"collecting \.\.\.|collected \d+ items|"
        r"--+ live log|platform \w+ -- Python",
        content, re.I,
    ))


def _filter_pytest(content: str) -> FilterResult:
    """Compress pytest output — failures only + summary."""
    lines = content.split("\n")

    # Extract summary line
    summary = None
    for line in lines:
        s = re.search(
            r"=*\s*(\d+\s*(?:passed|failed|error|skipped|warnings?)"
            r"(?:.*(?:passed|failed|error|skipped|warnings?))?)"
            r".*=*\s*$",
            line, re.I,
        )
        if s:
            summary = line.strip().strip("=").strip()
            break

    # Extract FAILED test names
    failed_tests = []
    for line in lines:
        m = re.search(r"FAILED\s+(\S+::\S+)", line)
        if m:
            name = m.group(1)
            if name not in failed_tests:
                failed_tests.append(name)

    # Extract ERROR test names
    error_tests = []
    for line in lines:
        m = re.search(r"ERROR\s+(\S+::\S+)", line)
        if m:
            name = m.group(1)
            if name not in error_tests:
                error_tests.append(name)

    # Extract failure details (assertion errors, tracebacks)
    failure_details = []
    in_failure = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("E ") or stripped.startswith(">"):
            failure_details.append(stripped[:200])
            in_failure = True
        elif in_failure and (
            re.search(r"AssertionError|ValueError|TypeError|KeyError|"
                      r"AttributeError|ImportError|SyntaxError|"
                      r"IndexError|NameError|RuntimeError", stripped)
        ):
            failure_details.append(stripped[:200])
            if len(failure_details) >= 5:
                break
        elif stripped.startswith("=") or stripped.startswith("-"):
            in_failure = False

    is_failure = bool(
        failed_tests or error_tests
        or (
            summary and re.search(r"\d+\s*(failed|error)", summary, re.I)
            and not re.search(r"0\s*(failed|error)", summary, re.I)
        )
    )

    if not summary:
        summary = f"pytest: {len(lines)} lines"

    parts = [summary]
    if failed_tests:
        parts.append("Failures:")
        parts.extend(f"  {t}" for t in failed_tests[:10])
    if error_tests:
        parts.append("Errors:")
        parts.extend(f"  {t}" for t in error_tests[:10])
    if failure_details:
        parts.extend(failure_details[:5])

    compressed = "\n".join(parts)
    return FilterResult(
        command="pytest",
        compressed=compressed,
        raw_tokens=_estimate_tokens(content),
        compressed_tokens=_estimate_tokens(compressed),
        is_failure=is_failure,
    )


def _detect_cargo_test(content: str) -> bool:
    return bool(re.search(
        r"running \d+ tests?|"
        r"test result: (ok|FAILED)|"
        r"test \S+ \.\.\. (ok|FAILED|ignored)",
        content, re.M,
    ))


def _filter_cargo_test(content: str) -> FilterResult:
    """Compress cargo test output."""
    lines = content.split("\n")

    # Extract result line
    result_match = re.search(
        r"test result: (\w+)\.\s+(\d+) passed;\s*(\d+) failed;\s*(\d+) ignored",
        content,
    )

    # Extract failed test names
    failed_tests = []
    for line in lines:
        m = re.search(r"test (\S+) \.\.\. FAILED", line)
        if m:
            failed_tests.append(m.group(1))

    # Extract panic/failure output
    failure_details = []
    in_panic = False
    for line in lines:
        if "panicked at" in line or "thread '" in line and "panicked" in line:
            failure_details.append(line.strip()[:200])
            in_panic = True
        elif in_panic and line.strip().startswith("note:") or (in_panic and "at " in line and ".rs:" in line):
            failure_details.append(line.strip()[:200])
            if len(failure_details) >= 5:
                break
        elif line.strip() == "" and in_panic:
            in_panic = False

    is_failure = bool(failed_tests) or (result_match and result_match.group(1) == "FAILED")

    if result_match:
        status, passed, failed, ignored = result_match.groups()
        summary = f"cargo test: {status} — {passed} passed, {failed} failed"
        if int(ignored) > 0:
            summary += f", {ignored} ignored"
    else:
        summary = f"cargo test: {len(lines)} lines"

    parts = [summary]
    if failed_tests:
        parts.append("Failures:")
        parts.extend(f"  {t}" for t in failed_tests[:10])
    if failure_details:
        parts.extend(failure_details[:5])

    compressed = "\n".join(parts)
    return FilterResult(
        command="cargo_test",
        compressed=compressed,
        raw_tokens=_estimate_tokens(content),
        compressed_tokens=_estimate_tokens(compressed),
        is_failure=is_failure,
    )


def _detect_go_test(content: str) -> bool:
    return bool(re.search(
        r"^=== RUN\s+|^--- (PASS|FAIL):|^ok\s+\S+\s+\d+\.\d+s|"
        r"^FAIL\s+\S+\s+\d+\.\d+s|^\?\s+\S+\s+\[no test files\]",
        content, re.M,
    ))


def _filter_go_test(content: str) -> FilterResult:
    """Compress go test output."""
    lines = content.split("\n")

    # Extract package results
    ok_packages = re.findall(r"^ok\s+(\S+)\s+(\d+\.\d+s)", content, re.M)
    fail_packages = re.findall(r"^FAIL\s+(\S+)\s+(\d+\.\d+s)", content, re.M)
    skip_packages = re.findall(r"^\?\s+(\S+)\s+\[no test files\]", content, re.M)

    # Extract failed test names
    failed_tests = re.findall(r"^--- FAIL: (\S+)", content, re.M)

    is_failure = bool(fail_packages or failed_tests)

    parts = []
    if ok_packages:
        parts.append(f"ok: {len(ok_packages)} package(s)")
    if fail_packages:
        parts.append(f"FAIL: {len(fail_packages)} package(s)")
    if skip_packages:
        parts.append(f"skip: {len(skip_packages)} (no tests)")

    if not parts:
        parts.append(f"go test: {len(lines)} lines")

    if failed_tests:
        parts.append("Failures:")
        parts.extend(f"  {t}" for t in failed_tests[:10])

    # Extract failure output
    failure_lines = []
    for line in lines:
        if re.search(r"Error|panic:|Fatal|got .* want", line, re.I):
            failure_lines.append(line.strip()[:200])
            if len(failure_lines) >= 5:
                break
    if failure_lines:
        parts.extend(failure_lines)

    compressed = "\n".join(parts)
    return FilterResult(
        command="go_test",
        compressed=compressed,
        raw_tokens=_estimate_tokens(content),
        compressed_tokens=_estimate_tokens(compressed),
        is_failure=is_failure,
    )


def _detect_jest(content: str) -> bool:
    return bool(re.search(
        r"(PASS|FAIL)\s+\S+\.\w+|"
        r"Tests:\s+\d+\s+(passed|failed)|"
        r"Test Suites:\s+\d+\s+(passed|failed)|"
        r"Snapshots:\s+\d+\s+(passed|failed)|"
        r"Time:\s+\d+\.?\d*\s*s|"
        r"Ran all test suites",
        content, re.I,
    ))


def _filter_jest(content: str) -> FilterResult:
    """Compress jest/vitest output."""
    lines = content.split("\n")

    # Extract summary
    suites = re.search(r"Test Suites:\s+(.+)", content)
    tests = re.search(r"Tests:\s+(.+)", content)
    time_match = re.search(r"Time:\s+(\d+\.?\d*\s*s)", content)

    # Extract failed tests
    failed_tests = re.findall(r"FAIL\s+(\S+)", content)
    failed_details = re.findall(r"●\s+(.+?)(?:\n|$)", content)

    is_failure = bool(failed_tests) or (
        tests and re.search(r"\d+\s+failed", tests.group(1), re.I)
    )

    parts = []
    if suites:
        parts.append(f"Suites: {suites.group(1).strip()}")
    if tests:
        parts.append(f"Tests: {tests.group(1).strip()}")
    if time_match:
        parts.append(f"Time: {time_match.group(1)}")

    if not parts:
        parts.append(f"jest: {len(lines)} lines")

    if failed_tests:
        parts.append("Failures:")
        parts.extend(f"  {t}" for t in failed_tests[:10])
    if failed_details and not failed_tests:
        parts.extend(f"  {d.strip()[:200]}" for d in failed_details[:5])

    compressed = "\n".join(parts)
    return FilterResult(
        command="jest",
        compressed=compressed,
        raw_tokens=_estimate_tokens(content),
        compressed_tokens=_estimate_tokens(compressed),
        is_failure=is_failure,
    )


def _detect_rspec(content: str) -> bool:
    return bool(re.search(
        r"\d+ examples?,\s*\d+ failures?|"
        r"Randomized with seed \d+",
        content, re.I,
    ))


def _filter_rspec(content: str) -> FilterResult:
    """Compress rspec output."""
    lines = content.split("\n")

    summary = re.search(r"(\d+ examples?,\s*\d+ failures?(?:.*?\d+ pending)?)", content)
    time_match = re.search(r"Finished in (\d+\.?\d* seconds)", content)
    failures = re.findall(r"(?:Failure|Error):\s+(.+?)(?:\n|$)", content)

    is_failure = bool(re.search(r"[1-9]\d*\s+failures?", content, re.I))

    parts = []
    if summary:
        parts.append(f"rspec: {summary.group(1)}")
    if time_match:
        parts.append(f"Time: {time_match.group(1)}")
    if not parts:
        parts.append(f"rspec: {len(lines)} lines")
    if failures:
        parts.append("Failures:")
        parts.extend(f"  {f.strip()[:200]}" for f in failures[:10])

    compressed = "\n".join(parts)
    return FilterResult(
        command="rspec",
        compressed=compressed,
        raw_tokens=_estimate_tokens(content),
        compressed_tokens=_estimate_tokens(compressed),
        is_failure=is_failure,
    )


# ---------------------------------------------------------------------------
# Build / lint filters
# ---------------------------------------------------------------------------

def _detect_cargo_build(content: str) -> bool:
    return bool(re.search(
        r"Compiling \S+ v\S+|"
        r"Finished (?:dev|release|test) \[|"
        r"error\[E\d+\]:|"
        r"warning\[\w+\]:",
        content, re.M,
    ))


def _filter_cargo_build(content: str) -> FilterResult:
    """Compress cargo build/clippy output."""
    lines = content.split("\n")

    errors = re.findall(r"^error(?:\[E\d+\])?: (.+)", content, re.M)
    warnings = re.findall(r"^warning(?:\[\w+\])?: (.+)", content, re.M)
    finished = re.search(r"Finished (dev|release|test) \[([^\]]+)\]", content)

    is_failure = bool(errors)

    parts = []
    if finished:
        parts.append(f"cargo build: {finished.group(1)} [{finished.group(2)}]")
    if errors:
        parts.append(f"errors: {len(errors)}")
        parts.extend(f"  {e[:200]}" for e in errors[:5])
    if warnings:
        parts.append(f"warnings: {len(warnings)}")

    if not parts:
        parts.append(f"cargo build: {len(lines)} lines")

    compressed = "\n".join(parts)
    cmd = "cargo_clippy" if "clippy" in content.lower() else "cargo_build"
    return FilterResult(
        command=cmd,
        compressed=compressed,
        raw_tokens=_estimate_tokens(content),
        compressed_tokens=_estimate_tokens(compressed),
        is_failure=is_failure,
    )


def _detect_ruff(content: str) -> bool:
    return bool(re.search(
        r"Found \d+ errors?|"
        r"All checks passed|"
        r"^\S+\.py:\d+:\d+: [A-Z]\d{3,4}\s|"
        r"ruff \d+\.\d+\.\d+",
        content, re.M,
    ))


def _filter_ruff(content: str) -> FilterResult:
    """Compress ruff/eslint/linter output."""
    lines = content.split("\n")

    all_pass = re.search(r"All checks passed|no issues found|0 errors", content, re.I)
    if all_pass:
        compressed = "lint: all checks passed"
        return FilterResult("ruff", compressed, _estimate_tokens(content),
                            _estimate_tokens(compressed), False)

    # Count by rule code
    violations = re.findall(r"^(\S+):(\d+):(\d+): ([A-Z]\d{3,4})\s+(.+)", content, re.M)
    by_rule: dict[str, int] = {}
    by_file: dict[str, int] = {}
    for filepath, _line, _col, code, _msg in violations:
        by_rule[code] = by_rule.get(code, 0) + 1
        by_file[filepath] = by_file.get(filepath, 0) + 1

    if violations:
        parts = [f"ruff: {len(violations)} violation(s)"]
        if by_rule:
            top_rules = sorted(by_rule.items(), key=lambda x: -x[1])[:5]
            parts.append("  by rule: " + ", ".join(f"{code}({n})" for code, n in top_rules))
        if by_file:
            top_files = sorted(by_file.items(), key=lambda x: -x[1])[:5]
            parts.append("  by file: " + ", ".join(f"{f}({n})" for f, n in top_files))
        compressed = "\n".join(parts)
    else:
        found = re.search(r"Found (\d+) errors?", content)
        if found:
            compressed = f"ruff: {found.group(1)} errors"
        else:
            compressed = f"ruff: {len(lines)} lines"

    return FilterResult("ruff", compressed, _estimate_tokens(content),
                        _estimate_tokens(compressed), True)


def _detect_golangci_lint(content: str) -> bool:
    return bool(re.search(
        r"^\S+\.go:\d+:\d+: .+ \(\w+\)$|"
        r"golangci-lint|"
        r"ERRO\[",
        content, re.M,
    ))


def _filter_golangci_lint(content: str) -> FilterResult:
    """Compress golangci-lint output."""
    violations = re.findall(r"^(\S+\.go):(\d+):(\d+): (.+) \((\w+)\)", content, re.M)
    by_linter: dict[str, int] = {}
    by_file: dict[str, int] = {}
    for filepath, _line, _col, _msg, linter in violations:
        by_linter[linter] = by_linter.get(linter, 0) + 1
        by_file[filepath] = by_file.get(filepath, 0) + 1

    if violations:
        parts = [f"golangci-lint: {len(violations)} issue(s)"]
        top = sorted(by_linter.items(), key=lambda x: -x[1])[:5]
        parts.append("  by linter: " + ", ".join(f"{l}({n})" for l, n in top))
        compressed = "\n".join(parts)
    else:
        compressed = "golangci-lint: no issues"

    return FilterResult("golangci_lint", compressed, _estimate_tokens(content),
                        _estimate_tokens(compressed), bool(violations))


# ---------------------------------------------------------------------------
# Container filters
# ---------------------------------------------------------------------------

def _detect_docker(content: str) -> bool:
    return bool(re.search(
        r"^CONTAINER ID\s+IMAGE|"  # docker ps header
        r"^REPOSITORY\s+TAG|"  # docker images header
        r"^NAMES?\s+IMAGE",  # docker compose header
        content, re.M,
    ))


def _filter_docker_ps(content: str) -> str:
    """Compress docker ps output."""
    lines = [l for l in content.split("\n") if l.strip()]
    # Skip header
    data_lines = lines[1:] if len(lines) > 1 else []

    if not data_lines:
        return "docker ps: no containers"

    containers = []
    for line in data_lines[:15]:
        # Extract key info: ID, IMAGE, STATUS, NAMES
        parts = line.split()
        if len(parts) >= 4:
            cid = parts[0][:12]
            image = parts[1][:30]
            # Find status (Up/Exited/Created)
            status = "unknown"
            for i, p in enumerate(parts):
                if p in ("Up", "Exited", "Created", "Restarting"):
                    status = p
                    if p == "Up" and i + 1 < len(parts):
                        status += " " + parts[i + 1]
                    elif p == "Exited" and i + 1 < len(parts):
                        status += "(" + parts[i + 1].rstrip(")") + ")"
                    break
            name = parts[-1][:30]
            containers.append(f"  {cid} {image} [{status}] {name}")

    result = f"docker ps: {len(data_lines)} container(s)\n" + "\n".join(containers)
    return result


def _filter_docker_images(content: str) -> str:
    """Compress docker images output."""
    lines = [l for l in content.split("\n") if l.strip()]
    data_lines = lines[1:] if len(lines) > 1 else []

    if not data_lines:
        return "docker images: none"

    images = []
    for line in data_lines[:10]:
        parts = line.split()
        if len(parts) >= 3:
            repo = parts[0][:30]
            tag = parts[1][:15]
            size = parts[-1] if len(parts) >= 5 else "?"
            images.append(f"  {repo}:{tag} ({size})")

    result = f"docker images: {len(data_lines)} image(s)\n" + "\n".join(images)
    if len(data_lines) > 10:
        result += f"\n  (+{len(data_lines) - 10} more)"
    return result


def _filter_docker(content: str) -> FilterResult:
    """Route docker output to sub-filter."""
    if re.search(r"^CONTAINER ID", content, re.M):
        compressed = _filter_docker_ps(content)
    elif re.search(r"^REPOSITORY\s+TAG", content, re.M):
        compressed = _filter_docker_images(content)
    else:
        # Docker logs — deduplicate
        lines = content.split("\n")
        unique_lines = list(dict.fromkeys(l.strip() for l in lines if l.strip()))
        if len(unique_lines) < len(lines) // 2:
            compressed = f"docker logs: {len(unique_lines)} unique / {len(lines)} total lines"
        else:
            last = [l for l in lines[-5:] if l.strip()]
            compressed = "\n".join(last[-5:]) if last else "docker: empty output"

    return FilterResult("docker", compressed, _estimate_tokens(content),
                        _estimate_tokens(compressed), False)


def _detect_kubectl(content: str) -> bool:
    return bool(re.search(
        r"^NAME\s+READY\s+STATUS|^NAME\s+TYPE\s+CLUSTER-IP|"
        r"^NAME\s+AGE\s+VERSION|"
        r"^\S+\s+\d+/\d+\s+Running",
        content, re.M,
    ))


def _filter_kubectl(content: str) -> FilterResult:
    """Compress kubectl output."""
    lines = [l for l in content.split("\n") if l.strip()]
    data_lines = lines[1:] if len(lines) > 1 else []

    if not data_lines:
        compressed = "kubectl: no resources"
    else:
        resources = []
        for line in data_lines[:15]:
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0][:30]
                status = " ".join(parts[1:4]) if len(parts) >= 4 else parts[1]
                resources.append(f"  {name} {status}")
        compressed = f"kubectl: {len(data_lines)} resource(s)\n" + "\n".join(resources)
        if len(data_lines) > 15:
            compressed += f"\n  (+{len(data_lines) - 15} more)"

    return FilterResult("kubectl", compressed, _estimate_tokens(content),
                        _estimate_tokens(compressed), False)


# ---------------------------------------------------------------------------
# Directory listing filters
# ---------------------------------------------------------------------------

def _detect_directory_listing(content: str) -> bool:
    return bool(re.search(
        r"^total \d+$|"
        r"^[dlrwx-]{10}\s|"
        r"^d[rwx-]{9}\s|"
        r"├──|└──|│",
        content, re.M,
    ))


def _filter_directory(content: str) -> FilterResult:
    """Compress directory listing output."""
    lines = content.split("\n")

    # tree-style output
    if re.search(r"├──|└──", content):
        dirs = len(re.findall(r"[├└]── \S+/$", content, re.M))
        files = len(re.findall(r"[├└]── \S+\.\w+$", content, re.M))
        compressed = f"tree: {dirs} dirs, {files} files"
        return FilterResult("tree", compressed, _estimate_tokens(content),
                            _estimate_tokens(compressed), False)

    # ls -la style
    entries = []
    for line in lines:
        if re.match(r"^[dlrwx-]{10}", line):
            parts = line.split()
            if len(parts) >= 9:
                perms = parts[0]
                name = parts[-1]
                is_dir = perms.startswith("d")
                entries.append(("dir" if is_dir else "file", name))

    if entries:
        dirs = [e[1] for e in entries if e[0] == "dir"]
        files = [e[1] for e in entries if e[0] == "file"]
        parts = [f"ls: {len(dirs)} dirs, {len(files)} files"]
        if dirs:
            parts.append(f"  dirs: {', '.join(d.rstrip('/') for d in dirs[:10])}")
        compressed = "\n".join(parts)
    else:
        non_empty = [l for l in lines if l.strip()]
        compressed = f"ls: {len(non_empty)} entries"

    return FilterResult("ls", compressed, _estimate_tokens(content),
                        _estimate_tokens(compressed), False)


# ---------------------------------------------------------------------------
# Package manager filters
# ---------------------------------------------------------------------------

def _detect_pip(content: str) -> bool:
    return bool(re.search(
        r"^Package\s+Version$|"
        r"Successfully installed|"
        r"Requirement already satisfied|"
        r"pip \d+\.\d+",
        content, re.M,
    ))


def _filter_pip(content: str) -> FilterResult:
    """Compress pip output."""
    lines = content.split("\n")
    packages = re.findall(r"^(\S+)\s+(\S+)$", content, re.M)
    packages = [(n, v) for n, v in packages if n != "Package" and not n.startswith("-")]

    if packages:
        compressed = f"pip list: {len(packages)} packages"
    elif re.search(r"Successfully installed", content):
        installed = re.search(r"Successfully installed (.+)", content)
        compressed = f"pip: installed {installed.group(1)[:200]}" if installed else "pip: installed"
    elif re.search(r"Requirement already satisfied", content):
        compressed = "pip: already satisfied"
    else:
        compressed = f"pip: {len(lines)} lines"

    return FilterResult("pip", compressed, _estimate_tokens(content),
                        _estimate_tokens(compressed), False)


# ---------------------------------------------------------------------------
# AWS CLI filters
# ---------------------------------------------------------------------------

def _detect_aws(content: str) -> bool:
    return bool(re.search(
        r'"(Reservations|Functions|Buckets|LogGroups|Tables|Roles|Stacks)":\s*\[|'
        r'"Arn":\s*"arn:aws:|'
        r'"Account":\s*"\d{12}"',
        content, re.M,
    ))


def _filter_aws(content: str) -> FilterResult:
    """Compress AWS CLI JSON output."""
    import json as _json

    data = None
    try:
        data = _json.loads(content)
    except (ValueError, _json.JSONDecodeError):
        pass

    if isinstance(data, dict):
        if "Reservations" in data:
            instances = []
            for r in data.get("Reservations", []):
                for i in r.get("Instances", []):
                    iid = i.get("InstanceId", "?")[:15]
                    state = i.get("State", {}).get("Name", "?")
                    itype = i.get("InstanceType", "?")
                    instances.append(f"  {iid} {itype} [{state}]")
            compressed = f"ec2: {len(instances)} instance(s)\n" + "\n".join(instances[:10])
            return FilterResult("aws_ec2", compressed, _estimate_tokens(content),
                                _estimate_tokens(compressed), False)

        if "Functions" in data:
            funcs = data["Functions"]
            flines = [f"  {f.get('FunctionName', '?')[:30]} ({f.get('Runtime', '?')})"
                      for f in funcs[:10]]
            compressed = f"lambda: {len(funcs)} function(s)\n" + "\n".join(flines)
            return FilterResult("aws_lambda", compressed, _estimate_tokens(content),
                                _estimate_tokens(compressed), False)

        if "Buckets" in data:
            buckets = data["Buckets"]
            blines = [f"  {b.get('Name', '?')}" for b in buckets[:10]]
            compressed = f"s3: {len(buckets)} bucket(s)\n" + "\n".join(blines)
            return FilterResult("aws_s3", compressed, _estimate_tokens(content),
                                _estimate_tokens(compressed), False)

        if "Account" in data and "Arn" in data:
            compressed = f"sts: {data.get('Arn', '?')}"
            return FilterResult("aws_sts", compressed, _estimate_tokens(content),
                                _estimate_tokens(compressed), False)

        if "Stacks" in data:
            stacks = data["Stacks"]
            slines = [f"  {s.get('StackName', '?')[:30]} [{s.get('StackStatus', '?')}]"
                      for s in stacks[:10]]
            compressed = f"cloudformation: {len(stacks)} stack(s)\n" + "\n".join(slines)
            return FilterResult("aws_cfn", compressed, _estimate_tokens(content),
                                _estimate_tokens(compressed), False)

        if "TableNames" in data:
            tables = data["TableNames"]
            compressed = f"dynamodb: {len(tables)} table(s): {', '.join(tables[:10])}"
            return FilterResult("aws_dynamodb", compressed, _estimate_tokens(content),
                                _estimate_tokens(compressed), False)

        if "Roles" in data:
            roles = data["Roles"]
            rlines = [f"  {r.get('RoleName', '?')[:30]}" for r in roles[:10]]
            compressed = f"iam: {len(roles)} role(s)\n" + "\n".join(rlines)
            return FilterResult("aws_iam", compressed, _estimate_tokens(content),
                                _estimate_tokens(compressed), False)

    if "LogGroups" in content:
        groups = re.findall(r'"logGroupName":\s*"([^"]+)"', content)
        compressed = f"logs: {len(groups)} group(s): {', '.join(groups[:10])}"
        return FilterResult("aws_logs", compressed, _estimate_tokens(content),
                            _estimate_tokens(compressed), False)

    compressed = f"aws: {len(content.split(chr(10)))} lines"
    return FilterResult("aws", compressed, _estimate_tokens(content),
                        _estimate_tokens(compressed), False)


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

_FILTERS: list[tuple] = [
    (_detect_cargo_test, _filter_cargo_test),  # Before pytest (cargo output contains "N passed")
    (_detect_go_test, _filter_go_test),
    (_detect_pytest, _filter_pytest),
    (_detect_jest, _filter_jest),
    (_detect_rspec, _filter_rspec),
    (_detect_cargo_build, _filter_cargo_build),
    (_detect_ruff, _filter_ruff),
    (_detect_golangci_lint, _filter_golangci_lint),
    (_detect_docker, _filter_docker),
    (_detect_kubectl, _filter_kubectl),
    (_detect_git, _filter_git),
    (_detect_directory_listing, _filter_directory),
    (_detect_pip, _filter_pip),
    (_detect_aws, _filter_aws),
]


def detect_and_filter(content: str) -> Optional[FilterResult]:
    """Detect the command that produced this output and apply a specific filter.

    Returns a FilterResult if a command was recognized, or None if no
    command-specific filter matched (caller should fall through to generic
    compression).
    """
    if not content or not content.strip():
        return None

    # Cap detection scan to first 4KB — command signatures appear in headers
    head = content[:4096]

    for detector, filter_fn in _FILTERS:
        try:
            if detector(head):
                return filter_fn(content)
        except Exception:
            continue

    return None


def list_supported_commands() -> list[str]:
    """Return the list of supported command families."""
    return [
        "git (status, log, diff, add, commit, push, pull)",
        "pytest",
        "cargo test",
        "go test",
        "jest / vitest",
        "rspec",
        "cargo build / cargo clippy",
        "ruff",
        "golangci-lint",
        "docker (ps, images, logs)",
        "kubectl",
        "ls / tree (directory listings)",
        "pip",
        "aws (ec2, lambda, s3, sts, cloudformation, dynamodb, iam, logs)",
    ]