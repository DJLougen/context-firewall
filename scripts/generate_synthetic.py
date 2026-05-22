"""Generate synthetic training data for the context firewall classifier.

Creates realistic agent session patterns with known labels.
"""

from __future__ import annotations

import random
from pathlib import Path

from context_firewall.io import make_row, write_jsonl
from context_firewall.labels import ContentType, Label


# ---------------------------------------------------------------------------
# Content templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = [
    "You are a helpful coding assistant. Help the user debug and fix code.",
    "You are an expert software engineer. Analyze code and suggest improvements.",
    "You are a Python specialist. Focus on Python code and best practices.",
    "You are a debugging assistant. Help identify and fix bugs in code.",
    "You are a code reviewer. Review code for correctness and style.",
]

USER_GOALS = [
    "Fix the bug in src/foo.py where bar() returns None.",
    "Add error handling to the database connection code.",
    "Refactor the authentication module to use JWT tokens.",
    "Write unit tests for the payment processing function.",
    "Optimize the database query that's causing slow performance.",
    "Fix the race condition in the async task queue.",
    "Add input validation to the API endpoint.",
    "Update the code to use the new API version.",
    "Fix the memory leak in the image processing pipeline.",
    "Add logging to the critical path of the order processing.",
]

TOOL_CALLS = [
    'tool_name: read_file\nargs: {path: "src/foo.py"}',
    'tool_name: run_tests\nargs: {command: "pytest tests/"}',
    'tool_name: apply_patch\nargs: {path: "src/bar.py", diff: "..."}',
    'tool_name: search\nargs: {query: "def authenticate"}',
    'tool_name: run_command\nargs: {command: "npm install"}',
]

TEST_OUTPUTS = [
    "pytest -v\ntest_foo.py::test_bar PASSED\ntest_foo.py::test_baz PASSED\n94 passed in 3.5s",
    "pytest -v\ntest_auth.py::test_login PASSED\ntest_auth.py::test_logout FAILED\nAssertionError: expected 200 got 401\n15 passed, 1 failed in 2.1s",
    "npm test\n\n  ✓ should add two numbers\n  ✓ should subtract\n  ✗ should multiply\n    Expected: 6\n    Received: 5\n\n2 passing, 1 failing",
    "cargo test\nrunning 42 tests\ntest math::add ... ok\ntest math::sub ... ok\ntest math::mul ... FAILED\n\ntest result: FAILED. 41 passed; 1 failed",
    "go test ./...\nok  	github.com/user/project/pkg	0.123s\nFAIL	github.com/user/project/api	0.456s",
]

FILE_CONTENTS_SMALL = [
    "# src/utils.py\ndef add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b",
    "class Config:\n    def __init__(self):\n        self.debug = False\n        self.port = 8080",
    "export function greet(name: string): string {\n  return `Hello, ${name}!`;\n}",
    "def process_data(items):\n    result = []\n    for item in items:\n        result.append(item * 2)\n    return result",
]

FILE_CONTENTS_LARGE = [
    "# src/database.py\n" + "class Database:\n" + "    def __init__(self):\n" + "        self.conn = None\n" + 
    "    def connect(self):\n" + "        pass\n" + "    def query(self, sql):\n" + "        pass\n" * 50,
    "class AuthService:\n" + "    def authenticate(self, token):\n" + "        pass\n" + 
    "    def authorize(self, user, resource):\n" + "        pass\n" * 60,
]

COMMAND_OUTPUTS = [
    "$ npm install\nadded 123 packages in 4.5s\nexit=0",
    "$ git status\nOn branch main\nnothing to commit, working tree clean\nexit=0",
    "$ python -m pytest\n94 passed in 3.5s\nexit=0",
    "$ cargo build\n   Compiling myapp v0.1.0\n    Finished dev [unoptimized + debuginfo] target(s) in 2.34s\nexit=0",
    "$ docker build -t myapp .\nSuccessfully built abc123\nexit=0",
]

ERROR_TRACES = [
    'Traceback (most recent call last):\n  File "src/foo.py", line 42, in bar\n    result = baz()\n  File "src/foo.py", line 50, in baz\n    return int("not a number")\nValueError: invalid literal for int() with base 10',
    'Error: Cannot read property "foo" of undefined\n    at processItem (src/utils.js:23:15)\n    at Array.map (<anonymous>)\n    at processData (src/utils.js:20:10)',
    'TypeError: expected str, got int\n    at parse_config (config.py:15)\n    at main (main.py:8)',
]

REASONING_SHORT = [
    "The issue is that bar() returns None. I should change it to return 42.",
    "The test is failing because the input validation is missing. I'll add a check.",
    "The database connection is timing out. I need to add retry logic.",
    "The API endpoint is not handling errors properly. I'll add error handling.",
]

REASONING_LONG = [
    "Let me analyze this step by step. First, I need to understand the bug. The user reports that bar() returns None when it should return 42. Looking at the code, I can see that bar() is defined as:\n\ndef bar():\n    return None\n\nThis is clearly wrong. The function should return 42. Let me think about why this might be happening. Perhaps there was a placeholder left in during development. The fix is straightforward: change the return value to 42. I should also add a test to ensure this doesn't regress.",
    "I need to refactor the authentication module. Currently it uses session cookies, but the requirement is to switch to JWT tokens. This involves several steps: 1) Generate JWT tokens on login, 2) Validate tokens on each request, 3) Handle token expiration and refresh. Let me think about the implementation. I'll need to add a JWT library, create a token generation function, and update the authentication middleware. The tricky part will be handling token refresh without disrupting the user session.",
]

PATCHES = [
    "diff --git a/src/foo.py b/src/foo.py\n--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1,3 +1,3 @@\n def bar():\n-    return None\n+    return 42",
    "diff --git a/src/auth.py b/src/auth.py\n--- a/src/auth.py\n+++ b/src/auth.py\n@@ -10,6 +10,8 @@\n def login(user, password):\n     if verify(user, password):\n+        token = generate_jwt(user)\n+        return {\"token\": token}\n         return {\"status\": \"ok\"}\n     return {\"status\": \"error\"}",
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_session(session_id: str, num_turns: int = 10) -> list[dict]:
    """Generate a single synthetic session."""
    rows = []
    turn = 0
    
    # System prompt (always first)
    rows.append(make_row(
        role="system",
        content=random.choice(SYSTEM_PROMPTS),
        content_type=ContentType.SYSTEM,
        label=Label.CORE,
        turn=turn,
        session_id=session_id,
    ))
    turn += 1
    
    # User goal
    rows.append(make_row(
        role="user",
        content=random.choice(USER_GOALS),
        content_type=ContentType.USER_GOAL,
        label=Label.CORE,
        turn=turn,
        session_id=session_id,
    ))
    turn += 1
    
    # Generate random turns
    for _ in range(num_turns):
        turn_type = random.choice([
            "tool_call", "test", "file_small", "file_large", 
            "command", "error", "reasoning_short", "reasoning_long", "patch"
        ])
        
        if turn_type == "tool_call":
            rows.append(make_row(
                role="assistant",
                content=random.choice(TOOL_CALLS),
                content_type=ContentType.TOOL_CALL,
                label=Label.DROP,
                turn=turn,
                session_id=session_id,
            ))
        elif turn_type == "test":
            rows.append(make_row(
                role="tool",
                content=random.choice(TEST_OUTPUTS),
                content_type=ContentType.TOOL_RESULT_TEST,
                label=Label.DISTILL,
                turn=turn,
                session_id=session_id,
            ))
        elif turn_type == "file_small":
            rows.append(make_row(
                role="tool",
                content=random.choice(FILE_CONTENTS_SMALL),
                content_type=ContentType.TOOL_RESULT_FILE,
                label=Label.DISTILL,
                turn=turn,
                session_id=session_id,
            ))
        elif turn_type == "file_large":
            rows.append(make_row(
                role="tool",
                content=random.choice(FILE_CONTENTS_LARGE),
                content_type=ContentType.TOOL_RESULT_FILE,
                label=Label.COMPACT,
                turn=turn,
                session_id=session_id,
            ))
        elif turn_type == "command":
            rows.append(make_row(
                role="tool",
                content=random.choice(COMMAND_OUTPUTS),
                content_type=ContentType.TOOL_RESULT_COMMAND,
                label=Label.DISTILL,
                turn=turn,
                session_id=session_id,
            ))
        elif turn_type == "error":
            # Recent errors are CORE, older are DISTILL
            label = Label.CORE if turn < 3 else Label.DISTILL
            rows.append(make_row(
                role="tool",
                content=random.choice(ERROR_TRACES),
                content_type=ContentType.TOOL_RESULT_ERROR,
                label=label,
                turn=turn,
                session_id=session_id,
            ))
        elif turn_type == "reasoning_short":
            rows.append(make_row(
                role="assistant",
                content=random.choice(REASONING_SHORT),
                content_type=ContentType.AGENT_REASONING,
                label=Label.DISTILL,
                turn=turn,
                session_id=session_id,
            ))
        elif turn_type == "reasoning_long":
            rows.append(make_row(
                role="assistant",
                content=random.choice(REASONING_LONG),
                content_type=ContentType.AGENT_REASONING,
                label=Label.COMPACT,
                turn=turn,
                session_id=session_id,
            ))
        elif turn_type == "patch":
            rows.append(make_row(
                role="assistant",
                content=random.choice(PATCHES),
                content_type=ContentType.AGENT_PATCH,
                label=Label.DISTILL,
                turn=turn,
                session_id=session_id,
            ))
        
        turn += 1
    
    return rows


def generate_dataset(num_sessions: int = 100, output_path: str = "examples/train.jsonl") -> None:
    """Generate a full training dataset."""
    all_rows = []
    
    for i in range(num_sessions):
        session_id = f"session_{i:04d}"
        num_turns = random.randint(8, 15)
        rows = generate_session(session_id, num_turns)
        all_rows.extend(rows)
    
    # Shuffle
    random.shuffle(all_rows)
    
    # Write
    write_jsonl(output_path, all_rows)
    print(f"Generated {len(all_rows)} rows to {output_path}")


if __name__ == "__main__":
    generate_dataset(100, "examples/train.jsonl")
    generate_dataset(20, "examples/eval.jsonl")
