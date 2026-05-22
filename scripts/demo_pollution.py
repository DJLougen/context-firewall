#!/usr/bin/env python3
"""
Demo: Show the raw messy agent trace vs the cleaned context firewall output.

This simulates a realistic 10-turn coding agent session and displays:
  LEFT  = what the LLM actually receives today (the pollution)
  RIGHT = what the LLM receives after the context firewall cleans it
"""

from __future__ import annotations

from context_firewall.firewall import ContextFirewall, Message
from context_firewall.labels import Label
from context_firewall.session import _estimate_tokens


# ---------------------------------------------------------------------------
# Realistic agent session trace (the mess agents actually produce)
# ---------------------------------------------------------------------------

TRACE = [
    # Turn 1: System prompt
    Message(role="system", content="""You are an expert software engineer. You have access to tools: read_file, run_tests, apply_patch, escalate.
Always read the file before editing. Always run tests after editing.
If tests fail, read the error carefully and fix it.
Never delete tests. Never skip verification.
When you are done, call escalate with a summary.
Your current working directory is /home/user/project.
The project uses Python 3.12, pytest, and ruff for linting.
The test suite has 200+ tests across 15 modules.
Remember to keep your changes minimal and focused on the issue."""),

    # Turn 2: User request
    Message(role="user", content="""Fix the bug in src/auth.py where the JWT token validation fails silently
when the token has an expired 'exp' claim. The function should raise
a TokenExpiredError instead of returning True. This is causing security
issues in production because expired tokens are being accepted.

The affected function is validate_token() around line 42.

Related test file: tests/test_auth.py"""),

    # Turn 3: Agent reads the file (full file dump)
    Message(role="tool", content="""# src/auth.py
import jwt
import time
from datetime import datetime, timedelta

SECRET_KEY = "super-secret-key-do-not-share"

class TokenExpiredError(Exception):
    pass

class TokenInvalidError(Exception):
    pass

def create_token(user_id: int, expires_in: int = 3600) -> str:
    \"\"\"Create a JWT token for a user.\"\"\"
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(seconds=expires_in),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def validate_token(token: str) -> dict:
    \"\"\"Validate a JWT token and return the payload.\"\"\"
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        # BUG: This should raise TokenExpiredError but instead returns True
        return True
    except jwt.InvalidTokenError as e:
        raise TokenInvalidError(f"Invalid token: {e}")

def refresh_token(token: str) -> str:
    \"\"\"Refresh an existing token.\"\"\"
    payload = validate_token(token)
    if isinstance(payload, bool):
        raise TokenExpiredError("Cannot refresh expired token")
    return create_token(payload["user_id"])

def revoke_token(token: str) -> None:
    \"\"\"Revoke a token (stub - needs Redis).\"\"\"
    pass

# Middleware for FastAPI
def auth_middleware(request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        raise TokenInvalidError("No token provided")
    payload = validate_token(token)
    if isinstance(payload, bool):
        return None  # BUG: Should reject
    request.user = payload
    return request

# Utility
def get_user_from_token(token: str) -> int:
    payload = validate_token(token)
    if isinstance(payload, bool):
        return -1  # BUG: Should raise
    return payload.get("user_id", -1)

def is_token_valid(token: str) -> bool:
    try:
        payload = validate_token(token)
        return not isinstance(payload, bool)
    except TokenInvalidError:
        return False
"""),

    # Turn 4: Agent runs tests (massive output)
    Message(role="tool", content="""$ cd /home/user/project && python -m pytest tests/test_auth.py -v
============================= test session starts ==============================
platform linux -- Python 3.12.0, pytest-7.4.3, pluggy-1.3.0
rootdir: /home/user/project
plugins: cov-4.1.0, xdist-3.5.0
collected 12 items

tests/test_auth.py::test_create_token PASSED                             [  8%]
tests/test_auth.py::test_create_token_expiry PASSED                      [ 16%]
tests/test_auth.py::test_validate_token_valid PASSED                     [ 25%]
tests/test_auth.py::test_validate_token_expired FAILED                     [ 33%]
tests/test_auth.py::test_validate_token_invalid FAILED                     [ 41%]
tests/test_auth.py::test_refresh_token PASSED                            [ 50%]
tests/test_auth.py::test_refresh_expired_token FAILED                      [ 58%]
tests/test_auth.py::test_revoke_token PASSED                             [ 66%]
tests/test_auth.py::test_auth_middleware_valid PASSED                    [ 75%]
tests/test_auth.py::test_auth_middleware_expired FAILED                    [ 83%]
tests/test_auth.py::test_get_user_from_token PASSED                      [ 91%]
tests/test_auth.py::test_is_token_valid PASSED                           [100%]

=================================== FAILURES ===================================
_______________________ test_validate_token_expired ________________________

    def test_validate_token_expired():
        token = create_token(1, expires_in=-1)
        with pytest.raises(TokenExpiredError):
>           validate_token(token)
E           assert True == raises(TokenExpiredError)

_______________________ test_validate_token_invalid ________________________

    def test_validate_token_invalid():
        with pytest.raises(TokenInvalidError):
>           validate_token("not.a.real.token")
E           TokenInvalidError: Invalid token: Not enough segments

_______________________ test_refresh_expired_token ________________________

    def test_refresh_expired_token():
        token = create_token(1, expires_in=-1)
        with pytest.raises(TokenExpiredError):
>           refresh_token(token)
E           TypeError: 'bool' object is not subscriptable
E           at refresh_token() line 38: return create_token(payload["user_id"])

_______________________ test_auth_middleware_expired ________________________

    def test_auth_middleware_expired():
        token = create_token(1, expires_in=-1)
        request = MockRequest(headers={"Authorization": f"Bearer {token}"})
>       result = auth_middleware(request)
E       assert result is None  # Should have raised TokenExpiredError

=========================== short test summary info ============================
FAILED tests/test_auth.py::test_validate_token_expired
FAILED tests/test_auth.py::test_validate_token_invalid
FAILED tests/test_auth.py::test_refresh_expired_token
FAILED tests/test_auth.py::test_auth_middleware_expired
========================= 4 failed, 8 passed in 1.23s =========================
exit=1"""),

    # Turn 5: Agent reasoning (verbose)
    Message(role="assistant", content="""Looking at the test failures, I can see the root cause. In validate_token(), 
when a jwt.ExpiredSignatureError is caught, the function returns True instead of raising TokenExpiredError. 
This is clearly the bug described in the issue.

The fix is straightforward: change `return True` to `raise TokenExpiredError("Token has expired")`.

But I also notice this has cascading effects:
1. refresh_token() calls validate_token() and expects a dict, but gets True for expired tokens
2. auth_middleware() checks `isinstance(payload, bool)` which is a workaround for the bug
3. get_user_from_token() also has the same isinstance check

After fixing validate_token(), the isinstance checks in refresh_token(), auth_middleware(), 
and get_user_from_token() become unnecessary. However, I should keep them as defensive coding 
in case jwt.decode returns something unexpected.

Actually, looking more carefully: after the fix, jwt.ExpiredSignatureError will raise TokenExpiredError 
from validate_token(), so the callers will get an exception instead of True. The isinstance(payload, bool) 
checks will never be True anymore (assuming no other code path returns bool). 

I think the cleanest fix is:
1. Fix validate_token() to raise TokenExpiredError
2. Remove the isinstance checks in callers since they're dead code after the fix
3. Update auth_middleware to let the exception propagate

Let me apply this fix now."""),

    # Turn 6: Agent applies patch
    Message(role="tool", content="""$ cd /home/user/project && git diff
diff --git a/src/auth.py b/src/auth.py
index 3a4f5b2..7c8d9e1 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -28,8 +28,7 @@ def validate_token(token: str) -> dict:
         payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
         return payload
     except jwt.ExpiredSignatureError:
-        # BUG: This should raise TokenExpiredError but instead returns True
-        return True
+        raise TokenExpiredError("Token has expired")
     except jwt.InvalidTokenError as e:
         raise TokenInvalidError(f"Invalid token: {e}")
 
@@ -36,8 +35,7 @@ def validate_token(token: str) -> dict:
 def refresh_token(token: str) -> str:
     \"\"\"Refresh an existing token.\"\"\"
     payload = validate_token(token)
-    if isinstance(payload, bool):
-        raise TokenExpiredError("Cannot refresh expired token")
     return create_token(payload["user_id"])
 
 def revoke_token(token: str) -> None:
@@ -48,8 +46,7 @@ def revoke_token(token: str) -> None:
 def auth_middleware(request):
     token = request.headers.get("Authorization", "").replace("Bearer ", "")
     if not token:
         raise TokenInvalidError("No token provided")
     payload = validate_token(token)
-    if isinstance(payload, bool):
-        return None  # BUG: Should reject
     request.user = payload
     return request
 
@@ -58,9 +55,7 @@ def auth_middleware(request):
 def get_user_from_token(token: str) -> int:
     payload = validate_token(token)
-    if isinstance(payload, bool):
-        return -1  # BUG: Should raise
     return payload.get("user_id", -1)
 
 def is_token_valid(token: str) -> bool:
     try:
         payload = validate_token(token)
-        return not isinstance(payload, bool)
+        return True
     except TokenInvalidError:
         return False
exit=0"""),

    # Turn 7: Run tests again (another massive dump)
    Message(role="tool", content="""$ cd /home/user/project && python -m pytest tests/test_auth.py -v
============================= test session starts ==============================
platform linux -- Python 3.12.0, pytest-7.4.3, pluggy-1.3.0
rootdir: /home/user/project
plugins: cov-4.1.0, xdist-3.5.0
collected 12 items

tests/test_auth.py::test_create_token PASSED                             [  8%]
tests/test_auth.py::test_create_token_expiry PASSED                      [ 16%]
tests/test_auth.py::test_validate_token_valid PASSED                     [ 25%]
tests/test_auth.py::test_validate_token_expired PASSED                     [ 33%]
tests/test_auth.py::test_validate_token_invalid PASSED                     [ 41%]
tests/test_auth.py::test_refresh_token PASSED                            [ 50%]
tests/test_auth.py::test_refresh_expired_token PASSED                    [ 58%]
tests/test_auth.py::test_revoke_token PASSED                             [ 66%]
tests/test_auth.py::test_auth_middleware_valid PASSED                    [ 75%]
tests/test_auth.py::test_auth_middleware_expired PASSED                  [ 83%]
tests/test_auth.py::test_get_user_from_token PASSED                      [ 91%]
tests/test_auth.py::test_is_token_valid PASSED                           [100%]

============================== 12 passed in 1.15s ==============================
exit=0"""),

    # Turn 8: Agent reads another file for context
    Message(role="tool", content="""# tests/test_auth.py
import pytest
from src.auth import (
    create_token, validate_token, refresh_token, revoke_token,
    auth_middleware, get_user_from_token, is_token_valid,
    TokenExpiredError, TokenInvalidError,
)
from unittest.mock import MagicMock

class MockRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}
        self.user = None

def test_create_token():
    token = create_token(42)
    assert isinstance(token, str)
    assert len(token) > 0

def test_create_token_expiry():
    token = create_token(42, expires_in=3600)
    payload = validate_token(token)
    assert payload["user_id"] == 42

def test_validate_token_valid():
    token = create_token(1)
    payload = validate_token(token)
    assert payload["user_id"] == 1

def test_validate_token_expired():
    token = create_token(1, expires_in=-1)
    with pytest.raises(TokenExpiredError):
        validate_token(token)

def test_validate_token_invalid():
    with pytest.raises(TokenInvalidError):
        validate_token("not.a.real.token")

def test_refresh_token():
    token = create_token(1)
    new_token = refresh_token(token)
    assert isinstance(new_token, str)

def test_refresh_expired_token():
    token = create_token(1, expires_in=-1)
    with pytest.raises(TokenExpiredError):
        refresh_token(token)

def test_revoke_token():
    token = create_token(1)
    revoke_token(token)  # Should not raise

def test_auth_middleware_valid():
    token = create_token(1)
    request = MockRequest(headers={"Authorization": f"Bearer {token}"})
    result = auth_middleware(request)
    assert result.user["user_id"] == 1

def test_auth_middleware_expired():
    token = create_token(1, expires_in=-1)
    request = MockRequest(headers={"Authorization": f"Bearer {token}"})
    with pytest.raises(TokenExpiredError):
        auth_middleware(request)

def test_get_user_from_token():
    token = create_token(42)
    assert get_user_from_token(token) == 42

def test_is_token_valid():
    token = create_token(1)
    assert is_token_valid(token) is True
    assert is_token_valid("bad") is False

# Integration tests
def test_full_flow():
    token = create_token(99)
    assert is_token_valid(token)
    payload = validate_token(token)
    assert payload["user_id"] == 99
    new_token = refresh_token(token)
    assert is_token_valid(new_token)
"""),

    # Turn 9: Run full test suite
    Message(role="tool", content="""$ cd /home/user/project && python -m pytest tests/ -v --tb=short
============================= test session starts ==============================
platform linux -- Python 3.12.0, pytest-7.4.3, pluggy-1.3.0
rootdir: /home/user/project
plugins: cov-4.1.0, xdist-3.5.0
collected 213 items

tests/test_api.py::test_health PASSED                                      [  0%]
tests/test_api.py::test_version PASSED                                     [  0%]
tests/test_api.py::test_root PASSED                                        [  1%]
tests/test_auth.py::test_create_token PASSED                             [  2%]
tests/test_auth.py::test_create_token_expiry PASSED                      [  2%]
tests/test_auth.py::test_validate_token_valid PASSED                     [  3%]
tests/test_auth.py::test_validate_token_expired PASSED                     [  3%]
tests/test_auth.py::test_validate_token_invalid PASSED                     [  4%]
tests/test_auth.py::test_refresh_token PASSED                            [  4%]
tests/test_auth.py::test_refresh_expired_token PASSED                    [  5%]
tests/test_auth.py::test_revoke_token PASSED                             [  5%]
tests/test_auth.py::test_auth_middleware_valid PASSED                    [  6%]
tests/test_auth.py::test_auth_middleware_expired PASSED                  [  6%]
tests/test_auth.py::test_get_user_from_token PASSED                      [  7%]
tests/test_auth.py::test_is_token_valid PASSED                           [  7%]
tests/test_auth.py::test_full_flow PASSED                                [  8%]
tests/test_cache.py::test_get_set PASSED                                   [  8%]
tests/test_cache.py::test_expiry PASSED                                   [  9%]
tests/test_cache.py::test_eviction PASSED                                 [  9%]
tests/test_cache.py::test_concurrent PASSED                              [ 10%]
... (183 more tests) ...
tests/test_workers.py::test_spawn PASSED                                   [ 98%]
tests/test_workers.py::test_join PASSED                                    [ 99%]
tests/test_workers.py::test_timeout PASSED                                 [ 99%]
tests/test_workers.py::test_crash PASSED                                   [100%]

============================== 213 passed in 4.72s ==============================
exit=0"""),

    # Turn 10: Agent summary
    Message(role="assistant", content="""## Summary

Fixed the JWT token validation bug in src/auth.py. The validate_token() function 
was returning True for expired tokens instead of raising TokenExpiredError.

### Changes:
1. **src/auth.py line 30**: Changed `return True` to `raise TokenExpiredError("Token has expired")`
2. **src/auth.py line 38**: Removed dead `isinstance(payload, bool)` check in refresh_token()
3. **src/auth.py line 48**: Removed dead isinstance check in auth_middleware()
4. **src/auth.py line 57**: Removed dead isinstance check in get_user_from_token()
5. **src/auth.py line 63**: Simplified is_token_valid() to return True on success

### Verification:
- All 12 auth tests pass (previously 4 failed)
- Full suite: 213 passed in 4.72s, no regressions

### Root cause:
The original code had a catch-all that returned True for ExpiredSignatureError, 
treating expired tokens as valid. This propagated through 4 functions that all 
had isinstance(payload, bool) workarounds instead of proper error handling."""),
]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

def hr(char="=", width=100):
    print(f"{DIM}{char * width}{RESET}")

def role_color(role: str) -> str:
    return {
        "system": CYAN,
        "user": GREEN,
        "assistant": YELLOW,
        "tool": RED,
    }.get(role, "")

def truncate(text: str, max_lines: int = 12) -> str:
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    shown = lines[:max_lines]
    omitted = len(lines) - max_lines
    shown.append(f"  ... ({omitted} more lines omitted) ...")
    return "\n".join(shown)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    print()
    print(f"{BOLD}{'=' * 100}")
    print(f"  CONTEXT FIREWALL DEMO: Raw Agent Trace vs Cleaned Context")
    print(f"{'=' * 100}{RESET}")
    print()
    print(f"  This demo shows a 10-turn coding agent session.")
    print(f"  {RED}LEFT{RESET}  = what the LLM receives today (every byte of tool output)")
    print(f"  {GREEN}RIGHT{RESET} = what the LLM receives through the context firewall")
    print()
    hr()

    fw = ContextFirewall()

    raw_tokens_total = 0
    clean_tokens_total = 0
    entries = []

    for msg in TRACE:
        compressed = fw.process(msg)
        raw_tokens = compressed.original_tokens
        clean_tokens = compressed.compressed_tokens
        raw_tokens_total += raw_tokens
        clean_tokens_total += clean_tokens
        entries.append((msg, compressed))

    # Display each turn
    for i, (msg, comp) in enumerate(entries):
        turn = i + 1
        rc = role_color(msg.role)
        raw_lines = msg.content.split("\n")
        clean_lines = comp.content.split("\n")

        raw_tok = comp.original_tokens
        clean_tok = comp.compressed_tokens
        ratio = raw_tok / max(clean_tok, 1)
        savings = raw_tok - clean_tok

        print()
        print(f"{BOLD}  TURN {turn}  |  {rc}{msg.role.upper()}{RESET}  |  "
              f"content_type={comp.content_type.value}  label={comp.label.value}")
        print(f"  {DIM}tokens: {raw_tok:,} raw  ->  {clean_tok:,} clean  "
              f"({ratio:.1f}x compression, {savings:,} tokens saved){RESET}")
        print()

        # Raw side
        print(f"  {RED}{BOLD}--- RAW (what LLM sees today) ---{RESET}")
        raw_shown = truncate(msg.content, max_lines=15)
        for line in raw_shown.split("\n"):
            print(f"  {RED}{DIM}|{RESET} {line}")
        if len(raw_lines) > 15:
            print(f"  {RED}{DIM}|  ... ({len(raw_lines) - 15} more lines){RESET}")

        print()

        # Clean side
        print(f"  {GREEN}{BOLD}--- CLEAN (after context firewall) ---{RESET}")
        for line in comp.content.split("\n"):
            print(f"  {GREEN}|{RESET} {line}")

        hr("-")

    # Final summary
    overall_ratio = raw_tokens_total / max(clean_tokens_total, 1)
    savings_total = raw_tokens_total - clean_tokens_total

    print()
    print(f"{BOLD}{'=' * 100}")
    print(f"  SESSION TOTALS")
    print(f"{'=' * 100}{RESET}")
    print()
    print(f"  {RED}Raw context (today):      {raw_tokens_total:>8,} tokens{RESET}")
    print(f"  {GREEN}Clean context (firewall): {clean_tokens_total:>8,} tokens{RESET}")
    print(f"  {BOLD}Compression ratio:        {overall_ratio:>7.1f}x{RESET}")
    print(f"  {BOLD}Tokens saved:             {savings_total:>8,} tokens{RESET}")
    print()
    print(f"  {DIM}At ~$0.01/1K tokens (GPT-4 class), this session costs:{RESET}")
    print(f"    Raw:   ${raw_tokens_total * 0.01 / 1000:.2f}")
    print(f"    Clean: ${clean_tokens_total * 0.01 / 1000:.2f}")
    print(f"    Saved: ${savings_total * 0.01 / 1000:.2f} ({savings_total/raw_tokens_total*100:.0f}%)")
    print()

    # Show what the LLM's context window looks like
    print(f"{BOLD}{'=' * 100}")
    print(f"  THE ACTUAL CONTEXT WINDOW (what the LLM sees with the firewall)")
    print(f"{'=' * 100}{RESET}")
    print()

    for msg, comp in entries:
        rc = role_color(msg.role)
        print(f"  {rc}[{msg.role}]{RESET} {comp.content}")
        print()

    hr()
    print(f"  {DIM}Total: {clean_tokens_total:,} tokens. "
          f"The LLM sees only this -- no noise, no bloat, no pollution.{RESET}")
    print()


if __name__ == "__main__":
    main()
