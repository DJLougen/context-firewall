"""Tests for command-specific output filters (detect_and_filter)."""

from honeycomb.command_filters import detect_and_filter, list_supported_commands, FilterResult


# ---------------------------------------------------------------------------
# Git filters
# ---------------------------------------------------------------------------

def test_git_status_clean():
    """Clean working tree should be detected as git_status with is_failure=False."""
    content = (
        "On branch main\n"
        "Your branch is up to date with 'origin/main'.\n\n"
        "nothing to commit, working tree clean\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "git_status"
    assert "clean" in result.compressed
    assert result.is_failure is False


def test_git_status_dirty():
    """Dirty working tree should list modified/untracked files."""
    content = (
        "On branch feature/auth\n"
        "Your branch is ahead of 'origin/feature/auth' by 3 commits.\n\n"
        "Changes to be committed:\n"
        "  (use \"git restore --staged <file>...\" to unstage)\n"
        "\tmodified:   src/auth.py\n"
        "\tnew file:   src/auth_utils.py\n\n"
        "Changes not staged for commit:\n"
        "  (use \"git add <file>...\" to update what will be committed)\n"
        "\tmodified:   src/main.py\n\n"
        "Untracked files:\n"
        "  (use \"git add <file>...\" to include in what will be committed)\n"
        "\ttemp.log\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "git_status"
    assert "feature/auth" in result.compressed
    assert result.raw_tokens > result.compressed_tokens


def test_git_log():
    """Git log output should be compressed to one-line-per-commit."""
    content = (
        "commit abc1234567890abcdef1234567890abcdef123456\n"
        "Author: Alice <alice@example.com>\n"
        "Date:   Mon Jan 15 10:30:00 2024 -0500\n\n"
        "    Add authentication module\n\n"
        "commit def4567890abcdef1234567890abcdef12345678\n"
        "Author: Bob <bob@example.com>\n"
        "Date:   Sun Jan 14 18:00:00 2024 -0500\n\n"
        "    Fix login redirect bug\n\n"
        "commit 111222333444555666777888999000aaabbbcccddd\n"
        "Author: Charlie <charlie@example.com>\n"
        "Date:   Sat Jan 13 09:00:00 2024 -0500\n\n"
        "    Update README\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "git_log"
    assert "3 commits" in result.compressed
    assert result.raw_tokens > result.compressed_tokens


def test_git_diff():
    """Git diff output should summarize files and +/- counts."""
    content = (
        "diff --git a/src/auth.py b/src/auth.py\n"
        "index 1234567..abcdefg 100644\n"
        "--- a/src/auth.py\n"
        "+++ b/src/auth.py\n"
        "@@ -10,6 +10,8 @@ def login():\n"
        "     username = input()\n"
        "+    password = getpass()\n"
        "+    token = generate_token(username, password)\n"
        "     return session\n"
        "-    return None\n"
        "diff --git a/src/utils.py b/src/utils.py\n"
        "index 2345678..bcdefgh 100644\n"
        "--- a/src/utils.py\n"
        "+++ b/src/utils.py\n"
        "@@ -1,3 +1,5 @@\n"
        "+import hashlib\n"
        " def hash_pw(pw):\n"
        "-    return pw\n"
        "+    return hashlib.sha256(pw.encode()).hexdigest()\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "git_diff"
    assert "2 file(s)" in result.compressed
    assert result.is_failure is False


def test_git_push():
    """Git push output should be detected and compressed."""
    content = (
        "Enumerating objects: 15, done.\n"
        "Counting objects: 100% (15/15), done.\n"
        "Delta compression using up to 12 threads\n"
        "Compressing objects: 100% (8/8), done.\n"
        "Writing objects: 100% (8/8), 2.5 KiB | 2.50 MiB/s, done.\n"
        "Total 8 (delta 5), reused 0 (delta 0), pack-reused 0\n"
        "remote: Resolving deltas: 100% (5/5), completed with 4 local objects.\n"
        "To github.com:acme/repo.git\n"
        "   abc1234..def5678  main -> main\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "git_push_pull"
    assert result.is_failure is False


def test_git_pull():
    """Git pull with fast-forward should be detected."""
    content = (
        "remote: Enumerating objects: 22, done.\n"
        "remote: Counting objects: 100% (18/18), done.\n"
        "remote: Compressing objects: 100% (5/5), done.\n"
        "Unpacking objects: 100% (12/12), 3.2 KiB | 1.60 MiB/s, done.\n"
        "From github.com:acme/repo\n"
        "   abc1234..def5678  main       -> origin/main\n"
        "Updating abc1234..def5678\n"
        "Fast-forward\n"
        " src/auth.py  | 15 +++++++++------\n"
        " src/utils.py |  3 ++-\n"
        " 2 files changed, 11 insertions(+), 7 deletions(-)\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "git_push_pull"
    assert "ok" in result.compressed.lower() or "fast-forward" in result.compressed.lower()


def test_git_add_commit():
    """Git commit output should extract sha and message."""
    content = (
        "[main abc1234] Add user authentication\n"
        " 5 files changed, 120 insertions(+), 3 deletions(-)\n"
        " create mode 100644 src/auth.py\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "git_add_commit"
    assert "abc1234" in result.compressed
    assert result.is_failure is False


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------

def test_pytest_pass():
    """Pytest all-passing output should be detected with is_failure=False."""
    content = (
        "============================= test session starts ==============================\n"
        "platform linux -- Python 3.12.1, pytest-8.0.0, pluggy-1.4.0\n"
        "rootdir: /home/user/project\n"
        "collected 42 items\n\n"
        "tests/test_auth.py .............                                        [ 30%]\n"
        "tests/test_api.py .................                                     [ 71%]\n"
        "tests/test_utils.py ............                                        [100%]\n\n"
        "============================== 42 passed in 3.21s ==============================\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "pytest"
    assert "42 passed" in result.compressed
    assert result.is_failure is False


def test_pytest_fail():
    """Pytest with failures should extract failed test names and is_failure=True."""
    content = (
        "============================= test session starts ==============================\n"
        "platform linux -- Python 3.12.1, pytest-8.0.0, pluggy-1.4.0\n"
        "collected 42 items\n\n"
        "tests/test_auth.py .............F                                       [ 33%]\n"
        "tests/test_api.py ......F........                                        [ 66%]\n"
        "tests/test_utils.py ............                                         [100%]\n\n"
        "=================================== FAILURES ===================================\n"
        "___________________________ test_login_redirect ______________________________\n\n"
        "E       AssertionError: expected 302 got 500\n"
        ">       assert response.status_code == 302\n\n"
        "FAILED tests/test_auth.py::test_login_redirect - AssertionError: expected 302 got 500\n"
        "FAILED tests/test_api.py::test_create_user - ValueError: invalid email\n"
        "========================= 2 failed, 40 passed in 4.12s =========================\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "pytest"
    assert result.is_failure is True
    assert "test_login_redirect" in result.compressed
    assert "test_create_user" in result.compressed


def test_cargo_test_pass():
    """Cargo test all-passing should be detected with is_failure=False."""
    content = (
        "   Compiling myapp v0.1.0 (/home/user/myapp)\n"
        "    Finished test [unoptimized + debuginfo] target(s) in 4.23s\n"
        "     Running unittests src/lib.rs (target/debug/deps/myapp-abc123)\n\n"
        "running 12 tests\n"
        "test auth::tests::test_login ... ok\n"
        "test auth::tests::test_logout ... ok\n"
        "test auth::tests::test_token_refresh ... ok\n"
        "test db::tests::test_connect ... ok\n"
        "test db::tests::test_query ... ok\n"
        "test db::tests::test_migrate ... ok\n"
        "test api::tests::test_get_users ... ok\n"
        "test api::tests::test_create_user ... ok\n"
        "test api::tests::test_delete_user ... ok\n"
        "test api::tests::test_update_user ... ok\n"
        "test utils::tests::test_hash ... ok\n"
        "test utils::tests::test_validate ... ok\n\n"
        "test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "cargo_test"
    assert result.is_failure is False
    assert "12 passed" in result.compressed


def test_cargo_test_fail():
    """Cargo test with failures should extract failed test names."""
    content = (
        "   Compiling myapp v0.1.0 (/home/user/myapp)\n"
        "    Finished test [unoptimized + debuginfo] target(s) in 3.11s\n"
        "     Running unittests src/lib.rs (target/debug/deps/myapp-def456)\n\n"
        "running 8 tests\n"
        "test auth::tests::test_login ... ok\n"
        "test auth::tests::test_logout ... ok\n"
        "test auth::tests::test_token_refresh ... FAILED\n"
        "test db::tests::test_connect ... ok\n"
        "test db::tests::test_query ... FAILED\n"
        "test api::tests::test_get_users ... ok\n"
        "test api::tests::test_create_user ... ok\n"
        "test api::tests::test_delete_user ... ok\n\n"
        "failures:\n"
        "thread 'auth::tests::test_token_refresh' panicked at 'assertion failed: token.is_valid()', src/auth.rs:42\n"
        "thread 'db::tests::test_query' panicked at 'called `Result::unwrap()` on an `Err` value: ConnectionRefused', src/db.rs:88\n\n"
        "test result: FAILED. 6 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "cargo_test"
    assert result.is_failure is True
    assert "test_token_refresh" in result.compressed
    assert "test_query" in result.compressed


def test_go_test():
    """Go test output should be detected and summarized by package."""
    content = (
        "=== RUN   TestLogin\n"
        "--- PASS: TestLogin (0.01s)\n"
        "=== RUN   TestLogout\n"
        "--- PASS: TestLogout (0.00s)\n"
        "=== RUN   TestCreateUser\n"
        "--- FAIL: TestCreateUser (0.03s)\n"
        "    user_test.go:42: Error: expected 201 got 500\n"
        "ok      github.com/acme/app/auth   0.045s\n"
        "FAIL    github.com/acme/app/api    0.032s\n"
        "ok      github.com/acme/app/db     0.012s\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "go_test"
    assert result.is_failure is True
    assert "TestCreateUser" in result.compressed


def test_jest():
    """Jest test output should be detected and compressed."""
    content = (
        "PASS src/components/__tests__/Button.test.tsx\n"
        "  Button\n"
        "    ✓ renders correctly (23 ms)\n"
        "    ✓ handles click events (15 ms)\n\n"
        "FAIL src/components/__tests__/Form.test.tsx\n"
        "  Form\n"
        "    ✓ renders all fields (18 ms)\n"
        "    ✕ submits data correctly (45 ms)\n\n"
        "  ● Form › submits data correctly\n"
        "    Expected: {\"name\": \"Alice\"}\n"
        "    Received: {\"name\": \"\"}\n\n"
        "Test Suites: 1 failed, 1 passed, 2 total\n"
        "Tests:       1 failed, 3 passed, 4 total\n"
        "Snapshots:   0 total\n"
        "Time:        2.345 s\n"
        "Ran all test suites.\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "jest"
    assert result.is_failure is True
    assert "1 failed" in result.compressed


# ---------------------------------------------------------------------------
# Build / lint
# ---------------------------------------------------------------------------

def test_cargo_build_success():
    """Successful cargo build should be detected with is_failure=False."""
    content = (
        "   Compiling serde v1.0.195\n"
        "   Compiling tokio v1.35.1\n"
        "   Compiling reqwest v0.11.23\n"
        "   Compiling myapp v0.1.0 (/home/user/myapp)\n"
        "    Finished dev [unoptimized + debuginfo] target(s) in 12.45s\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "cargo_build"
    assert result.is_failure is False
    assert "dev" in result.compressed


def test_cargo_build_error():
    """Cargo build with errors should set is_failure=True and list errors."""
    content = (
        "   Compiling myapp v0.1.0 (/home/user/myapp)\n"
        "error[E0308]: mismatched types\n"
        "  --> src/main.rs:15:9\n"
        "   |\n"
        "15 |         return 42;\n"
        "   |                ^^ expected String, found integer\n"
        "\n"
        "error[E0433]: failed to resolve: use of undeclared crate or module `foo`\n"
        "  --> src/lib.rs:3:5\n"
        "   |\n"
        "3  | use foo::bar;\n"
        "   |     ^^^ use of undeclared crate or module `foo`\n"
        "\n"
        "error: could not compile `myapp` (bin \"myapp\") due to 2 previous errors\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "cargo_build"
    assert result.is_failure is True
    assert "2" in result.compressed  # error count


def test_ruff_pass():
    """Ruff with all checks passed should be detected with is_failure=False."""
    content = "All checks passed!\n"
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "ruff"
    assert result.is_failure is False
    assert "passed" in result.compressed.lower()


def test_ruff_violations():
    """Ruff with violations should count them and set is_failure=True."""
    content = (
        "src/main.py:10:5: F401 `os` imported but unused\n"
        "src/main.py:25:80: E501 Line too long (120 > 79)\n"
        "src/utils.py:3:1: F401 `sys` imported but unused\n"
        "src/utils.py:15:5: E722 Do not use bare `except`\n"
        "Found 4 errors.\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "ruff"
    assert result.is_failure is True
    assert "4" in result.compressed


def test_golangci_lint():
    """golangci-lint output should be detected and violations summarized."""
    content = (
        "cmd/server.go:42:5: exported function Init should have comment (revive)\n"
        "internal/auth.go:15:2: var-naming: don't use ALL_CAPS in Go names (stylecheck)\n"
        "internal/db.go:88:10: Error return value of `rows.Close` is not checked (errcheck)\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "golangci_lint"
    assert result.is_failure is True
    assert "3" in result.compressed


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

def test_docker_ps():
    """Docker ps output should be detected and containers summarized."""
    content = (
        "CONTAINER ID   IMAGE          COMMAND                  CREATED          STATUS          PORTS                    NAMES\n"
        "a1b2c3d4e5f6   nginx:latest   \"/docker-entrypoint.…\"   2 hours ago      Up 2 hours      0.0.0.0:80->80/tcp       web-server\n"
        "b2c3d4e5f6a7   postgres:16    \"docker-entrypoint.s…\"   2 hours ago      Up 2 hours      0.0.0.0:5432->5432/tcp   db\n"
        "c3d4e5f6a7b8   redis:7        \"docker-entrypoint.s…\"   2 hours ago      Up 2 hours      0.0.0.0:6379->6379/tcp   cache\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "docker"
    assert "3 container" in result.compressed
    assert result.is_failure is False


def test_docker_images():
    """Docker images output should be detected and images summarized."""
    content = (
        "REPOSITORY    TAG        IMAGE ID       CREATED        SIZE\n"
        "nginx         latest     a1b2c3d4e5f6   2 weeks ago    187MB\n"
        "postgres      16         b2c3d4e5f6a7   3 weeks ago    432MB\n"
        "redis         7          c3d4e5f6a7b8   1 month ago    117MB\n"
        "python        3.12-slim  d4e5f6a7b8c9   1 month ago    125MB\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "docker"
    assert "4 image" in result.compressed


# ---------------------------------------------------------------------------
# Directory listings
# ---------------------------------------------------------------------------

def test_ls_la():
    """ls -la style output should be detected and summarized."""
    content = (
        "total 48\n"
        "drwxr-xr-x  6 user user 4096 Jan 15 10:30 .\n"
        "drwxr-xr-x 12 user user 4096 Jan 15 09:00 ..\n"
        "-rw-r--r--  1 user user  220 Jan 14 12:00 .gitignore\n"
        "-rw-r--r--  1 user user 1063 Jan 14 12:00 LICENSE\n"
        "-rw-r--r--  1 user user 2456 Jan 15 10:30 README.md\n"
        "drwxr-xr-x  3 user user 4096 Jan 14 12:00 src\n"
        "drwxr-xr-x  4 user user 4096 Jan 15 10:00 tests\n"
        "-rwxr-xr-x  1 user user  891 Jan 15 10:30 setup.py\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "ls"
    assert "dirs" in result.compressed
    assert "files" in result.compressed
    assert result.is_failure is False


def test_tree():
    """Tree-style output should be detected and count dirs/files."""
    content = (
        "src/\n"
        "├── __init__.py\n"
        "├── main.py\n"
        "├── auth/\n"
        "│   ├── __init__.py\n"
        "│   ├── login.py\n"
        "│   └── logout.py\n"
        "├── db/\n"
        "│   ├── __init__.py\n"
        "│   └── models.py\n"
        "└── utils/\n"
        "    ├── __init__.py\n"
        "    └── helpers.py\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "tree"
    assert "dirs" in result.compressed
    assert "files" in result.compressed


# ---------------------------------------------------------------------------
# Package managers
# ---------------------------------------------------------------------------

def test_pip_list():
    """pip list output should be detected and package count extracted."""
    content = (
        "Package         Version\n"
        "--------------- -------\n"
        "certifi         2024.2.2\n"
        "charset-normalizer 3.3.2\n"
        "idna            3.6\n"
        "pip             24.0\n"
        "requests        2.31.0\n"
        "setuptools      69.0.3\n"
        "urllib3         2.1.0\n"
        "wheel           0.42.0\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "pip"
    assert "8 packages" in result.compressed
    assert result.is_failure is False


# ---------------------------------------------------------------------------
# AWS
# ---------------------------------------------------------------------------

def test_aws_ec2_describe_instances():
    """EC2 describe-instances JSON should be detected and instances listed."""
    content = (
        '{\n'
        '    "Reservations": [\n'
        '        {\n'
        '            "Instances": [\n'
        '                {\n'
        '                    "InstanceId": "i-0abc123def456789",\n'
        '                    "InstanceType": "t3.micro",\n'
        '                    "State": {"Name": "running"}\n'
        '                },\n'
        '                {\n'
        '                    "InstanceId": "i-0def456abc789012",\n'
        '                    "InstanceType": "t3.small",\n'
        '                    "State": {"Name": "stopped"}\n'
        '                }\n'
        '            ]\n'
        '        }\n'
        '    ]\n'
        '}\n'
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "aws_ec2"
    assert "2 instance" in result.compressed
    assert result.is_failure is False


def test_aws_lambda_list_functions():
    """Lambda list-functions JSON should be detected and functions listed."""
    content = (
        '{\n'
        '    "Functions": [\n'
        '        {"FunctionName": "process-orders", "Runtime": "python3.12"},\n'
        '        {"FunctionName": "send-notifications", "Runtime": "nodejs20.x"},\n'
        '        {"FunctionName": "cleanup-sessions", "Runtime": "python3.12"}\n'
        '    ]\n'
        '}\n'
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "aws_lambda"
    assert "3 function" in result.compressed


def test_aws_sts_get_caller_identity():
    """STS get-caller-identity should be detected and Arn extracted."""
    content = (
        '{\n'
        '    "UserId": "AIDA1234567890EXAMPLE",\n'
        '    "Account": "123456789012",\n'
        '    "Arn": "arn:aws:iam::123456789012:user/alice"\n'
        '}\n'
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.command == "aws_sts"
    assert "arn:aws" in result.compressed


# ---------------------------------------------------------------------------
# Detection failures
# ---------------------------------------------------------------------------

def test_unrecognized_content_returns_none():
    """Content matching no command should return None."""
    assert detect_and_filter("hello world, this is just random text") is None


def test_empty_content_returns_none():
    """Empty or whitespace-only content should return None."""
    assert detect_and_filter("") is None
    assert detect_and_filter("   \n\n  ") is None


def test_generic_programming_output_returns_none():
    """Generic programming output without command markers returns None."""
    content = (
        "Starting application server...\n"
        "Listening on port 8080\n"
        "Connection accepted from 192.168.1.10\n"
        "Processing request GET /api/users\n"
        "Response sent: 200 OK (45ms)\n"
    )
    assert detect_and_filter(content) is None


# ---------------------------------------------------------------------------
# list_supported_commands
# ---------------------------------------------------------------------------

def test_list_supported_commands_nonempty():
    """list_supported_commands should return a non-empty list of strings."""
    cmds = list_supported_commands()
    assert isinstance(cmds, list)
    assert len(cmds) > 0
    assert all(isinstance(c, str) for c in cmds)


def test_list_supported_commands_contains_git():
    """list_supported_commands should mention git."""
    cmds = list_supported_commands()
    assert any("git" in c.lower() for c in cmds)


def test_list_supported_commands_contains_pytest():
    """list_supported_commands should mention pytest."""
    cmds = list_supported_commands()
    assert any("pytest" in c.lower() for c in cmds)


# ---------------------------------------------------------------------------
# Compression ratios
# ---------------------------------------------------------------------------

def test_compression_ratio_git_status():
    """Non-trivial git status should have raw_tokens > compressed_tokens."""
    content = (
        "On branch develop\n"
        "Your branch is behind 'origin/develop' by 12 commits, and can be fast-forwarded.\n"
        "  (use \"git pull\" to update your local branch)\n\n"
        "Changes to be committed:\n"
        "  (use \"git restore --staged <file>...\" to unstage)\n"
        "\tmodified:   src/handlers/auth_handler.py\n"
        "\tmodified:   src/handlers/user_handler.py\n"
        "\tnew file:   src/middleware/rate_limiter.py\n\n"
        "Changes not staged for commit:\n"
        "  (use \"git add <file>...\" to update what will be committed)\n"
        "  (use \"git restore <file>...\" to discard changes in working directory)\n"
        "\tmodified:   src/main.py\n"
        "\tmodified:   config/settings.yaml\n\n"
        "Untracked files:\n"
        "  (use \"git add <file>...\" to include in what will be committed)\n"
        "\tdebug.log\n"
        "\ttemp_output/\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.raw_tokens > result.compressed_tokens


def test_compression_ratio_pytest():
    """Non-trivial pytest output should have raw_tokens > compressed_tokens."""
    lines = ["============================= test session starts =============================="]
    lines.append("platform linux -- Python 3.12.1, pytest-8.0.0, pluggy-1.4.0")
    lines.append("collected 100 items")
    lines.append("")
    for i in range(50):
        lines.append(f"tests/test_module_{i}.py ..{'.' * (i % 5)}  [{' ' * (i % 3)}{min(100, (i+1)*2)}%]")
    lines.append("")
    lines.append("============================== 100 passed in 8.45s ==============================")
    content = "\n".join(lines)
    result = detect_and_filter(content)
    assert result is not None
    assert result.raw_tokens > result.compressed_tokens


def test_compression_ratio_docker_ps():
    """Docker ps with multiple containers should compress significantly."""
    lines = ["CONTAINER ID   IMAGE          COMMAND                  CREATED          STATUS          PORTS                    NAMES"]
    for i in range(10):
        lines.append(
            f"a{i}b{i}c{i}d{i}e{i}f{i}   nginx:{i}.0      \"/docker-entrypoint.…\"   {i} hours ago    Up {i} hours    0.0.0.0:{8080+i}->{80+i}/tcp  server-{i}"
        )
    content = "\n".join(lines)
    result = detect_and_filter(content)
    assert result is not None
    assert result.raw_tokens > result.compressed_tokens


# ---------------------------------------------------------------------------
# FilterResult dataclass
# ---------------------------------------------------------------------------

def test_filter_result_fields():
    """FilterResult should have all expected fields."""
    result = detect_and_filter(
        "On branch main\nnothing to commit, working tree clean\n"
    )
    assert result is not None
    assert isinstance(result, FilterResult)
    assert isinstance(result.command, str)
    assert isinstance(result.compressed, str)
    assert isinstance(result.raw_tokens, int)
    assert isinstance(result.compressed_tokens, int)
    assert isinstance(result.is_failure, bool)


def test_git_push_failure():
    """Git push rejection should be detected as failure."""
    content = (
        "To github.com:acme/repo.git\n"
        " ! [rejected]        main -> main (non-fast-forward)\n"
        "error: failed to push some refs to 'github.com:acme/repo.git'\n"
        "hint: Updates were rejected because the tip of your current branch is behind\n"
    )
    result = detect_and_filter(content)
    assert result is not None
    assert result.is_failure is True
