"""
Local code execution with timeout and memory limits.

Adapted from Mathematical-Reasoning-RL-Scaling-Law/verl/utils/reward_score/coder1/unsafe_local_exec.py
"""
import os
import subprocess
import shlex
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Tuple, Optional

# Constants
CLI_ARG_SIZE_LIMIT = 1024 * 3  # Max code size for command line argument
MEMORY_LIMIT_KB = 10 * 1024 * 1024  # 10GB in KB
DEFAULT_TIMEOUT = 30  # seconds
ERROR_MSG_PREFIX = "[EXECUTION ERROR] "


def _wrap_command_with_ulimit(command: list) -> list:
    """Wrap command with ulimit for memory limiting."""
    cmd_str = ' '.join(shlex.quote(c) for c in command)
    return ["bash", "-c", f"ulimit -v {MEMORY_LIMIT_KB}; exec {cmd_str}"]


def code_exec_local(
    code: str,
    stdin: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    functional: Optional[str] = None,
    python_env: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Execute Python code locally with safety limits.

    Args:
        code: Python code to execute
        stdin: Optional stdin input
        timeout: Execution timeout in seconds (default: 30)
        functional: Optional functional test code to append (for HumanEval-style tests)
        python_env: Optional path to Python environment (default: /usr/bin/python3)

    Returns:
        Tuple of (success: bool, output: str)
        - success: True if return code is 0
        - output: stdout on success, or error message on failure

    Example:
        >>> success, output = code_exec_local("print('hello')")
        >>> success
        True
        >>> output
        'hello\\n'

        >>> success, output = code_exec_local("def add(a,b): return a+b", functional="assert add(1,2)==3")
        >>> success
        True
    """
    # Setup environment
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"]

    # Determine Python executable
    if python_env is None:
        python_executable = "/usr/bin/python3"
    else:
        python_executable = os.path.join(python_env, "python3")

    # If functional tests provided, append them to code
    if functional:
        full_code = code + "\n" + functional
    else:
        full_code = code

    try:
        # Choose execution method based on code size
        if len(full_code) < CLI_ARG_SIZE_LIMIT:
            # Small code: pass via command line
            command = ["timeout", str(timeout), python_executable, "-c", full_code]
            command = _wrap_command_with_ulimit(command)
            result = subprocess.run(
                command,
                input=(stdin.encode() if stdin else None),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                check=False,
            )
        else:
            # Large code: write to temp file
            with NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                tmp.write(full_code)
                tmp.flush()
                tmp_path = tmp.name

            try:
                command = ["timeout", str(timeout), python_executable, tmp_path]
                command = _wrap_command_with_ulimit(command)
                result = subprocess.run(
                    command,
                    input=(stdin.encode() if stdin else None),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    check=False,
                )
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # Process result
        stderr = result.stderr.decode().strip()
        stdout = result.stdout.decode()

        if result.returncode == 0:
            return True, stdout
        else:
            return False, ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"

    except Exception as e:
        return False, ERROR_MSG_PREFIX + f"Exception during execution: {str(e)}"


def fuzzy_equal(actual: str, expected: str, tolerance: float = 1e-6) -> bool:
    """
    Compare two outputs with fuzzy matching.

    Handles:
    1. Integer and floating-point number comparison with tolerance
    2. Case-insensitive comparison for yes/no

    Args:
        actual: The actual output from code execution
        expected: The expected output
        tolerance: Tolerance for floating point number comparison

    Returns:
        bool: True if outputs are approximately equal
    """
    # Normalize line endings
    actual = actual.strip().replace("\r\n", "\n")
    expected = expected.strip().replace("\r\n", "\n")

    # If exact match after normalization, return early
    if actual == expected:
        return True

    # Split into lines
    actual_lines = actual.split("\n")
    expected_lines = expected.split("\n")

    # If different number of lines, they're definitely not equal
    if len(actual_lines) != len(expected_lines):
        return False

    # Compare each line
    for actual_line, expected_line in zip(actual_lines, expected_lines):
        # If lines match exactly, continue
        if actual_line == expected_line:
            continue

        # Split into tokens by whitespace
        actual_tokens = actual_line.split()
        expected_tokens = expected_line.split()

        # If different number of tokens, they're not equal
        if len(actual_tokens) != len(expected_tokens):
            return False

        # Compare each token
        for actual_token, expected_token in zip(actual_tokens, expected_tokens):
            # If tokens match exactly, continue
            if actual_token == expected_token:
                continue

            # For yes/no, use case-insensitive comparison
            if actual_token.lower() in ["yes", "no"] and expected_token.lower() in ["yes", "no"]:
                if actual_token.lower() != expected_token.lower():
                    return False
                continue

            # Try numeric comparison
            try:
                actual_num = float(actual_token)
                expected_num = float(expected_token)
                diff = abs(actual_num - expected_num)

                if diff > tolerance:
                    return False
            except ValueError:
                # Not numeric values, and they don't match exactly
                return False

    return True
