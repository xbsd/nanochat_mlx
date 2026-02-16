"""
Sandboxed code execution for HumanEval evaluation.
Ported from nanochat/execution.py - pure Python (multiprocessing, signal, tempfile).

Provides safe execution of model-generated code against test cases
by running in a separate process with timeout and resource limits.
"""

import os
import sys
import signal
import tempfile
import multiprocessing
import contextlib
import io
import traceback
from typing import Optional, Dict, Any


# Timeout for code execution (seconds)
DEFAULT_TIMEOUT = 10.0


def _run_code(code: str, timeout: float, result_queue: multiprocessing.Queue):
    """
    Execute code in a restricted environment.
    This function runs in a child process.

    Args:
        code: Python code string to execute
        timeout: Maximum execution time in seconds
        result_queue: Queue to put the result into
    """
    # Set up alarm-based timeout (Unix only)
    def timeout_handler(signum, frame):
        raise TimeoutError("Code execution timed out")

    try:
        # Set the alarm
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout) + 1)

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Create a restricted globals dict
        exec_globals = {
            "__builtins__": __builtins__,
        }

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(code, exec_globals)

        result_queue.put({
            "passed": True,
            "result": "passed",
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
        })

    except TimeoutError:
        result_queue.put({
            "passed": False,
            "result": "timed out",
            "error": "Execution timed out",
        })
    except AssertionError as e:
        result_queue.put({
            "passed": False,
            "result": "failed",
            "error": f"AssertionError: {str(e)}",
        })
    except Exception as e:
        result_queue.put({
            "passed": False,
            "result": "failed",
            "error": f"{type(e).__name__}: {str(e)}",
            "traceback": traceback.format_exc(),
        })
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)


def unsafe_execute(code: str, timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    Execute code in a separate process with timeout.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        dict with 'passed' (bool), 'result' (str), and optionally 'error'
    """
    result_queue = multiprocessing.Queue()

    # Run in a separate process
    proc = multiprocessing.Process(
        target=_run_code,
        args=(code, timeout, result_queue),
    )
    proc.start()
    proc.join(timeout=timeout + 2)  # Extra buffer for process overhead

    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        return {
            "passed": False,
            "result": "timed out",
            "error": "Process killed after timeout",
        }

    if not result_queue.empty():
        return result_queue.get()

    return {
        "passed": False,
        "result": "failed",
        "error": "No result returned from execution",
    }


def check_correctness(
    problem: Dict[str, str],
    completion: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Evaluate a code completion against test cases.

    This is the main entry point for HumanEval evaluation.
    Combines the problem prompt with the completion and test cases,
    then runs everything in a sandboxed process.

    Args:
        problem: Dict with 'prompt', 'test', and 'entry_point' keys
            - prompt: The function signature and docstring
            - test: The test code (calls to check() function)
            - entry_point: The function name being tested
        completion: The model's code completion (function body)
        timeout: Maximum execution time in seconds

    Returns:
        dict with:
            - 'passed': bool indicating if all tests passed
            - 'result': str description ('passed', 'failed', 'timed out')
            - 'error': str error message (if failed)
    """
    prompt = problem["prompt"]
    test = problem["test"]
    entry_point = problem["entry_point"]

    # Build the full code to execute
    # The check function for HumanEval tests
    check_program = f"""
{prompt}{completion}

{test}

check({entry_point})
"""

    return unsafe_execute(check_program, timeout=timeout)


def execute_code(code: str, timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    Execute arbitrary Python code in a sandboxed process.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        dict with execution result
    """
    return unsafe_execute(code, timeout=timeout)


def run_with_timeout(func, args=(), kwargs=None, timeout=DEFAULT_TIMEOUT):
    """
    Run a function with a timeout using multiprocessing.

    Args:
        func: The function to run
        args: Positional arguments
        kwargs: Keyword arguments
        timeout: Maximum execution time in seconds

    Returns:
        The function's return value, or None if timed out
    """
    if kwargs is None:
        kwargs = {}

    result_queue = multiprocessing.Queue()

    def wrapper(q, *a, **kw):
        try:
            result = func(*a, **kw)
            q.put(("success", result))
        except Exception as e:
            q.put(("error", e))

    proc = multiprocessing.Process(
        target=wrapper,
        args=(result_queue, *args),
        kwargs=kwargs,
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        return None

    if not result_queue.empty():
        status, result = result_queue.get()
        if status == "success":
            return result
        else:
            raise result

    return None
