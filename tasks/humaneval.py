"""
HumanEval coding benchmark task for nanochat_mlx.
Ported from nanochat/tasks/humaneval.py - uses datasets + sandboxed code execution.

HumanEval is a set of 164 hand-written Python programming problems.
Each problem has a function signature, docstring with examples, and test cases.
Code is executed in a sandboxed environment for safety.
"""

import re
from datasets import load_dataset

from tasks.common import Task


class HumanEval(Task):
    """
    HumanEval code generation evaluation task.

    Each example provides a function signature and docstring.
    The model should complete the function body.
    Evaluation runs the generated code against hidden test cases
    in a sandboxed environment.
    """

    def __init__(self, split="test", start=None, stop=None, step=None):
        """
        Args:
            split: Dataset split (HumanEval only has 'test')
            start: Start index for slicing
            stop: Stop index for slicing
            step: Step size for slicing
        """
        super().__init__(start=start, stop=stop, step=step)
        self.dataset = load_dataset("openai/openai_humaneval", split=split)

    @property
    def num_examples(self):
        return len(self.dataset)

    def get_example(self, index):
        """
        Get a HumanEval example.

        Returns:
            dict with:
                - conversation: user asks to complete the function
                - ideal: the canonical solution
                - prompt: the function signature + docstring
                - test: the test code
                - entry_point: the function name to test
                - task_id: the HumanEval task ID
        """
        item = self.dataset[index]
        prompt = item["prompt"]
        canonical_solution = item["canonical_solution"]
        test = item["test"]
        entry_point = item["entry_point"]
        task_id = item["task_id"]

        # Build the conversation
        user_content = (
            "Complete the following Python function. "
            "Return ONLY the function body (the code that goes after the function signature). "
            "Do not include the function signature or any explanation.\n\n"
            f"```python\n{prompt}```"
        )

        conversation = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": ""},
            ]
        }

        return {
            "conversation": conversation,
            "ideal": canonical_solution,
            "prompt": prompt,
            "test": test,
            "entry_point": entry_point,
            "task_id": task_id,
        }

    def evaluate(self, index, output):
        """
        Evaluate model output by running generated code against test cases.

        Args:
            index: Sliced index
            output: Model's generated code

        Returns:
            dict with 'correct', 'score', 'error' (if any)
        """
        from nanochat_mlx.execution import check_correctness

        example = self[index]
        prompt = example["prompt"]
        test = example["test"]
        entry_point = example["entry_point"]

        # Extract code from the output
        code = _extract_code(output)

        # Combine prompt + generated code
        full_code = prompt + code

        # Run in sandbox with test cases
        result = check_correctness(
            problem={
                "prompt": prompt,
                "test": test,
                "entry_point": entry_point,
            },
            completion=code,
            timeout=10.0,
        )

        passed = result.get("passed", False)

        return {
            "correct": passed,
            "score": 1.0 if passed else 0.0,
            "result": result.get("result", "unknown"),
            "error": result.get("error", None),
        }


def _extract_code(output):
    """
    Extract Python code from model output.
    Handles code blocks and raw code.

    Args:
        output: Model's generated text

    Returns:
        Extracted code string
    """
    # Try to extract from markdown code block
    match = re.search(r'```(?:python)?\s*\n?(.*?)```', output, re.DOTALL)
    if match:
        return match.group(1).strip() + "\n"

    # If no code block, try to find indented code (function body)
    lines = output.strip().split('\n')
    code_lines = []
    for line in lines:
        # Skip empty lines at the start
        if not code_lines and not line.strip():
            continue
        # Stop at non-code text after code has started
        if code_lines and not line.strip() and not any(l.strip() for l in lines[lines.index(line):]):
            break
        code_lines.append(line)

    if code_lines:
        return '\n'.join(code_lines) + "\n"

    return output + "\n"
