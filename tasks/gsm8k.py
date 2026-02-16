"""
GSM8K evaluation task for nanochat_mlx.
Ported from nanochat/tasks/gsm8k.py - pure Python + datasets library.

GSM8K is a dataset of 8.5K grade school math word problems.
Answers are extracted using the #### pattern.
Tool calls are parsed with <<expr=result>> pattern.
"""

import re
import math
from datasets import load_dataset

from tasks.common import Task


def extract_answer(text):
    """
    Extract the final numerical answer from GSM8K-style text.
    Looks for the #### pattern that marks the final answer.

    Args:
        text: The answer text containing #### marker

    Returns:
        The extracted answer string, or None if not found
    """
    # Look for #### pattern
    match = re.search(r'####\s*(.*?)$', text, re.MULTILINE)
    if match:
        answer = match.group(1).strip()
        # Remove commas from numbers
        answer = answer.replace(",", "")
        return answer
    return None


def extract_answer_from_output(output):
    """
    Extract the final numerical answer from model output.
    Tries multiple patterns:
    1. #### marker
    2. "The answer is X" pattern
    3. Last number in the text

    Args:
        output: The model's generated output

    Returns:
        The extracted answer string, or None
    """
    # Try #### pattern first
    answer = extract_answer(output)
    if answer is not None:
        return answer

    # Try "the answer is" pattern
    match = re.search(r'[Tt]he\s+answer\s+is\s*:?\s*\$?([+-]?[\d,]+\.?\d*)', output)
    if match:
        return match.group(1).replace(",", "")

    # Try to find the last number in the text
    numbers = re.findall(r'[+-]?\d[\d,]*\.?\d*', output)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def parse_tool_calls(text):
    """
    Parse tool calls in the format <<expr=result>>.
    Used for calculator-augmented math solving.

    Args:
        text: Text potentially containing <<expr=result>> patterns

    Returns:
        List of (expression, result) tuples
    """
    pattern = r'<<(.+?)=(.+?)>>'
    matches = re.findall(pattern, text)
    return [(expr.strip(), result.strip()) for expr, result in matches]


def evaluate_tool_calls(text):
    """
    Evaluate tool calls in text and replace with computed results.
    Handles <<expr>> patterns by computing the expression.

    Args:
        text: Text with tool call patterns

    Returns:
        Text with tool calls replaced by their results
    """
    def replace_match(match):
        expr = match.group(1)
        if '=' in expr:
            # Already has result, keep as-is
            return match.group(0)
        try:
            result = eval(expr)
            return f"<<{expr}={result}>>"
        except Exception:
            return match.group(0)

    return re.sub(r'<<(.+?)>>', replace_match, text)


def is_correct(predicted, expected):
    """
    Check if predicted answer matches expected answer.
    Handles numeric comparison with tolerance.

    Args:
        predicted: The predicted answer string
        expected: The expected answer string

    Returns:
        bool indicating whether answers match
    """
    if predicted is None or expected is None:
        return False

    # Clean up
    predicted = predicted.strip().rstrip(".")
    expected = expected.strip().rstrip(".")

    # Direct string match
    if predicted == expected:
        return True

    # Try numeric comparison
    try:
        pred_num = float(predicted.replace(",", ""))
        exp_num = float(expected.replace(",", ""))
        return math.isclose(pred_num, exp_num, rel_tol=1e-5)
    except (ValueError, TypeError):
        return False


class GSM8K(Task):
    """
    GSM8K math word problem evaluation task.

    Each example is a conversation where the user asks a math question
    and the model should provide a step-by-step solution ending with
    #### followed by the numerical answer.
    """

    def __init__(self, split="test", start=None, stop=None, step=None):
        """
        Args:
            split: Dataset split to use ('train' or 'test')
            start: Start index for slicing
            stop: Stop index for slicing
            step: Step size for slicing
        """
        super().__init__(start=start, stop=stop, step=step)
        self.dataset = load_dataset("openai/gsm8k", "main", split=split)

    @property
    def num_examples(self):
        return len(self.dataset)

    def get_example(self, index):
        """
        Get a GSM8K example.

        Returns:
            dict with:
                - conversation: user asks the question
                - ideal: the expected numerical answer
                - solution: the full solution text
        """
        item = self.dataset[index]
        question = item["question"]
        answer_text = item["answer"]
        ideal = extract_answer(answer_text)

        conversation = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": ""},
            ]
        }

        return {
            "conversation": conversation,
            "ideal": ideal,
            "solution": answer_text,
        }

    def evaluate(self, index, output):
        """
        Evaluate model output for a GSM8K example.

        Args:
            index: Sliced index of the example
            output: Model's generated text

        Returns:
            dict with 'correct' (bool) and 'score' (float 0 or 1)
        """
        example = self[index]
        expected = example["ideal"]
        predicted = extract_answer_from_output(output)
        correct = is_correct(predicted, expected)

        return {
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "predicted": predicted,
            "expected": expected,
        }

    def reward(self, index, output):
        """
        Compute reward for RL training.

        Args:
            index: Sliced index
            output: Model output text

        Returns:
            float: 1.0 if correct, 0.0 otherwise
        """
        result = self.evaluate(index, output)
        return result["score"]
