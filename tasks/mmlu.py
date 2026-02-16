"""
MMLU (Massive Multitask Language Understanding) evaluation task for nanochat_mlx.
Ported from nanochat/tasks/mmlu.py - pure Python + datasets library.

MMLU tests knowledge across 57 subjects ranging from STEM to humanities.
Uses categorical (multiple-choice) evaluation.
"""

from datasets import load_dataset

from tasks.common import Task, render_mc, MC_LABELS


class MMLU(Task):
    """
    MMLU multiple-choice evaluation task.

    Each example presents a question with 4 choices (A, B, C, D).
    The model should output the letter of the correct answer.
    Evaluation is categorical: exact match on the answer label.
    """

    def __init__(self, split="test", subject=None, start=None, stop=None, step=None):
        """
        Args:
            split: Dataset split ('test', 'validation', 'dev', or 'auxiliary_train')
            subject: Optional specific MMLU subject to filter on
            start: Start index for slicing
            stop: Stop index for slicing
            step: Step size for slicing
        """
        super().__init__(start=start, stop=stop, step=step)
        if subject is not None:
            self.dataset = load_dataset("cais/mmlu", subject, split=split)
        else:
            self.dataset = load_dataset("cais/mmlu", "all", split=split)
        self.subject = subject

    @property
    def num_examples(self):
        return len(self.dataset)

    def get_example(self, index):
        """
        Get an MMLU example.

        Returns:
            dict with:
                - conversation: user asks the MC question
                - ideal: the correct answer label (A, B, C, D)
                - subject: the MMLU subject
        """
        item = self.dataset[index]
        question = item["question"]
        choices = item["choices"]
        answer_index = item["answer"]
        subject = item.get("subject", self.subject or "unknown")

        formatted_question, answer_label = render_mc(question, choices, answer_index)

        # Build the prompt with instructions
        prompt = f"The following is a multiple choice question about {subject.replace('_', ' ')}.\n\n"
        prompt += formatted_question
        prompt += "\n\nAnswer with just the letter (A, B, C, or D)."

        conversation = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ""},
            ]
        }

        return {
            "conversation": conversation,
            "ideal": answer_label,
            "subject": subject,
        }

    def evaluate(self, index, output):
        """
        Evaluate model output for an MMLU example.
        Categorical evaluation: checks if the output contains the correct letter.

        Args:
            index: Sliced index
            output: Model's generated text

        Returns:
            dict with 'correct', 'score', 'predicted', 'expected'
        """
        example = self[index]
        expected = example["ideal"]

        # Extract the predicted answer label from model output
        predicted = _extract_mc_answer(output)
        correct = predicted is not None and predicted == expected

        return {
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "predicted": predicted,
            "expected": expected,
        }


def _extract_mc_answer(output):
    """
    Extract a multiple-choice answer label from model output.
    Looks for a single letter A-J at the start or after common patterns.

    Args:
        output: Model's generated text

    Returns:
        The extracted label string or None
    """
    import re

    output = output.strip()

    # Try: just a single letter
    if len(output) == 1 and output.upper() in MC_LABELS:
        return output.upper()

    # Try: starts with a letter followed by period, paren, colon, or space
    match = re.match(r'^([A-Ja-j])\s*[\.\)\:\,]', output)
    if match:
        return match.group(1).upper()

    # Try: "The answer is X" pattern
    match = re.search(r'[Tt]he\s+answer\s+is\s*:?\s*\(?([A-Ja-j])\)?', output)
    if match:
        return match.group(1).upper()

    # Try: "Answer: X" pattern
    match = re.search(r'[Aa]nswer\s*:\s*\(?([A-Ja-j])\)?', output)
    if match:
        return match.group(1).upper()

    # Try: first single uppercase letter in the output
    match = re.search(r'\b([A-D])\b', output)
    if match:
        return match.group(1)

    return None
