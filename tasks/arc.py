"""
ARC (AI2 Reasoning Challenge) evaluation task for nanochat_mlx.
Ported from nanochat/tasks/arc.py - pure Python + datasets library.

ARC contains grade-school level multiple-choice science questions.
Two difficulty levels: ARC-Easy and ARC-Challenge.
Uses categorical (multiple-choice) evaluation.
"""

from datasets import load_dataset

from tasks.common import Task, render_mc, MC_LABELS


class ARC(Task):
    """
    ARC multiple-choice evaluation task.

    Each example presents a science question with multiple choices.
    The model should output the letter of the correct answer.
    Evaluation is categorical: exact match on the answer label.
    """

    def __init__(self, difficulty="Challenge", split="test", start=None, stop=None, step=None):
        """
        Args:
            difficulty: 'Challenge' or 'Easy'
            split: Dataset split ('test', 'validation', 'train')
            start: Start index for slicing
            stop: Stop index for slicing
            step: Step size for slicing
        """
        super().__init__(start=start, stop=stop, step=step)
        assert difficulty in ("Challenge", "Easy"), f"Invalid difficulty: {difficulty}"
        self.difficulty = difficulty
        config_name = f"ARC-{difficulty}"
        self.dataset = load_dataset("allenai/ai2_arc", config_name, split=split)

    @property
    def num_examples(self):
        return len(self.dataset)

    def get_example(self, index):
        """
        Get an ARC example.

        Returns:
            dict with:
                - conversation: user asks the MC question
                - ideal: the correct answer label
        """
        item = self.dataset[index]
        question = item["question"]
        choices_data = item["choices"]
        answer_key = item["answerKey"]

        # ARC choices come as a dict with 'text' and 'label' lists
        choice_texts = choices_data["text"]
        choice_labels = choices_data["label"]

        # Find the correct answer index
        # Labels can be "A", "B", "C", "D" or "1", "2", "3", "4"
        answer_index = None
        for i, label in enumerate(choice_labels):
            if label == answer_key:
                answer_index = i
                break

        if answer_index is None:
            # Try numeric mapping
            try:
                answer_index = int(answer_key) - 1
            except ValueError:
                # Try letter mapping
                answer_index = ord(answer_key.upper()) - ord('A')

        formatted_question, answer_label = render_mc(question, choice_texts, answer_index)

        prompt = formatted_question
        prompt += "\n\nAnswer with just the letter."

        conversation = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ""},
            ]
        }

        return {
            "conversation": conversation,
            "ideal": answer_label,
        }

    def evaluate(self, index, output):
        """
        Evaluate model output for an ARC example.
        Categorical evaluation: checks if output contains correct letter.

        Args:
            index: Sliced index
            output: Model's generated text

        Returns:
            dict with 'correct', 'score', 'predicted', 'expected'
        """
        example = self[index]
        expected = example["ideal"]
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

    Args:
        output: Model's generated text

    Returns:
        The extracted label string or None
    """
    import re

    output = output.strip()

    # Just a single letter
    if len(output) == 1 and output.upper() in MC_LABELS:
        return output.upper()

    # Starts with a letter followed by punctuation
    match = re.match(r'^([A-Ja-j])\s*[\.\)\:\,]', output)
    if match:
        return match.group(1).upper()

    # "The answer is X" pattern
    match = re.search(r'[Tt]he\s+answer\s+is\s*:?\s*\(?([A-Ja-j])\)?', output)
    if match:
        return match.group(1).upper()

    # "Answer: X" pattern
    match = re.search(r'[Aa]nswer\s*:\s*\(?([A-Ja-j])\)?', output)
    if match:
        return match.group(1).upper()

    # First standalone uppercase letter
    match = re.search(r'\b([A-D])\b', output)
    if match:
        return match.group(1)

    return None
