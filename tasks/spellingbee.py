"""
SpellingBee and SimpleSpelling tasks for nanochat_mlx.
Ported from nanochat/tasks/spellingbee.py.

Synthetic letter-counting and spelling tasks with multilingual templates.
Tests the model's ability to spell words and count letters.
"""

import os
import re
import json
import random

from tasks.common import Task
from nanochat_mlx.common import download_file_with_lock


# Multilingual prompt templates for spelling tasks
SPELLING_TEMPLATES = [
    # English
    "How many times does the letter '{letter}' appear in the word '{word}'?",
    "Count the number of '{letter}' letters in '{word}'.",
    "In the word '{word}', how many '{letter}' letters are there?",
    "What is the count of the letter '{letter}' in '{word}'?",
    "Tell me how many times '{letter}' occurs in the word '{word}'.",
    # Variations
    "How many '{letter}'s are in '{word}'?",
    "Can you count the letter '{letter}' in the word '{word}'?",
    "How many occurrences of '{letter}' are in '{word}'?",
]

SIMPLE_SPELLING_TEMPLATES = [
    "Spell the word '{word}' one letter at a time, separated by dashes.",
    "Break down the word '{word}' into individual letters separated by dashes.",
    "Please spell out '{word}' letter by letter with dashes between each letter.",
    "Write the letters of '{word}' separated by dashes.",
]

# Common English words for the spelling task
DEFAULT_WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
DEFAULT_WORD_LIST_FILE = "spelling_words.txt"


def load_word_list(min_length=3, max_length=12, max_words=10000, seed=42):
    """
    Load a word list for spelling tasks.

    Downloads from a URL if not cached, then filters and samples.

    Args:
        min_length: Minimum word length
        max_length: Maximum word length
        max_words: Maximum number of words to keep
        seed: Random seed for sampling

    Returns:
        List of words
    """
    file_path = download_file_with_lock(DEFAULT_WORD_LIST_URL, DEFAULT_WORD_LIST_FILE)

    with open(file_path, "r") as f:
        all_words = [line.strip().lower() for line in f if line.strip()]

    # Filter by length
    words = [w for w in all_words if min_length <= len(w) <= max_length]

    # Sample deterministically
    rng = random.Random(seed)
    if len(words) > max_words:
        words = rng.sample(words, max_words)
    else:
        rng.shuffle(words)

    return words


class SpellingBee(Task):
    """
    SpellingBee letter-counting task.

    Given a word and a letter, the model must count how many times
    that letter appears in the word. Tests character-level understanding.
    """

    def __init__(self, num_examples_val=1000, seed=42, start=None, stop=None, step=None):
        """
        Args:
            num_examples_val: Number of examples to generate
            seed: Random seed for reproducibility
            start: Start index for slicing
            stop: Stop index for slicing
            step: Step size for slicing
        """
        super().__init__(start=start, stop=stop, step=step)
        self.seed = seed
        self._num_examples = num_examples_val
        self._examples = None

    def _generate_examples(self):
        """Lazily generate examples."""
        if self._examples is not None:
            return

        words = load_word_list(seed=self.seed)
        rng = random.Random(self.seed)
        self._examples = []

        for _ in range(self._num_examples):
            word = rng.choice(words)
            # Pick a letter - sometimes one that's in the word, sometimes not
            if rng.random() < 0.8:
                # Pick a letter that's in the word
                letter = rng.choice(word)
            else:
                # Pick a random letter (might not be in the word)
                letter = rng.choice("abcdefghijklmnopqrstuvwxyz")

            count = word.lower().count(letter.lower())
            template = rng.choice(SPELLING_TEMPLATES)
            question = template.format(letter=letter, word=word)

            self._examples.append({
                "word": word,
                "letter": letter,
                "count": count,
                "question": question,
            })

    @property
    def num_examples(self):
        self._generate_examples()
        return len(self._examples)

    def get_example(self, index):
        """
        Get a SpellingBee example.

        Returns:
            dict with:
                - conversation: user asks about letter count
                - ideal: the correct count as string
        """
        self._generate_examples()
        item = self._examples[index]

        conversation = {
            "messages": [
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": ""},
            ]
        }

        return {
            "conversation": conversation,
            "ideal": str(item["count"]),
            "word": item["word"],
            "letter": item["letter"],
        }

    def evaluate(self, index, output):
        """
        Evaluate model output for letter counting.

        Args:
            index: Sliced index
            output: Model's generated text

        Returns:
            dict with 'correct', 'score', 'predicted', 'expected'
        """
        example = self[index]
        expected = example["ideal"]

        # Extract a number from the output
        predicted = _extract_number(output)
        correct = predicted is not None and predicted == expected

        return {
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "predicted": predicted,
            "expected": expected,
        }


class SimpleSpelling(Task):
    """
    SimpleSpelling task.

    Given a word, the model must spell it out letter by letter
    separated by dashes (e.g., "hello" -> "h-e-l-l-o").
    Tests basic character-level understanding.
    """

    def __init__(self, num_examples_val=1000, seed=42, start=None, stop=None, step=None):
        """
        Args:
            num_examples_val: Number of examples to generate
            seed: Random seed for reproducibility
            start: Start index for slicing
            stop: Stop index for slicing
            step: Step size for slicing
        """
        super().__init__(start=start, stop=stop, step=step)
        self.seed = seed
        self._num_examples = num_examples_val
        self._examples = None

    def _generate_examples(self):
        """Lazily generate examples."""
        if self._examples is not None:
            return

        words = load_word_list(seed=self.seed)
        rng = random.Random(self.seed + 1)  # Different seed from SpellingBee
        self._examples = []

        for _ in range(self._num_examples):
            word = rng.choice(words)
            template = rng.choice(SIMPLE_SPELLING_TEMPLATES)
            question = template.format(word=word)
            ideal = "-".join(word.lower())

            self._examples.append({
                "word": word,
                "question": question,
                "ideal": ideal,
            })

    @property
    def num_examples(self):
        self._generate_examples()
        return len(self._examples)

    def get_example(self, index):
        """
        Get a SimpleSpelling example.

        Returns:
            dict with:
                - conversation: user asks to spell the word
                - ideal: the correct spelling with dashes
        """
        self._generate_examples()
        item = self._examples[index]

        conversation = {
            "messages": [
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": ""},
            ]
        }

        return {
            "conversation": conversation,
            "ideal": item["ideal"],
            "word": item["word"],
        }

    def evaluate(self, index, output):
        """
        Evaluate model output for spelling.

        Args:
            index: Sliced index
            output: Model's generated text

        Returns:
            dict with 'correct', 'score', 'predicted', 'expected'
        """
        example = self[index]
        expected = example["ideal"]

        # Extract the dashed spelling from the output
        predicted = _extract_spelling(output)
        correct = predicted is not None and predicted.lower() == expected.lower()

        return {
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "predicted": predicted,
            "expected": expected,
        }


def _extract_number(output):
    """
    Extract a number from model output.

    Args:
        output: Model's generated text

    Returns:
        Number as string, or None
    """
    output = output.strip()

    # Try: just a number
    if output.isdigit():
        return output

    # Try: "The answer is X" pattern
    match = re.search(r'[Tt]he\s+(?:answer|count|number)\s+is\s*:?\s*(\d+)', output)
    if match:
        return match.group(1)

    # Try: first number in the output
    match = re.search(r'\b(\d+)\b', output)
    if match:
        return match.group(1)

    # Try: number words
    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7",
        "eight": "8", "nine": "9", "ten": "10",
    }
    output_lower = output.lower()
    for word, num in number_words.items():
        if word in output_lower:
            return num

    return None


def _extract_spelling(output):
    """
    Extract dashed spelling from model output.
    Looks for patterns like "h-e-l-l-o".

    Args:
        output: Model's generated text

    Returns:
        The dashed spelling string, or None
    """
    output = output.strip()

    # Look for a dash-separated sequence of single letters
    match = re.search(r'([a-zA-Z](?:\s*-\s*[a-zA-Z])+)', output)
    if match:
        spelling = match.group(1)
        # Normalize: remove spaces around dashes
        spelling = re.sub(r'\s*-\s*', '-', spelling)
        return spelling.lower()

    return None
