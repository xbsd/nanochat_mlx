"""
Common task infrastructure for nanochat_mlx evaluation tasks.
Ported from nanochat/tasks/common.py - pure Python, no framework dependency.

Provides:
- Task base class with slicing, indexing, and evaluation interface
- TaskMixture for shuffled multi-task mixtures
- TaskSequence for sequential curriculum learning
- render_mc helper for multiple-choice formatting
"""

import random
from abc import ABC, abstractmethod


# Multiple choice labels
MC_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def render_mc(question, choices, answer_index=None):
    """
    Render a multiple-choice question with labeled choices.

    Args:
        question: The question text
        choices: List of choice strings
        answer_index: Optional index of the correct answer

    Returns:
        If answer_index is None: formatted question string
        If answer_index is given: (formatted_question, answer_label) tuple
    """
    lines = [question, ""]
    for i, choice in enumerate(choices):
        label = MC_LABELS[i]
        lines.append(f"{label}. {choice}")
    formatted = "\n".join(lines)
    if answer_index is not None:
        answer_label = MC_LABELS[answer_index]
        return formatted, answer_label
    return formatted


class Task(ABC):
    """
    Base class for evaluation tasks.

    Supports slicing via start/stop/step to select subsets of examples.
    Subclasses must implement:
        - num_examples: property returning total number of examples
        - get_example(index): returns a dict for the given raw index
        - evaluate(index, output): returns a dict with evaluation results
    """

    def __init__(self, start=None, stop=None, step=None):
        """
        Args:
            start: Start index for slicing (inclusive)
            stop: Stop index for slicing (exclusive)
            step: Step size for slicing
        """
        self._start = start
        self._stop = stop
        self._step = step
        self._indices = None  # lazily computed

    def _get_indices(self):
        """Compute and cache the slice indices."""
        if self._indices is None:
            n = self.num_examples
            start = self._start if self._start is not None else 0
            stop = self._stop if self._stop is not None else n
            step = self._step if self._step is not None else 1
            # Clamp
            start = max(0, min(start, n))
            stop = max(0, min(stop, n))
            self._indices = list(range(start, stop, step))
        return self._indices

    def __len__(self):
        """Return the number of examples after slicing."""
        return len(self._get_indices())

    def __getitem__(self, index):
        """
        Get an example by sliced index.

        Args:
            index: Index into the sliced view

        Returns:
            dict with at least 'conversation' and optionally 'ideal' keys
        """
        indices = self._get_indices()
        if index < 0:
            index = len(indices) + index
        if index < 0 or index >= len(indices):
            raise IndexError(f"Index {index} out of range for task with {len(indices)} examples")
        raw_index = indices[index]
        return self.get_example(raw_index)

    @property
    @abstractmethod
    def num_examples(self):
        """Return the total number of raw examples (before slicing)."""
        pass

    @abstractmethod
    def get_example(self, index):
        """
        Get a raw example by its original index.

        Args:
            index: Raw index into the dataset

        Returns:
            dict with:
                - 'conversation': dict with 'messages' key (list of role/content dicts)
                - 'ideal': the expected answer (string or label)
                - optionally other metadata
        """
        pass

    @abstractmethod
    def evaluate(self, index, output):
        """
        Evaluate a model output for the given example.

        Args:
            index: The sliced index of the example
            output: The model's output string

        Returns:
            dict with evaluation results, typically including:
                - 'correct': bool indicating correctness
                - 'score': float score (0.0 to 1.0)
                - other task-specific metrics
        """
        pass

    def reward(self, index, output):
        """
        Compute a scalar reward for RL training. Default uses evaluate().

        Args:
            index: The sliced index of the example
            output: The model's output string

        Returns:
            float reward value
        """
        result = self.evaluate(index, output)
        return float(result.get("score", result.get("correct", 0.0)))


class TaskMixture:
    """
    A shuffled mixture of multiple tasks.

    Combines examples from multiple tasks into a single dataset,
    shuffled together. Each example tracks which task it came from.
    """

    def __init__(self, tasks, seed=42):
        """
        Args:
            tasks: List of (name, task) tuples or list of Task objects
            seed: Random seed for shuffling
        """
        self.tasks = []
        self.task_names = []
        self.examples = []  # list of (task_index, example_index)

        for item in tasks:
            if isinstance(item, tuple):
                name, task = item
            else:
                name = item.__class__.__name__
                task = item
            self.task_names.append(name)
            self.tasks.append(task)
            task_idx = len(self.tasks) - 1
            for i in range(len(task)):
                self.examples.append((task_idx, i))

        # Shuffle with fixed seed for reproducibility
        rng = random.Random(seed)
        rng.shuffle(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        task_idx, example_idx = self.examples[index]
        example = self.tasks[task_idx][example_idx]
        example["_task_name"] = self.task_names[task_idx]
        example["_task_index"] = task_idx
        example["_example_index"] = example_idx
        return example

    def evaluate(self, index, output):
        task_idx, example_idx = self.examples[index]
        return self.tasks[task_idx].evaluate(example_idx, output)

    def reward(self, index, output):
        task_idx, example_idx = self.examples[index]
        return self.tasks[task_idx].reward(example_idx, output)


class TaskSequence:
    """
    Sequential curriculum of tasks.

    Tasks are presented one after another (not shuffled).
    Useful for curriculum learning where task order matters.
    """

    def __init__(self, tasks):
        """
        Args:
            tasks: List of (name, task) tuples or list of Task objects
        """
        self.tasks = []
        self.task_names = []
        self.examples = []  # list of (task_index, example_index)

        for item in tasks:
            if isinstance(item, tuple):
                name, task = item
            else:
                name = item.__class__.__name__
                task = item
            self.task_names.append(name)
            self.tasks.append(task)
            task_idx = len(self.tasks) - 1
            for i in range(len(task)):
                self.examples.append((task_idx, i))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        task_idx, example_idx = self.examples[index]
        example = self.tasks[task_idx][example_idx]
        example["_task_name"] = self.task_names[task_idx]
        example["_task_index"] = task_idx
        example["_example_index"] = example_idx
        return example

    def evaluate(self, index, output):
        task_idx, example_idx = self.examples[index]
        return self.tasks[task_idx].evaluate(example_idx, output)

    def reward(self, index, output):
        task_idx, example_idx = self.examples[index]
        return self.tasks[task_idx].reward(example_idx, output)
