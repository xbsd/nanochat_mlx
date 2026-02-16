"""
CustomJSON task for nanochat_mlx.
Ported from nanochat/tasks/customjson.py.

Loads evaluation or training data from JSONL files.
Each line should be a JSON object with a 'messages' field
(list of role/content dicts) and optionally an 'ideal' field.
"""

import json

from tasks.common import Task


class CustomJSON(Task):
    """
    Custom task loaded from a JSONL file.

    Supports flexible JSONL formats:
    1. Chat format: {"messages": [{"role": "user", "content": "..."}, ...]}
    2. Chat format with ideal: {"messages": [...], "ideal": "expected answer"}
    3. Simple QA: {"question": "...", "answer": "..."}

    The JSONL file is loaded entirely into memory on init.
    """

    def __init__(self, jsonl_path, start=None, stop=None, step=None):
        """
        Args:
            jsonl_path: Path to the JSONL file
            start: Start index for slicing
            stop: Stop index for slicing
            step: Step size for slicing
        """
        super().__init__(start=start, stop=stop, step=step)
        self.jsonl_path = jsonl_path
        self._data = self._load_jsonl(jsonl_path)

    @staticmethod
    def _load_jsonl(path):
        """
        Load a JSONL file into a list of dicts.

        Args:
            path: Path to JSONL file

        Returns:
            List of parsed JSON objects
        """
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}")
        return data

    @property
    def num_examples(self):
        return len(self._data)

    def get_example(self, index):
        """
        Get a CustomJSON example.

        Supports multiple formats:
        1. {"messages": [...], "ideal": "..."} - standard chat format
        2. {"messages": [...]} - chat format without ideal
        3. {"question": "...", "answer": "..."} - simple QA format
        4. {"prompt": "...", "completion": "..."} - prompt/completion format

        Returns:
            dict with:
                - conversation: dict with 'messages' key
                - ideal: expected answer (if available)
        """
        item = self._data[index]

        # Standard chat format
        if "messages" in item:
            messages = item["messages"]
            # Ensure messages have the right structure
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

            conversation = {"messages": formatted_messages}
            ideal = item.get("ideal", None)

        # Simple QA format
        elif "question" in item:
            question = item["question"]
            answer = item.get("answer", "")
            conversation = {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": ""},
                ]
            }
            ideal = answer if answer else None

        # Prompt/completion format
        elif "prompt" in item:
            prompt = item["prompt"]
            completion = item.get("completion", "")
            conversation = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": ""},
                ]
            }
            ideal = completion if completion else None

        else:
            raise ValueError(
                f"Unrecognized format in {self.jsonl_path} at index {index}. "
                "Expected 'messages', 'question', or 'prompt' key."
            )

        result = {
            "conversation": conversation,
            "ideal": ideal,
        }

        # Pass through any extra fields
        for key in item:
            if key not in ("messages", "ideal", "question", "answer", "prompt", "completion"):
                result[key] = item[key]

        return result

    def evaluate(self, index, output):
        """
        Evaluate model output. Uses exact match if ideal is available.

        Args:
            index: Sliced index
            output: Model's generated text

        Returns:
            dict with 'correct', 'score'
        """
        example = self[index]
        ideal = example.get("ideal")

        if ideal is None:
            return {
                "correct": False,
                "score": 0.0,
                "note": "No ideal answer provided for evaluation.",
            }

        # Simple exact match (case-insensitive, stripped)
        correct = output.strip().lower() == ideal.strip().lower()

        return {
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "predicted": output.strip(),
            "expected": ideal.strip(),
        }
