"""
SmolTalk conversational dataset task for nanochat_mlx.
Ported from nanochat/tasks/smoltalk.py - simple wrapper around HuggingFace dataset.

SmolTalk is a large-scale conversational dataset from HuggingFace.
Used for supervised fine-tuning on multi-turn conversations.
"""

from datasets import load_dataset

from tasks.common import Task


class SmolTalk(Task):
    """
    SmolTalk conversational dataset.

    Each example is a multi-turn conversation between user and assistant.
    Primarily used for SFT (supervised fine-tuning) rather than evaluation.
    """

    def __init__(self, split="train", start=None, stop=None, step=None, subset="all"):
        """
        Args:
            split: Dataset split ('train', 'test')
            start: Start index for slicing
            stop: Stop index for slicing
            step: Step size for slicing
            subset: Dataset subset/config name
        """
        super().__init__(start=start, stop=stop, step=step)
        self.dataset = load_dataset("HuggingFaceTB/smoltalk", subset, split=split)

    @property
    def num_examples(self):
        return len(self.dataset)

    def get_example(self, index):
        """
        Get a SmolTalk conversation example.

        Returns:
            dict with:
                - conversation: the multi-turn conversation
                - ideal: the last assistant message (for evaluation)
        """
        item = self.dataset[index]
        messages = item["messages"]

        # Ensure messages have the right format
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        conversation = {
            "messages": formatted_messages,
        }

        # The ideal is the last assistant response
        ideal = None
        for msg in reversed(formatted_messages):
            if msg["role"] == "assistant":
                ideal = msg["content"]
                break

        return {
            "conversation": conversation,
            "ideal": ideal,
        }

    def evaluate(self, index, output):
        """
        Evaluate model output for SmolTalk.
        SmolTalk doesn't have a rigorous evaluation metric;
        returns a placeholder score.

        Args:
            index: Sliced index
            output: Model's generated text

        Returns:
            dict with 'score' (always 0 for now, since this is a training task)
        """
        return {
            "correct": False,
            "score": 0.0,
            "note": "SmolTalk is a training task; evaluation is not meaningful.",
        }
