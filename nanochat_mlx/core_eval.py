"""
CORE metric evaluation ported to MLX.
Ported from nanochat/core_eval.py.

Key changes from PyTorch original:
- Replace torch.tensor with mx.array
- Replace torch.full with mx.full
- Replace torch.roll with array slicing
- Replace torch.nn.functional.cross_entropy with manual computation using MLX
- Remove dist.barrier() and dist.all_reduce() (single device)
- forward_model returns losses and predictions using MLX ops
- stack_sequences uses mx.full and mx.array
"""

import time
import logging
import numpy as np

import mlx.core as mx
import mlx.nn as nn

from nanochat_mlx.common import print0

logger = logging.getLogger(__name__)


def stack_sequences(token_ids_list, mask_list, pad_token_id=0, max_seq_len=None):
    """
    Stack variable-length sequences into padded batches.

    Args:
        token_ids_list: List of token ID lists (variable length)
        mask_list: List of mask lists (same lengths as token_ids_list)
        pad_token_id: Token ID to use for padding
        max_seq_len: Maximum sequence length (None = use longest)

    Returns:
        (input_ids, masks) as mx.array tensors of shape (batch, seq_len)
    """
    batch_size = len(token_ids_list)

    if max_seq_len is None:
        max_seq_len = max(len(ids) for ids in token_ids_list)

    # Create padded arrays
    input_ids = mx.full((batch_size, max_seq_len), pad_token_id, dtype=mx.int32)
    masks = mx.zeros((batch_size, max_seq_len), dtype=mx.int32)

    # Fill in the actual values using numpy for efficient construction
    input_ids_np = np.full((batch_size, max_seq_len), pad_token_id, dtype=np.int32)
    masks_np = np.zeros((batch_size, max_seq_len), dtype=np.int32)

    for i, (ids, mask) in enumerate(zip(token_ids_list, mask_list)):
        seq_len = min(len(ids), max_seq_len)
        input_ids_np[i, :seq_len] = ids[:seq_len]
        masks_np[i, :seq_len] = mask[:seq_len]

    input_ids = mx.array(input_ids_np)
    masks = mx.array(masks_np)

    return input_ids, masks


def cross_entropy_loss(logits, targets, reduction='none'):
    """
    Compute cross-entropy loss using MLX operations.
    Replaces torch.nn.functional.cross_entropy.

    Args:
        logits: (batch, seq_len, vocab_size) or (seq_len, vocab_size) float array
        targets: (batch, seq_len) or (seq_len,) int array
        reduction: 'none' returns per-element losses, 'mean' returns mean

    Returns:
        Loss array. Shape depends on reduction:
            - 'none': same shape as targets
            - 'mean': scalar
    """
    # Use MLX's built-in cross entropy
    # nn.losses.cross_entropy expects logits and targets
    if logits.ndim == 3:
        # (batch, seq_len, vocab_size) -> reshape for computation
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        # Manual cross entropy: -log_softmax[target]
        log_probs = logits_flat - mx.logsumexp(logits_flat, axis=-1, keepdims=True)
        # Gather the log probs for the target tokens
        losses = -mx.take_along_axis(log_probs, targets_flat[:, None], axis=-1).squeeze(-1)
        losses = losses.reshape(batch_size, seq_len)
    elif logits.ndim == 2:
        # (seq_len, vocab_size)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        losses = -mx.take_along_axis(log_probs, targets[:, None], axis=-1).squeeze(-1)
    else:
        raise ValueError(f"Expected 2D or 3D logits, got {logits.ndim}D")

    if reduction == 'mean':
        return mx.mean(losses)
    elif reduction == 'none':
        return losses
    elif reduction == 'sum':
        return mx.sum(losses)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def forward_model(model, input_ids):
    """
    Run forward pass and compute per-token losses and predictions.

    The model is expected to return logits of shape (batch, seq_len, vocab_size).
    We compute cross-entropy loss for predicting the next token:
        - inputs:  tokens[:-1]
        - targets: tokens[1:]

    Args:
        model: The language model (callable, returns logits)
        input_ids: mx.array of shape (batch, seq_len) with token IDs

    Returns:
        losses: mx.array of shape (batch, seq_len-1) per-token losses
        predictions: mx.array of shape (batch, seq_len-1) predicted token IDs
    """
    # Forward pass
    logits = model(input_ids)  # (batch, seq_len, vocab_size)

    # Shift: predict token[t+1] from token[t]
    # logits for positions 0..T-2 predict tokens at positions 1..T-1
    shift_logits = logits[:, :-1, :]   # (batch, seq_len-1, vocab_size)
    shift_targets = input_ids[:, 1:]    # (batch, seq_len-1)

    # Compute per-token cross-entropy loss
    losses = cross_entropy_loss(shift_logits, shift_targets, reduction='none')

    # Get predictions (argmax of logits)
    predictions = mx.argmax(shift_logits, axis=-1)

    # Evaluate to materialize the computation graph
    mx.eval(losses, predictions)

    return losses, predictions


def evaluate_example(model, tokenizer, example, max_seq_len=2048):
    """
    Evaluate a single example using the CORE metric.

    The CORE metric measures:
    1. Loss on masked (assistant) tokens
    2. Accuracy on masked tokens
    3. Whether the example is fully correct

    Args:
        model: The language model
        tokenizer: The tokenizer with render_conversation method
        example: Dict with 'conversation' key
        max_seq_len: Maximum sequence length

    Returns:
        dict with:
            - 'loss': mean loss on masked tokens
            - 'accuracy': fraction of masked tokens predicted correctly
            - 'correct': whether all masked tokens are correct
            - 'num_tokens': number of masked tokens
            - 'total_loss': sum of losses on masked tokens
    """
    conversation = example["conversation"]

    # Render the conversation to token IDs and mask
    ids, mask = tokenizer.render_conversation(conversation, max_tokens=max_seq_len)

    if len(ids) < 2:
        return {
            "loss": float('inf'),
            "accuracy": 0.0,
            "correct": False,
            "num_tokens": 0,
            "total_loss": 0.0,
        }

    # Stack into batch of 1
    input_ids, masks = stack_sequences([ids], [mask], max_seq_len=max_seq_len)

    # Forward pass
    losses, predictions = forward_model(model, input_ids)

    # The mask for loss computation: shift mask by 1 to align with losses
    # mask[t] tells us if token[t] is a target token
    # losses[t] is the loss for predicting token[t+1] from token[t]
    # So we need mask[1:] to know which losses to count
    shift_mask = masks[:, 1:]  # (batch, seq_len-1)

    # Convert to numpy for computation
    losses_np = np.array(losses)
    preds_np = np.array(predictions)
    targets_np = np.array(input_ids[:, 1:])
    mask_np = np.array(shift_mask)

    # Compute masked metrics
    masked_losses = losses_np[mask_np == 1]
    masked_preds = preds_np[mask_np == 1]
    masked_targets = targets_np[mask_np == 1]

    num_tokens = len(masked_losses)
    if num_tokens == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "correct": True,
            "num_tokens": 0,
            "total_loss": 0.0,
        }

    total_loss = float(masked_losses.sum())
    mean_loss = total_loss / num_tokens
    correct_tokens = (masked_preds == masked_targets).sum()
    accuracy = float(correct_tokens) / num_tokens
    all_correct = bool(correct_tokens == num_tokens)

    return {
        "loss": mean_loss,
        "accuracy": accuracy,
        "correct": all_correct,
        "num_tokens": num_tokens,
        "total_loss": total_loss,
    }


def evaluate_task(model, tokenizer, task, max_seq_len=2048, max_examples=None, batch_size=1):
    """
    Evaluate a task using the CORE metric.

    Iterates over all examples in the task, computes per-example metrics,
    and aggregates them into task-level statistics.

    Args:
        model: The language model
        tokenizer: The tokenizer
        task: A Task object (or any object with __len__ and __getitem__)
        max_seq_len: Maximum sequence length
        max_examples: Maximum number of examples to evaluate (None = all)
        batch_size: Number of examples per batch (currently only 1 is fully supported)

    Returns:
        dict with:
            - 'mean_loss': average loss across all masked tokens
            - 'mean_accuracy': average accuracy across examples
            - 'fraction_correct': fraction of examples fully correct
            - 'num_examples': number of examples evaluated
            - 'total_tokens': total number of masked tokens
            - 'examples': list of per-example results
    """
    num_examples = len(task)
    if max_examples is not None:
        num_examples = min(num_examples, max_examples)

    results = []
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    t0 = time.time()

    for i in range(num_examples):
        example = task[i]
        result = evaluate_example(model, tokenizer, example, max_seq_len=max_seq_len)
        results.append(result)

        total_loss += result["total_loss"]
        total_tokens += result["num_tokens"]
        if result["correct"]:
            total_correct += 1

        # Log progress periodically
        if (i + 1) % 100 == 0 or (i + 1) == num_examples:
            elapsed = time.time() - t0
            examples_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                f"CORE eval: {i + 1}/{num_examples} examples, "
                f"{examples_per_sec:.1f} examples/sec"
            )

    # Aggregate metrics
    mean_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    mean_accuracy = (
        sum(r["accuracy"] for r in results) / len(results) if results else 0.0
    )
    fraction_correct = total_correct / len(results) if results else 0.0

    elapsed = time.time() - t0
    print0(f"CORE evaluation completed: {num_examples} examples in {elapsed:.1f}s")
    print0(f"  Mean loss: {mean_loss:.4f}")
    print0(f"  Mean accuracy: {mean_accuracy:.4f}")
    print0(f"  Fraction correct: {fraction_correct:.4f}")
    print0(f"  Total tokens evaluated: {total_tokens}")

    return {
        "mean_loss": mean_loss,
        "mean_accuracy": mean_accuracy,
        "fraction_correct": fraction_correct,
        "num_examples": num_examples,
        "total_tokens": total_tokens,
        "examples": results,
    }


def evaluate_batch(model, tokenizer, examples, max_seq_len=2048):
    """
    Evaluate a batch of examples together for efficiency.

    Args:
        model: The language model
        tokenizer: The tokenizer
        examples: List of example dicts
        max_seq_len: Maximum sequence length

    Returns:
        List of per-example result dicts
    """
    if not examples:
        return []

    # Render all conversations
    all_ids = []
    all_masks = []
    for example in examples:
        ids, mask = tokenizer.render_conversation(
            example["conversation"], max_tokens=max_seq_len
        )
        all_ids.append(ids)
        all_masks.append(mask)

    # Stack into a batch
    input_ids, masks = stack_sequences(all_ids, all_masks, max_seq_len=max_seq_len)

    # Forward pass
    losses, predictions = forward_model(model, input_ids)

    # Shift mask
    shift_mask = masks[:, 1:]

    # Convert to numpy
    losses_np = np.array(losses)
    preds_np = np.array(predictions)
    targets_np = np.array(input_ids[:, 1:])
    mask_np = np.array(shift_mask)

    # Compute per-example metrics
    results = []
    for i in range(len(examples)):
        m = mask_np[i] == 1
        num_tokens = int(m.sum())

        if num_tokens == 0:
            results.append({
                "loss": 0.0,
                "accuracy": 0.0,
                "correct": True,
                "num_tokens": 0,
                "total_loss": 0.0,
            })
            continue

        masked_losses = losses_np[i][m]
        masked_preds = preds_np[i][m]
        masked_targets = targets_np[i][m]

        total_loss = float(masked_losses.sum())
        mean_loss = total_loss / num_tokens
        correct_tokens = int((masked_preds == masked_targets).sum())
        accuracy = correct_tokens / num_tokens
        all_correct = correct_tokens == num_tokens

        results.append({
            "loss": mean_loss,
            "accuracy": accuracy,
            "correct": bool(all_correct),
            "num_tokens": num_tokens,
            "total_loss": total_loss,
        })

    return results
