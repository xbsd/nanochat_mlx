"""
BPB (Bits Per Byte) evaluation ported to MLX.
Ported from nanochat/loss_eval.py.

Key changes from PyTorch original:
- Replace torch operations with mx equivalents
- Remove distributed communication (dist.barrier, dist.all_reduce)
- Use mx.array for total_nats and total_bytes
- model(x, y, loss_reduction='none') returns per-token losses

BPB measures the model's compression efficiency in bits per byte of text,
providing a tokenizer-agnostic metric of language modeling quality.
"""

import time
import logging
import math
import numpy as np

import mlx.core as mx

from nanochat_mlx.common import print0

logger = logging.getLogger(__name__)


def compute_bpb(
    model,
    data_loader,
    token_bytes,
    max_steps=None,
):
    """
    Compute Bits Per Byte (BPB) on a dataset.

    BPB = total_nats / (total_bytes * ln(2))

    This measures how many bits the model needs per byte of text,
    providing a tokenizer-independent metric.

    Args:
        model: The language model. Should be callable as model(input_ids)
               returning logits of shape (batch, seq_len, vocab_size).
        data_loader: Iterator yielding (input_ids, targets) batches.
                     input_ids: mx.array of shape (batch, seq_len)
                     targets: mx.array of shape (batch, seq_len) or None
        token_bytes: mx.array of shape (vocab_size,) giving the byte length
                     of each token.
        max_steps: Maximum number of batches to evaluate (None = all)

    Returns:
        dict with:
            - 'bpb': bits per byte
            - 'loss': mean cross-entropy loss in nats
            - 'total_nats': total nats accumulated
            - 'total_bytes': total bytes accumulated
            - 'num_steps': number of batches processed
            - 'num_tokens': total number of tokens processed
    """
    total_nats = 0.0
    total_bytes = 0.0
    total_tokens = 0
    num_steps = 0

    t0 = time.time()

    for batch in data_loader:
        if max_steps is not None and num_steps >= max_steps:
            break

        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                input_ids, targets = batch
            else:
                input_ids = batch[0]
                targets = None
        else:
            input_ids = batch
            targets = None

        # If targets not provided, use next-token prediction
        if targets is None:
            targets = input_ids[:, 1:]
            input_ids_for_model = input_ids[:, :-1]
        else:
            input_ids_for_model = input_ids

        # Forward pass
        logits = model(input_ids_for_model)  # (batch, seq_len, vocab_size)

        # Compute per-token cross-entropy loss
        batch_size, seq_len, vocab_size = logits.shape

        # Manual cross-entropy
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        # Gather log probs for target tokens
        # targets shape: (batch, seq_len)
        target_log_probs = mx.take_along_axis(
            log_probs.reshape(-1, vocab_size),
            targets.reshape(-1, 1),
            axis=-1,
        ).squeeze(-1)  # (batch * seq_len,)

        per_token_nats = -target_log_probs  # (batch * seq_len,)
        per_token_nats = per_token_nats.reshape(batch_size, seq_len)

        # Compute bytes per token using the token_bytes lookup
        # targets contains the token IDs whose byte counts we need
        targets_np = np.array(targets)
        token_bytes_np = np.array(token_bytes)

        # Clamp indices to valid range
        targets_clamped = np.clip(targets_np, 0, len(token_bytes_np) - 1)
        per_token_bytes = token_bytes_np[targets_clamped]  # (batch, seq_len)

        # Accumulate
        per_token_nats_np = np.array(per_token_nats)
        batch_nats = float(per_token_nats_np.sum())
        batch_bytes = float(per_token_bytes.sum())
        batch_tokens = int(np.prod(targets_np.shape))

        total_nats += batch_nats
        total_bytes += batch_bytes
        total_tokens += batch_tokens
        num_steps += 1

        # Log progress periodically
        if num_steps % 50 == 0:
            elapsed = time.time() - t0
            current_bpb = total_nats / (total_bytes * math.log(2)) if total_bytes > 0 else float('inf')
            logger.info(
                f"BPB eval step {num_steps}: "
                f"bpb={current_bpb:.4f}, "
                f"tokens={total_tokens}, "
                f"{elapsed:.1f}s elapsed"
            )

    # Final computation
    if total_bytes > 0:
        bpb = total_nats / (total_bytes * math.log(2))
    else:
        bpb = float('inf')

    mean_loss = total_nats / total_tokens if total_tokens > 0 else float('inf')

    elapsed = time.time() - t0
    print0(f"BPB evaluation completed: {num_steps} steps in {elapsed:.1f}s")
    print0(f"  BPB: {bpb:.4f}")
    print0(f"  Mean loss (nats): {mean_loss:.4f}")
    print0(f"  Total tokens: {total_tokens}")
    print0(f"  Total bytes: {total_bytes:.0f}")

    return {
        "bpb": bpb,
        "loss": mean_loss,
        "total_nats": total_nats,
        "total_bytes": total_bytes,
        "num_steps": num_steps,
        "num_tokens": total_tokens,
    }


def evaluate_loss(
    model,
    data_loader,
    max_steps=None,
):
    """
    Evaluate plain cross-entropy loss (without BPB calculation).

    Simpler alternative when token_bytes is not available.

    Args:
        model: The language model
        data_loader: Iterator yielding batches
        max_steps: Maximum number of batches

    Returns:
        dict with:
            - 'loss': mean cross-entropy loss in nats
            - 'perplexity': exp(loss)
            - 'num_steps': number of batches processed
            - 'num_tokens': total tokens processed
    """
    total_loss = 0.0
    total_tokens = 0
    num_steps = 0

    t0 = time.time()

    for batch in data_loader:
        if max_steps is not None and num_steps >= max_steps:
            break

        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                input_ids, targets = batch
            else:
                input_ids = batch[0]
                targets = None
        else:
            input_ids = batch
            targets = None

        if targets is None:
            targets = input_ids[:, 1:]
            input_ids_for_model = input_ids[:, :-1]
        else:
            input_ids_for_model = input_ids

        # Forward pass
        logits = model(input_ids_for_model)

        # Compute per-token cross-entropy
        batch_size, seq_len, vocab_size = logits.shape
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        target_log_probs = mx.take_along_axis(
            log_probs.reshape(-1, vocab_size),
            targets.reshape(-1, 1),
            axis=-1,
        ).squeeze(-1)

        per_token_loss = -target_log_probs
        mx.eval(per_token_loss)

        batch_loss = float(np.array(per_token_loss).sum())
        batch_tokens = int(np.prod(np.array(targets).shape))

        total_loss += batch_loss
        total_tokens += batch_tokens
        num_steps += 1

        if num_steps % 50 == 0:
            elapsed = time.time() - t0
            current_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
            logger.info(
                f"Loss eval step {num_steps}: "
                f"loss={current_loss:.4f}, "
                f"tokens={total_tokens}, "
                f"{elapsed:.1f}s elapsed"
            )

    mean_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(mean_loss) if mean_loss < 100 else float('inf')

    elapsed = time.time() - t0
    print0(f"Loss evaluation completed: {num_steps} steps in {elapsed:.1f}s")
    print0(f"  Mean loss: {mean_loss:.4f}")
    print0(f"  Perplexity: {perplexity:.2f}")
    print0(f"  Total tokens: {total_tokens}")

    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "num_steps": num_steps,
        "num_tokens": total_tokens,
    }
