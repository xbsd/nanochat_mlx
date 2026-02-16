"""
Supervised fine-tuning (SFT) script for nanochat_mlx.
Ported from nanochat/scripts/chat_sft.py (PyTorch) to Apple MLX.

Key differences from PyTorch version:
- nn.value_and_grad + manual gradient accumulation instead of torch autograd
- mx.eval() after each micro-step for lazy evaluation
- No DDP, no FP8, no autocast
- Memory tracking via psutil instead of torch.cuda
- Single device (Apple Silicon unified memory)

Loads a pretrained base model checkpoint and fine-tunes on an SFT data mixture
(SmolTalk, MMLU, GSM8K, SpellingBee, SimpleSpelling).

Uses BOS-aligned bestfit packing with padding (targets masked with -1).
LR schedule: constant for 80%, then linear ramp to 0.
Training ends after one epoch through the dataset (last_step flag).
"""

import os
import sys
import time
import math
import argparse
import random

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from nanochat_mlx.common import mlx_init, print0, DummyWandb, get_base_dir, get_memory_usage
from nanochat_mlx.tokenizer import get_tokenizer, get_token_bytes
from nanochat_mlx.checkpoint_manager import save_checkpoint, load_model
from nanochat_mlx.optim import MuonAdamW
from nanochat_mlx.gpt import GPT, GPTConfig

from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk

from tasks.spellingbee import SimpleSpelling, SpellingBee


# ---------------------------------------------------------------------------
# SFT Dataloader: BOS-aligned bestfit packing with padding
# ---------------------------------------------------------------------------

def sft_data_generator_bos_bestfit(tokenizer, B, T, split, buffer_size=100):
    """
    Generates (inputs, targets) batches from the SFT task mixture.

    Uses bestfit packing with conversations from TaskMixture.
    Each row starts with BOS. Conversations are packed greedily into rows
    using best-fit, with padding positions masked with -1 in targets.

    Args:
        tokenizer: The tokenizer (must support render_conversation)
        B: Batch size
        T: Sequence length
        split: 'train' or 'val'
        buffer_size: Number of conversations to buffer for bestfit packing

    Yields:
        inputs: mx.array (B, T) int32
        targets: mx.array (B, T) int32, padding positions masked with -1
    """
    # Build the SFT task mixture
    tasks = [
        ("SmolTalk", SmolTalk(split=split)),
        ("MMLU", MMLU(split="auxiliary_train" if split == "train" else "test")),
        ("GSM8K", GSM8K(split="train" if split == "train" else "test")),
        ("SpellingBee", SpellingBee()),
        ("SimpleSpelling", SimpleSpelling()),
    ]
    mixture = TaskMixture(tasks, seed=42)
    total_examples = len(mixture)

    # Track how many examples we have consumed for epoch detection
    example_idx = 0

    row_capacity = T + 1  # +1 because inputs=row[:-1], targets=row[1:]

    # Pre-tokenize conversations into a buffer of (tokens, mask) pairs
    # tokens: list of ints, mask: list of 0/1 where 1 = train-on (assistant output)
    doc_buffer = []  # list of (tokens, mask) tuples

    def refill_buffer():
        nonlocal example_idx
        while len(doc_buffer) < buffer_size and example_idx < total_examples:
            example = mixture[example_idx]
            example_idx += 1
            conversation = example["conversation"]
            try:
                ids, mask = tokenizer.render_conversation(conversation, max_tokens=row_capacity)
            except (AssertionError, KeyError, ValueError):
                # Skip malformed conversations
                continue
            if len(ids) > 1:
                doc_buffer.append((ids, mask))

    # Pre-allocate row buffers
    # For SFT packing, we need to track which tokens are padding vs real
    # rows_tokens[b] = list of token ids (length row_capacity)
    # rows_mask[b] = list of 0/1 mask values (length row_capacity), 0 = don't train

    pad_token = 0  # use token 0 as padding (will be masked out in targets)

    while True:
        # Initialize rows with padding
        rows_tokens = [[pad_token] * row_capacity for _ in range(B)]
        rows_mask = [[0] * row_capacity for _ in range(B)]

        last_step = False

        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                refill_buffer()

                if len(doc_buffer) == 0:
                    # Dataset exhausted - we've gone through one epoch
                    last_step = True
                    break

                remaining = row_capacity - pos

                # Find largest doc that fits entirely (best-fit)
                best_idx = -1
                best_len = 0
                for i, (doc_ids, doc_mask) in enumerate(doc_buffer):
                    doc_len = len(doc_ids)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    # Place the best-fitting document
                    doc_ids, doc_mask = doc_buffer.pop(best_idx)
                    doc_len = len(doc_ids)
                    rows_tokens[row_idx][pos:pos + doc_len] = doc_ids
                    rows_mask[row_idx][pos:pos + doc_len] = doc_mask
                    pos += doc_len
                else:
                    # No doc fits entirely - crop the shortest one to fill remaining
                    if len(doc_buffer) > 0:
                        shortest_idx = min(range(len(doc_buffer)),
                                           key=lambda i: len(doc_buffer[i][0]))
                        doc_ids, doc_mask = doc_buffer.pop(shortest_idx)
                        rows_tokens[row_idx][pos:pos + remaining] = doc_ids[:remaining]
                        rows_mask[row_idx][pos:pos + remaining] = doc_mask[:remaining]
                    pos = row_capacity  # row is full

            if last_step:
                break

        # Convert to mx.array
        # batch_tokens: (B, T+1)
        batch_tokens = mx.array(rows_tokens, dtype=mx.int32)
        batch_mask = mx.array(rows_mask, dtype=mx.int32)

        # inputs = tokens[:, :-1], targets = tokens[:, 1:]
        inputs = batch_tokens[:, :-1]    # (B, T)
        targets = batch_tokens[:, 1:]    # (B, T)
        mask = batch_mask[:, 1:]         # (B, T) - mask for targets

        # Mask out padding and non-trainable positions in targets with -1
        # Only train on positions where mask == 1
        targets = mx.where(mask == 1, targets, mx.array(-1, dtype=mx.int32))

        yield inputs, targets, last_step

        if last_step:
            return


# ---------------------------------------------------------------------------
# LR Schedule: constant for 80%, then linear ramp to 0
# ---------------------------------------------------------------------------

def get_lr_multiplier(step, total_steps):
    """
    LR schedule for SFT:
    - Constant LR (multiplier 1.0) for the first 80% of training
    - Linear decay from 1.0 to 0.0 over the last 20%
    """
    ramp_start = int(0.8 * total_steps)
    if step < ramp_start:
        return 1.0
    else:
        # Linear ramp from 1.0 at ramp_start to 0.0 at total_steps
        progress = (step - ramp_start) / max(total_steps - ramp_start, 1)
        return max(0.0, 1.0 - progress)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NanoChat MLX SFT Training")

    # Model / checkpoint
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained base model checkpoint directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for SFT checkpoints (default: runs/chat_sft_<timestamp>)")

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Micro-batch size per step")
    parser.add_argument("--sequence-length", type=int, default=2048,
                        help="Sequence length for training")
    parser.add_argument("--grad-accum-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override base learning rate (default: use model's setup_optimizer defaults)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for Muon groups")

    # Evaluation
    parser.add_argument("--eval-every", type=int, default=100,
                        help="Evaluate every N optimizer steps")
    parser.add_argument("--save-every", type=int, default=500,
                        help="Save checkpoint every N optimizer steps")

    # Logging
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="nanochat-mlx-sft",
                        help="wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="wandb run name")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--estimated-total-steps", type=int, default=0,
                        help="Estimated total optimizer steps (for LR schedule); "
                             "0 = estimate from dataset size")

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    mlx_init(seed=args.seed)
    random.seed(args.seed)

    # Setup output directory
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("runs", f"chat_sft_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    print0(f"Output directory: {args.output_dir}")

    # -----------------------------------------------------------------------
    # Load pretrained model
    # -----------------------------------------------------------------------
    print0(f"Loading pretrained model from: {args.checkpoint}")
    model, config = load_model(args.checkpoint)
    print0(f"Model config: {config}")

    B = args.batch_size
    T = args.sequence_length
    grad_accum_steps = args.grad_accum_steps
    effective_batch_size = B * grad_accum_steps

    print0(f"Micro-batch size: {B}")
    print0(f"Sequence length: {T}")
    print0(f"Gradient accumulation steps: {grad_accum_steps}")
    print0(f"Effective batch size: {effective_batch_size}")

    # -----------------------------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------------------------
    tokenizer = get_tokenizer()
    print0(f"Vocab size: {tokenizer.get_vocab_size()}")

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------
    opt_config = model.setup_optimizer(weight_decay=args.weight_decay)
    param_groups = opt_config["param_groups"]

    # Override LR if requested
    if args.lr is not None:
        for group in param_groups:
            group["lr"] = args.lr

    optimizer = MuonAdamW(param_groups)

    # Count parameters
    from mlx.utils import tree_flatten
    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print0(f"Total parameters: {total_params:,}")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    print0("Setting up SFT data mixture...")
    train_gen = sft_data_generator_bos_bestfit(tokenizer, B, T, split="train")

    # Estimate total steps if not provided
    # Build the mixture briefly just to get the length
    if args.estimated_total_steps <= 0:
        tmp_tasks = [
            ("SmolTalk", SmolTalk(split="train")),
            ("MMLU", MMLU(split="auxiliary_train")),
            ("GSM8K", GSM8K(split="train")),
            ("SpellingBee", SpellingBee()),
            ("SimpleSpelling", SimpleSpelling()),
        ]
        tmp_mixture = TaskMixture(tmp_tasks, seed=42)
        total_conversations = len(tmp_mixture)
        # Rough estimate: average ~300 tokens per conversation, pack into B*T
        avg_tokens_per_conv = 300
        tokens_per_step = B * T * grad_accum_steps
        total_steps_estimate = max(1, (total_conversations * avg_tokens_per_conv) // tokens_per_step)
        del tmp_mixture, tmp_tasks
        print0(f"Estimated total optimizer steps: {total_steps_estimate} "
               f"(from {total_conversations} conversations)")
    else:
        total_steps_estimate = args.estimated_total_steps
        print0(f"Using provided total steps estimate: {total_steps_estimate}")

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
    else:
        wandb_run = DummyWandb()

    # -----------------------------------------------------------------------
    # Define loss function for value_and_grad
    # -----------------------------------------------------------------------
    def loss_fn(model, inputs, targets):
        """Compute cross-entropy loss with -1 masking."""
        loss = model(inputs, targets=targets, loss_reduction="mean")
        return loss

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print0("Starting SFT training...")
    print0("=" * 80)

    step = 0
    tokens_seen = 0
    t0 = time.time()
    last_step = False

    while not last_step:
        step += 1
        step_t0 = time.time()

        # Update learning rate
        lr_mult = get_lr_multiplier(step, total_steps_estimate)
        optimizer.update_lr(lr_mult)

        # ---- Gradient accumulation loop ----
        # Same pattern as base_train.py: accumulate, then divide
        accumulated_grads = None
        total_loss = 0.0

        for micro_step in range(grad_accum_steps):
            try:
                inputs, targets, last_step_flag = next(train_gen)
            except StopIteration:
                last_step = True
                break

            if last_step_flag:
                last_step = True

            # Forward + backward
            loss_val, grads = loss_and_grad_fn(model, inputs, targets)
            # CRITICAL: evaluate to prevent computation graph explosion
            mx.eval(loss_val)
            total_loss += loss_val.item()

            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(lambda a, b: a + b,
                                             accumulated_grads, grads)

            tokens_seen += B * T

        if accumulated_grads is None:
            break

        # Average gradients over accumulation steps
        if grad_accum_steps > 1:
            accumulated_grads = tree_map(
                lambda g: g * (1.0 / grad_accum_steps), accumulated_grads
            )
        accumulated_loss = total_loss / grad_accum_steps

        # ---- Optimizer step ----
        optimizer.update(model, accumulated_grads)
        # CRITICAL: evaluate all updated parameters and optimizer state
        mx.eval(model.parameters(), optimizer.state)

        step_dt = time.time() - step_t0
        elapsed = time.time() - t0

        # ---- Logging ----
        tokens_per_sec = (B * T * grad_accum_steps) / step_dt if step_dt > 0 else 0
        current_lr = optimizer.param_groups[0]["lr"]
        mem_bytes = get_memory_usage()
        mem_gb = mem_bytes / (1024 ** 3)

        if step % 10 == 0 or step == 1 or last_step:
            print0(
                f"step {step:6d} | "
                f"loss {accumulated_loss:.4f} | "
                f"lr {current_lr:.6f} (mult={lr_mult:.3f}) | "
                f"tokens/s {tokens_per_sec:,.0f} | "
                f"mem {mem_gb:.1f} GB | "
                f"elapsed {elapsed:.1f}s"
            )

        wandb_run.log({
            "train/loss": accumulated_loss,
            "train/lr": current_lr,
            "train/lr_mult": lr_mult,
            "train/tokens_per_sec": tokens_per_sec,
            "train/tokens_seen": tokens_seen,
            "train/step": step,
            "train/memory_gb": mem_gb,
        })

        # ---- Evaluation ----
        if step % args.eval_every == 0 or last_step:
            print0(f"Running evaluation at step {step}...")
            eval_gen = sft_data_generator_bos_bestfit(
                tokenizer, B, T, split="val", buffer_size=50
            )

            eval_losses = []
            num_eval_batches = 10
            for eval_step in range(num_eval_batches):
                try:
                    eval_inputs, eval_targets, _ = next(eval_gen)
                except StopIteration:
                    break
                eval_loss = model(eval_inputs, targets=eval_targets, loss_reduction="mean")
                mx.eval(eval_loss)
                eval_losses.append(eval_loss.item())

            if eval_losses:
                avg_eval_loss = sum(eval_losses) / len(eval_losses)
                print0(f"  eval loss: {avg_eval_loss:.4f} (over {len(eval_losses)} batches)")
                wandb_run.log({
                    "eval/loss": avg_eval_loss,
                    "eval/step": step,
                })

        # ---- Save checkpoint ----
        if step % args.save_every == 0 or last_step:
            ckpt_dir = os.path.join(args.output_dir, f"step_{step:06d}")
            print0(f"Saving checkpoint to {ckpt_dir}")
            save_checkpoint(
                model=model,
                config=config,
                path=ckpt_dir,
                step=step,
                metadata={
                    "sft_step": step,
                    "tokens_seen": tokens_seen,
                    "loss": accumulated_loss,
                    "lr_mult": lr_mult,
                },
            )

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    total_time = time.time() - t0
    print0("=" * 80)
    print0(f"SFT training complete!")
    print0(f"  Total steps: {step}")
    print0(f"  Total tokens: {tokens_seen:,}")
    print0(f"  Total time: {total_time:.1f}s")
    print0(f"  Avg tokens/s: {tokens_seen / total_time:,.0f}")
    print0(f"  Final loss: {accumulated_loss:.4f}")

    # Save final checkpoint
    final_dir = os.path.join(args.output_dir, "final")
    print0(f"Saving final checkpoint to {final_dir}")
    save_checkpoint(
        model=model,
        config=config,
        path=final_dir,
        step=step,
        metadata={
            "sft_step": step,
            "tokens_seen": tokens_seen,
            "loss": accumulated_loss,
            "is_final": True,
        },
    )

    wandb_run.finish()
    print0("Done.")


if __name__ == "__main__":
    main()
