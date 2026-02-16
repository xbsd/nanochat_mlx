#!/usr/bin/env python3
"""
NanoChat MLX base pretraining script.
Ported from the original PyTorch base_train.py to Apple MLX.

Key differences from the PyTorch version:
- No torch.compile, DDP, FP8, meta device, or autocast
- Gradient accumulation uses nn.value_and_grad with explicit graph eval
- Memory monitoring via psutil instead of torch.cuda
- Single device only (no multi-GPU)

Usage:
    python scripts/base_train.py [options]

Examples:
    # Quick test run
    python scripts/base_train.py --num_iterations 100 --batch_size 4

    # Full training run with wandb
    python scripts/base_train.py --wandb_project nanochat --wandb_run_name base_v1
"""

import os
import sys
import math
import time
import json
import argparse
import platform
import subprocess

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from nanochat_mlx.gpt import GPT, GPTConfig
from nanochat_mlx.dataloader import (
    tokenizing_data_loader_with_state_bos_bestfit,
    tokenizing_data_loader_bos_bestfit,
)
from nanochat_mlx.common import (
    mlx_init,
    print0,
    DummyWandb,
    print_banner,
    get_base_dir,
    get_memory_usage,
    get_peak_flops,
)
from nanochat_mlx.tokenizer import get_tokenizer, get_token_bytes
from nanochat_mlx.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat_mlx.optim import MuonAdamW, build_param_groups

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="NanoChat MLX base pretraining")

    # Model architecture
    parser.add_argument("--model_name", type=str, default="", help="Model name for logging")
    parser.add_argument("--n_embd", type=int, default=768, help="Model dimension")
    parser.add_argument("--n_layer", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--n_kv_head", type=int, default=6, help="Number of KV heads (for GQA)")
    parser.add_argument("--depth", type=int, default=None, help="Alias for --n_layer")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, help="Micro-batch size per step")
    parser.add_argument("--device-batch-size", type=int, default=None, help="Alias for --batch_size")
    parser.add_argument("--sequence_length", type=int, default=2048, help="Sequence length T")
    parser.add_argument("--num_iterations", type=int, default=4578, help="Total training steps")
    parser.add_argument("--warmup_iters", type=int, default=0, help="LR warmup steps")
    parser.add_argument("--warmdown_iters", type=int, default=1450, help="LR warmdown steps")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (for Muon)")
    parser.add_argument("--lr_scale", type=float, default=1.0, help="Global LR scale for AdamW groups")

    # Gradient accumulation
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Number of micro-steps for gradient accumulation")

    # Target tokens for scaling law calculation
    parser.add_argument("--target_tokens", type=float, default=None,
                        help="Target total tokens (e.g. 5.0e9). Overrides num_iterations.")

    # Evaluation
    parser.add_argument("--eval-every", type=int, default=250, help="Evaluate val loss every N steps")
    parser.add_argument("--val_tokens", type=int, default=10485760, help="Tokens for val BPB evaluation")
    parser.add_argument("--sample_every", type=int, default=0, help="Generate sample every N steps (0=disabled)")

    # Checkpointing
    parser.add_argument("--save-every", type=int, default=0, help="Save checkpoint every N steps (0=disabled)")
    parser.add_argument("--save_best", action="store_true", default=False,
                        help="Save checkpoint on best val loss")
    parser.add_argument("--run", type=str, default="base_run",
                        help="Run name for checkpoint directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint directory to resume from")

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="", help="Wandb project name (empty=disabled)")
    parser.add_argument("--wandb_run_name", type=str, default="", help="Wandb run name")

    # System
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of tokenizer threads for dataloader")
    parser.add_argument("--tokenizer_batch_size", type=int, default=128,
                        help="Batch size for tokenizer in dataloader")

    args = parser.parse_args()

    # Handle aliases
    if args.depth is not None:
        args.n_layer = args.depth
    if getattr(args, 'device_batch_size', None) is not None:
        args.batch_size = args.device_batch_size

    return args

# -----------------------------------------------------------------------------
# Loss function
# -----------------------------------------------------------------------------

def compute_loss(model, x, y):
    """Compute cross-entropy loss. The model handles softcapping internally."""
    return model(x, targets=y)

# -----------------------------------------------------------------------------
# Learning rate, momentum, and weight decay schedules
# -----------------------------------------------------------------------------

def get_lr_multiplier(step, warmup_iters, num_iterations, warmdown_iters):
    """LR schedule: linear warmup, constant, cosine warmdown."""
    if warmup_iters > 0 and step < warmup_iters:
        return (step + 1) / warmup_iters
    warmdown_start = num_iterations - warmdown_iters
    if warmdown_iters > 0 and step >= warmdown_start:
        decay_ratio = (step - warmdown_start) / warmdown_iters
        return 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return 1.0


def get_momentum(step, num_iterations, warmdown_iters):
    """Momentum schedule for Muon: constant 0.95, then ramp to 0.80."""
    warmdown_start = num_iterations - warmdown_iters
    if warmdown_iters > 0 and step >= warmdown_start:
        decay_ratio = (step - warmdown_start) / warmdown_iters
        return 0.95 - 0.15 * decay_ratio
    return 0.95


def get_weight_decay(step, num_iterations, warmdown_iters, base_wd):
    """Weight decay schedule: ramp from 0 to base_wd during warmdown."""
    warmdown_start = num_iterations - warmdown_iters
    if warmdown_iters > 0 and step >= warmdown_start:
        decay_ratio = (step - warmdown_start) / warmdown_iters
        return base_wd * decay_ratio
    return 0.0

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def count_parameters(model):
    """Count total trainable parameters."""
    total = 0
    for p in model.trainable_parameters():
        total += p.size
    return total


def get_device_name():
    """Try to get the Apple Silicon chip name."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def compute_mfu(tokens_per_sec, num_params, peak_flops):
    """Compute model FLOPs utilization (MFU)."""
    if peak_flops == float('inf') or peak_flops == 0:
        return 0.0
    flops_achieved = 6 * num_params * tokens_per_sec
    return flops_achieved / peak_flops


def calculate_training_params(target_tokens, batch_size, sequence_length, grad_accum_steps):
    """Given a target token budget, calculate iterations and warmdown."""
    tokens_per_step = batch_size * sequence_length * grad_accum_steps
    num_iterations = int(target_tokens / tokens_per_step)
    warmdown_iters = max(1, int(0.2 * num_iterations))
    return num_iterations, warmdown_iters

# -----------------------------------------------------------------------------
# Sampling / generation
# -----------------------------------------------------------------------------

def generate_sample(model, tokenizer, vocab_size, prompt="The", max_new_tokens=128,
                    temperature=0.8, top_k=50):
    """Generate a text sample from the model for qualitative monitoring."""
    token_ids = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
    tokens = mx.array([token_ids], dtype=mx.int32)

    for _ in range(max_new_tokens):
        logits = model(tokens)
        logits = logits[:, -1, :]
        logits = logits / temperature

        if top_k > 0:
            top_k_vals = mx.topk(logits, k=min(top_k, logits.shape[-1]))
            threshold = top_k_vals[:, -1:]
            logits = mx.where(logits < threshold, mx.array(float('-inf')), logits)

        token = mx.random.categorical(logits)
        token = token.reshape(1, 1)
        mx.eval(token)

        tokens = mx.concatenate([tokens, token.astype(mx.int32)], axis=1)

        tok_id = token.item()
        if tok_id == tokenizer.get_bos_token_id():
            break

    generated_ids = tokens[0].tolist()
    text = tokenizer.decode(generated_ids)
    return text

# -----------------------------------------------------------------------------
# Main training function
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    # Initialize MLX
    print_banner()
    mlx_init(seed=args.seed)

    # Load tokenizer
    print0("Loading tokenizer...")
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Tokenizer vocab size: {vocab_size}")

    # Load token bytes for BPB evaluation
    try:
        token_bytes = get_token_bytes()
        print0(f"Loaded token_bytes array, shape: {token_bytes.shape}")
    except FileNotFoundError:
        print0("WARNING: token_bytes not found, BPB evaluation will be unavailable")
        token_bytes = None

    # Scaling law calculations
    tokens_per_step = args.batch_size * args.sequence_length * args.grad_accum_steps

    if args.target_tokens is not None:
        num_iterations, warmdown_iters = calculate_training_params(
            args.target_tokens, args.batch_size, args.sequence_length, args.grad_accum_steps
        )
        args.num_iterations = num_iterations
        args.warmdown_iters = warmdown_iters
        print0(f"Scaling law: {args.target_tokens:.2e} target tokens -> "
               f"{num_iterations} iterations, {warmdown_iters} warmdown iters")

    total_tokens = args.num_iterations * tokens_per_step
    print0(f"Training configuration:")
    print0(f"  Batch size: {args.batch_size}")
    print0(f"  Sequence length: {args.sequence_length}")
    print0(f"  Grad accum steps: {args.grad_accum_steps}")
    print0(f"  Tokens per step: {tokens_per_step:,}")
    print0(f"  Total iterations: {args.num_iterations}")
    print0(f"  Warmup iterations: {args.warmup_iters}")
    print0(f"  Warmdown iterations: {args.warmdown_iters}")
    print0(f"  Total tokens: {total_tokens:,.0f} ({total_tokens:.2e})")

    # Build model
    print0("\nBuilding model...")
    config = GPTConfig(
        vocab_size=vocab_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
    )
    model = GPT(config)
    model.init_weights()
    mx.eval(model.parameters())

    num_params = count_parameters(model)
    print0(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)")
    print0(f"Model config: n_embd={config.n_embd}, n_layer={config.n_layer}, "
           f"n_head={config.n_head}, n_kv_head={config.n_kv_head}")

    # Device info and peak FLOPS
    device_name = get_device_name()
    peak_flops = get_peak_flops(device_name)
    print0(f"Device: {device_name}")
    if peak_flops != float('inf'):
        print0(f"Peak BF16 FLOPS: {peak_flops:.2e}")

    # Set up optimizer
    print0("\nSetting up optimizer...")
    param_groups = build_param_groups(model, lr_scale=args.lr_scale)

    for i, group in enumerate(param_groups):
        n_params = 0
        for name in group["param_names"]:
            try:
                from nanochat_mlx.optim import _get_nested
                p = _get_nested(model.parameters(), name)
                n_params += p.size
            except (KeyError, IndexError):
                pass
        print0(f"  Group {i} ({group['kind']}): {len(group['param_names'])} params, "
               f"{n_params:,} elements, lr={group['lr']:.6f}")

    optimizer = MuonAdamW(param_groups)

    # Resume from checkpoint if requested
    start_step = 0
    best_val_loss = float('inf')
    dataloader_state_dict = None

    if args.resume is not None:
        print0(f"\nResuming from checkpoint: {args.resume}")
        checkpoint_data = load_checkpoint(args.resume, model, optimizer)
        if checkpoint_data is not None:
            start_step = checkpoint_data.get("step", 0)
            best_val_loss = checkpoint_data.get("best_val_loss", float('inf'))
            dataloader_state_dict = checkpoint_data.get("dataloader_state_dict", None)
            print0(f"Resumed from step {start_step}, best_val_loss={best_val_loss:.4f}")
        else:
            print0("WARNING: Failed to load checkpoint, starting from scratch")

    # Set up data loaders
    print0("\nSetting up dataloaders...")
    train_loader = tokenizing_data_loader_with_state_bos_bestfit(
        tokenizer=tokenizer,
        B=args.batch_size,
        T=args.sequence_length,
        split="train",
        tokenizer_threads=args.num_workers,
        tokenizer_batch_size=args.tokenizer_batch_size,
        resume_state_dict=dataloader_state_dict,
    )

    x, y, dataloader_state_dict = next(train_loader)
    print0(f"First batch loaded: x.shape={x.shape}, y.shape={y.shape}")

    # Set up wandb
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or args.run,
            config=vars(args),
        )
        print0(f"Wandb initialized: {wandb_run.url}")
    else:
        wandb_run = DummyWandb()
        print0("Wandb disabled")

    # Set up checkpoint directory
    base_dir = get_base_dir()
    run_dir = os.path.join(base_dir, "runs", args.run)
    os.makedirs(run_dir, exist_ok=True)

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print0(f"Config saved to {config_path}")

    # Create loss+grad function
    loss_and_grad_fn = nn.value_and_grad(model, compute_loss)

    # Training loop
    print0(f"\n{'='*80}")
    print0(f"Starting training from step {start_step} to {args.num_iterations}")
    print0(f"{'='*80}\n")

    training_time_ms = 0.0
    total_tokens_processed = start_step * tokens_per_step
    log_interval = 10
    val_loss_every = getattr(args, 'eval_every', 250)

    for step in range(start_step, args.num_iterations):
        step_start_time = time.time()
        is_last_step = (step == args.num_iterations - 1)

        # Update schedules
        lr_mult = get_lr_multiplier(step, args.warmup_iters, args.num_iterations, args.warmdown_iters)
        optimizer.update_lr(lr_mult)
        mom = get_momentum(step, args.num_iterations, args.warmdown_iters)
        optimizer.update_muon_momentum(mom)
        wd = get_weight_decay(step, args.num_iterations, args.warmdown_iters, args.weight_decay)
        optimizer.update_weight_decay(wd)

        # Evaluation
        if val_loss_every > 0 and (step % val_loss_every == 0 or is_last_step):
            t_eval_start = time.time()

            if token_bytes is not None:
                try:
                    from nanochat_mlx.loss_eval import compute_bpb
                    val_loader = tokenizing_data_loader_bos_bestfit(
                        tokenizer=tokenizer, B=args.batch_size,
                        T=args.sequence_length, split="val",
                        tokenizer_threads=args.num_workers,
                        tokenizer_batch_size=args.tokenizer_batch_size,
                    )
                    max_val_steps = max(1, args.val_tokens // (args.batch_size * args.sequence_length))
                    bpb_result = compute_bpb(model, val_loader, token_bytes, max_steps=max_val_steps)
                    val_bpb_val = bpb_result["bpb"]
                    t_eval_end = time.time()
                    print0(f"step {step:>6d} | val_bpb: {val_bpb_val:.4f} | "
                           f"eval time: {t_eval_end - t_eval_start:.1f}s")
                    wandb_run.log({"val/bpb": val_bpb_val, "step": step})

                    if val_bpb_val < best_val_loss:
                        best_val_loss = val_bpb_val
                        if args.save_best:
                            best_dir = os.path.join(run_dir, "best")
                            save_checkpoint(best_dir, model, optimizer,
                                            step=step, best_val_loss=best_val_loss,
                                            dataloader_state_dict=dataloader_state_dict,
                                            config=vars(args))
                            print0(f"  Saved best checkpoint (bpb={best_val_loss:.4f})")
                except Exception as e:
                    print0(f"  BPB evaluation failed: {e}")

        # Text generation sample
        if args.sample_every > 0 and (step % args.sample_every == 0 or is_last_step) and step > 0:
            try:
                sample_text = generate_sample(model, tokenizer, vocab_size)
                print0(f"\n--- Sample at step {step} ---")
                print0(sample_text)
                print0(f"--- End sample ---\n")
            except Exception as e:
                print0(f"  Sampling failed: {e}")

        # Periodic checkpoint saving
        save_every = getattr(args, 'save_every', 0)
        if save_every > 0 and step > 0 and (step % save_every == 0 or is_last_step):
            ckpt_dir = os.path.join(run_dir, f"step_{step:06d}")
            save_checkpoint(ckpt_dir, model, optimizer,
                            step=step, best_val_loss=best_val_loss,
                            dataloader_state_dict=dataloader_state_dict,
                            config=vars(args))
            print0(f"  Saved checkpoint at step {step}")

        # Forward + backward with gradient accumulation
        accumulated_grads = None
        total_loss = 0.0

        for micro_step in range(args.grad_accum_steps):
            loss, grads = loss_and_grad_fn(model, x, y)
            mx.eval(loss)
            total_loss += loss.item()

            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(lambda a, b: a + b, accumulated_grads, grads)

            x, y, dataloader_state_dict = next(train_loader)

        if args.grad_accum_steps > 1:
            accumulated_grads = tree_map(
                lambda g: g * (1.0 / args.grad_accum_steps), accumulated_grads
            )
        avg_loss = total_loss / args.grad_accum_steps

        # Optimizer step
        optimizer.update(model, accumulated_grads)
        mx.eval(model.parameters(), optimizer.state)

        # Timing and logging
        step_end_time = time.time()
        step_time_ms = (step_end_time - step_start_time) * 1000
        training_time_ms += step_time_ms
        total_tokens_processed += tokens_per_step

        tokens_per_sec = tokens_per_step / (step_time_ms / 1000) if step_time_ms > 0 else 0
        mfu = compute_mfu(tokens_per_sec, num_params, peak_flops)

        if step % log_interval == 0 or is_last_step:
            mem_bytes = get_memory_usage()
            mem_gb = mem_bytes / (1024 ** 3) if mem_bytes > 0 else 0
            current_lr = optimizer.param_groups[0]["lr"]

            print0(
                f"step {step:>6d}/{args.num_iterations} | "
                f"loss: {avg_loss:.4f} | "
                f"lr: {current_lr:.2e} | "
                f"dt: {step_time_ms:.0f}ms | "
                f"tok/s: {tokens_per_sec:,.0f} | "
                f"MFU: {mfu*100:.1f}% | "
                f"mem: {mem_gb:.1f}GB"
            )

        wandb_run.log({
            "train/loss": avg_loss,
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/step_time_ms": step_time_ms,
            "train/tokens_per_sec": tokens_per_sec,
            "train/mfu": mfu,
            "train/total_tokens": total_tokens_processed,
            "step": step,
        })

    # Training complete
    total_time_sec = training_time_ms / 1000
    total_time_hr = total_time_sec / 3600

    print0(f"\n{'='*80}")
    print0(f"Training complete!")
    print0(f"{'='*80}")
    print0(f"  Total steps: {args.num_iterations}")
    print0(f"  Total tokens: {total_tokens_processed:,}")
    print0(f"  Total time: {total_time_hr:.2f}h")
    print0(f"  Best val BPB: {best_val_loss:.4f}")

    # Save final checkpoint
    final_dir = os.path.join(run_dir, "final")
    save_checkpoint(final_dir, model, optimizer,
                    step=args.num_iterations, best_val_loss=best_val_loss,
                    dataloader_state_dict=dataloader_state_dict,
                    config=vars(args))
    print0(f"  Final checkpoint saved to {final_dir}")

    wandb_run.finish()
    print0("Done!")


if __name__ == "__main__":
    main()
