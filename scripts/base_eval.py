"""
Unified base model evaluation script for nanochat_mlx.
Ported from nanochat/scripts/base_eval.py (PyTorch) to Apple MLX.

Key differences from PyTorch version:
- No DDP/distributed code (single device)
- No torch.amp.autocast or FP8
- mx.array operations instead of torch tensors
- No .to(device) calls (MLX unified memory)
- Uses MLX lazy evaluation with mx.eval() for synchronization

Evaluates pretrained (base) models on:
- Validation set perplexity / BPB (bits per byte)
- HellaSwag (categorical, 10-shot or 0-shot)
- MMLU (categorical, 5-shot or 0-shot)
- ARC-Easy / ARC-Challenge (categorical)
- BPB on FineWeb-Edu validation set

Computes the BaseCORE metric (weighted average across benchmarks).

Usage:
    python scripts/base_eval.py --checkpoint runs/base_train_xxx/step_010000
    python scripts/base_eval.py --checkpoint runs/base_train_xxx/step_010000 --tasks bpb hellaswag
"""

import os
import sys
import time
import math
import json
import argparse
from collections import defaultdict

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from nanochat_mlx.common import mlx_init, print0, get_base_dir
from nanochat_mlx.checkpoint_manager import load_model
from nanochat_mlx.tokenizer import get_tokenizer, get_token_bytes
from nanochat_mlx.gpt import GPT, GPTConfig, norm
from nanochat_mlx.dataloader import tokenizing_data_loader_bos_bestfit


# ---------------------------------------------------------------------------
# BPB evaluation (bits per byte on validation set)
# ---------------------------------------------------------------------------

def evaluate_val_bpb(model, tokenizer, token_bytes, num_batches=20,
                     batch_size=4, seq_len=2048):
    """
    Evaluate bits-per-byte (BPB) on the validation set.

    BPB is computed as: mean(cross_entropy_per_token * log2(e) / bytes_per_token)
    This normalizes the loss by the number of bytes each token represents,
    giving a measure independent of tokenization.

    Args:
        model: The GPT model
        tokenizer: The tokenizer
        token_bytes: mx.array of byte counts per token ID
        num_batches: Number of validation batches to evaluate
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        dict with 'bpb' (float) and 'loss' (float, raw CE loss)
    """
    print0(f"Evaluating BPB on validation set ({num_batches} batches)...")

    val_loader = tokenizing_data_loader_bos_bestfit(
        tokenizer, batch_size, seq_len, split="val"
    )

    total_loss = 0.0
    total_tokens = 0
    total_bpb_numerator = 0.0
    total_bpb_denominator = 0.0

    for batch_idx in range(num_batches):
        inputs, targets = next(val_loader)

        # Forward pass with per-token loss
        logits = model(inputs)
        logits = logits.astype(mx.float32)

        # Compute per-token cross-entropy
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)

        per_token_loss = nn.losses.cross_entropy(
            logits_flat, targets_flat, reduction="none"
        )

        # Valid mask (targets != -1)
        valid_mask = (targets_flat != -1).astype(mx.float32)
        per_token_loss = per_token_loss * valid_mask

        # Accumulate raw loss
        batch_loss = mx.sum(per_token_loss).item()
        batch_tokens = mx.sum(valid_mask).item()
        total_loss += batch_loss
        total_tokens += batch_tokens

        # BPB: weight each token's loss by 1/bytes_per_token
        # token_bytes[target_id] gives the byte-length of each target token
        target_bytes = token_bytes[targets_flat]  # (B*T,)
        target_bytes = target_bytes.astype(mx.float32)
        # Avoid division by zero for padding tokens
        target_bytes = mx.maximum(target_bytes, mx.array(1.0))

        # BPB per token = CE_loss * log2(e) / bytes
        log2_e = 1.0 / math.log(2.0)
        bpb_per_token = per_token_loss * log2_e / target_bytes
        total_bpb_numerator += mx.sum(bpb_per_token).item()
        total_bpb_denominator += batch_tokens

        mx.eval(per_token_loss)

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    avg_bpb = total_bpb_numerator / total_bpb_denominator if total_bpb_denominator > 0 else float("inf")

    print0(f"  Val loss: {avg_loss:.4f}, BPB: {avg_bpb:.4f}")

    return {
        "bpb": avg_bpb,
        "loss": avg_loss,
        "total_tokens": int(total_tokens),
    }


# ---------------------------------------------------------------------------
# HellaSwag evaluation (categorical, completion scoring)
# ---------------------------------------------------------------------------

def evaluate_hellaswag(model, tokenizer, max_examples=None, num_shots=0):
    """
    Evaluate on HellaSwag: given a context, pick the most likely continuation
    from 4 choices.

    For each example, we compute the average log-probability of each ending
    continuation conditioned on the context, and select the highest.

    Args:
        model: The GPT model
        tokenizer: The tokenizer
        max_examples: Max examples to evaluate (None = all)
        num_shots: Number of few-shot examples (0 for zero-shot)

    Returns:
        dict with 'accuracy', 'correct', 'total'
    """
    from datasets import load_dataset

    print0(f"Evaluating HellaSwag ({num_shots}-shot)...")
    dataset = load_dataset("Rowan/hellaswag", split="validation")

    num_examples = len(dataset) if max_examples is None else min(max_examples, len(dataset))

    correct = 0
    total = 0

    for idx in range(num_examples):
        item = dataset[idx]
        ctx = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])

        # Build the context prefix (optionally with few-shot examples)
        prefix = ctx

        # Tokenize prefix
        prefix_tokens = tokenizer.encode(prefix, prepend=tokenizer.get_bos_token_id())
        prefix_len = len(prefix_tokens)

        # Score each ending
        ending_scores = []
        for ending in endings:
            # Tokenize the ending (no BOS, it continues from prefix)
            ending_tokens = tokenizer.encode(ending)
            if not ending_tokens:
                ending_scores.append(float("-inf"))
                continue

            # Full sequence = prefix + ending
            full_tokens = prefix_tokens + ending_tokens
            # Truncate to model's max sequence length
            max_len = 2048
            if len(full_tokens) > max_len:
                full_tokens = full_tokens[:max_len]
                # Recompute ending region
                ending_len_actual = len(full_tokens) - prefix_len
                if ending_len_actual <= 0:
                    ending_scores.append(float("-inf"))
                    continue
            else:
                ending_len_actual = len(ending_tokens)

            # Forward pass
            input_ids = mx.array([full_tokens[:-1]], dtype=mx.int32)
            target_ids = mx.array([full_tokens[1:]], dtype=mx.int32)

            logits = model(input_ids)  # (1, T, vocab)
            logits = logits.astype(mx.float32)

            # Log-softmax
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

            # Score only the ending tokens (starting from prefix_len - 1 in target space)
            start = prefix_len - 1
            end = start + ending_len_actual
            ending_logits = log_probs[0, start:end, :]
            ending_targets = target_ids[0, start:end]

            # Gather log-probs for actual tokens
            token_lps = []
            for t in range(ending_targets.shape[0]):
                tid = ending_targets[t].item()
                lp = ending_logits[t, tid].item()
                token_lps.append(lp)

            # Average log-prob (length-normalized)
            avg_lp = sum(token_lps) / len(token_lps) if token_lps else float("-inf")
            ending_scores.append(avg_lp)

        mx.eval(logits)

        # Pick best ending
        predicted = max(range(len(ending_scores)), key=lambda i: ending_scores[i])

        if predicted == label:
            correct += 1
        total += 1

        if total % 100 == 0:
            acc = correct / total if total > 0 else 0
            print0(f"  ... {total}/{num_examples} done, accuracy: {acc:.3f}")

    accuracy = correct / total if total > 0 else 0.0
    print0(f"HellaSwag: accuracy = {accuracy:.4f} ({correct}/{total})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


# ---------------------------------------------------------------------------
# MMLU evaluation (categorical, for base models)
# ---------------------------------------------------------------------------

def evaluate_mmlu_base(model, tokenizer, max_examples=None, num_shots=5):
    """
    Evaluate on MMLU using completion scoring (appropriate for base models).

    For base models, we use the standard completion-based scoring:
    given a question + choices formatted as text, compute log-probability
    of each answer letter (A, B, C, D).

    Args:
        model: The GPT model
        tokenizer: The tokenizer
        max_examples: Max examples per subject (None = all)
        num_shots: Number of few-shot examples

    Returns:
        dict with 'accuracy', 'correct', 'total', 'per_subject' results
    """
    from datasets import load_dataset
    from tasks.common import render_mc

    print0(f"Evaluating MMLU ({num_shots}-shot, base-model scoring)...")

    dataset = load_dataset("cais/mmlu", "all", split="test")

    # Group by subject for per-subject reporting
    subject_results = defaultdict(lambda: {"correct": 0, "total": 0})

    # Load few-shot examples if needed
    few_shot_examples = {}
    if num_shots > 0:
        dev_dataset = load_dataset("cais/mmlu", "all", split="validation")
        for item in dev_dataset:
            subject = item["subject"]
            if subject not in few_shot_examples:
                few_shot_examples[subject] = []
            if len(few_shot_examples[subject]) < num_shots:
                few_shot_examples[subject].append(item)

    num_examples = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    correct = 0
    total = 0

    for idx in range(num_examples):
        item = dataset[idx]
        question = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]
        subject = item["subject"]

        # Build prompt with few-shot examples
        prompt_parts = []
        if num_shots > 0 and subject in few_shot_examples:
            for shot in few_shot_examples[subject]:
                shot_q, shot_a = render_mc(
                    shot["question"], shot["choices"], shot["answer"]
                )
                prompt_parts.append(f"{shot_q}\nAnswer: {shot_a}")

        # Format the test question
        test_q = render_mc(question, choices)
        prompt_parts.append(f"{test_q}\nAnswer:")
        prompt_text = "\n\n".join(prompt_parts)

        # Tokenize prompt
        prompt_tokens = tokenizer.encode(prompt_text, prepend=tokenizer.get_bos_token_id())

        # Score each answer choice (A, B, C, D)
        answer_labels = ["A", "B", "C", "D"]
        choice_scores = []

        for label in answer_labels[:len(choices)]:
            # Tokenize the answer label
            label_tokens = tokenizer.encode(f" {label}")
            if not label_tokens:
                choice_scores.append(float("-inf"))
                continue

            # Forward pass on prompt
            full_tokens = prompt_tokens + label_tokens
            max_len = 2048
            if len(full_tokens) > max_len:
                # Truncate prefix, keep label tokens
                keep = max_len - len(label_tokens)
                full_tokens = prompt_tokens[-keep:] + label_tokens

            input_ids = mx.array([full_tokens[:-1]], dtype=mx.int32)
            target_ids = mx.array([full_tokens[1:]], dtype=mx.int32)

            logits = model(input_ids)
            logits = logits.astype(mx.float32)

            # Log-softmax
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

            # Score the label tokens
            start = len(prompt_tokens) - 1
            label_logprobs = []
            for t, lt in enumerate(label_tokens):
                pos = start + t
                if pos < log_probs.shape[1]:
                    lp = log_probs[0, pos, lt].item()
                    label_logprobs.append(lp)

            avg_lp = sum(label_logprobs) / len(label_logprobs) if label_logprobs else float("-inf")
            choice_scores.append(avg_lp)

        mx.eval(logits)

        # Pick best
        predicted_idx = max(range(len(choice_scores)), key=lambda i: choice_scores[i])

        is_correct = (predicted_idx == answer_idx)
        if is_correct:
            correct += 1
            subject_results[subject]["correct"] += 1
        total += 1
        subject_results[subject]["total"] += 1

        if total % 200 == 0:
            acc = correct / total if total > 0 else 0
            print0(f"  ... {total}/{num_examples} done, accuracy: {acc:.3f}")

    accuracy = correct / total if total > 0 else 0.0

    # Per-subject summary
    per_subject = {}
    for subject, sr in subject_results.items():
        sub_acc = sr["correct"] / sr["total"] if sr["total"] > 0 else 0
        per_subject[subject] = {"accuracy": sub_acc, **sr}

    print0(f"MMLU: accuracy = {accuracy:.4f} ({correct}/{total})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_subject": per_subject,
    }


# ---------------------------------------------------------------------------
# ARC evaluation (categorical, for base models)
# ---------------------------------------------------------------------------

def evaluate_arc_base(model, tokenizer, difficulty="easy", max_examples=None, num_shots=0):
    """
    Evaluate on ARC using completion scoring (appropriate for base models).

    Args:
        model: The GPT model
        tokenizer: The tokenizer
        difficulty: 'easy' or 'challenge'
        max_examples: Max examples (None = all)
        num_shots: Number of few-shot examples

    Returns:
        dict with 'accuracy', 'correct', 'total'
    """
    from datasets import load_dataset
    from tasks.common import render_mc

    config_name = "ARC-Easy" if difficulty == "easy" else "ARC-Challenge"
    print0(f"Evaluating ARC-{difficulty.capitalize()} ({num_shots}-shot, base scoring)...")

    dataset = load_dataset("allenai/ai2_arc", config_name, split="test")
    num_examples = len(dataset) if max_examples is None else min(max_examples, len(dataset))

    correct = 0
    total = 0

    for idx in range(num_examples):
        item = dataset[idx]
        question = item["question"]
        choices_text = item["choices"]["text"]
        choices_labels = item["choices"]["label"]
        answer_key = item["answerKey"]

        # Find answer index
        try:
            answer_index = choices_labels.index(answer_key)
        except ValueError:
            continue

        # Format as MC question
        formatted_q, _ = render_mc(question, choices_text, answer_index)
        prompt_text = f"{formatted_q}\nAnswer:"

        # Tokenize prompt
        prompt_tokens = tokenizer.encode(prompt_text, prepend=tokenizer.get_bos_token_id())

        # Score each choice label
        choice_scores = []
        for label in choices_labels:
            label_tokens = tokenizer.encode(f" {label}")
            if not label_tokens:
                choice_scores.append(float("-inf"))
                continue

            full_tokens = prompt_tokens + label_tokens
            max_len = 2048
            if len(full_tokens) > max_len:
                keep = max_len - len(label_tokens)
                full_tokens = prompt_tokens[-keep:] + label_tokens

            input_ids = mx.array([full_tokens[:-1]], dtype=mx.int32)

            logits = model(input_ids)
            logits = logits.astype(mx.float32)

            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

            start = len(prompt_tokens) - 1
            label_logprobs = []
            for t, lt in enumerate(label_tokens):
                pos = start + t
                if pos < log_probs.shape[1]:
                    lp = log_probs[0, pos, lt].item()
                    label_logprobs.append(lp)

            avg_lp = sum(label_logprobs) / len(label_logprobs) if label_logprobs else float("-inf")
            choice_scores.append(avg_lp)

        mx.eval(logits)

        predicted_idx = max(range(len(choice_scores)), key=lambda i: choice_scores[i])
        if predicted_idx == answer_index:
            correct += 1
        total += 1

        if total % 100 == 0:
            acc = correct / total if total > 0 else 0
            print0(f"  ... {total}/{num_examples} done, accuracy: {acc:.3f}")

    accuracy = correct / total if total > 0 else 0.0
    print0(f"ARC-{difficulty.capitalize()}: accuracy = {accuracy:.4f} ({correct}/{total})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


# ---------------------------------------------------------------------------
# BaseCORE metric
# ---------------------------------------------------------------------------

def compute_base_core(results):
    """
    Compute the BaseCORE metric: weighted average across benchmarks.

    Weights reflect the relative importance/difficulty:
    - BPB: included but not weighted into the average (it's a different scale)
    - HellaSwag, MMLU, ARC-E, ARC-C: equally weighted

    Args:
        results: dict mapping task_name -> result dict

    Returns:
        float: BaseCORE score
    """
    # Define weights for each task
    task_weights = {
        "hellaswag": 1.0,
        "mmlu": 1.0,
        "arc_easy": 1.0,
        "arc_challenge": 1.0,
    }

    weighted_sum = 0.0
    total_weight = 0.0

    print0("\nBaseCORE components:")
    for task_name, result in results.items():
        acc = result.get("accuracy", result.get("bpb", 0.0))
        weight = task_weights.get(task_name, 0.0)

        if task_name == "bpb":
            print0(f"  {task_name:20s}: {result.get('bpb', 0.0):.4f} BPB (not in CORE)")
        else:
            print0(f"  {task_name:20s}: {acc:.4f} (weight={weight:.1f})")

        if weight > 0:
            weighted_sum += acc * weight
            total_weight += weight

    core = weighted_sum / total_weight if total_weight > 0 else 0.0
    print0(f"  {'BaseCORE':20s}: {core:.4f}")
    return core


# ---------------------------------------------------------------------------
# Task dispatcher
# ---------------------------------------------------------------------------

TASK_REGISTRY = {
    "bpb": {
        "description": "Bits-per-byte on FineWeb-Edu validation set",
        "eval_type": "bpb",
    },
    "hellaswag": {
        "description": "HellaSwag completion selection",
        "eval_type": "hellaswag",
    },
    "mmlu": {
        "description": "MMLU multiple choice",
        "eval_type": "mmlu",
    },
    "arc_easy": {
        "description": "ARC-Easy multiple choice",
        "eval_type": "arc",
        "difficulty": "easy",
    },
    "arc_challenge": {
        "description": "ARC-Challenge multiple choice",
        "eval_type": "arc",
        "difficulty": "challenge",
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NanoChat MLX Base Model Evaluation")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Tasks to evaluate (default: all). "
                             "Options: bpb, hellaswag, mmlu, arc_easy, arc_challenge")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Max examples per task (default: all)")
    parser.add_argument("--num-shots", type=int, default=0,
                        help="Number of few-shot examples for MC tasks (default: 0)")
    parser.add_argument("--bpb-batches", type=int, default=20,
                        help="Number of validation batches for BPB (default: 20)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for BPB evaluation")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Sequence length for BPB evaluation")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-example results")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    mlx_init(seed=args.seed)

    # Default task list
    if args.tasks is None:
        args.tasks = ["bpb", "hellaswag", "mmlu", "arc_easy", "arc_challenge"]

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print0(f"Loading model from: {args.checkpoint}")
    t0 = time.time()
    model, config = load_model(args.checkpoint)
    load_time = time.time() - t0
    print0(f"Model loaded in {load_time:.1f}s")

    total_params = sum(p.size for _, p in model.parameters())
    print0(f"Total parameters: {total_params:,}")

    # Load tokenizer
    tokenizer = get_tokenizer()

    # Load token bytes for BPB computation
    try:
        token_bytes = get_token_bytes()
    except FileNotFoundError:
        token_bytes = None
        if "bpb" in [t.lower() for t in args.tasks]:
            print0("Warning: token_bytes not found, BPB evaluation will be skipped")

    # -----------------------------------------------------------------------
    # Run evaluations
    # -----------------------------------------------------------------------
    print0("=" * 80)
    print0("Starting base model evaluation")
    print0("=" * 80)

    all_results = {}
    total_t0 = time.time()

    for task_name in args.tasks:
        task_lower = task_name.lower().replace("-", "_")

        if task_lower not in TASK_REGISTRY:
            print0(f"Warning: unknown task '{task_name}', skipping")
            continue

        task_info = TASK_REGISTRY[task_lower]
        print0(f"\n--- {task_name} ({task_info['description']}) ---")
        task_t0 = time.time()

        if task_info["eval_type"] == "bpb":
            if token_bytes is None:
                print0("  Skipping BPB (token_bytes not available)")
                continue
            result = evaluate_val_bpb(
                model=model,
                tokenizer=tokenizer,
                token_bytes=token_bytes,
                num_batches=args.bpb_batches,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
            )
        elif task_info["eval_type"] == "hellaswag":
            result = evaluate_hellaswag(
                model=model,
                tokenizer=tokenizer,
                max_examples=args.max_examples,
                num_shots=args.num_shots,
            )
        elif task_info["eval_type"] == "mmlu":
            result = evaluate_mmlu_base(
                model=model,
                tokenizer=tokenizer,
                max_examples=args.max_examples,
                num_shots=max(args.num_shots, 5) if args.num_shots > 0 else 0,
            )
        elif task_info["eval_type"] == "arc":
            result = evaluate_arc_base(
                model=model,
                tokenizer=tokenizer,
                difficulty=task_info["difficulty"],
                max_examples=args.max_examples,
                num_shots=args.num_shots,
            )
        else:
            print0(f"  Unknown eval_type: {task_info['eval_type']}")
            continue

        result["time"] = time.time() - task_t0
        all_results[task_lower] = result

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_time = time.time() - total_t0
    print0("\n" + "=" * 80)
    print0("EVALUATION SUMMARY")
    print0("=" * 80)

    base_core = compute_base_core(all_results)

    print0(f"\nTotal evaluation time: {total_time:.1f}s")
    print0(f"Checkpoint: {args.checkpoint}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    summary = {
        "checkpoint": args.checkpoint,
        "base_core": base_core,
        "total_time": total_time,
        "num_shots": args.num_shots,
        "tasks": {},
    }
    for task_name, result in all_results.items():
        # Remove per_subject for cleaner JSON (it can be huge for MMLU)
        clean_result = {k: v for k, v in result.items() if k != "per_subject"}
        summary["tasks"][task_name] = clean_result

    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print0(f"\nResults saved to: {args.output}")
    else:
        # Try to save to checkpoint directory
        output_path = os.path.join(args.checkpoint, "base_eval_results.json")
        try:
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2)
            print0(f"\nResults saved to: {output_path}")
        except OSError:
            print0("\nCould not save results to checkpoint directory.")

    # Print machine-parseable one-liner
    task_strs = []
    for tn, tr in all_results.items():
        if "bpb" in tr:
            task_strs.append(f"{tn}={tr['bpb']:.4f}")
        elif "accuracy" in tr:
            task_strs.append(f"{tn}={tr['accuracy']:.4f}")
    print0(f"\nRESULT: BaseCORE={base_core:.4f} | {' | '.join(task_strs)}")
    print0("\nDone.")


if __name__ == "__main__":
    main()
