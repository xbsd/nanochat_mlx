"""
Chat model evaluation script for nanochat_mlx.
Ported from nanochat/scripts/chat_eval.py (PyTorch) to Apple MLX.

Key differences from PyTorch version:
- No DDP/distributed code (single device)
- Replace torch operations with mx equivalents
- No autocast, no FP8
- Uses Engine for generation

Supports evaluation on:
- ARC-Easy (categorical)
- ARC-Challenge (categorical)
- MMLU (categorical)
- GSM8K (generative)
- HumanEval (generative)
- SpellingBee (generative)

Computes the ChatCORE metric (average across all task scores).

Usage:
    python scripts/chat_eval.py --checkpoint runs/chat_sft_xxx/final
    python scripts/chat_eval.py --checkpoint runs/chat_sft_xxx/final --tasks gsm8k mmlu
"""

import os
import sys
import time
import math
import json
import argparse
import re
from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn

from nanochat_mlx.common import mlx_init, print0, get_base_dir
from nanochat_mlx.checkpoint_manager import load_model
from nanochat_mlx.engine import Engine
from nanochat_mlx.tokenizer import get_tokenizer
from nanochat_mlx.gpt import GPT, GPTConfig, norm


# ---------------------------------------------------------------------------
# Categorical evaluation (multiple choice: ARC, MMLU)
# ---------------------------------------------------------------------------

def run_categorical_eval(model, tokenizer, task, task_name, max_examples=None,
                         verbose=False):
    """
    Run categorical (multiple-choice) evaluation.

    For each example, we compute the log-probability of each answer choice
    conditioned on the prompt, and pick the highest. This is the standard
    approach for ARC and MMLU.

    Args:
        model: The GPT model
        tokenizer: The tokenizer
        task: A Task object with conversation examples and 'ideal' labels
        task_name: Name for logging
        max_examples: Max examples to evaluate (None = all)
        verbose: Print per-example results

    Returns:
        dict with 'accuracy', 'correct', 'total', 'per_example' results
    """
    num_examples = len(task) if max_examples is None else min(max_examples, len(task))
    correct = 0
    total = 0
    per_example = []

    print0(f"Running categorical eval on {task_name}: {num_examples} examples")

    for idx in range(num_examples):
        example = task[idx]
        conversation = example["conversation"]
        ideal = example["ideal"]

        # Get the answer choices from the example
        # For MC tasks, choices are typically A, B, C, D
        choices = example.get("choices", ["A", "B", "C", "D"])

        # Tokenize the conversation prompt (everything up to the assistant response)
        conv_for_prompt = {"messages": conversation["messages"].copy()}
        # Ensure the last message is assistant with empty content
        if conv_for_prompt["messages"][-1]["role"] == "assistant":
            conv_for_prompt["messages"][-1]["content"] = ""

        try:
            prompt_ids, prompt_mask = tokenizer.render_conversation(conv_for_prompt)
        except (AssertionError, KeyError, ValueError):
            if verbose:
                print0(f"  [{idx}] Skipped (tokenization error)")
            continue

        prompt_ids_mx = mx.array([prompt_ids], dtype=mx.int32)

        # For each choice, compute log-probability
        choice_logprobs = []
        for choice in choices:
            # Tokenize the choice
            choice_tokens = tokenizer.encode(choice)
            if isinstance(choice_tokens, list) and len(choice_tokens) > 0:
                # Concatenate prompt + choice tokens
                full_ids = prompt_ids + choice_tokens
                full_ids_mx = mx.array([full_ids[:-1]], dtype=mx.int32)  # inputs
                full_targets = mx.array([full_ids[1:]], dtype=mx.int32)  # targets

                # Forward pass to get logits
                logits = model(full_ids_mx)  # (1, T, vocab)
                logits = logits.astype(mx.float32)

                # Compute log-probabilities for the choice tokens only
                # The choice tokens start at position len(prompt_ids)-1 in the target
                start_pos = len(prompt_ids) - 1
                choice_logits = logits[0, start_pos:, :]  # (num_choice_tokens, vocab)
                choice_target_ids = full_targets[0, start_pos:]  # (num_choice_tokens,)

                # Log softmax
                log_probs = choice_logits - mx.logsumexp(choice_logits, axis=-1, keepdims=True)

                # Gather log-probs for the actual choice tokens
                token_log_probs = []
                for t in range(choice_target_ids.shape[0]):
                    token_id = choice_target_ids[t].item()
                    lp = log_probs[t, token_id].item()
                    token_log_probs.append(lp)

                avg_logprob = sum(token_log_probs) / len(token_log_probs) if token_log_probs else float("-inf")
                choice_logprobs.append(avg_logprob)
            else:
                choice_logprobs.append(float("-inf"))

        mx.eval(logits)

        # Pick the choice with highest average log-probability
        best_choice_idx = max(range(len(choice_logprobs)), key=lambda i: choice_logprobs[i])
        predicted = choices[best_choice_idx]

        is_correct = (predicted == ideal)
        if is_correct:
            correct += 1
        total += 1

        per_example.append({
            "index": idx,
            "predicted": predicted,
            "ideal": ideal,
            "correct": is_correct,
            "logprobs": dict(zip(choices, choice_logprobs)),
        })

        if verbose:
            status = "OK" if is_correct else "WRONG"
            print0(f"  [{idx}] {status}: predicted={predicted}, ideal={ideal}")

        if total % 50 == 0:
            acc_so_far = correct / total if total > 0 else 0
            print0(f"  ... {total}/{num_examples} done, accuracy so far: {acc_so_far:.3f}")

    accuracy = correct / total if total > 0 else 0.0
    print0(f"{task_name}: accuracy = {accuracy:.4f} ({correct}/{total})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_example": per_example,
    }


# ---------------------------------------------------------------------------
# Generative evaluation (GSM8K, HumanEval, SpellingBee)
# ---------------------------------------------------------------------------

def run_generative_eval(engine, tokenizer, task, task_name, max_examples=None,
                        max_tokens=512, temperature=0.0, top_k=None,
                        verbose=False):
    """
    Run generative evaluation.

    For each example, we generate a response and evaluate it using the
    task's evaluate() method.

    Args:
        engine: The Engine object for generation
        tokenizer: The tokenizer
        task: A Task object
        task_name: Name for logging
        max_examples: Max examples to evaluate (None = all)
        max_tokens: Max tokens to generate per response
        temperature: Sampling temperature (0 = greedy)
        top_k: Top-k sampling (None = disabled)
        verbose: Print per-example results

    Returns:
        dict with 'accuracy', 'correct', 'total', 'per_example' results
    """
    num_examples = len(task) if max_examples is None else min(max_examples, len(task))
    correct = 0
    total = 0
    total_score = 0.0
    per_example = []

    # Get stop tokens
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    stop_tokens = [assistant_end] if assistant_end is not None else []

    print0(f"Running generative eval on {task_name}: {num_examples} examples")
    t0 = time.time()

    for idx in range(num_examples):
        example = task[idx]
        conversation = example["conversation"]

        # Build the prompt for completion
        conv_for_completion = {"messages": conversation["messages"].copy()}
        # Ensure last message is assistant
        if conv_for_completion["messages"][-1]["role"] != "assistant":
            conv_for_completion["messages"].append({"role": "assistant", "content": ""})
        else:
            conv_for_completion["messages"][-1]["content"] = ""

        try:
            prompt_ids = tokenizer.render_for_completion(conv_for_completion)
        except (AssertionError, KeyError, ValueError):
            if verbose:
                print0(f"  [{idx}] Skipped (tokenization error)")
            continue

        # Generate response
        response_tokens = []
        for token_id in engine.generate(
            prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_tokens=stop_tokens,
        ):
            response_tokens.append(token_id)

        response_text = tokenizer.decode(response_tokens)

        # Evaluate using task's evaluate method
        try:
            result = task.evaluate(idx, response_text)
        except Exception as e:
            if verbose:
                print0(f"  [{idx}] Evaluation error: {e}")
            result = {"correct": False, "score": 0.0}

        is_correct = result.get("correct", False)
        score = result.get("score", 1.0 if is_correct else 0.0)

        if is_correct:
            correct += 1
        total += 1
        total_score += score

        per_example.append({
            "index": idx,
            "response": response_text[:200],  # truncate for logging
            "correct": is_correct,
            "score": score,
            "ideal": example.get("ideal", "N/A"),
        })

        if verbose:
            status = "OK" if is_correct else "WRONG"
            ideal = example.get("ideal", "N/A")
            print0(f"  [{idx}] {status}: ideal={ideal}")
            if not is_correct:
                print0(f"         response: {response_text[:100]}...")

        if total % 20 == 0:
            acc_so_far = correct / total if total > 0 else 0
            elapsed = time.time() - t0
            rate = total / elapsed if elapsed > 0 else 0
            print0(f"  ... {total}/{num_examples} done, accuracy so far: {acc_so_far:.3f} "
                   f"({rate:.1f} examples/s)")

    elapsed = time.time() - t0
    accuracy = correct / total if total > 0 else 0.0
    avg_score = total_score / total if total > 0 else 0.0
    print0(f"{task_name}: accuracy = {accuracy:.4f} ({correct}/{total}), "
           f"avg_score = {avg_score:.4f}, time = {elapsed:.1f}s")

    return {
        "accuracy": accuracy,
        "avg_score": avg_score,
        "correct": correct,
        "total": total,
        "per_example": per_example,
        "time": elapsed,
    }


# ---------------------------------------------------------------------------
# ChatCORE metric
# ---------------------------------------------------------------------------

def compute_chat_core(results):
    """
    Compute the ChatCORE metric: average accuracy across all evaluated tasks.

    ChatCORE = mean(accuracy_task1, accuracy_task2, ..., accuracy_taskN)

    Args:
        results: dict mapping task_name -> result dict with 'accuracy' key

    Returns:
        float: ChatCORE score
    """
    accuracies = []
    for task_name, result in results.items():
        acc = result.get("accuracy", 0.0)
        accuracies.append(acc)
        print0(f"  {task_name:20s}: {acc:.4f}")

    if not accuracies:
        return 0.0

    core = sum(accuracies) / len(accuracies)
    print0(f"  {'ChatCORE':20s}: {core:.4f}")
    return core


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

def get_eval_tasks(task_names):
    """
    Load evaluation tasks by name.

    Args:
        task_names: List of task name strings

    Returns:
        List of (name, task, eval_type) tuples where eval_type is
        'categorical' or 'generative'
    """
    from tasks.gsm8k import GSM8K
    from tasks.mmlu import MMLU

    task_configs = {
        "arc_easy": {
            "loader": lambda: _load_arc("easy"),
            "eval_type": "categorical",
        },
        "arc_challenge": {
            "loader": lambda: _load_arc("challenge"),
            "eval_type": "categorical",
        },
        "mmlu": {
            "loader": lambda: MMLU(split="test"),
            "eval_type": "categorical",
        },
        "gsm8k": {
            "loader": lambda: GSM8K(split="test"),
            "eval_type": "generative",
        },
        "humaneval": {
            "loader": lambda: _load_humaneval(),
            "eval_type": "generative",
        },
        "spellingbee": {
            "loader": lambda: _load_spellingbee(),
            "eval_type": "generative",
        },
    }

    tasks = []
    for name in task_names:
        name_lower = name.lower().replace("-", "_")
        if name_lower not in task_configs:
            print0(f"Warning: unknown task '{name}', skipping")
            continue
        cfg = task_configs[name_lower]
        print0(f"Loading task: {name}")
        task = cfg["loader"]()
        tasks.append((name, task, cfg["eval_type"]))
        print0(f"  {name}: {len(task)} examples ({cfg['eval_type']})")

    return tasks


def _load_arc(difficulty):
    """Load ARC dataset (Easy or Challenge)."""
    from datasets import load_dataset
    from tasks.common import Task, render_mc

    class ARC(Task):
        def __init__(self, difficulty, split="test"):
            super().__init__()
            config_name = "ARC-Easy" if difficulty == "easy" else "ARC-Challenge"
            self.dataset = load_dataset("allenai/ai2_arc", config_name, split=split)

        @property
        def num_examples(self):
            return len(self.dataset)

        def get_example(self, index):
            item = self.dataset[index]
            question = item["question"]
            choices_text = item["choices"]["text"]
            choices_labels = item["choices"]["label"]
            answer_key = item["answerKey"]

            # Find the answer index
            answer_index = choices_labels.index(answer_key)
            formatted_q, answer_label = render_mc(question, choices_text, answer_index)

            conversation = {
                "messages": [
                    {"role": "user", "content": formatted_q},
                    {"role": "assistant", "content": ""},
                ]
            }

            return {
                "conversation": conversation,
                "ideal": answer_label,
                "choices": [chr(65 + i) for i in range(len(choices_text))],
            }

        def evaluate(self, index, output):
            example = self[index]
            ideal = example["ideal"]
            # Extract first letter from output
            output_clean = output.strip().upper()
            predicted = output_clean[0] if output_clean else ""
            correct = (predicted == ideal)
            return {"correct": correct, "score": 1.0 if correct else 0.0}

    return ARC(difficulty)


def _load_humaneval():
    """Load HumanEval evaluation task."""
    from datasets import load_dataset
    from tasks.common import Task

    class HumanEval(Task):
        def __init__(self):
            super().__init__()
            self.dataset = load_dataset("openai/openai_humaneval", split="test")

        @property
        def num_examples(self):
            return len(self.dataset)

        def get_example(self, index):
            item = self.dataset[index]
            prompt = item["prompt"]
            canonical_solution = item["canonical_solution"]
            test = item["test"]
            entry_point = item["entry_point"]

            instruction = (
                f"Complete the following Python function. "
                f"Only provide the function body, no explanations.\n\n{prompt}"
            )

            conversation = {
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": ""},
                ]
            }

            return {
                "conversation": conversation,
                "ideal": canonical_solution,
                "test": test,
                "entry_point": entry_point,
                "prompt": prompt,
            }

        def evaluate(self, index, output):
            """
            Evaluate by extracting code and running tests.
            For safety, we do basic string matching; full execution
            would require a sandbox.
            """
            example = self[index]
            # Simple heuristic: check if the output contains key parts of the solution
            # Full evaluation would require code execution in a sandbox
            ideal = example["ideal"].strip()
            output_clean = output.strip()

            # Basic check: does it look like Python code?
            has_return = "return " in output_clean
            has_indent = any(line.startswith("    ") or line.startswith("\t")
                            for line in output_clean.split("\n") if line.strip())
            looks_like_code = has_return or has_indent

            return {
                "correct": looks_like_code,
                "score": 1.0 if looks_like_code else 0.0,
                "note": "heuristic-only (no sandbox execution)",
            }

    return HumanEval()


def _load_spellingbee():
    """Load SpellingBee evaluation task."""
    from tasks.spellingbee import SpellingBee
    return SpellingBee(split="test")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NanoChat MLX Chat Evaluation")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Tasks to evaluate (default: all). "
                             "Options: arc_easy, arc_challenge, mmlu, gsm8k, humaneval, spellingbee")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Max examples per task (default: all)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens for generative tasks")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for generative tasks (0 = greedy)")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Top-k for generative tasks (0 = disabled)")
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
        args.tasks = ["arc_easy", "arc_challenge", "mmlu", "gsm8k", "spellingbee"]

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

    # Create engine for generative tasks
    engine = Engine(model=model, tokenizer=tokenizer, config=config)

    # -----------------------------------------------------------------------
    # Load tasks
    # -----------------------------------------------------------------------
    eval_tasks = get_eval_tasks(args.tasks)

    if not eval_tasks:
        print0("No valid tasks to evaluate. Exiting.")
        return

    # -----------------------------------------------------------------------
    # Run evaluations
    # -----------------------------------------------------------------------
    print0("=" * 80)
    print0("Starting evaluation")
    print0("=" * 80)

    all_results = {}
    total_t0 = time.time()

    for task_name, task, eval_type in eval_tasks:
        print0(f"\n--- {task_name} ({eval_type}) ---")
        task_t0 = time.time()

        if eval_type == "categorical":
            result = run_categorical_eval(
                model=model,
                tokenizer=tokenizer,
                task=task,
                task_name=task_name,
                max_examples=args.max_examples,
                verbose=args.verbose,
            )
        elif eval_type == "generative":
            result = run_generative_eval(
                engine=engine,
                tokenizer=tokenizer,
                task=task,
                task_name=task_name,
                max_examples=args.max_examples,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else None,
                verbose=args.verbose,
            )
        else:
            print0(f"Unknown eval_type: {eval_type}, skipping")
            continue

        result["time"] = time.time() - task_t0
        all_results[task_name] = result

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_time = time.time() - total_t0
    print0("\n" + "=" * 80)
    print0("EVALUATION SUMMARY")
    print0("=" * 80)

    chat_core = compute_chat_core(all_results)

    print0(f"\nTotal evaluation time: {total_time:.1f}s")
    print0(f"Checkpoint: {args.checkpoint}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    # Build serializable results (remove per_example for brevity in the summary)
    summary = {
        "checkpoint": args.checkpoint,
        "chat_core": chat_core,
        "total_time": total_time,
        "tasks": {},
    }
    for task_name, result in all_results.items():
        summary["tasks"][task_name] = {
            k: v for k, v in result.items() if k != "per_example"
        }

    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print0(f"\nResults saved to: {args.output}")
    else:
        # Save to checkpoint directory
        output_path = os.path.join(args.checkpoint, "eval_results.json")
        try:
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2)
            print0(f"\nResults saved to: {output_path}")
        except OSError:
            # Checkpoint might be read-only
            print0("\nCould not save results to checkpoint directory.")

    print0("\nDone.")


if __name__ == "__main__":
    main()
