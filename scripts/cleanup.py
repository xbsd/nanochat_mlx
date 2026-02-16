"""
Cleanup script for nanochat_mlx.

Removes downloaded data, cached files, training checkpoints, and
HuggingFace dataset caches created during training and evaluation.

Usage:
    python -m scripts.cleanup              # Interactive (prompts before deleting)
    python -m scripts.cleanup --all        # Delete everything without prompting
    python -m scripts.cleanup --dry-run    # Show what would be deleted
    python -m scripts.cleanup --data       # Only delete training data shards
    python -m scripts.cleanup --runs       # Only delete training run checkpoints
    python -m scripts.cleanup --tokenizer  # Only delete tokenizer files
    python -m scripts.cleanup --hf-cache   # Only delete HuggingFace datasets cache
    python -m scripts.cleanup --sft        # Only delete local SFT run outputs
"""

import os
import sys
import glob
import shutil
import argparse


def get_base_dir():
    """Mirror of nanochat_mlx.common.get_base_dir() to avoid import dependencies."""
    if os.environ.get("NANOCHAT_BASE_DIR"):
        return os.environ["NANOCHAT_BASE_DIR"]
    return os.path.join(os.path.expanduser("~"), ".cache", "nanochat")


def dir_size(path):
    """Return total size of a directory in bytes."""
    total = 0
    if not os.path.exists(path):
        return 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def fmt_size(nbytes):
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def find_targets(base_dir):
    """Find all cleanup targets and return as list of (name, path, description) tuples."""
    targets = []

    # 1. Training data shards
    data_dir = os.path.join(base_dir, "base_data")
    if os.path.exists(data_dir):
        n_files = len(glob.glob(os.path.join(data_dir, "*.parquet")))
        targets.append((
            "data",
            data_dir,
            f"Training data shards ({n_files} parquet files, {fmt_size(dir_size(data_dir))})",
        ))

    # 2. Tokenizer files
    tok_dir = os.path.join(base_dir, "tokenizer")
    if os.path.exists(tok_dir):
        targets.append((
            "tokenizer",
            tok_dir,
            f"Tokenizer files ({fmt_size(dir_size(tok_dir))})",
        ))

    # 3. Training run checkpoints
    runs_dir = os.path.join(base_dir, "runs")
    if os.path.exists(runs_dir):
        run_names = [d for d in os.listdir(runs_dir)
                     if os.path.isdir(os.path.join(runs_dir, d))]
        targets.append((
            "runs",
            runs_dir,
            f"Training run checkpoints ({len(run_names)} runs: {', '.join(sorted(run_names)) or 'none'}, {fmt_size(dir_size(runs_dir))})",
        ))

    # 4. Legacy checkpoint dirs
    for name, dirname in [("base_checkpoints", "base_checkpoints"),
                          ("sft_checkpoints", "chatsft_checkpoints"),
                          ("rl_checkpoints", "chatrl_checkpoints")]:
        ckpt_dir = os.path.join(base_dir, dirname)
        if os.path.exists(ckpt_dir):
            targets.append((
                "runs",
                ckpt_dir,
                f"Legacy {name} ({fmt_size(dir_size(ckpt_dir))})",
            ))

    # 5. Evaluation report
    report_path = os.path.join(base_dir, "report.json")
    if os.path.exists(report_path):
        targets.append((
            "runs",
            report_path,
            "Evaluation report (report.json)",
        ))

    # 6. Spelling words file
    spelling_path = os.path.join(base_dir, "spelling_words.txt")
    if os.path.exists(spelling_path):
        targets.append((
            "data",
            spelling_path,
            f"Spelling words list ({fmt_size(os.path.getsize(spelling_path))})",
        ))

    # 7. Local SFT run outputs (in cwd)
    cwd_runs = os.path.join(os.getcwd(), "runs")
    if os.path.exists(cwd_runs):
        sft_dirs = [d for d in os.listdir(cwd_runs)
                    if d.startswith("chat_sft_") and os.path.isdir(os.path.join(cwd_runs, d))]
        if sft_dirs:
            targets.append((
                "sft",
                cwd_runs,
                f"Local SFT outputs ({len(sft_dirs)} runs in ./runs/, {fmt_size(dir_size(cwd_runs))})",
            ))

    # 8. HuggingFace datasets cache
    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")
    if os.path.exists(hf_cache):
        targets.append((
            "hf-cache",
            hf_cache,
            f"HuggingFace datasets cache ({fmt_size(dir_size(hf_cache))})",
        ))

    # 9. HuggingFace hub cache (downloaded model files, dataset files)
    hf_hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    if os.path.exists(hf_hub):
        targets.append((
            "hf-cache",
            hf_hub,
            f"HuggingFace hub cache ({fmt_size(dir_size(hf_hub))})",
        ))

    return targets


def delete_path(path, dry_run=False):
    """Delete a file or directory."""
    if dry_run:
        print(f"  [dry-run] Would delete: {path}")
        return
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)
    print(f"  Deleted: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up nanochat_mlx downloaded data, caches, and checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--all", action="store_true",
                        help="Delete everything without prompting")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without actually deleting")
    parser.add_argument("--data", action="store_true",
                        help="Delete training data shards and spelling words")
    parser.add_argument("--runs", action="store_true",
                        help="Delete training run checkpoints and evaluation reports")
    parser.add_argument("--tokenizer", action="store_true",
                        help="Delete tokenizer files")
    parser.add_argument("--hf-cache", action="store_true",
                        help="Delete HuggingFace datasets and hub cache")
    parser.add_argument("--sft", action="store_true",
                        help="Delete local SFT run outputs in ./runs/")

    args = parser.parse_args()

    base_dir = get_base_dir()
    targets = find_targets(base_dir)

    if not targets:
        print("Nothing to clean up.")
        return

    # Determine which categories to delete
    specific = args.data or args.runs or args.tokenizer or args.hf_cache or args.sft
    if args.all:
        selected_categories = {"data", "tokenizer", "runs", "hf-cache", "sft"}
    elif specific:
        selected_categories = set()
        if args.data:
            selected_categories.add("data")
        if args.runs:
            selected_categories.add("runs")
        if args.tokenizer:
            selected_categories.add("tokenizer")
        if args.hf_cache:
            selected_categories.add("hf-cache")
        if args.sft:
            selected_categories.add("sft")
    else:
        selected_categories = None  # interactive mode

    # Show what we found
    print(f"\nNanoChat MLX data directory: {base_dir}")
    print(f"{'=' * 60}")

    if selected_categories is not None:
        # Non-interactive: filter and delete
        to_delete = [(name, path, desc) for name, path, desc in targets
                     if name in selected_categories]
        if not to_delete:
            print("No matching files found for the selected categories.")
            return

        for name, path, desc in to_delete:
            print(f"\n  {desc}")
            delete_path(path, dry_run=args.dry_run)
    else:
        # Interactive mode
        print("\nFound the following items:\n")
        for i, (name, path, desc) in enumerate(targets, 1):
            print(f"  {i}. [{name}] {desc}")
            print(f"     {path}")

        print(f"\nOptions:")
        print(f"  a - Delete ALL of the above")
        print(f"  1-{len(targets)} - Delete specific item(s) (comma-separated)")
        print(f"  q - Quit without deleting\n")

        choice = input("Choice: ").strip().lower()

        if choice == "q" or choice == "":
            print("Cancelled.")
            return
        elif choice == "a":
            indices = list(range(len(targets)))
        else:
            try:
                indices = [int(x.strip()) - 1 for x in choice.split(",")]
                for idx in indices:
                    if idx < 0 or idx >= len(targets):
                        print(f"Invalid selection: {idx + 1}")
                        return
            except ValueError:
                print("Invalid input.")
                return

        print()
        for idx in indices:
            name, path, desc = targets[idx]
            print(f"  Deleting: {desc}")
            delete_path(path, dry_run=args.dry_run)

    # Clean up empty nanochat dir
    if not args.dry_run and os.path.exists(base_dir) and not os.listdir(base_dir):
        os.rmdir(base_dir)
        print(f"\n  Removed empty directory: {base_dir}")

    print(f"\n{'=' * 60}")
    if args.dry_run:
        print("Dry run complete. No files were deleted.")
    else:
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
