"""
Checkpoint management for nanochat_mlx.
Save and load model weights using safetensors format.

Ported from nanochat/checkpoint_manager.py - replaces PyTorch save/load
with MLX-native safetensors and numpy serialization.

Key differences from the PyTorch version:
- Model weights saved/loaded via model.save_weights() / model.load_weights()
  using the safetensors format.
- Optimizer state saved as .npz files.
- Metadata saved as JSON.
- No _orig_mod. prefix stripping (no torch.compile on MLX).
- No meta device pattern - GPT is created directly.
- No DDP rank handling (MLX is single-device).
"""

import os
import json
import logging

import mlx.core as mx
import numpy as np

from nanochat_mlx.common import get_base_dir

logger = logging.getLogger(__name__)


def save_checkpoint(
    path,
    model,
    optimizer=None,
    step=0,
    best_val_loss=None,
    dataloader_state_dict=None,
    config=None,
    metadata=None,
):
    """Save a checkpoint to disk.

    Saves model weights as safetensors and metadata as JSON.

    Args:
        path: Directory to save into.
        model: The GPT model.
        optimizer: Optional MuonAdamW optimizer (state saved as npz).
        step: Current training step.
        best_val_loss: Best validation loss so far.
        dataloader_state_dict: Dataloader resumption state dict.
        config: Model or training config dict (or dataclass).
        metadata: Additional metadata dict.
    """
    os.makedirs(path, exist_ok=True)

    # Save model weights
    weights_path = os.path.join(path, "model.safetensors")
    model.save_weights(weights_path)

    # Build metadata
    meta = {"step": step}
    if best_val_loss is not None:
        meta["best_val_loss"] = best_val_loss
    if dataloader_state_dict is not None:
        meta["dataloader_state_dict"] = dataloader_state_dict
    if config is not None:
        if hasattr(config, "__dict__"):
            meta["config"] = vars(config) if not isinstance(config, dict) else config
        else:
            meta["config"] = config
    if metadata is not None:
        meta.update(metadata)

    # Save model config separately for easy reconstruction
    if hasattr(model, "config"):
        from dataclasses import asdict
        try:
            meta["model_config"] = asdict(model.config)
        except Exception:
            meta["model_config"] = vars(model.config)

    meta_path = os.path.join(path, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # Save optimizer state if provided
    if optimizer is not None and hasattr(optimizer, "state") and optimizer.state:
        opt_path = os.path.join(path, "optimizer.npz")
        opt_arrays = {}
        for name, state_dict in optimizer.state.items():
            for key, val in state_dict.items():
                if isinstance(val, mx.array):
                    arr = val.astype(mx.float32) if val.dtype == mx.bfloat16 else val
                    opt_arrays[f"{name}@@{key}"] = np.array(arr)
        if opt_arrays:
            np.savez(opt_path, **opt_arrays)
        # Also save step count
        if hasattr(optimizer, "step_count"):
            meta["optimizer_step_count"] = optimizer.step_count
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, default=str)

    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(path, model, optimizer=None):
    """Load a checkpoint from disk.

    Args:
        path: Directory containing the checkpoint.
        model: The GPT model to load weights into.
        optimizer: Optional optimizer to load state into.

    Returns:
        Metadata dict with keys like 'step', 'best_val_loss', etc.
        Returns None if the checkpoint doesn't exist.
    """
    weights_path = os.path.join(path, "model.safetensors")
    if not os.path.exists(weights_path):
        logger.warning(f"No checkpoint found at {path}")
        return None

    # Load model weights
    model.load_weights(weights_path)

    # Load metadata
    meta = {}
    meta_path = os.path.join(path, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    # Load optimizer state if available and requested
    if optimizer is not None:
        opt_path = os.path.join(path, "optimizer.npz")
        if os.path.exists(opt_path):
            data = np.load(opt_path)
            for full_key in data.files:
                parts = full_key.split("@@", 1)
                if len(parts) == 2:
                    name, key = parts
                    if name not in optimizer.state:
                        optimizer.state[name] = {}
                    optimizer.state[name][key] = mx.array(data[full_key])
            if hasattr(optimizer, "step_count"):
                optimizer.step_count = meta.get("optimizer_step_count",
                                                 meta.get("step", 0))
            logger.info(f"Loaded optimizer state from {opt_path}")
        else:
            logger.warning(f"Optimizer state not found at {opt_path}")

    return meta


def load_model(path, phase=None):
    """Load a model from a checkpoint directory or source name.

    Args:
        path: Either a directory path or a source name ("base", "sft", "rl").
        phase: Optional phase string ("train" or "eval"). Used when loading
               by source name.

    Returns:
        (model, config) tuple when loading from a path.
        (model, tokenizer, meta) tuple when loading by source name.
    """
    from nanochat_mlx.gpt import GPT, GPTConfig

    # Check if path is a source name
    source_dirs = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }
    if path in source_dirs:
        return _load_model_by_source(path, phase or "eval")

    # Load from directory path
    meta_path = os.path.join(path, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        model_config_raw = meta.get("model_config", {})
    else:
        meta = {}
        model_config_raw = {}

    # Build GPTConfig from saved values, mapping alternative field names
    config_kwargs = {}
    field_map = {
        "vocab_size": "vocab_size",
        "sequence_len": "sequence_len",
        "sequence_length": "sequence_len",
        "n_layer": "n_layer",
        "n_layers": "n_layer",
        "n_head": "n_head",
        "n_heads": "n_head",
        "n_kv_head": "n_kv_head",
        "n_kv_heads": "n_kv_head",
        "n_embd": "n_embd",
        "d_model": "n_embd",
        "window_pattern": "window_pattern",
    }
    for src_key, dst_key in field_map.items():
        if src_key in model_config_raw and dst_key not in config_kwargs:
            config_kwargs[dst_key] = model_config_raw[src_key]

    config = GPTConfig(**config_kwargs) if config_kwargs else GPTConfig()
    model = GPT(config)

    # Load weights
    weights_path = os.path.join(path, "model.safetensors")
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        logger.warning(f"No weights at {weights_path}, using random init")
        model.init_weights()

    mx.eval(model.parameters())
    return model, config


def _load_model_by_source(source, phase):
    """Load model using source name ('base', 'sft', 'rl') from standard dirs."""
    import glob
    import re
    from nanochat_mlx.gpt import GPT, GPTConfig
    from nanochat_mlx.tokenizer import get_tokenizer

    source_dirs = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, source_dirs[source])

    if not os.path.exists(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoints dir not found: {checkpoints_dir}")

    # Find the best model tag (largest depth)
    model_tags = [
        f for f in os.listdir(checkpoints_dir)
        if os.path.isdir(os.path.join(checkpoints_dir, f))
    ]
    if not model_tags:
        raise FileNotFoundError(f"No model dirs in {checkpoints_dir}")

    # Try depth-based naming (d12, d24, etc.)
    depth_tags = []
    for tag in model_tags:
        m = re.match(r"d(\d+)", tag)
        if m:
            depth_tags.append((int(m.group(1)), tag))
    if depth_tags:
        depth_tags.sort(reverse=True)
        model_tag = depth_tags[0][1]
    else:
        model_tag = sorted(model_tags)[-1]

    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)

    # Find latest step
    step_files = glob.glob(os.path.join(checkpoint_dir, "model_*.safetensors"))
    step_files += glob.glob(os.path.join(checkpoint_dir, "model_*.npz"))
    if not step_files:
        # Try flat directory structure
        if os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")):
            model, config = load_model(checkpoint_dir)
            tokenizer = get_tokenizer()
            return model, tokenizer, {}
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")

    steps = []
    for f in step_files:
        basename = os.path.basename(f)
        step_str = basename.split("_")[-1].split(".")[0]
        try:
            steps.append(int(step_str))
        except ValueError:
            continue
    step = max(steps)

    # Load metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    else:
        meta = {}

    # Build model
    model_config_kwargs = meta.get("model_config", {})
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"

    config = GPTConfig(**model_config_kwargs) if model_config_kwargs else GPTConfig()
    model = GPT(config)

    # Load weights
    weights_path = os.path.join(checkpoint_dir, f"model_{step:06d}.safetensors")
    if os.path.exists(weights_path):
        model.load_weights(weights_path, strict=False)
    else:
        npz_path = os.path.join(checkpoint_dir, f"model_{step:06d}.npz")
        if os.path.exists(npz_path):
            weights = dict(mx.load(npz_path))
            model.load_weights(list(weights.items()), strict=False)

    mx.eval(model.parameters())
    tokenizer = get_tokenizer()
    return model, tokenizer, meta
