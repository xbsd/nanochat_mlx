"""
Full pipeline dry run for NanoChat MLX.
Tests the entire pipeline with a tiny model and synthetic data.

Run as:
    python -m scripts.dry_run

This verifies:
1. Model creation and weight initialization
2. Forward pass produces correct shapes
3. Loss computation and gradient flow
4. Optimizer update step
5. Loss decreases over multiple steps
6. Checkpoint save and load roundtrip
7. Inference with KV cache
8. All components work together end-to-end

Target: completes in <30 seconds, uses <1GB memory
"""

import os
import sys
import time
import tempfile
import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from nanochat_mlx.gpt import GPT, GPTConfig
from nanochat_mlx.optim import MuonAdamW
from nanochat_mlx.engine import KVCache
from nanochat_mlx.common import print0


def test_model_creation():
    """Test 1: Model creation and weight initialization."""
    print0("=" * 60)
    print0("Test 1: Model creation and weight initialization")
    print0("=" * 60)

    config = GPTConfig(
        sequence_len=128,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="SL",
    )
    model = GPT(config)
    model.init_weights()

    # Verify model exists and has the right config
    assert model.config.n_layer == 2
    assert model.config.n_embd == 64
    assert model.config.n_head == 2

    print0(f"  Model created with config: depth={config.n_layer}, dim={config.n_embd}")
    print0("  PASSED")
    return model, config


def test_forward_pass(model, config):
    """Test 2: Forward pass produces correct output shape."""
    print0("=" * 60)
    print0("Test 2: Forward pass shape check")
    print0("=" * 60)

    B, T = 2, 64
    x = mx.random.randint(0, config.vocab_size, shape=(B, T))

    logits = model(x)
    mx.eval(logits)

    expected_shape = (B, T, config.vocab_size)
    # The model may pad vocab size - logits could be larger
    assert logits.shape[0] == B, f"Batch dim mismatch: {logits.shape[0]} != {B}"
    assert logits.shape[1] == T, f"Seq dim mismatch: {logits.shape[1]} != {T}"
    assert logits.shape[2] >= config.vocab_size, f"Vocab dim too small: {logits.shape[2]} < {config.vocab_size}"

    print0(f"  Input shape: ({B}, {T})")
    print0(f"  Output shape: {logits.shape}")
    print0("  PASSED")


def test_loss_computation(model, config):
    """Test 3: Loss computation works."""
    print0("=" * 60)
    print0("Test 3: Loss computation")
    print0("=" * 60)

    B, T = 2, 64
    x = mx.random.randint(0, config.vocab_size, shape=(B, T))
    y = mx.random.randint(0, config.vocab_size, shape=(B, T))

    loss = model(x, targets=y)
    mx.eval(loss)

    loss_val = loss.item()
    assert loss_val > 0, f"Loss should be positive, got {loss_val}"
    assert not np.isnan(loss_val), "Loss is NaN"
    assert not np.isinf(loss_val), "Loss is infinite"

    # For random predictions over vocab_size=256, loss should be around ln(256) ~ 5.5
    assert loss_val < 10.0, f"Loss seems too high: {loss_val}"

    print0(f"  Loss value: {loss_val:.4f} (expected ~{np.log(config.vocab_size):.2f} for random)")
    print0("  PASSED")


def test_gradient_flow(model, config):
    """Test 4: Gradients flow through the model."""
    print0("=" * 60)
    print0("Test 4: Gradient flow")
    print0("=" * 60)

    B, T = 2, 64
    x = mx.random.randint(0, config.vocab_size, shape=(B, T))
    y = mx.random.randint(0, config.vocab_size, shape=(B, T))

    def loss_fn(model, x, y):
        return model(x, targets=y)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model, x, y)
    mx.eval(loss, grads)

    # Check that at least some gradients are non-zero
    has_nonzero_grads = False

    def check_grads(g):
        nonlocal has_nonzero_grads
        if isinstance(g, mx.array):
            if mx.any(g != 0).item():
                has_nonzero_grads = True
        return g

    tree_map(check_grads, grads)
    assert has_nonzero_grads, "All gradients are zero - something is wrong"

    print0(f"  Loss: {loss.item():.4f}")
    print0("  Non-zero gradients detected: YES")
    print0("  PASSED")
    return loss_fn


def test_optimizer_step(model, config, loss_fn):
    """Test 5: Optimizer updates parameters and loss decreases."""
    print0("=" * 60)
    print0("Test 5: Optimizer step and loss decrease")
    print0("=" * 60)

    # Create optimizer
    param_groups = model.get_param_groups(
        unembedding_lr=0.001,
        embedding_lr=0.01,
        matrix_lr=0.01,
        weight_decay=0.0,
        adam_betas=(0.9, 0.999),
        scalar_lr=0.01,
    )
    optimizer = MuonAdamW(param_groups)

    # Fixed synthetic data
    mx.random.seed(42)
    B, T = 2, 64
    x = mx.random.randint(0, config.vocab_size, shape=(B, T))
    y = mx.random.randint(0, config.vocab_size, shape=(B, T))
    mx.eval(x, y)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    losses = []
    num_steps = 10
    for step in range(num_steps):
        loss, grads = loss_and_grad_fn(model, x, y)
        mx.eval(loss)
        losses.append(loss.item())

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    print0(f"  Step  0 loss: {losses[0]:.4f}")
    print0(f"  Step {num_steps-1} loss: {losses[-1]:.4f}")

    # Loss should generally decrease (allow some noise)
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    print0("  Loss decreased: YES")
    print0("  PASSED")
    return losses


def test_checkpoint_roundtrip(model, config):
    """Test 6: Checkpoint save and load roundtrip."""
    print0("=" * 60)
    print0("Test 6: Checkpoint save/load roundtrip")
    print0("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save model weights
        weights_path = os.path.join(tmpdir, "model.safetensors")
        model.save_weights(weights_path)
        print0(f"  Saved weights to {weights_path}")

        # Get output before reload
        B, T = 1, 32
        x = mx.random.randint(0, config.vocab_size, shape=(B, T))
        mx.eval(x)
        out_before = model(x)
        mx.eval(out_before)

        # Create new model and load weights
        model2 = GPT(config)
        model2.load_weights(weights_path)

        out_after = model2(x)
        mx.eval(out_after)

        # Check outputs match
        diff = mx.abs(out_before - out_after).max().item()
        assert diff < 1e-4, f"Outputs differ after load: max diff = {diff}"

        print0(f"  Max output difference after load: {diff:.2e}")
        print0("  PASSED")


def test_kv_cache_inference(model, config):
    """Test 7: KV cache inference produces valid tokens."""
    print0("=" * 60)
    print0("Test 7: KV cache inference")
    print0("=" * 60)

    kv_cache = KVCache(config.n_layer)

    # Prefill with a short sequence
    prompt = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    logits = model(prompt, kv_cache=kv_cache)
    mx.eval(logits)

    assert logits.shape == (1, 5, config.vocab_size), f"Prefill shape wrong: {logits.shape}"

    # Generate a few tokens autoregressively
    next_token = mx.argmax(logits[:, -1:, :], axis=-1)
    generated = [next_token.item()]

    for _ in range(4):
        logits = model(next_token, kv_cache=kv_cache)
        mx.eval(logits)
        next_token = mx.argmax(logits[:, -1:, :], axis=-1)
        generated.append(next_token.item())

    print0(f"  Prompt: [1, 2, 3, 4, 5]")
    print0(f"  Generated: {generated}")
    assert len(generated) == 5, f"Expected 5 tokens, got {len(generated)}"
    assert all(0 <= t < config.vocab_size for t in generated), "Generated token out of range"

    print0("  PASSED")


def main():
    print0("=" * 60)
    print0("NanoChat MLX - Full Pipeline Dry Run")
    print0("=" * 60)
    print0()

    t_start = time.time()
    mx.random.seed(42)

    # Test 1: Model creation
    model, config = test_model_creation()
    print0()

    # Test 2: Forward pass
    test_forward_pass(model, config)
    print0()

    # Test 3: Loss computation
    test_loss_computation(model, config)
    print0()

    # Test 4: Gradient flow
    loss_fn = test_gradient_flow(model, config)
    print0()

    # Reinitialize model for clean optimizer test
    model = GPT(config)
    model.init_weights()

    # Test 5: Optimizer step
    losses = test_optimizer_step(model, config, loss_fn)
    print0()

    # Test 6: Checkpoint roundtrip
    test_checkpoint_roundtrip(model, config)
    print0()

    # Test 7: KV cache inference
    # Reinitialize for clean inference test
    model = GPT(config)
    model.init_weights()
    test_kv_cache_inference(model, config)
    print0()

    t_end = time.time()
    elapsed = t_end - t_start

    print0("=" * 60)
    print0(f"ALL TESTS PASSED in {elapsed:.1f}s")
    print0("=" * 60)

    # Memory report
    try:
        from nanochat_mlx.common import get_memory_usage
        mem_mb = get_memory_usage() / 1024 / 1024
        print0(f"Current memory usage: {mem_mb:.1f} MB")
    except Exception:
        pass


if __name__ == "__main__":
    main()
