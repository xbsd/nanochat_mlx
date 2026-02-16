"""
Unit tests for the GPT model architecture.
"""

import pytest
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from nanochat_mlx.gpt import GPT, GPTConfig, norm, apply_rotary_emb, has_ve


@pytest.fixture
def tiny_config():
    return GPTConfig(
        sequence_len=128,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="SL",
    )


@pytest.fixture
def tiny_model(tiny_config):
    model = GPT(tiny_config)
    model.init_weights()
    return model


class TestGPTConfig:
    def test_default_config(self):
        config = GPTConfig()
        assert config.sequence_len == 2048
        assert config.vocab_size == 32768
        assert config.n_layer == 12
        assert config.n_head == 6
        assert config.n_kv_head == 6
        assert config.n_embd == 768
        assert config.window_pattern == "SSSL"

    def test_custom_config(self, tiny_config):
        assert tiny_config.n_layer == 2
        assert tiny_config.n_embd == 64


class TestNorm:
    def test_norm_output_shape(self):
        x = mx.random.normal(shape=(2, 10, 64))
        y = norm(x)
        mx.eval(y)
        assert y.shape == x.shape

    def test_norm_approximately_unit_variance(self):
        x = mx.random.normal(shape=(2, 10, 64))
        y = norm(x)
        mx.eval(y)
        # RMSNorm should produce approximately unit RMS
        rms = mx.sqrt(mx.mean(y * y, axis=-1))
        mx.eval(rms)
        np.testing.assert_allclose(rms, 1.0, atol=0.1)


class TestRotaryEmbeddings:
    def test_rotary_shape(self):
        x = mx.random.normal(shape=(2, 10, 4, 16))  # B, T, H, D
        cos = mx.ones((1, 10, 1, 8))  # 1, T, 1, D/2
        sin = mx.zeros((1, 10, 1, 8))
        y = apply_rotary_emb(x, cos, sin)
        mx.eval(y)
        assert y.shape == x.shape

    def test_identity_rotation(self):
        """cos=1, sin=0 should be identity."""
        x = mx.random.normal(shape=(1, 5, 2, 8))
        cos = mx.ones((1, 5, 1, 4))
        sin = mx.zeros((1, 5, 1, 4))
        y = apply_rotary_emb(x, cos, sin)
        mx.eval(x, y)
        np.testing.assert_allclose(np.array(y), np.array(x), atol=1e-5)


class TestHasVE:
    def test_alternating_pattern(self):
        # For n_layer=4, layer 3 is always included
        # Alternating: layers matching (n_layer-1)%2
        n_layer = 4
        ve_layers = [i for i in range(n_layer) if has_ve(i, n_layer)]
        assert 3 in ve_layers  # last layer always included
        assert len(ve_layers) == 2  # alternating = half the layers


class TestGPTModel:
    def test_model_creation(self, tiny_config):
        model = GPT(tiny_config)
        model.init_weights()
        assert model.config == tiny_config

    def test_forward_shape(self, tiny_model, tiny_config):
        B, T = 2, 32
        x = mx.random.randint(0, tiny_config.vocab_size, shape=(B, T))
        logits = tiny_model(x)
        mx.eval(logits)
        assert logits.shape[0] == B
        assert logits.shape[1] == T
        # Padded vocab size may be larger
        assert logits.shape[2] >= tiny_config.vocab_size

    def test_forward_with_targets(self, tiny_model, tiny_config):
        B, T = 2, 32
        x = mx.random.randint(0, tiny_config.vocab_size, shape=(B, T))
        y = mx.random.randint(0, tiny_config.vocab_size, shape=(B, T))
        loss = tiny_model(x, targets=y)
        mx.eval(loss)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0

    def test_gradient_computation(self, tiny_model, tiny_config):
        B, T = 2, 32
        x = mx.random.randint(0, tiny_config.vocab_size, shape=(B, T))
        y = mx.random.randint(0, tiny_config.vocab_size, shape=(B, T))

        def loss_fn(model, x, y):
            return model(x, targets=y)

        loss_and_grad = nn.value_and_grad(tiny_model, loss_fn)
        loss, grads = loss_and_grad(tiny_model, x, y)
        mx.eval(loss, grads)
        assert loss.item() > 0

    def test_estimate_flops(self, tiny_model):
        flops = tiny_model.estimate_flops()
        assert flops > 0

    def test_num_scaling_params(self, tiny_model):
        counts = tiny_model.num_scaling_params()
        assert counts['total'] > 0
        assert counts['transformer_matrices'] > 0
        assert counts['lm_head'] > 0
        assert counts['wte'] > 0


class TestSlidingWindow:
    def test_window_sizes_computed(self, tiny_model):
        assert len(tiny_model._window_sizes) == tiny_model.config.n_layer
        # Last layer should always be full context
        last_window = tiny_model._window_sizes[-1]
        assert last_window == tiny_model.config.sequence_len

    def test_pattern_sssl(self):
        config = GPTConfig(
            sequence_len=128, vocab_size=256,
            n_layer=4, n_head=2, n_kv_head=2, n_embd=64,
            window_pattern="SSSL",
        )
        model = GPT(config)
        ws = model._window_sizes
        assert ws[0] == 64  # S = half context
        assert ws[1] == 64  # S
        assert ws[2] == 64  # S
        assert ws[3] == 128  # Last layer always L
