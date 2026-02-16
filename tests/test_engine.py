"""
Unit tests for the inference engine with KV cache.
"""

import pytest
import numpy as np
import mlx.core as mx

from nanochat_mlx.gpt import GPT, GPTConfig
from nanochat_mlx.engine import KVCache, sample_next_token


@pytest.fixture
def tiny_config():
    return GPTConfig(
        sequence_len=128,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
    )


@pytest.fixture
def tiny_model(tiny_config):
    model = GPT(tiny_config)
    model.init_weights()
    return model


class TestKVCache:
    def test_creation(self):
        cache = KVCache(num_layers=4)
        assert cache.get_pos() == 0
        assert len(cache.keys) == 4

    def test_update_and_get(self):
        cache = KVCache(num_layers=2)
        k = mx.random.normal(shape=(1, 2, 5, 32))  # B, H, T, D
        v = mx.random.normal(shape=(1, 2, 5, 32))
        cache.update(0, k, v)
        mx.eval(k, v)

        k_out, v_out = cache.get(0)
        assert k_out.shape == (1, 2, 5, 32)
        assert v_out.shape == (1, 2, 5, 32)

    def test_concatenation(self):
        cache = KVCache(num_layers=1)
        k1 = mx.random.normal(shape=(1, 2, 3, 16))
        v1 = mx.random.normal(shape=(1, 2, 3, 16))
        cache.update(0, k1, v1)

        k2 = mx.random.normal(shape=(1, 2, 1, 16))
        v2 = mx.random.normal(shape=(1, 2, 1, 16))
        cache.update(0, k2, v2)

        k_out, v_out = cache.get(0)
        mx.eval(k_out, v_out)
        assert k_out.shape == (1, 2, 4, 16)  # 3 + 1
        assert v_out.shape == (1, 2, 4, 16)

    def test_advance(self):
        cache = KVCache(num_layers=1)
        assert cache.get_pos() == 0
        cache.advance(5)
        assert cache.get_pos() == 5
        cache.advance(3)
        assert cache.get_pos() == 8

    def test_reset(self):
        cache = KVCache(num_layers=2)
        k = mx.random.normal(shape=(1, 2, 5, 16))
        v = mx.random.normal(shape=(1, 2, 5, 16))
        cache.update(0, k, v)
        cache.advance(5)

        cache.reset()
        assert cache.get_pos() == 0
        assert cache.keys[0] is None


class TestSampling:
    def test_temperature_zero_deterministic(self):
        logits = mx.array([[1.0, 2.0, 3.0, 0.5]])
        token1 = sample_next_token(logits, temperature=0.0)
        token2 = sample_next_token(logits, temperature=0.0)
        mx.eval(token1, token2)
        assert token1.item() == token2.item() == 2  # argmax

    def test_top_k(self):
        mx.random.seed(42)
        logits = mx.array([[10.0, 0.0, 0.0, 0.0, 0.0]])
        token = sample_next_token(logits, temperature=1.0, top_k=1)
        mx.eval(token)
        assert token.item() == 0  # Only top-1 candidate

    def test_output_shape(self):
        logits = mx.random.normal(shape=(3, 100))
        token = sample_next_token(logits, temperature=1.0)
        mx.eval(token)
        assert token.shape == (3, 1)


class TestInference:
    def test_prefill_and_generate(self, tiny_model, tiny_config):
        """Test that prefill + autoregressive generation works."""
        kv_cache = KVCache(tiny_config.n_layer)

        # Prefill
        prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
        logits = tiny_model(prompt, kv_cache=kv_cache)
        mx.eval(logits)

        assert logits.shape[1] == 3
        assert kv_cache.get_pos() == 3  # auto-inferred from cache

        # Generate one token
        next_token = mx.argmax(logits[:, -1:, :], axis=-1)
        logits2 = tiny_model(next_token, kv_cache=kv_cache)
        mx.eval(logits2)

        assert logits2.shape[1] == 1
        assert kv_cache.get_pos() == 4

    def test_consistency(self, tiny_model, tiny_config):
        """Test that full forward and cached forward produce same logits for last token."""
        seq = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)

        # Full forward (no cache)
        logits_full = tiny_model(seq)
        mx.eval(logits_full)
        last_logit_full = logits_full[:, -1, :]

        # Cached forward
        kv_cache = KVCache(tiny_config.n_layer)
        # Prefill with first 4 tokens
        logits_cached = tiny_model(seq[:, :4], kv_cache=kv_cache)
        mx.eval(logits_cached)
        # Then generate 5th token (pos auto-inferred from cache)
        logits_last = tiny_model(seq[:, 4:5], kv_cache=kv_cache)
        mx.eval(logits_last)
        last_logit_cached = logits_last[:, -1, :]

        # Should be close (not exact due to floating point)
        diff = mx.abs(last_logit_full - last_logit_cached).max().item()
        assert diff < 0.1, f"Cached vs full mismatch: {diff}"
