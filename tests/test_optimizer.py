"""
Unit tests for the MuonAdamW optimizer.
"""

import pytest
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from nanochat_mlx.gpt import GPT, GPTConfig
from nanochat_mlx.optim import MuonAdamW


@pytest.fixture
def tiny_config():
    return GPTConfig(
        sequence_len=64,
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


class TestMuonAdamW:
    def test_creation(self, tiny_model):
        param_groups = tiny_model.get_param_groups(
            unembedding_lr=0.001,
            embedding_lr=0.01,
            matrix_lr=0.01,
            weight_decay=0.0,
            adam_betas=(0.9, 0.999),
            scalar_lr=0.01,
        )
        optimizer = MuonAdamW(param_groups)
        assert optimizer is not None

    def test_update_step(self, tiny_model, tiny_config):
        param_groups = tiny_model.get_param_groups(
            unembedding_lr=0.01,
            embedding_lr=0.01,
            matrix_lr=0.01,
            weight_decay=0.0,
            adam_betas=(0.9, 0.999),
            scalar_lr=0.01,
        )
        optimizer = MuonAdamW(param_groups)

        # Compute loss and grads
        x = mx.random.randint(0, tiny_config.vocab_size, shape=(2, 32))
        y = mx.random.randint(0, tiny_config.vocab_size, shape=(2, 32))

        def loss_fn(model, x, y):
            return model(x, targets=y)

        loss_and_grad_fn = nn.value_and_grad(tiny_model, loss_fn)
        loss, grads = loss_and_grad_fn(tiny_model, x, y)
        mx.eval(loss, grads)

        loss_before = loss.item()

        # Update
        optimizer.update(tiny_model, grads)
        mx.eval(tiny_model.parameters(), optimizer.state)

        # Compute loss again - should be different
        loss2 = loss_fn(tiny_model, x, y)
        mx.eval(loss2)
        assert loss2.item() != loss_before  # Parameters changed

    def test_loss_decreases(self, tiny_model, tiny_config):
        param_groups = tiny_model.get_param_groups(
            unembedding_lr=0.01,
            embedding_lr=0.01,
            matrix_lr=0.01,
            weight_decay=0.0,
            adam_betas=(0.9, 0.999),
            scalar_lr=0.01,
        )
        optimizer = MuonAdamW(param_groups)

        # Fixed data
        mx.random.seed(123)
        x = mx.random.randint(0, tiny_config.vocab_size, shape=(2, 32))
        y = mx.random.randint(0, tiny_config.vocab_size, shape=(2, 32))
        mx.eval(x, y)

        def loss_fn(model, x, y):
            return model(x, targets=y)

        loss_and_grad_fn = nn.value_and_grad(tiny_model, loss_fn)

        losses = []
        for _ in range(5):
            loss, grads = loss_and_grad_fn(tiny_model, x, y)
            mx.eval(loss)
            losses.append(loss.item())
            optimizer.update(tiny_model, grads)
            mx.eval(tiny_model.parameters(), optimizer.state)

        # Loss should decrease
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

    def test_lr_update(self, tiny_model):
        param_groups = tiny_model.get_param_groups(
            unembedding_lr=0.01,
            embedding_lr=0.01,
            matrix_lr=0.01,
            weight_decay=0.0,
            adam_betas=(0.9, 0.999),
            scalar_lr=0.01,
        )
        optimizer = MuonAdamW(param_groups)
        optimizer.update_lr(0.5)  # Scale LRs by 0.5
        # Should not crash
