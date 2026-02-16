"""
GPT model for nanochat_mlx.
Ported from nanochat/gpt.py (PyTorch) to Apple MLX.

Key differences from the PyTorch version:
- Uses mlx.nn.Module with __call__ instead of forward
- No nn.Parameter; uses plain mx.array attributes
- Functional RMSNorm (no learnable params)
- Manual RoPE implementation with mx.arange/cos/sin
- mx.fast.scaled_dot_product_attention with explicit mask for sliding window
- Squared ReLU via mx.maximum(x, 0) ** 2
- Logit softcapping at 15
- No torch.compile, no DDP, no meta device
- Lazy evaluation (no need for @torch.no_grad)
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from nanochat_mlx.common import print0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm(x: mx.array) -> mx.array:
    """Functional RMSNorm without learnable parameters."""
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Determine whether a layer uses value embeddings (ResFormer-style).
    Alternating layers get value embeddings; the parity is chosen so the
    last layer always has one."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply rotary positional embeddings.

    x: (B, T, n_heads, head_dim)
    cos, sin: (1, T, 1, head_dim//2)
    """
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return mx.concatenate([y1, y2], axis=3)


# ---------------------------------------------------------------------------
# CausalSelfAttention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # Value embedding gate (only on layers that use VE)
        self.ve_gate_channels = 32
        if has_ve(layer_idx, config.n_layer):
            self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
        else:
            self.ve_gate = None

    def __call__(
        self,
        x: mx.array,
        ve: Optional[mx.array],
        cos: mx.array,
        sin: mx.array,
        additive_mask: Optional[mx.array],
        kv_cache=None,
    ) -> mx.array:
        B, T, C = x.shape

        # Project Q, K, V
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Apply value embedding gate
        if ve is not None and self.ve_gate is not None:
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * mx.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            # gate: (B, T, n_kv_head) -> (B, T, n_kv_head, 1) for broadcasting
            v = v + mx.expand_dims(gate, axis=-1) * ve

        # Apply RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # QK-norm: apply RMSNorm to Q and K after RoPE
        q = norm(q)
        k = norm(k)

        # Transpose q to (B, n_head, T, head_dim) for SDPA
        q = q.transpose(0, 2, 1, 3)

        if kv_cache is not None:
            # Store k/v in transposed format: (B, n_kv_head, T, head_dim)
            k_t = k.transpose(0, 2, 1, 3)
            v_t = v.transpose(0, 2, 1, 3)
            kv_cache.update(self.layer_idx, k_t, v_t)
            # Get returns (B, n_kv_head, T_total, head_dim) already transposed
            k, v = kv_cache.get(self.layer_idx)
        else:
            # Transpose k/v to (B, n_kv_head, T, head_dim) for SDPA
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        # For GQA, mx.fast.scaled_dot_product_attention handles n_kv_head < n_head natively
        scale = self.head_dim ** -0.5
        if additive_mask is not None:
            y = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=additive_mask
            )
        else:
            # Inference with KV cache: no mask needed (single token queries are causal by nature)
            y = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale
            )

        # Transpose back and reshape: (B, n_head, T, head_dim) -> (B, T, n_embd)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)

        # Output projection
        y = self.c_proj(y)
        return y


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.c_fc(x)
        # Squared ReLU activation
        x = mx.maximum(x, 0) ** 2
        x = self.c_proj(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int, padded_vocab_size: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

        # Value embedding (ResFormer-style) — only on alternating layers
        if has_ve(layer_idx, config.n_layer):
            head_dim = config.n_embd // config.n_head
            kv_dim = config.n_kv_head * head_dim
            self.ve = nn.Embedding(padded_vocab_size, kv_dim)
        else:
            self.ve = None

    def __call__(
        self,
        x: mx.array,
        idx: mx.array,
        cos: mx.array,
        sin: mx.array,
        additive_mask: Optional[mx.array],
        kv_cache=None,
    ) -> mx.array:
        ve = self.ve(idx) if self.ve is not None else None
        x = x + self.attn(norm(x), ve, cos, sin, additive_mask, kv_cache)
        x = x + self.mlp(norm(x))
        return x


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, config: GPTConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config

        # Compute sliding window sizes per layer (underscore prefix to keep out of module tree)
        self._window_sizes = self._compute_window_sizes(config)

        # Pad vocab for memory alignment efficiency
        self.padded_vocab_size = (
            (config.vocab_size + pad_vocab_size_to - 1)
            // pad_vocab_size_to
            * pad_vocab_size_to
        )
        if self.padded_vocab_size != config.vocab_size:
            print0(
                f"Padding vocab_size from {config.vocab_size} to {self.padded_vocab_size} for efficiency"
            )

        # Token embedding
        self.wte = nn.Embedding(self.padded_vocab_size, config.n_embd)

        # Transformer blocks (each block owns its own value embedding if applicable)
        self.h = [Block(config, layer_idx, self.padded_vocab_size) for layer_idx in range(config.n_layer)]

        # Language model head (untied from wte)
        self.lm_head = nn.Linear(config.n_embd, self.padded_vocab_size, bias=False)

        # Per-layer residual stream scalars (no nn.Parameter in MLX - use mx.array)
        self.resid_lambdas = mx.ones((config.n_layer,))
        self.x0_lambdas = mx.zeros((config.n_layer,))

        # Precompute rotary embeddings (cache 10x the sequence length)
        head_dim = config.n_embd // config.n_head
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self._cos = cos
        self._sin = sin

    def _precompute_rotary_embeddings(
        self, seq_len: int, head_dim: int, base: float = 10000.0
    ) -> tuple:
        """Precompute cos and sin for rotary embeddings.

        Returns cos, sin each of shape (1, seq_len, 1, head_dim//2) in bfloat16.
        """
        channel_range = mx.arange(0, head_dim, 2).astype(mx.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = mx.arange(seq_len).astype(mx.float32)
        # Outer product: (seq_len,) x (head_dim//2,) -> (seq_len, head_dim//2)
        freqs = t[:, None] * inv_freq[None, :]
        cos = mx.cos(freqs).astype(mx.bfloat16)
        sin = mx.sin(freqs).astype(mx.bfloat16)
        # Reshape to (1, seq_len, 1, head_dim//2) for broadcasting
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config: GPTConfig) -> list:
        """Compute per-layer window sizes from the window pattern string.

        Pattern characters:
            'S' -> short window (sequence_len // 2)
            'L' -> long window (sequence_len, i.e. full context)

        The pattern repeats cyclically across layers.
        The last layer is always forced to 'L' (full context).

        Returns a list of window size integers, one per layer.
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), (
            f"Invalid window_pattern: {pattern}. Use only S and L."
        )
        long_window = config.sequence_len
        short_window = long_window // 2

        char_to_window = {
            "L": long_window,
            "S": short_window,
        }

        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Last layer always full context
        window_sizes[-1] = long_window
        return window_sizes

    def _build_sliding_window_mask(self, T: int, window: int) -> mx.array:
        """Build an additive attention mask for sliding window + causal.

        Returns an additive mask of shape (T, T) where allowed positions are 0
        and blocked positions are -1e9.
        """
        row_idx = mx.arange(T, dtype=mx.int32)[:, None]
        col_idx = mx.arange(T, dtype=mx.int32)[None, :]
        # Causal: col <= row. Sliding window: row - col <= window
        mask = (col_idx <= row_idx) & ((row_idx - col_idx) <= window)
        additive_mask = mx.where(mask, mx.array(0.0), mx.array(-1e9))
        return additive_mask

    def init_weights(self):
        """Initialize all model weights. Must be called after construction."""
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        # Token embedding: normal(0, 1)
        self.wte.weight = mx.random.normal(shape=self.wte.weight.shape).astype(
            mx.bfloat16
        )

        # LM head: normal(0, 0.001)
        self.lm_head.weight = mx.random.normal(shape=self.lm_head.weight.shape) * 0.001

        # Transformer blocks
        for block in self.h:
            # Attention projections: uniform(-s, s)
            block.attn.c_q.weight = mx.random.uniform(
                low=-s, high=s, shape=block.attn.c_q.weight.shape
            )
            block.attn.c_k.weight = mx.random.uniform(
                low=-s, high=s, shape=block.attn.c_k.weight.shape
            )
            block.attn.c_v.weight = mx.random.uniform(
                low=-s, high=s, shape=block.attn.c_v.weight.shape
            )
            # Output projection: zeros
            block.attn.c_proj.weight = mx.zeros_like(block.attn.c_proj.weight)

            # MLP
            block.mlp.c_fc.weight = mx.random.uniform(
                low=-s, high=s, shape=block.mlp.c_fc.weight.shape
            )
            block.mlp.c_proj.weight = mx.zeros_like(block.mlp.c_proj.weight)

            # VE gate: zeros
            if block.attn.ve_gate is not None:
                block.attn.ve_gate.weight = mx.zeros_like(
                    block.attn.ve_gate.weight
                )

            # Value embedding: uniform(-s, s)
            if block.ve is not None:
                block.ve.weight = mx.random.uniform(
                    low=-s, high=s, shape=block.ve.weight.shape
                ).astype(mx.bfloat16)

        # Residual lambdas
        self.resid_lambdas = mx.ones((self.config.n_layer,))
        self.x0_lambdas = mx.full((self.config.n_layer,), 0.1)

        # Re-precompute rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self._cos = cos
        self._sin = sin

    def estimate_flops(self) -> int:
        """Estimate FLOPs per token for the forward pass (6N rule + attention)."""
        # Count all trainable parameters
        nparams = sum(p.size for _, p in tree_flatten(self.parameters()))

        # Exclude embedding-like parameters from the 6N calculation
        wte_numel = self.wte.weight.size
        value_embeds_numel = sum(
            block.ve.weight.size for block in self.h if block.ve is not None
        )
        scalars_numel = self.resid_lambdas.size + self.x0_lambdas.size
        nparams_exclude = wte_numel + value_embeds_numel + scalars_numel

        # Attention FLOPs (per token)
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window in self._window_sizes:
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq

        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self) -> dict:
        """Return a breakdown of parameter counts by category."""
        wte = self.wte.weight.size
        value_embeds = sum(
            block.ve.weight.size for block in self.h if block.ve is not None
        )
        lm_head = self.lm_head.weight.size

        # Transformer block matrices: all parameters inside self.h
        # tree_flatten on the h subtree gives us flat (name, array) pairs
        h_params = self.parameters().get("h", [])
        transformer_matrices = sum(p.size for _, p in tree_flatten(h_params))

        scalars = self.resid_lambdas.size + self.x0_lambdas.size

        total = wte + value_embeds + lm_head + transformer_matrices + scalars

        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def setup_optimizer(
        self,
        unembedding_lr: float = 0.004,
        embedding_lr: float = 0.2,
        matrix_lr: float = 0.02,
        weight_decay: float = 0.0,
        adam_betas: tuple = (0.8, 0.95),
        scalar_lr: float = 0.5,
    ) -> dict:
        """Return a dict describing parameter groups for optimizer construction.

        In MLX we don't construct the optimizer here (the training script does
        that), but we return the configuration that the optimizer needs:
        a list of parameter group dicts with keys like 'kind', 'param_names',
        'lr', 'betas', etc.
        """
        model_dim = self.config.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(
            f"Scaling the LR for the AdamW parameters "
            f"proportional to 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}"
        )

        # Collect parameter names by category
        lm_head_names = ["lm_head.weight"]
        embedding_names = ["wte.weight"]
        value_embed_names = [
            f"h.{i}.ve.weight" for i, block in enumerate(self.h) if block.ve is not None
        ]
        resid_names = ["resid_lambdas"]
        x0_names = ["x0_lambdas"]

        # Matrix params: everything in self.h EXCEPT value embeddings (ve.weight)
        # tree_flatten on the h subtree gives us flat (name, array) pairs
        h_params = self.parameters().get("h", [])
        ve_name_set = set(value_embed_names)
        matrix_names = [
            f"h.{name}" for name, _ in tree_flatten(h_params)
            if f"h.{name}" not in ve_name_set
        ]

        # VE gate params are inside the blocks, so they are already in matrix_names
        # (they get the muon treatment along with other block parameters)

        param_groups = [
            dict(
                kind="adamw",
                param_names=lm_head_names,
                lr=unembedding_lr * dmodel_lr_scale,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                param_names=embedding_names,
                lr=embedding_lr * dmodel_lr_scale,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                param_names=value_embed_names,
                lr=embedding_lr * dmodel_lr_scale,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                param_names=resid_names,
                lr=scalar_lr * 0.01,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                param_names=x0_names,
                lr=scalar_lr,
                betas=(0.96, 0.95),
                eps=1e-10,
                weight_decay=0.0,
            ),
        ]

        # Group matrix params by shape for muon optimizer
        shape_to_names: dict[tuple, list] = {}
        all_params_flat = dict(tree_flatten(self.parameters()))
        for name in matrix_names:
            p = all_params_flat[name]
            s = tuple(p.shape)
            shape_to_names.setdefault(s, []).append(name)

        for shape in sorted(shape_to_names.keys()):
            param_groups.append(
                dict(
                    kind="muon",
                    param_names=shape_to_names[shape],
                    lr=matrix_lr,
                    momentum=0.95,
                    ns_steps=5,
                    beta2=0.95,
                    weight_decay=weight_decay,
                )
            )

        return {
            "param_groups": param_groups,
            "dmodel_lr_scale": dmodel_lr_scale,
        }

    def get_param_groups(self, **kwargs) -> list:
        """Convenience alias: returns just the param_groups list from setup_optimizer."""
        return self.setup_optimizer(**kwargs)["param_groups"]

    def __call__(
        self,
        idx: mx.array,
        targets: Optional[mx.array] = None,
        kv_cache=None,
        loss_reduction: str = "mean",
    ):
        """Forward pass.

        Args:
            idx: Input token indices, shape (B, T).
            targets: Target token indices for loss computation, shape (B, T).
                     If None, return logits instead.
            kv_cache: Optional KVCache object for inference.
            loss_reduction: 'mean' or 'none' for cross-entropy loss.

        Returns:
            If targets is not None: scalar loss (or per-token losses if reduction='none').
            Otherwise: logits of shape (B, T, vocab_size).
        """
        B, T = idx.shape

        assert T <= self._cos.shape[1], (
            f"Sequence length grew beyond the rotary embeddings cache: "
            f"{T} > {self._cos.shape[1]}"
        )

        # Determine RoPE position offset from KV cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos = self._cos[:, T0 : T0 + T]
        sin = self._sin[:, T0 : T0 + T]

        # Token embedding + RMSNorm
        x = self.wte(idx)
        x = norm(x)
        x0 = x

        # Build attention masks (one per unique window size, only for training)
        if kv_cache is None:
            mask_cache = {}
            for w in self._window_sizes:
                if w not in mask_cache:
                    mask_cache[w] = self._build_sliding_window_mask(T, w)
        else:
            mask_cache = {}

        # Transformer blocks
        for i, block in enumerate(self.h):
            # Per-layer residual scaling
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0

            # Get the mask for this layer's window size
            additive_mask = mask_cache.get(self._window_sizes[i]) if kv_cache is None else None

            x = block(x, idx, cos, sin, additive_mask, kv_cache)

        # Final norm
        x = norm(x)

        # Logit softcapping
        softcap = 15
        logits = self.lm_head(x)
        # Slice to actual vocab size (remove padding)
        logits = logits[..., : self.config.vocab_size]
        logits = logits.astype(mx.float32)
        logits = softcap * mx.tanh(logits / softcap)

        if targets is not None:
            # Compute cross-entropy loss
            # mlx.nn.losses.cross_entropy expects (N, C) logits and (N,) targets
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)

            # Compute per-token loss, then mask out ignore_index=-1
            per_token_loss = nn.losses.cross_entropy(
                logits_flat, targets_flat, reduction="none"
            )

            # Mask: ignore tokens where target == -1
            valid_mask = (targets_flat != -1).astype(mx.float32)
            per_token_loss = per_token_loss * valid_mask

            if loss_reduction == "mean":
                # Average over valid tokens only
                loss = mx.sum(per_token_loss) / mx.maximum(
                    mx.sum(valid_mask), mx.array(1.0)
                )
            else:
                loss = per_token_loss
            return loss
        else:
            return logits

    def generate(self, tokens: list, max_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None, seed: int = 42):
        """Generate tokens autoregressively.

        Args:
            tokens: List of initial token ids.
            max_tokens: Number of tokens to generate.
            temperature: Sampling temperature. 0 for greedy.
            top_k: If set, restrict sampling to top-k logits.
            seed: Random seed for reproducibility.

        Yields:
            Generated token ids one at a time.
        """
        mx.random.seed(seed)
        ids = mx.array([tokens], dtype=mx.int32)

        for _ in range(max_tokens):
            logits = self(ids)
            logits = logits[:, -1, :]  # (B, vocab_size)

            if top_k is not None and top_k > 0:
                # Keep only top-k logits
                k = min(top_k, logits.shape[-1])
                # Sort descending, get the k-th largest value as threshold
                top_values = mx.sort(logits, axis=-1)[..., -k:]
                threshold = top_values[..., 0:1]
                logits = mx.where(logits < threshold, mx.array(-float("inf")), logits)

            if temperature > 0:
                logits = logits / temperature
                # Sample from categorical distribution
                token = mx.random.categorical(logits, axis=-1)
                token = token.reshape(1, 1)
            else:
                token = mx.argmax(logits, axis=-1, keepdims=True)

            ids = mx.concatenate([ids, token.astype(mx.int32)], axis=1)
            yield token.item()
