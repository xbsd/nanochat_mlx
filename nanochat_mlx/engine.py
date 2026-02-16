"""
Engine for efficient inference of nanochat models on Apple MLX.

Ported from nanochat/engine.py - replaces PyTorch with MLX operations.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.
- KV cache uses concatenation pattern (MLX arrays are immutable).
- No torch.inference_mode needed (MLX gradients are opt-in).
"""

import warnings
import threading
from collections import deque

import mlx.core as mx

from nanochat_mlx.checkpoint_manager import load_model


# -----------------------------------------------------------------------------
# Calculator tool helpers

def eval_with_timeout(formula, max_time=3):
    """Evaluate a formula with a timeout using a thread-based approach.

    Unlike the PyTorch version which uses SIGALRM (Unix-only, main thread only),
    this uses a simple thread with a timeout join, which works cross-platform.
    """
    result_box = [None]
    error_box = [None]

    def _eval_target():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                result_box[0] = eval(formula, {"__builtins__": {}}, {})
        except Exception as e:
            error_box[0] = e

    thread = threading.Thread(target=_eval_target, daemon=True)
    thread.start()
    thread.join(timeout=max_time)

    if thread.is_alive():
        # Timed out - we cannot forcefully kill the thread, but since it's a
        # daemon thread it won't prevent process exit.
        return None
    if error_box[0] is not None:
        return None
    return result_box[0]


def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Check if it's a pure math expression (old behavior)
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # Disallow dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    if '.count(' not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)


# -----------------------------------------------------------------------------
class KVCache:
    """
    KV Cache for MLX inference using concatenation pattern.

    Unlike the PyTorch version which pre-allocates fixed-size buffers for FA3,
    MLX arrays are immutable so we use concatenation to grow the cache.
    The tensors are stored as (B, n_kv_head, T, head_dim) to match the
    attention layout used in the MLX model.
    """

    def __init__(self, num_layers):
        self.keys = [None] * num_layers
        self.values = [None] * num_layers
        self.pos = 0

    def get_pos(self):
        """Get current position (number of tokens cached).

        Inferred from the actual cache contents when available, so callers
        don't need to explicitly call advance() after each step.
        Falls back to self.pos for cases where advance() is called manually.
        """
        if self.keys[0] is not None:
            return self.keys[0].shape[2]  # T dimension in (B, H, T, D) layout
        return self.pos

    def update(self, layer_idx, new_k, new_v):
        """Update the cache for a given layer with new key/value tensors.

        Args:
            layer_idx: Which transformer layer this belongs to.
            new_k: New keys of shape (B, n_kv_head, T_new, head_dim).
            new_v: New values of shape (B, n_kv_head, T_new, head_dim).
        """
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = new_k
            self.values[layer_idx] = new_v
        else:
            self.keys[layer_idx] = mx.concatenate(
                [self.keys[layer_idx], new_k], axis=2
            )
            self.values[layer_idx] = mx.concatenate(
                [self.values[layer_idx], new_v], axis=2
            )

    def get(self, layer_idx):
        """Return (keys, values) for a given layer."""
        return self.keys[layer_idx], self.values[layer_idx]

    def advance(self, n):
        """Advance the position counter by n tokens."""
        self.pos += n

    def reset(self):
        """Reset cache to empty state."""
        self.keys = [None] * len(self.keys)
        self.values = [None] * len(self.values)
        self.pos = 0

    def prefill(self, other):
        """Copy cached KV from another cache into this one.

        Used when we do batch=1 prefill and then want to generate
        multiple samples in parallel. The source cache keys/values are
        broadcast-expanded along the batch dimension to match self's
        batch size.

        Args:
            other: Another KVCache to copy from.
        """
        assert self.pos == 0, "Cannot prefill a non-empty KV cache"
        n_layers = len(self.keys)
        assert n_layers == len(other.keys)
        for i in range(n_layers):
            if other.keys[i] is not None:
                # other.keys[i] is (1, n_kv_head, T, head_dim)
                # We just copy the reference - MLX will broadcast when needed
                self.keys[i] = other.keys[i]
                self.values[i] = other.values[i]
        self.pos = other.pos


# -----------------------------------------------------------------------------
def sample_next_token(logits, temperature=1.0, top_k=None):
    """Sample a single next token from given logits.

    Args:
        logits: Logits of shape (B, vocab_size).
        temperature: Sampling temperature. 0.0 = greedy.
        top_k: If set, only consider the top-k logits.

    Returns:
        Token IDs of shape (B, 1).
    """
    assert temperature >= 0.0, "temperature must be non-negative"

    if temperature == 0.0:
        return mx.argmax(logits, axis=-1, keepdims=True)

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.shape[-1])
        # mx.topk returns the top-k values sorted descending
        vals = mx.topk(logits, k=k, axis=-1)  # (B, k)
        # Create threshold from the k-th largest value (last in sorted order)
        threshold = vals[:, -1:]  # (B, 1)
        logits = mx.where(logits < threshold, mx.array(-1e9), logits)

    logits = logits / temperature
    # mx.random.categorical expects log-probabilities (unnormalized is fine,
    # it applies softmax internally via the log-sum-exp trick)
    token = mx.random.categorical(logits)  # (B,)
    return token[:, None]  # (B, 1)


# -----------------------------------------------------------------------------

class RowState:
    """Per-row state tracking during generation."""

    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []  # Current token sequence for this row
        self.forced_tokens = deque()  # Queue of tokens to force inject
        self.in_python_block = False  # Whether we are inside a python block
        self.python_expr_tokens = []  # Tokens of the current python expression
        self.completed = False  # Whether this row has completed generation


class Engine:
    """Inference engine with KV cache support for MLX models.

    Provides streaming generation (generate) and batch generation (generate_batch)
    with tool use (calculator) support.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer  # needed for tool use

    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0,
                 top_k=None, seed=42):
        """Streaming generation with KV cache and tool use support.

        Performs a single batch=1 prefill of the prompt, then clones the KV
        cache for each sample and generates tokens autoregressively.

        Args:
            tokens: List of token IDs (prompt).
            num_samples: Number of parallel samples to generate.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
            top_k: If set, only consider the top-k logits.
            seed: Random seed for reproducibility.

        Yields:
            (token_column, token_masks): Lists of length num_samples.
                token_column[i] = next token ID for sample i.
                token_masks[i] = 1 if sampled, 0 if forced (tool use).
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int), \
            "expecting list of ints"

        # Set the random seed
        mx.random.seed(seed)

        # Get the special tokens we need to coordinate the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")  # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id()  # if sampled, ends row

        # 1) Run a batch-1 prefill of the prompt tokens
        m = self.model.config
        kv_cache_prefill = KVCache(num_layers=m.n_layer)

        ids = mx.array([tokens], dtype=mx.int32)
        logits = self.model(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :]  # (1, vocab_size)
        # Expand logits for all samples
        logits = mx.broadcast_to(logits, (num_samples, logits.shape[-1]))
        mx.eval(logits)

        # 2) Replicate the KV cache for each sample/row
        kv_cache_decode = KVCache(num_layers=m.n_layer)
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill  # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            # Sample the next token for each row
            next_ids = sample_next_token(logits, temperature, top_k)  # (B, 1)
            mx.eval(next_ids)
            sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state, optional tool use
            token_column = []  # contains the next token id along each row
            token_masks = []  # contains the mask (was it sampled (1) or forced (0)?)

            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)

                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)

                # On <|assistant_end|> or <|bos|>, mark the row as completed
                if next_token == assistant_end or next_token == bos:
                    state.completed = True

                # Handle tool logic
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1

            # Prepare logits for next iteration
            ids = mx.array(token_column, dtype=mx.int32).reshape(-1, 1)  # (B, 1)
            logits = self.model(ids, kv_cache=kv_cache_decode)[:, -1, :]  # (B, vocab_size)
            mx.eval(logits)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """Non-streaming batch generation that returns the final token sequences.

        Terminal tokens (assistant_end, bos) are not included in the results.

        Args:
            tokens: List of token IDs (prompt).
            num_samples: Number of parallel samples to generate.
            **kwargs: Additional arguments passed to generate().

        Returns:
            (results, masks): Lists of lists.
                results[i] = full token sequence for sample i (prompt + generated).
                masks[i] = mask for each token (0 = prompt/forced, 1 = sampled).
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples

        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break

        return results, masks


if __name__ == "__main__":
    """
    Quick inline test to verify the Engine generates correctly.
    """
    import time

    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()

    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)

    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)

    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs)

    t0 = time.time()
    for token_column, token_masks in stream:
        token = token_column[0]  # only print out the first row
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    print(f"Tokens generated: {len(generated_tokens)}")
