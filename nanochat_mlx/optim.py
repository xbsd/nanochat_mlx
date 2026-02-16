"""
MuonAdamW optimizer for Apple MLX.
Ported from nanochat's PyTorch optimizer - removes all CUDA/DDP/torch.compile dependencies.

Combines two optimizers:
  - AdamW: for embeddings, scalars, and other non-matrix parameters
  - Muon (Momentum Orthogonalized by Newton-Schulz): for 2D matrix parameters

Muon uses the Polar Express Sign Method for Newton-Schulz orthogonalization
with NorMuon variance reduction and cautious weight decay.
"""

import math
from typing import Any

import mlx.core as mx

# ---------------------------------------------------------------------------
# Polar Express Sign Method coefficients (5 iterations)
# Each tuple is (a, b, c) used in the Newton-Schulz iteration:
#   X_{k+1} = a * X_k + X_k @ (b * A + c * A^2)     (tall)
#   X_{k+1} = a * X_k + (b * A + c * A^2) @ X_k      (wide)
# where A = X^T X (tall) or X X^T (wide).
# ---------------------------------------------------------------------------
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

# ---------------------------------------------------------------------------
# Default hyper-parameters for each parameter group kind
# ---------------------------------------------------------------------------
DEFAULT_ADAMW_BETAS = (0.9, 0.95)
DEFAULT_ADAMW_EPS = 1e-8
DEFAULT_ADAMW_WD = 0.0
DEFAULT_MUON_MOMENTUM = 0.95
DEFAULT_MUON_NS_STEPS = 5
DEFAULT_MUON_BETA2 = 0.7
DEFAULT_MUON_WD = 0.0


# ========================== helper utilities ===============================

def _get_nested(tree: dict, path: str) -> Any:
    """Navigate a nested dict/list tree by a dot-separated *path*.

    Example
    -------
    >>> _get_nested({"layers": [{"attn": {"w": 1}}]}, "layers.0.attn.w")
    1
    """
    parts = path.split(".")
    node = tree
    for part in parts:
        if isinstance(node, (list, tuple)):
            node = node[int(part)]
        else:
            node = node[part]
    return node


def _set_nested(tree: dict, path: str, value: Any) -> None:
    """Set a *value* inside a nested dict/list tree at *path* (in-place)."""
    parts = path.split(".")
    node = tree
    for part in parts[:-1]:
        if isinstance(node, (list, tuple)):
            node = node[int(part)]
        else:
            node = node[part]
    last = parts[-1]
    if isinstance(node, (list, tuple)):
        node[int(last)] = value
    else:
        node[last] = value


def _filter_grads(grads: dict, names: list[str]) -> dict[str, mx.array]:
    """Extract gradient arrays for a list of parameter name paths.

    Returns a flat dict ``{name: grad_array}`` containing only the entries
    where the gradient is not ``None``.
    """
    out: dict[str, mx.array] = {}
    for name in names:
        try:
            g = _get_nested(grads, name)
        except (KeyError, IndexError, TypeError):
            continue
        if g is not None:
            out[name] = g
    return out


def _zero_grads(grads: dict, names: list[str]) -> None:
    """Zero-out gradients at the given *names* inside *grads* (in-place).

    This is useful after an optimizer group has consumed its gradients so that
    a later group does not accidentally double-count them.
    """
    for name in names:
        try:
            g = _get_nested(grads, name)
        except (KeyError, IndexError, TypeError):
            continue
        if g is not None:
            _set_nested(grads, name, mx.zeros_like(g))


# ========================== AdamW step =====================================

def _adamw_step(
    param: mx.array,
    grad: mx.array,
    exp_avg: mx.array,
    exp_avg_sq: mx.array,
    step_count: int,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """One step of decoupled AdamW with bias correction.

    Returns the updated (param, exp_avg, exp_avg_sq).
    """
    beta1, beta2 = betas

    # Decoupled weight decay (applied *before* the Adam update direction is
    # computed, matching the standard AdamW formulation).
    if weight_decay != 0.0:
        param = param * (1.0 - lr * weight_decay)

    # Exponential moving averages
    exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
    exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * (grad * grad)

    # Bias correction
    bc1 = 1.0 - beta1 ** step_count
    bc2 = 1.0 - beta2 ** step_count
    corrected_avg = exp_avg / bc1
    corrected_avg_sq = exp_avg_sq / bc2

    # Update
    denom = mx.sqrt(corrected_avg_sq) + eps
    param = param - lr * corrected_avg / denom

    return param, exp_avg, exp_avg_sq


# ========================== Muon step ======================================

def _polar_express(X: mx.array, ns_steps: int) -> mx.array:
    """Orthogonalise *X* via the Polar Express Sign Method.

    *X* has shape ``(..., rows, cols)`` where the leading dimensions are a
    batch of matrices.  The routine chooses tall-vs-wide orientation
    automatically so that the Gram matrix is as small as possible.
    """
    orig_dtype = X.dtype
    X = X.astype(mx.float32)
    # Normalise each matrix in the batch
    norms = mx.linalg.norm(X, axis=(-2, -1), keepdims=True)
    X = X / (norms * 1.02 + 1e-6)

    for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
        if X.shape[-2] > X.shape[-1]:
            # Tall matrix: form the smaller X^T X
            A = X.swapaxes(-2, -1) @ X          # (..., cols, cols)
            B = b * A + c * (A @ A)
            X = a * X + X @ B
        else:
            # Wide (or square) matrix: form the smaller X X^T
            A = X @ X.swapaxes(-2, -1)           # (..., rows, rows)
            B = b * A + c * (A @ A)
            X = a * X + B @ X

    return X.astype(orig_dtype)


def _muon_step(
    stacked_grads: mx.array,
    stacked_params: mx.array,
    momentum_buf: mx.array,
    second_momentum_buf: mx.array,
    momentum: float,
    lr: float,
    wd: float,
    beta2: float,
    ns_steps: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """One Muon step for a *batch* of same-shape 2-D parameters.

    Parameters
    ----------
    stacked_grads : mx.array
        Gradients stacked along a new leading axis, shape ``(N, rows, cols)``.
    stacked_params : mx.array
        Current parameter values, same shape.
    momentum_buf : mx.array
        Running Nesterov momentum buffer, same shape.
    second_momentum_buf : mx.array
        Running second-moment estimate for NorMuon variance scaling, same shape.
    momentum, lr, wd, beta2, ns_steps : scalars
        Hyper-parameters for this group.

    Returns
    -------
    (updated_params, momentum_buf, second_momentum_buf, update_direction)
    """
    # ---- 1. Nesterov momentum ----
    momentum_buf = momentum * momentum_buf + (1.0 - momentum) * stacked_grads
    g = momentum * momentum_buf + (1.0 - momentum) * stacked_grads  # lookahead

    # ---- 2. Polar Express orthogonalisation ----
    g = _polar_express(g, ns_steps)
    g = g.astype(stacked_params.dtype)

    # ---- 3. NorMuon variance reduction ----
    # Per-neuron (per-row) adaptive rescaling.
    # We maintain a running EMA of the squared row-norms of the update
    # direction and rescale so that the effective step has unit variance.
    row_rms = mx.mean(g * g, axis=-1, keepdims=True)  # (N, rows, 1)
    second_momentum_buf = beta2 * second_momentum_buf + (1.0 - beta2) * row_rms
    # Rescale g so each neuron's update has approximately unit RMS
    scale = 1.0 / (mx.sqrt(second_momentum_buf + 1e-8))
    g = g * scale

    # Normalise so that the mean absolute update magnitude across the whole
    # batch matches what we would get without NorMuon (prevents the overall
    # step-size from drifting).
    g = g * (mx.sqrt(mx.array(float(g.shape[-1]))))

    # ---- 4. Cautious weight decay + parameter update ----
    # Cautious: only apply weight decay in directions that *agree* with the
    # update direction (i.e. where the update would move the parameter
    # towards zero anyway).
    mask = (g * stacked_params) >= 0
    mask = mask.astype(stacked_params.dtype)
    stacked_params = stacked_params - lr * g - lr * wd * stacked_params * mask

    return stacked_params, momentum_buf, second_momentum_buf, g


# ========================== MuonAdamW class ================================

class MuonAdamW:
    """Combined Muon + AdamW optimizer for MLX.

    Parameters
    ----------
    param_groups : list[dict]
        Each dict must contain at least:
          - ``kind`` : ``"adamw"`` or ``"muon"``
          - ``param_names`` : list of dot-separated parameter name paths

        Optional keys (with defaults):
          **AdamW groups**
          - ``lr``           : learning rate (required)
          - ``betas``        : (beta1, beta2) – default ``(0.9, 0.95)``
          - ``eps``          : epsilon – default ``1e-8``
          - ``weight_decay`` : default ``0.0``

          **Muon groups**
          - ``lr``           : learning rate (required)
          - ``momentum``     : Nesterov momentum coefficient – default ``0.95``
          - ``ns_steps``     : Newton-Schulz iterations – default ``5``
          - ``beta2``        : second-moment EMA coeff for NorMuon – default ``0.7``
          - ``weight_decay`` : cautious weight decay – default ``0.0``

    Example
    -------
    >>> groups = build_param_groups(model, lr_scale=1.0)
    >>> opt = MuonAdamW(groups)
    >>> # training loop:
    >>> grads = mx.grad(loss_fn)(model)
    >>> opt.update(model, grads)
    """

    def __init__(self, param_groups: list[dict]):
        self.param_groups: list[dict] = []
        self.state: dict[str, dict[str, Any]] = {}
        self.step_count: int = 0

        for group in param_groups:
            g = dict(group)  # shallow copy so we don't mutate the caller's dict
            g.setdefault("betas", DEFAULT_ADAMW_BETAS)
            g.setdefault("eps", DEFAULT_ADAMW_EPS)
            g.setdefault("weight_decay", DEFAULT_ADAMW_WD)
            g.setdefault("momentum", DEFAULT_MUON_MOMENTUM)
            g.setdefault("ns_steps", DEFAULT_MUON_NS_STEPS)
            g.setdefault("beta2", DEFAULT_MUON_BETA2)
            # Store the *base* lr so that ``update_lr`` can scale relative
            # to the originally requested rate.
            g["base_lr"] = g["lr"]
            self.param_groups.append(g)

    # ----- state helpers ---------------------------------------------------

    def _init_adamw_state(self, name: str, param: mx.array) -> dict:
        """Lazily create AdamW state buffers for *name*."""
        if name not in self.state:
            self.state[name] = {
                "kind": "adamw",
                "exp_avg": mx.zeros_like(param),
                "exp_avg_sq": mx.zeros_like(param),
            }
        return self.state[name]

    def _init_muon_state(self, name: str, param: mx.array) -> dict:
        """Lazily create Muon state buffers for *name*."""
        if name not in self.state:
            rows, cols = param.shape[-2], param.shape[-1]
            self.state[name] = {
                "kind": "muon",
                "momentum_buf": mx.zeros_like(param),
                "second_momentum_buf": mx.zeros((rows, 1), dtype=param.dtype),
            }
        return self.state[name]

    # ----- public API ------------------------------------------------------

    def update(self, model, gradients: dict) -> None:
        """Apply one optimiser step to *model* given *gradients*.

        *gradients* must be a nested dict tree with the same structure as
        ``model.parameters()`` (as returned by ``mx.grad`` or
        ``nn.value_and_grad``).
        """
        self.step_count += 1

        params = model.parameters()

        for group in self.param_groups:
            kind = group["kind"]
            names = group["param_names"]
            lr = group["lr"]

            matched_grads = _filter_grads(gradients, names)
            if not matched_grads:
                continue

            if kind == "adamw":
                self._step_adamw(params, matched_grads, group)
            elif kind == "muon":
                self._step_muon(params, matched_grads, group)
            else:
                raise ValueError(f"Unknown optimizer kind: {kind!r}")

            # Zero consumed gradients so later groups don't re-use them.
            _zero_grads(gradients, list(matched_grads.keys()))

        # Write the updated parameters back into the model.
        model.update(params)

    def update_lr(self, lr_multiplier: float) -> None:
        """Scale all learning rates by *lr_multiplier* relative to base LR."""
        for group in self.param_groups:
            group["lr"] = group["base_lr"] * lr_multiplier

    def update_muon_momentum(self, momentum: float) -> None:
        """Set the Nesterov momentum coefficient for all Muon groups."""
        for group in self.param_groups:
            if group["kind"] == "muon":
                group["momentum"] = momentum

    def update_weight_decay(self, weight_decay: float) -> None:
        """Set the weight-decay coefficient for all Muon groups."""
        for group in self.param_groups:
            if group["kind"] == "muon":
                group["weight_decay"] = weight_decay

    # ----- internal step implementations -----------------------------------

    def _step_adamw(self, params: dict, matched_grads: dict, group: dict) -> None:
        """Run one AdamW step for every parameter in *matched_grads*."""
        lr = group["lr"]
        betas = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]

        for name, grad in matched_grads.items():
            param = _get_nested(params, name)
            s = self._init_adamw_state(name, param)

            new_param, new_avg, new_avg_sq = _adamw_step(
                param, grad,
                s["exp_avg"], s["exp_avg_sq"],
                self.step_count,
                lr, betas, eps, wd,
            )

            _set_nested(params, name, new_param)
            s["exp_avg"] = new_avg
            s["exp_avg_sq"] = new_avg_sq

    def _step_muon(self, params: dict, matched_grads: dict, group: dict) -> None:
        """Run one Muon step for the parameters in *matched_grads*.

        Parameters that share the same ``(rows, cols)`` shape are *stacked*
        into a single batch so that the Newton-Schulz iterations run once per
        unique shape (much faster on the GPU).
        """
        lr = group["lr"]
        momentum = group["momentum"]
        wd = group["weight_decay"]
        beta2 = group["beta2"]
        ns_steps = group["ns_steps"]

        # Group parameters by shape so we can batch them.
        shape_buckets: dict[tuple[int, ...], list[str]] = {}
        for name, grad in matched_grads.items():
            key = tuple(grad.shape)
            shape_buckets.setdefault(key, []).append(name)

        for shape, bucket_names in shape_buckets.items():
            # Collect arrays for this bucket.
            grads_list: list[mx.array] = []
            params_list: list[mx.array] = []
            mom_list: list[mx.array] = []
            sec_mom_list: list[mx.array] = []

            for name in bucket_names:
                g = matched_grads[name]
                p = _get_nested(params, name)
                s = self._init_muon_state(name, p)
                grads_list.append(g)
                params_list.append(p)
                mom_list.append(s["momentum_buf"])
                sec_mom_list.append(s["second_momentum_buf"])

            # Stack into batched tensors: (N, rows, cols)
            stacked_grads = mx.stack(grads_list, axis=0)
            stacked_params = mx.stack(params_list, axis=0)
            stacked_mom = mx.stack(mom_list, axis=0)
            stacked_sec_mom = mx.stack(sec_mom_list, axis=0)

            new_params, new_mom, new_sec_mom, _ = _muon_step(
                stacked_grads, stacked_params,
                stacked_mom, stacked_sec_mom,
                momentum, lr, wd, beta2, ns_steps,
            )

            # Un-stack and write back.
            for i, name in enumerate(bucket_names):
                _set_nested(params, name, new_params[i])
                s = self.state[name]
                s["momentum_buf"] = new_mom[i]
                s["second_momentum_buf"] = new_sec_mom[i]


# ========================== convenience builder ============================

def build_param_groups(model, lr_scale: float = 1.0) -> list[dict]:
    """Build the standard 6-group MuonAdamW configuration.

    The grouping follows the nanochat convention:

    1. ``lm_head``        -- unembedding -- AdamW, lr=0.004 * scale
    2. ``wte``            -- token embed -- AdamW, lr=0.3   * scale
    3. ``value_embeds``   -- value embeds -- AdamW, lr=0.3  * scale
    4. ``resid_lambdas``  -- residual lambdas -- AdamW, lr=0.005 * scale,
                            betas=(0.8, 0.95)
    5. ``x0_lambdas``     -- x0 lambdas -- AdamW, lr=0.5   * scale,
                            betas=(0.96, 0.95)
    6. **all 2-D matrices** -- Muon, lr=0.02

    Parameters
    ----------
    model : nn.Module
        The model whose ``parameters()`` / ``trainable_parameters()`` are
        inspected to discover parameter names.
    lr_scale : float
        A global scaling factor applied to AdamW learning rates.

    Returns
    -------
    list[dict]
        Ready-to-use ``param_groups`` for :class:`MuonAdamW`.
    """
    all_params = model.trainable_parameters()
    # ``trainable_parameters()`` returns a flat list of arrays; we need
    # the *names*.  ``model.parameters()`` returns the nested tree.
    # We'll walk it ourselves.
    param_tree = model.parameters()

    # Collect all (name, array) pairs via a walk.
    named: list[tuple[str, mx.array]] = []
    _walk_params(param_tree, "", named)

    lm_head_names: list[str] = []
    wte_names: list[str] = []
    value_embed_names: list[str] = []
    resid_lambda_names: list[str] = []
    x0_lambda_names: list[str] = []
    muon_names: list[str] = []

    assigned: set[str] = set()

    for name, arr in named:
        # --- rule-based assignment (order matters) ---
        if "lm_head" in name:
            lm_head_names.append(name)
            assigned.add(name)
        elif "wte" in name or "embed_tokens" in name:
            wte_names.append(name)
            assigned.add(name)
        elif "value_embed" in name:
            value_embed_names.append(name)
            assigned.add(name)
        elif "resid_lambda" in name:
            resid_lambda_names.append(name)
            assigned.add(name)
        elif "x0_lambda" in name:
            x0_lambda_names.append(name)
            assigned.add(name)

    # Everything remaining that is 2-D goes to Muon; anything else (biases,
    # layer-norm scales, 1-D embeddings, etc.) goes to a default AdamW group.
    default_adamw_names: list[str] = []
    for name, arr in named:
        if name in assigned:
            continue
        if arr.ndim == 2:
            muon_names.append(name)
        else:
            default_adamw_names.append(name)

    groups: list[dict] = []

    if lm_head_names:
        groups.append({
            "kind": "adamw",
            "param_names": lm_head_names,
            "lr": 0.004 * lr_scale,
            "betas": DEFAULT_ADAMW_BETAS,
            "eps": DEFAULT_ADAMW_EPS,
            "weight_decay": DEFAULT_ADAMW_WD,
        })

    if wte_names:
        groups.append({
            "kind": "adamw",
            "param_names": wte_names,
            "lr": 0.3 * lr_scale,
            "betas": DEFAULT_ADAMW_BETAS,
            "eps": DEFAULT_ADAMW_EPS,
            "weight_decay": DEFAULT_ADAMW_WD,
        })

    if value_embed_names:
        groups.append({
            "kind": "adamw",
            "param_names": value_embed_names,
            "lr": 0.3 * lr_scale,
            "betas": DEFAULT_ADAMW_BETAS,
            "eps": DEFAULT_ADAMW_EPS,
            "weight_decay": DEFAULT_ADAMW_WD,
        })

    if resid_lambda_names:
        groups.append({
            "kind": "adamw",
            "param_names": resid_lambda_names,
            "lr": 0.5 * 0.01 * lr_scale,
            "betas": (0.8, 0.95),
            "eps": DEFAULT_ADAMW_EPS,
            "weight_decay": DEFAULT_ADAMW_WD,
        })

    if x0_lambda_names:
        groups.append({
            "kind": "adamw",
            "param_names": x0_lambda_names,
            "lr": 0.5 * lr_scale,
            "betas": (0.96, 0.95),
            "eps": DEFAULT_ADAMW_EPS,
            "weight_decay": DEFAULT_ADAMW_WD,
        })

    if muon_names:
        groups.append({
            "kind": "muon",
            "param_names": muon_names,
            "lr": 0.02,
            "momentum": DEFAULT_MUON_MOMENTUM,
            "ns_steps": DEFAULT_MUON_NS_STEPS,
            "beta2": DEFAULT_MUON_BETA2,
            "weight_decay": DEFAULT_MUON_WD,
        })

    # Catch-all for 1-D / scalar params that didn't match any named group
    if default_adamw_names:
        groups.append({
            "kind": "adamw",
            "param_names": default_adamw_names,
            "lr": 0.3 * lr_scale,
            "betas": DEFAULT_ADAMW_BETAS,
            "eps": DEFAULT_ADAMW_EPS,
            "weight_decay": DEFAULT_ADAMW_WD,
        })

    return groups


def _walk_params(
    tree: Any,
    prefix: str,
    out: list[tuple[str, mx.array]],
) -> None:
    """Recursively walk a nested parameter tree and collect ``(name, array)`` pairs."""
    if isinstance(tree, mx.array):
        out.append((prefix, tree))
    elif isinstance(tree, dict):
        for key, val in tree.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            _walk_params(val, child_prefix, out)
    elif isinstance(tree, (list, tuple)):
        for idx, val in enumerate(tree):
            child_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            _walk_params(val, child_prefix, out)
