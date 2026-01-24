"""Low-level LoRA operations with optional Metal kernel dispatch."""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING, Any

# Conditional MLX import (only available on macOS with Apple Silicon)
_IS_MACOS = platform.system() == "Darwin"

if _IS_MACOS:
    try:
        import mlx.core as mx
        _MLX_AVAILABLE = True
    except ImportError:
        mx = None  # type: ignore
        _MLX_AVAILABLE = False
else:
    mx = None  # type: ignore
    _MLX_AVAILABLE = False

from .kernels import is_metal_available, lora_backward_metal, lora_forward_metal

# Type alias for array (allows static type checking while supporting runtime None)
if TYPE_CHECKING:
    import mlx.core
    Array = mlx.core.array
else:
    Array = Any


def lora_forward(
    x: Array,
    W0: Array,  # noqa: N803
    A: Array,  # noqa: N803
    B: Array,  # noqa: N803
    alpha: float = 16.0,
    dropout: float = 0.0,
    training: bool = False,
    use_metal: bool = True,
) -> Array:
    """LoRA forward: h = W0 @ x + (alpha/rank) * B @ A @ x

    Parameters
    ----------
    x : array
        Input tensor, shape [batch, seq, in_features] or [seq, in_features]
    W0 : array
        Base weight matrix [out_features, in_features]
    A : array
        LoRA down projection [rank, in_features]
    B : array
        LoRA up projection [out_features, rank]
    alpha : float
        LoRA scaling factor
    dropout : float
        Dropout probability (training only)
    training : bool
        Training mode flag
    use_metal : bool
        Use custom Metal kernels when available (default: True)

    Returns
    -------
    array
        Output tensor with same batch/seq dims and out_features last dim
    """
    original_ndim = x.ndim
    if x.ndim == 2:
        x = x[None, :, :]

    _, rank = B.shape
    scale = alpha / rank

    # Try Metal kernel path (inference only)
    if use_metal and is_metal_available() and not training:
        try:
            return lora_forward_metal(x=x, w0=W0, a=A, b=B, alpha=alpha)
        except Exception:
            pass  # Fall back to MLX

    # Pure MLX path
    w0x = mx.matmul(x, W0.T)
    ax = mx.matmul(x, A.T)
    bax = mx.matmul(ax, B.T)
    output = w0x + scale * bax

    if training and dropout > 0:
        mask = mx.random.bernoulli(1 - dropout, bax.shape)
        output = w0x + scale * bax * mask / (1.0 - dropout)

    if original_ndim == 2:
        output = output.squeeze(0)

    return output


def lora_forward_inference(x: Array, w_merged: Array) -> Array:
    """Inference forward with merged weights."""
    original_ndim = x.ndim
    if x.ndim == 2:
        x = x[None, :, :]
    output = mx.matmul(x, w_merged.T)
    if original_ndim == 2:
        output = output.squeeze(0)
    return output


def lora_backward(
    grad_output: Array,
    x: Array,
    A: Array,  # noqa: N803
    B: Array,  # noqa: N803
    alpha: float = 16.0,
    clip_value: float = 1.0,
    use_metal: bool = True,
) -> tuple[Array, Array]:
    """Compute gradients for A and B using efficient batched matmul.

    Parameters
    ----------
    use_metal : bool
        Use Metal kernels when available (default: True)
    """
    # Try Metal kernel path
    if use_metal and is_metal_available():
        try:
            return lora_backward_metal(
                grad_output=grad_output, x=x, a=A, b=B,
                alpha=alpha, clip_value=clip_value,
            )
        except Exception:
            pass

    # Pure MLX batched computation
    rank, _ = A.shape
    scale = alpha / rank

    # grad_B = scale * sum_batch(grad_output.T @ (x @ A.T))
    ax = mx.matmul(x, A.T)
    grad_b = scale * mx.sum(mx.matmul(mx.transpose(grad_output, (0, 2, 1)), ax), axis=0)

    # grad_A = scale * sum_batch((grad_output @ B).T @ x)
    bt_grad = mx.matmul(grad_output, B)
    grad_a = scale * mx.sum(mx.matmul(mx.transpose(bt_grad, (0, 2, 1)), x), axis=0)

    return mx.clip(grad_a, -clip_value, clip_value), mx.clip(grad_b, -clip_value, clip_value)


# Alias for backwards compatibility
lora_backward_efficient = lora_backward


def merge_lora_weights(
    W0: Array,  # noqa: N803
    A: Array,  # noqa: N803
    B: Array,  # noqa: N803
    alpha: float = 16.0,
) -> Array:
    """Merge LoRA into base weights: W' = W0 + (alpha/rank) * B @ A"""
    rank = A.shape[0]
    scale = alpha / rank
    ba = mx.matmul(B, A)
    return W0 + scale * ba


def unmerge_lora_weights(
    w_merged: Array,
    W0: Array,  # noqa: N803
    A: Array,  # noqa: N803
    B: Array,  # noqa: N803
    alpha: float = 16.0,
) -> tuple[Array, Array]:
    """Recover LoRA from merged weights using SVD."""
    rank = A.shape[0]
    scale = alpha / rank
    delta = (w_merged - W0) / scale

    u_mat, s_vec, vt_mat = mx.linalg.svd(delta)
    sqrt_s = mx.sqrt(s_vec[:rank])

    a_recovered = mx.diag(sqrt_s) @ vt_mat[:rank, :]
    b_recovered = u_mat[:, :rank] @ mx.diag(sqrt_s)

    return a_recovered, b_recovered
