"""Low-level LoRA operations with optional Metal kernel dispatch."""

from pathlib import Path

import mlx.core as mx

# Import Metal kernel functions
from .kernels import is_metal_available, lora_backward_metal, lora_forward_metal


def _get_kernel_source() -> str:
    """Load Metal kernel sources (for reference/documentation)."""
    kernel_dir = Path(__file__).parent.parent.parent / "kernels"
    sources = []
    for filename in ["lora_kernels.metal", "lora_train.metal", "lora_quantized.metal"]:
        kernel_path = kernel_dir / filename
        if kernel_path.exists():
            sources.append(kernel_path.read_text())
    return "\n".join(sources)


_KERNEL_SOURCE: str | None = None


def get_kernel_source() -> str:
    global _KERNEL_SOURCE
    if _KERNEL_SOURCE is None:
        _KERNEL_SOURCE = _get_kernel_source()
    return _KERNEL_SOURCE


def lora_forward(
    x: mx.array,
    W0: mx.array,  # noqa: N803
    A: mx.array,  # noqa: N803
    B: mx.array,  # noqa: N803
    alpha: float = 16.0,
    dropout: float = 0.0,
    training: bool = False,
    use_metal: bool = True,
) -> mx.array:
    """LoRA forward: h = W0 @ x + (alpha/rank) * B @ A @ x

    Parameters
    ----------
    x : mx.array
        Input tensor, shape [batch, seq, in_features] or [seq, in_features]
    W0 : mx.array
        Base weight matrix [out_features, in_features]
    A : mx.array
        LoRA down projection [rank, in_features]
    B : mx.array
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
    mx.array
        Output tensor with same batch/seq dims and out_features last dim
    """
    if x.ndim == 2:
        x = x[None, :, :]

    batch_size, seq_len, in_features = x.shape
    out_features, rank = B.shape
    scale = alpha / rank

    # Try Metal kernel path
    if use_metal and is_metal_available() and not training:
        try:
            output = lora_forward_metal(
                x=x, w0=W0, a=A, b=B,
                alpha=alpha,
                batch_size=batch_size,
                seq_len=seq_len,
                use_simd=(in_features >= 256),
            )
            return output
        except Exception:
            # Fall back to MLX on any kernel error
            pass

    # Pure MLX fallback
    ax = mx.matmul(x, A.T)
    bax = mx.matmul(ax, B.T)
    w0x = mx.matmul(x, W0.T)

    output = w0x + scale * bax

    if training and dropout > 0:
        mask = mx.random.bernoulli(1 - dropout, bax.shape)
        dropout_scale = 1.0 / (1.0 - dropout)
        output = w0x + scale * bax * mask * dropout_scale

    return output


def lora_forward_inference(x: mx.array, w_merged: mx.array) -> mx.array:  # noqa: N803
    """Inference forward with merged weights."""
    if x.ndim == 2:
        x = x[None, :, :]
    return mx.matmul(x, w_merged.T)


def lora_backward(
    grad_output: mx.array,
    x: mx.array,
    A: mx.array,  # noqa: N803
    B: mx.array,  # noqa: N803
    alpha: float = 16.0,
    clip_value: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """Compute gradients for A and B."""
    batch_size, seq_len, out_features = grad_output.shape
    rank, in_features = A.shape
    scale = alpha / rank

    ax = mx.matmul(x, A.T)
    bt_grad = mx.matmul(grad_output, B)

    grad_b = mx.zeros((out_features, rank))
    grad_a = mx.zeros((rank, in_features))

    for b in range(batch_size):
        grad_b = grad_b + mx.matmul(grad_output[b].T, ax[b])
        grad_a = grad_a + mx.matmul(bt_grad[b].T, x[b])

    grad_b = mx.clip(scale * grad_b, -clip_value, clip_value)
    grad_a = mx.clip(scale * grad_a, -clip_value, clip_value)

    return grad_a, grad_b


def lora_backward_efficient(
    grad_output: mx.array,
    x: mx.array,
    A: mx.array,  # noqa: N803
    B: mx.array,  # noqa: N803
    alpha: float = 16.0,
    clip_value: float = 1.0,
    use_metal: bool = True,
) -> tuple[mx.array, mx.array]:
    """Memory-efficient backward using batched matmul.

    Parameters
    ----------
    use_metal : bool
        Use Metal kernels when available (default: True)
    """
    # Try Metal kernel path
    if use_metal and is_metal_available():
        try:
            return lora_backward_metal(
                grad_output=grad_output,
                x=x, a=A, b=B,
                alpha=alpha,
                clip_value=clip_value,
            )
        except Exception:
            pass

    # Pure MLX fallback
    batch_size, seq_len, out_features = grad_output.shape
    rank, in_features = A.shape
    scale = alpha / rank

    ax = mx.matmul(x, A.T)
    grad_output_t = mx.transpose(grad_output, (0, 2, 1))
    grad_b_batched = mx.matmul(grad_output_t, ax)
    grad_b = scale * mx.sum(grad_b_batched, axis=0)

    bt_grad = mx.matmul(grad_output, B)
    bt_grad_t = mx.transpose(bt_grad, (0, 2, 1))
    grad_a_batched = mx.matmul(bt_grad_t, x)
    grad_a = scale * mx.sum(grad_a_batched, axis=0)

    grad_a = mx.clip(grad_a, -clip_value, clip_value)
    grad_b = mx.clip(grad_b, -clip_value, clip_value)

    return grad_a, grad_b


def merge_lora_weights(
    W0: mx.array,  # noqa: N803
    A: mx.array,  # noqa: N803
    B: mx.array,  # noqa: N803
    alpha: float = 16.0,
) -> mx.array:
    """Merge LoRA into base weights: W' = W0 + (alpha/rank) * B @ A"""
    rank = A.shape[0]
    scale = alpha / rank
    ba = mx.matmul(B, A)
    return W0 + scale * ba


def unmerge_lora_weights(
    w_merged: mx.array,
    W0: mx.array,  # noqa: N803
    A: mx.array,  # noqa: N803
    B: mx.array,  # noqa: N803
    alpha: float = 16.0,
) -> tuple[mx.array, mx.array]:
    """Recover LoRA from merged weights using SVD."""
    rank = A.shape[0]
    scale = alpha / rank
    delta = (w_merged - W0) / scale

    u_mat, s_vec, vt_mat = mx.linalg.svd(delta)
    sqrt_s = mx.sqrt(s_vec[:rank])

    a_recovered = mx.diag(sqrt_s) @ vt_mat[:rank, :]
    b_recovered = u_mat[:, :rank] @ mx.diag(sqrt_s)

    return a_recovered, b_recovered
