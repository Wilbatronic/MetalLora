"""Low-level LoRA operations."""

import mlx.core as mx
from pathlib import Path
from typing import Optional, Tuple


def _get_kernel_source() -> str:
    """Load Metal kernel sources."""
    kernel_dir = Path(__file__).parent.parent.parent / "kernels"
    sources = []
    for filename in ["lora_kernels.metal", "lora_train.metal", "lora_quantized.metal"]:
        kernel_path = kernel_dir / filename
        if kernel_path.exists():
            sources.append(kernel_path.read_text())
    return "\n".join(sources)


_KERNEL_SOURCE: Optional[str] = None


def get_kernel_source() -> str:
    global _KERNEL_SOURCE
    if _KERNEL_SOURCE is None:
        _KERNEL_SOURCE = _get_kernel_source()
    return _KERNEL_SOURCE


def lora_forward(
    x: mx.array,
    W0: mx.array,
    A: mx.array,
    B: mx.array,
    alpha: float = 16.0,
    dropout: float = 0.0,
    training: bool = False,
) -> mx.array:
    """LoRA forward: h = W0 @ x + (alpha/rank) * B @ A @ x"""
    if x.ndim == 2:
        x = x[None, :, :]
    
    batch_size, seq_len, in_features = x.shape
    out_features, rank = B.shape
    scale = alpha / rank
    
    Ax = mx.matmul(x, A.T)
    BAx = mx.matmul(Ax, B.T)
    W0x = mx.matmul(x, W0.T)
    
    output = W0x + scale * BAx
    
    if training and dropout > 0:
        mask = mx.random.bernoulli(1 - dropout, BAx.shape)
        dropout_scale = 1.0 / (1.0 - dropout)
        output = W0x + scale * BAx * mask * dropout_scale
    
    return output


def lora_forward_inference(x: mx.array, W_merged: mx.array) -> mx.array:
    """Inference forward with merged weights."""
    if x.ndim == 2:
        x = x[None, :, :]
    return mx.matmul(x, W_merged.T)


def lora_backward(
    grad_output: mx.array,
    x: mx.array,
    A: mx.array,
    B: mx.array,
    alpha: float = 16.0,
    clip_value: float = 1.0,
) -> Tuple[mx.array, mx.array]:
    """Compute gradients for A and B."""
    batch_size, seq_len, out_features = grad_output.shape
    rank, in_features = A.shape
    scale = alpha / rank
    
    Ax = mx.matmul(x, A.T)
    Bt_grad = mx.matmul(grad_output, B)
    
    grad_B = mx.zeros((out_features, rank))
    grad_A = mx.zeros((rank, in_features))
    
    for b in range(batch_size):
        grad_B = grad_B + mx.matmul(grad_output[b].T, Ax[b])
        grad_A = grad_A + mx.matmul(Bt_grad[b].T, x[b])
    
    grad_B = mx.clip(scale * grad_B, -clip_value, clip_value)
    grad_A = mx.clip(scale * grad_A, -clip_value, clip_value)
    
    return grad_A, grad_B


def lora_backward_efficient(
    grad_output: mx.array,
    x: mx.array,
    A: mx.array,
    B: mx.array,
    alpha: float = 16.0,
    clip_value: float = 1.0,
) -> Tuple[mx.array, mx.array]:
    """Memory-efficient backward using batched matmul."""
    batch_size, seq_len, out_features = grad_output.shape
    rank, in_features = A.shape
    scale = alpha / rank
    
    Ax = mx.matmul(x, A.T)
    grad_output_t = mx.transpose(grad_output, (0, 2, 1))
    grad_B_batched = mx.matmul(grad_output_t, Ax)
    grad_B = scale * mx.sum(grad_B_batched, axis=0)
    
    Bt_grad = mx.matmul(grad_output, B)
    Bt_grad_t = mx.transpose(Bt_grad, (0, 2, 1))
    grad_A_batched = mx.matmul(Bt_grad_t, x)
    grad_A = scale * mx.sum(grad_A_batched, axis=0)
    
    grad_A = mx.clip(grad_A, -clip_value, clip_value)
    grad_B = mx.clip(grad_B, -clip_value, clip_value)
    
    return grad_A, grad_B


def merge_lora_weights(
    W0: mx.array,
    A: mx.array,
    B: mx.array,
    alpha: float = 16.0,
) -> mx.array:
    """Merge LoRA into base weights: W' = W0 + (alpha/rank) * B @ A"""
    rank = A.shape[0]
    scale = alpha / rank
    BA = mx.matmul(B, A)
    return W0 + scale * BA


def unmerge_lora_weights(
    W_merged: mx.array,
    W0: mx.array,
    A: mx.array,
    B: mx.array,
    alpha: float = 16.0,
) -> Tuple[mx.array, mx.array]:
    """Recover LoRA from merged weights using SVD."""
    rank = A.shape[0]
    scale = alpha / rank
    delta = (W_merged - W0) / scale
    
    U, S, Vt = mx.linalg.svd(delta)
    sqrt_S = mx.sqrt(S[:rank])
    
    A_recovered = mx.diag(sqrt_S) @ Vt[:rank, :]
    B_recovered = U[:, :rank] @ mx.diag(sqrt_S)
    
    return A_recovered, B_recovered
