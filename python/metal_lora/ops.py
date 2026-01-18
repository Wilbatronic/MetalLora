"""
MetalLoRA - Low-level operations wrapping Metal kernel calls

This module provides the low-level interface between Python/MLX and the
custom Metal kernels. It handles:
  - Kernel source loading and compilation
  - Buffer setup and grid/threadgroup configuration
  - Kernel dispatch with proper synchronization
"""

import mlx.core as mx
from pathlib import Path
from typing import Optional, Tuple
import math

# ============================================================================
# KERNEL SOURCE LOADING
# ============================================================================

def _get_kernel_source() -> str:
    """Load all Metal kernel sources as a single string."""
    kernel_dir = Path(__file__).parent.parent.parent / "kernels"
    
    sources = []
    kernel_files = [
        "lora_common.metal",
        "lora_forward.metal", 
        "lora_backward.metal",
        "gemm_lowrank.metal",
    ]
    
    for filename in kernel_files:
        kernel_path = kernel_dir / filename
        if kernel_path.exists():
            sources.append(kernel_path.read_text())
        else:
            raise FileNotFoundError(f"Kernel file not found: {kernel_path}")
    
    return "\n".join(sources)


# Cache compiled kernel source
_KERNEL_SOURCE: Optional[str] = None

def get_kernel_source() -> str:
    """Get cached kernel source, loading if necessary."""
    global _KERNEL_SOURCE
    if _KERNEL_SOURCE is None:
        _KERNEL_SOURCE = _get_kernel_source()
    return _KERNEL_SOURCE


# ============================================================================
# CONFIGURATION STRUCTURES
# ============================================================================

def _make_lora_config(
    batch_size: int,
    seq_len: int,
    in_features: int,
    out_features: int,
    rank: int,
    alpha: float = 16.0,
    dropout: float = 0.0,
    seed: int = 0,
) -> mx.array:
    """Create LoRAConfig structure as MLX array for kernel binding."""
    # Pack config into uint32/float32 array matching Metal struct layout
    # struct LoRAConfig { uint batch_size, seq_len, in_features, out_features, rank;
    #                     float alpha, dropout_prob; uint seed; }
    config_data = [
        batch_size,
        seq_len, 
        in_features,
        out_features,
        rank,
    ]
    # Convert to bytes and add floats
    import struct
    packed = struct.pack('5I2fI', 
        batch_size, seq_len, in_features, out_features, rank,
        alpha, dropout, seed
    )
    return mx.array(list(packed), dtype=mx.uint8)


def _make_grad_config(
    learning_rate: float = 1e-4,
    clip_value: float = 1.0,
    weight_decay: float = 0.0,
    use_adam: bool = False,
) -> mx.array:
    """Create GradConfig structure for backward kernels."""
    import struct
    packed = struct.pack('3f?', learning_rate, clip_value, weight_decay, use_adam)
    return mx.array(list(packed), dtype=mx.uint8)


# ============================================================================
# FORWARD OPERATIONS
# ============================================================================

def lora_forward(
    x: mx.array,
    W0: mx.array,
    A: mx.array,
    B: mx.array,
    alpha: float = 16.0,
    dropout: float = 0.0,
    training: bool = False,
) -> mx.array:
    """
    Optimized LoRA forward pass.
    
    Computes: h = W₀x + (α/r) * BAx
    
    Args:
        x: Input tensor [batch, seq_len, in_features]
        W0: Base weight matrix [out_features, in_features]
        A: LoRA down-projection [rank, in_features]
        B: LoRA up-projection [out_features, rank]
        alpha: LoRA scaling factor
        dropout: Dropout probability (0 = disabled)
        training: Whether in training mode (affects dropout)
    
    Returns:
        Output tensor [batch, seq_len, out_features]
    """
    # Validate input shapes
    if x.ndim == 2:
        x = x[None, :, :]  # Add batch dimension
    
    batch_size, seq_len, in_features = x.shape
    out_features, rank = B.shape
    
    assert A.shape == (rank, in_features), f"A shape mismatch: {A.shape} vs ({rank}, {in_features})"
    assert W0.shape == (out_features, in_features), f"W0 shape mismatch: {W0.shape}"
    
    # For now, use MLX primitives with optimized computation order
    # This will be replaced with custom Metal kernel dispatch when using MLX's
    # metal.kernel() API is more stable
    
    # Compute LoRA contribution: (α/r) * B @ A @ x
    scale = alpha / rank
    
    # Fused computation: minimize intermediate tensors
    # Step 1: Ax = A @ x.T for each sample, then transpose back
    # More efficient: use einsum or batched matmul
    
    # x is [B, S, K], A is [R, K]
    # We want Ax = [B, S, R]
    Ax = mx.matmul(x, A.T)  # [B, S, K] @ [K, R] = [B, S, R]
    
    # B is [D, R], Ax is [B, S, R]
    # BAx = [B, S, D]
    BAx = mx.matmul(Ax, B.T)  # [B, S, R] @ [R, D] = [B, S, D]
    
    # W0 @ x
    W0x = mx.matmul(x, W0.T)  # [B, S, K] @ [K, D] = [B, S, D]
    
    # Combine
    output = W0x + scale * BAx
    
    # Apply dropout during training
    if training and dropout > 0:
        mask = mx.random.bernoulli(1 - dropout, BAx.shape)
        dropout_scale = 1.0 / (1.0 - dropout)
        output = W0x + scale * BAx * mask * dropout_scale
    
    return output


def lora_forward_inference(
    x: mx.array,
    W_merged: mx.array,
) -> mx.array:
    """
    Inference-optimized forward pass with merged weights.
    
    For inference, LoRA weights can be merged: W' = W₀ + (α/r) * BA
    This reduces the forward pass to a single matrix multiplication.
    
    Args:
        x: Input tensor [batch, seq_len, in_features]
        W_merged: Merged weight matrix [out_features, in_features]
    
    Returns:
        Output tensor [batch, seq_len, out_features]
    """
    if x.ndim == 2:
        x = x[None, :, :]
    return mx.matmul(x, W_merged.T)


# ============================================================================
# BACKWARD OPERATIONS
# ============================================================================

def lora_backward(
    grad_output: mx.array,
    x: mx.array,
    A: mx.array,
    B: mx.array,
    alpha: float = 16.0,
    clip_value: float = 1.0,
) -> Tuple[mx.array, mx.array]:
    """
    Optimized LoRA backward pass.
    
    Computes gradients for A and B matrices:
        ∂L/∂B = (α/r) * ∂L/∂h ⊗ Ax
        ∂L/∂A = (α/r) * Bᵀ @ ∂L/∂h ⊗ x
    
    Args:
        grad_output: Gradient of loss w.r.t. output [batch, seq, out_features]
        x: Cached input activations [batch, seq, in_features]
        A: LoRA down-projection [rank, in_features]
        B: LoRA up-projection [out_features, rank]
        alpha: LoRA scaling factor
        clip_value: Gradient clipping threshold
    
    Returns:
        Tuple of (grad_A, grad_B)
    """
    batch_size, seq_len, out_features = grad_output.shape
    rank, in_features = A.shape
    
    scale = alpha / rank
    
    # Compute Ax for grad_B computation
    # x: [B, S, K], A: [R, K] -> Ax: [B, S, R]
    Ax = mx.matmul(x, A.T)
    
    # Compute Bᵀ @ grad_output for grad_A computation
    # B: [D, R] -> Bᵀ: [R, D]
    # grad_output: [B, S, D]
    # Bt_grad: [B, S, R]
    Bt_grad = mx.matmul(grad_output, B)  # [B, S, D] @ [D, R] = [B, S, R]
    
    # grad_B = scale * sum over batch of (grad_output.T @ Ax)
    # grad_output: [B, S, D], Ax: [B, S, R]
    # For each sample: grad_output[i].T @ Ax[i] is [D, S] @ [S, R] = [D, R]
    # Sum over batch
    grad_B = mx.zeros((out_features, rank))
    for b in range(batch_size):
        # [S, D].T @ [S, R] = [D, R]
        grad_B = grad_B + mx.matmul(grad_output[b].T, Ax[b])
    grad_B = scale * grad_B
    
    # grad_A = scale * sum over batch of (Bt_grad.T @ x)
    # Bt_grad: [B, S, R], x: [B, S, K]
    # For each sample: Bt_grad[i].T @ x[i] is [R, S] @ [S, K] = [R, K]
    grad_A = mx.zeros((rank, in_features))
    for b in range(batch_size):
        grad_A = grad_A + mx.matmul(Bt_grad[b].T, x[b])
    grad_A = scale * grad_A
    
    # Gradient clipping
    grad_A = mx.clip(grad_A, -clip_value, clip_value)
    grad_B = mx.clip(grad_B, -clip_value, clip_value)
    
    return grad_A, grad_B


def lora_backward_efficient(
    grad_output: mx.array,
    x: mx.array,
    A: mx.array,
    B: mx.array,
    alpha: float = 16.0,
    clip_value: float = 1.0,
) -> Tuple[mx.array, mx.array]:
    """
    More memory-efficient backward using einsum.
    """
    batch_size, seq_len, out_features = grad_output.shape
    rank, in_features = A.shape
    scale = alpha / rank
    
    # Ax: [B, S, R]
    Ax = mx.matmul(x, A.T)
    
    # grad_B = scale * einsum('bsd,bsr->dr', grad_output, Ax)
    # Reshape for batched matmul: [B, D, S] @ [B, S, R] = [B, D, R]
    grad_output_t = mx.transpose(grad_output, (0, 2, 1))  # [B, D, S]
    grad_B_batched = mx.matmul(grad_output_t, Ax)  # [B, D, R]
    grad_B = scale * mx.sum(grad_B_batched, axis=0)  # [D, R]
    
    # Bt_grad: [B, S, R]
    Bt_grad = mx.matmul(grad_output, B)
    
    # grad_A = scale * einsum('bsr,bsk->rk', Bt_grad, x)
    Bt_grad_t = mx.transpose(Bt_grad, (0, 2, 1))  # [B, R, S]
    grad_A_batched = mx.matmul(Bt_grad_t, x)  # [B, R, K]
    grad_A = scale * mx.sum(grad_A_batched, axis=0)  # [R, K]
    
    # Clip
    grad_A = mx.clip(grad_A, -clip_value, clip_value)
    grad_B = mx.clip(grad_B, -clip_value, clip_value)
    
    return grad_A, grad_B


# ============================================================================
# WEIGHT MERGING
# ============================================================================

def merge_lora_weights(
    W0: mx.array,
    A: mx.array,
    B: mx.array,
    alpha: float = 16.0,
) -> mx.array:
    """
    Merge LoRA weights into base weights for inference.
    
    Computes: W' = W₀ + (α/r) * BA
    
    Args:
        W0: Base weight matrix [out_features, in_features]
        A: LoRA down-projection [rank, in_features]
        B: LoRA up-projection [out_features, rank]
        alpha: LoRA scaling factor
    
    Returns:
        Merged weight matrix [out_features, in_features]
    """
    rank = A.shape[0]
    scale = alpha / rank
    
    # BA: [D, R] @ [R, K] = [D, K]
    BA = mx.matmul(B, A)
    
    return W0 + scale * BA


def unmerge_lora_weights(
    W_merged: mx.array,
    W0: mx.array,
    A: mx.array,
    B: mx.array,
    alpha: float = 16.0,
) -> Tuple[mx.array, mx.array]:
    """
    Recover LoRA matrices A and B from merged weights.
    
    Note: This recovers the LoRA delta but cannot uniquely determine A and B.
    Returns an SVD-based decomposition of the delta.
    
    Args:
        W_merged: Merged weight matrix
        W0: Original base weight matrix
        A: Original A shape reference (for rank)
        B: Original B shape reference
        alpha: LoRA scaling factor
    
    Returns:
        Tuple of (A_recovered, B_recovered) such that W_merged ≈ W0 + (α/r) * B @ A
    """
    rank = A.shape[0]
    scale = alpha / rank
    
    # Extract delta
    delta = (W_merged - W0) / scale
    
    # SVD decomposition
    U, S, Vt = mx.linalg.svd(delta)
    
    # Take top-r singular values
    sqrt_S = mx.sqrt(S[:rank])
    
    # Reconstruct A and B
    A_recovered = mx.diag(sqrt_S) @ Vt[:rank, :]  # [R, K]
    B_recovered = U[:, :rank] @ mx.diag(sqrt_S)   # [D, R]
    
    return A_recovered, B_recovered
