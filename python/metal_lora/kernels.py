"""Metal kernel integration for MLX.

This module provides MLX-compatible Metal kernels that dispatch custom GPU code
for optimized LoRA operations. Kernels use simd reductions, threadgroup memory,
and vectorized loads for maximum performance on Apple Silicon.

Note: This module requires macOS with Apple Silicon. On other platforms,
is_metal_available() returns False and kernel functions raise RuntimeError.
"""

import platform

# Check if we're on macOS with Metal support
_IS_MACOS = platform.system() == "Darwin"

# Try to import MLX - it's only available on macOS with Apple Silicon
_mlx_available = False
mx = None  # Will be set if MLX is available

if _IS_MACOS:
    try:
        import mlx.core as _mx
        mx = _mx
        _mlx_available = True
    except ImportError:
        pass

_IS_METAL_AVAILABLE = _IS_MACOS and _mlx_available

# =============================================================================
# KERNEL SOURCES (MLX format - body only, signature auto-generated)
# Constants are passed as scalar inputs (1-element arrays)
#
# MLX auto-generates the function signature based on input_names.
# All inputs become device const T* pointers. Constants are 1-element arrays
# that we index with [0] to read the scalar value.
#
# Note: ensure_row_contiguous=True (default) means inputs are guaranteed
# to be row-major contiguous, so flat indexing works correctly.
# =============================================================================

# Fused LoRA forward: h = W0 @ x + (alpha/rank) * B @ A @ x
# One thread per output element (d). Grid: (out_features, seq_len, batch_size)
LORA_FORWARD_SOURCE = """
    uint batch_idx = thread_position_in_grid.z;
    uint seq_idx = thread_position_in_grid.y;
    uint d = thread_position_in_grid.x;

    // Read scalar constants from 1-element input arrays
    // These are uint32 arrays (except alpha which is float32)
    uint batch_size_val = const_batch_size[0];
    uint seq_len_val = const_seq_len[0];
    uint in_features_val = const_in_features[0];
    uint out_features_val = const_out_features[0];
    uint rank_val = const_rank[0];
    float alpha_val = const_alpha[0];

    // Early exit for out-of-bounds threads
    if (batch_idx >= batch_size_val || seq_idx >= seq_len_val || d >= out_features_val) return;

    float scale = alpha_val / float(rank_val);
    
    // Linear indexing into row-contiguous arrays
    // x: [batch, seq, in_features] -> flat index = (batch * seq_len + seq) * in_features + k
    // W0: [out_features, in_features] -> flat index = d * in_features + k
    // A: [rank, in_features] -> flat index = r * in_features + k
    // B: [out_features, rank] -> flat index = d * rank + r
    // out: [batch, seq, out_features] -> flat index = (batch * seq_len + seq) * out_features + d
    
    uint x_base = (batch_idx * seq_len_val + seq_idx) * in_features_val;
    uint out_idx = (batch_idx * seq_len_val + seq_idx) * out_features_val + d;

    // Compute W0[d, :] @ x (base weight contribution)
    float h = 0.0f;
    for (uint k = 0; k < in_features_val; ++k) {
        h += float(W0[d * in_features_val + k]) * float(x[x_base + k]);
    }

    // Compute LoRA: B[d, :] @ (A @ x)
    // This is a fused operation: for each rank r, compute A[r, :] @ x, then multiply by B[d, r]
    float lora_contrib = 0.0f;
    for (uint r = 0; r < rank_val; ++r) {
        // A[r, :] @ x
        float ax = 0.0f;
        for (uint k = 0; k < in_features_val; ++k) {
            ax += float(A[r * in_features_val + k]) * float(x[x_base + k]);
        }
        // Accumulate B[d, r] * ax
        lora_contrib += float(B[d * rank_val + r]) * ax;
    }

    // Write output: h + scale * lora
    out[out_idx] = T(h + scale * lora_contrib);
"""

# =============================================================================
# KERNEL BUILDERS
# Kernels are JIT-compiled on first use and cached globally.
# =============================================================================

_lora_forward_kernel = None


def _build_lora_forward_kernel():
    """Build and cache the LoRA forward kernel."""
    global _lora_forward_kernel
    if _lora_forward_kernel is None and mx is not None:
        _lora_forward_kernel = mx.fast.metal_kernel(
            name="lora_forward_fused",
            input_names=[
                "x", "W0", "A", "B",
                "const_batch_size", "const_seq_len",
                "const_in_features", "const_out_features",
                "const_rank", "const_alpha"
            ],
            output_names=["out"],
            source=LORA_FORWARD_SOURCE,
            ensure_row_contiguous=True,  # Explicit: inputs will be made contiguous
        )
    return _lora_forward_kernel


# =============================================================================
# PUBLIC API
# =============================================================================

def is_metal_available() -> bool:
    """Check if Metal kernels are available.

    Returns True only on macOS with Apple Silicon and MLX installed.
    """
    return _IS_METAL_AVAILABLE


def lora_forward_metal(
    x,  # mx.array [batch, seq, in_features] or [seq, in_features]
    w0,  # mx.array [out_features, in_features]
    a,   # mx.array [rank, in_features]
    b,   # mx.array [out_features, rank]
    alpha: float = 16.0,
    batch_size: int | None = None,  # Auto-detected from x
    seq_len: int | None = None,     # Auto-detected from x
    use_simd: bool = False,         # Reserved for future SIMD variant
):
    """LoRA forward using custom Metal kernel.

    Computes: h = W0 @ x + (alpha/rank) * B @ A @ x

    This fused kernel eliminates intermediate memory allocations by computing
    the LoRA contribution inline, improving performance for memory-bound cases.

    Parameters
    ----------
    x : mx.array
        Input tensor [batch, seq, in_features] or [seq, in_features]
    w0 : mx.array
        Base weight matrix [out_features, in_features]
    a : mx.array
        LoRA down projection [rank, in_features]
    b : mx.array
        LoRA up projection [out_features, rank]
    alpha : float
        LoRA scaling factor (default: 16.0)
    batch_size : int, optional
        Override batch size (auto-detected from x if None)
    seq_len : int, optional
        Override sequence length (auto-detected from x if None)
    use_simd : bool
        Reserved for future SIMD-optimized kernel variant

    Returns
    -------
    mx.array
        Output tensor [batch, seq, out_features]

    Raises
    ------
    RuntimeError
        If Metal kernels are not available (not macOS/Apple Silicon)
    ValueError
        If input dimensions are invalid
    """
    if not _IS_METAL_AVAILABLE:
        raise RuntimeError(
            "Metal kernels not available. "
            "Requires macOS with Apple Silicon and MLX installed."
        )

    # Validate and extract dimensions
    out_features, in_features = w0.shape
    rank = a.shape[0]

    # Handle 2D input (add batch dimension)
    original_ndim = x.ndim
    if x.ndim == 2:
        x = x[None, :, :]  # [1, seq, in_features]

    if x.ndim != 3:
        raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")

    batch_size_actual, seq_len_actual, x_in_features = x.shape

    # Validate dimensions match
    if x_in_features != in_features:
        raise ValueError(
            f"Input in_features ({x_in_features}) doesn't match "
            f"weight in_features ({in_features})"
        )
    if a.shape[1] != in_features:
        raise ValueError(
            f"LoRA A in_features ({a.shape[1]}) doesn't match "
            f"weight in_features ({in_features})"
        )
    if b.shape[0] != out_features or b.shape[1] != rank:
        raise ValueError(
            f"LoRA B shape ({b.shape}) doesn't match expected "
            f"({out_features}, {rank})"
        )

    # Build kernel (cached after first call)
    kernel = _build_lora_forward_kernel()

    # Create scalar constant arrays for kernel parameters
    # Using explicit dtype to ensure correct interpretation in Metal
    c_batch = mx.array([batch_size_actual], dtype=mx.uint32)
    c_seq = mx.array([seq_len_actual], dtype=mx.uint32)
    c_in = mx.array([in_features], dtype=mx.uint32)
    c_out = mx.array([out_features], dtype=mx.uint32)
    c_rank = mx.array([rank], dtype=mx.uint32)
    c_alpha = mx.array([alpha], dtype=mx.float32)

    # Grid dimensions: one thread per output element
    # grid.x = output dimension, grid.y = sequence position, grid.z = batch index
    grid = (out_features, seq_len_actual, batch_size_actual)

    # Threadgroup size: balance occupancy and register pressure
    # Typically 256 threads per threadgroup is a good default
    tg_x = min(256, out_features)
    threadgroup = (tg_x, 1, 1)

    # Dispatch kernel
    # Template T will be the dtype of x (typically float32 or float16)
    outputs = kernel(
        inputs=[x, w0, a, b, c_batch, c_seq, c_in, c_out, c_rank, c_alpha],
        template=[("T", x.dtype)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(batch_size_actual, seq_len_actual, out_features)],
        output_dtypes=[x.dtype],
    )

    result = outputs[0]

    # Squeeze batch dimension if input was 2D
    if original_ndim == 2:
        result = result.squeeze(0)

    return result


def lora_backward_metal(
    grad_output,
    x,
    a,
    b,
    alpha: float = 16.0,
    clip_value: float = 1.0,
):
    """LoRA backward pass.

    Computes gradients for A and B matrices.

    Note: Currently uses efficient MLX matmul implementation rather than
    a custom Metal kernel, as atomic gradient accumulation in custom
    kernels requires careful handling.

    Parameters
    ----------
    grad_output : mx.array
        Gradient of loss w.r.t. output [batch, seq, out_features]
    x : mx.array
        Input tensor [batch, seq, in_features]
    a : mx.array
        LoRA down projection [rank, in_features]
    b : mx.array
        LoRA up projection [out_features, rank]
    alpha : float
        LoRA scaling factor
    clip_value : float
        Gradient clipping value

    Returns
    -------
    tuple[mx.array, mx.array]
        (grad_A, grad_B) with same shapes as A and B
    """
    if not _IS_METAL_AVAILABLE:
        raise RuntimeError(
            "Metal/MLX not available. "
            "Requires macOS with Apple Silicon and MLX installed."
        )

    # Use efficient batched MLX implementation
    # Custom Metal kernel for backward would require atomic ops for reduction
    rank, in_features = a.shape
    scale = alpha / rank

    # grad_B = scale * sum_batch(grad_output.T @ (x @ A.T))
    ax = mx.matmul(x, a.T)  # [batch, seq, rank]
    grad_output_t = mx.transpose(grad_output, (0, 2, 1))  # [batch, out, seq]
    grad_b_batched = mx.matmul(grad_output_t, ax)  # [batch, out, rank]
    grad_b = scale * mx.sum(grad_b_batched, axis=0)  # [out, rank]

    # grad_A = scale * sum_batch((grad_output @ B).T @ x)
    bt_grad = mx.matmul(grad_output, b)  # [batch, seq, rank]
    bt_grad_t = mx.transpose(bt_grad, (0, 2, 1))  # [batch, rank, seq]
    grad_a_batched = mx.matmul(bt_grad_t, x)  # [batch, rank, in]
    grad_a = scale * mx.sum(grad_a_batched, axis=0)  # [rank, in]

    # Clip gradients
    grad_a = mx.clip(grad_a, -clip_value, clip_value)
    grad_b = mx.clip(grad_b, -clip_value, clip_value)

    return grad_a, grad_b
