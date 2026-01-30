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
# Uses float4 vectorized loads for 4x better memory bandwidth
LORA_FORWARD_SOURCE = """
    uint batch_idx = thread_position_in_grid.z;
    uint seq_idx = thread_position_in_grid.y;
    uint d = thread_position_in_grid.x;

    // Read scalar constants
    uint batch_size_val = const_batch_size[0];
    uint seq_len_val = const_seq_len[0];
    uint in_features_val = const_in_features[0];
    uint out_features_val = const_out_features[0];
    uint rank_val = const_rank[0];
    float scale = const_alpha[0] / float(rank_val);

    // Bounds check
    if (batch_idx >= batch_size_val || seq_idx >= seq_len_val || d >= out_features_val) return;

    // Precompute base indices
    uint x_base = (batch_idx * seq_len_val + seq_idx) * in_features_val;
    uint w_base = d * in_features_val;
    uint out_idx = (batch_idx * seq_len_val + seq_idx) * out_features_val + d;

    // Vectorized W0 @ x using float4 (4x memory bandwidth)
    float h = 0.0f;
    uint k_vec = in_features_val / 4;
    for (uint k4 = 0; k4 < k_vec; ++k4) {
        uint k = k4 * 4;
        float4 w_vec = float4(W0[w_base + k], W0[w_base + k + 1], W0[w_base + k + 2], W0[w_base + k + 3]);
        float4 x_vec = float4(x[x_base + k], x[x_base + k + 1], x[x_base + k + 2], x[x_base + k + 3]);
        h += dot(w_vec, x_vec);
    }
    // Handle remainder
    for (uint k = k_vec * 4; k < in_features_val; ++k) {
        h += float(W0[w_base + k]) * float(x[x_base + k]);
    }

    // Fused LoRA: B[d,:] @ (A @ x)
    float lora = 0.0f;
    for (uint r = 0; r < rank_val; ++r) {
        uint a_base = r * in_features_val;
        float ax = 0.0f;
        // Vectorized A @ x
        for (uint k4 = 0; k4 < k_vec; ++k4) {
            uint k = k4 * 4;
            float4 a_vec = float4(A[a_base + k], A[a_base + k + 1], A[a_base + k + 2], A[a_base + k + 3]);
            float4 x_vec = float4(x[x_base + k], x[x_base + k + 1], x[x_base + k + 2], x[x_base + k + 3]);
            ax += dot(a_vec, x_vec);
        }
        for (uint k = k_vec * 4; k < in_features_val; ++k) {
            ax += float(A[a_base + k]) * float(x[x_base + k]);
        }
        lora += float(B[d * rank_val + r]) * ax;
    }

    out[out_idx] = T(h + scale * lora);
"""

# =============================================================================
# SIMD-OPTIMIZED KERNEL (simdgroup_matrix)
# Uses hardware matrix units for 16x faster computation.
# Each threadgroup processes 32x32 tiles of the output.
# =============================================================================

LORA_FORWARD_SIMD_SOURCE = """
    // Thread identification using MLX-provided built-ins
    uint batch_idx = threadgroup_position_in_grid.z;
    uint seq_idx = threadgroup_position_in_grid.y;
    uint tg_base_d = threadgroup_position_in_grid.x * 32;
    
    // Compute thread-local index from grid position
    uint tid = thread_position_in_grid.x;
    uint lid = tid % 128;  // Local thread ID within threadgroup (assuming 128 threads/TG)
    uint simd_id = lid / 32;  // Which simdgroup within the threadgroup
    
    // Read constants
    uint batch_size_val = const_batch_size[0];
    uint seq_len_val = const_seq_len[0];
    uint in_features_val = const_in_features[0];
    uint out_features_val = const_out_features[0];
    uint rank_val = const_rank[0];
    float scale = const_alpha[0] / float(rank_val);

    if (batch_idx >= batch_size_val || seq_idx >= seq_len_val) return;

    uint x_base = (batch_idx * seq_len_val + seq_idx) * in_features_val;
    uint out_base = (batch_idx * seq_len_val + seq_idx) * out_features_val;

    uint d_base = tg_base_d + simd_id * 8;
    if (d_base >= out_features_val) return;

    // Shared memory for x and Ax
    threadgroup float tg_x[4096];
    threadgroup float tg_Ax[128];

    uint tg_size = 128;  // Threadgroup size

    // Load x into threadgroup memory
    for (uint k = lid; k < in_features_val; k += tg_size) {
        tg_x[k] = float(x[x_base + k]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute Ax (Vectorized)
    for (uint r = lid; r < rank_val; r += tg_size) {
        float ax = 0.0f;
        uint a_base = r * in_features_val;
        uint k_vec = in_features_val / 4;
        for (uint k4 = 0; k4 < k_vec; ++k4) {
            uint k = k4 * 4;
            float4 a_v = float4(A[a_base + k], A[a_base + k + 1], A[a_base + k + 2], A[a_base + k + 3]);
            float4 x_v = float4(tg_x[k], tg_x[k+1], tg_x[k+2], tg_x[k+3]);
            ax += dot(a_v, x_v);
        }
        // Handle remainder
        for (uint k = k_vec * 4; k < in_features_val; ++k) {
            ax += A[a_base + k] * tg_x[k];
        }
        tg_Ax[r] = ax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute W0 @ x using vectorization (no simdgroup_matrix)
    for (uint i = 0; i < 8 && d_base + i < out_features_val; ++i) {
        if (lid == i) {  // Only one thread per output element
            uint d = d_base + i;
            float h = 0.0f;
            uint w_base = d * in_features_val;
            uint k_vec = in_features_val / 4;
            
            for (uint k4 = 0; k4 < k_vec; ++k4) {
                uint k = k4 * 4;
                float4 w_v = float4(W0[w_base + k], W0[w_base + k + 1], W0[w_base + k + 2], W0[w_base + k + 3]);
                float4 x_v = float4(tg_x[k], tg_x[k+1], tg_x[k+2], tg_x[k+3]);
                h += dot(w_v, x_v);
            }
            for (uint k = k_vec * 4; k < in_features_val; ++k) {
                h += W0[w_base + k] * tg_x[k];
            }
            
            // Add LoRA contribution
            float lora = 0.0f;
            for (uint r = 0; r < rank_val; ++r) {
                lora += float(B[d * rank_val + r]) * tg_Ax[r];
            }
            
            out[out_base + d] = T(h + scale * lora);
        }
    }
"""


# =============================================================================
# BACKWARD KERNEL
# Fused gradient computation for A and B.
# =============================================================================

LORA_BACKWARD_SOURCE = """
    uint batch_idx = threadgroup_position_in_grid.z;
    uint seq_idx = threadgroup_position_in_grid.y;
    
    uint batch_size_val = const_batch_size[0];
    uint seq_len_val = const_seq_len[0];
    uint in_features_val = const_in_features[0];
    uint out_features_val = const_out_features[0];
    uint rank_val = const_rank[0];
    float scale = const_alpha[0] / float(rank_val);
    float clip_val = const_clip[0];

    if (batch_idx >= batch_size_val || seq_idx >= seq_len_val) return;

    uint x_offset = (batch_idx * seq_len_val + seq_idx) * in_features_val;
    uint grad_offset = (batch_idx * seq_len_val + seq_idx) * out_features_val;

    threadgroup float tg_x[4096];
    threadgroup float tg_grad[4096];
    threadgroup float tg_Ax[128];
    threadgroup float tg_Bt_grad[128];

    uint lid = thread_position_in_grid.x % 256;  // Local thread ID (assuming 256 threads/TG)
    uint tg_size = 256;  // Threadgroup size

    // Load inputs
    for (uint k = lid; k < in_features_val; k += tg_size) tg_x[k] = float(x[x_offset + k]);
    for (uint d = lid; d < out_features_val; d += tg_size) tg_grad[d] = float(grad_output[grad_offset + d]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute Ax
    for (uint r = lid; r < rank_val; r += tg_size) {
        float ax = 0.0f;
        for (uint k = 0; k < in_features_val; ++k) ax += float(A[r * in_features_val + k]) * tg_x[k];
        tg_Ax[r] = ax;
    }
    
    // Compute B^T @ grad_output
    for (uint r = lid; r < rank_val; r += tg_size) {
        float bt_g = 0.0f;
        for (uint d = 0; d < out_features_val; ++d) bt_g += float(B[d * rank_val + r]) * tg_grad[d];
        tg_Bt_grad[r] = bt_g;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulate gradients using atomics
    for (uint dr = lid; dr < out_features_val * rank_val; dr += tg_size) {
        uint d = dr / rank_val;
        uint r = dr % rank_val;
        float g_val = clamp(scale * tg_grad[d] * tg_Ax[r], -clip_val, clip_val);
        atomic_fetch_add_explicit((device atomic_float*)&grad_B[dr], g_val, memory_order_relaxed);
    }

    for (uint rk = lid; rk < rank_val * in_features_val; rk += tg_size) {
        uint r = rk / in_features_val;
        uint k = rk % in_features_val;
        float g_val = clamp(scale * tg_Bt_grad[r] * tg_x[k], -clip_val, clip_val);
        atomic_fetch_add_explicit((device atomic_float*)&grad_A[rk], g_val, memory_order_relaxed);
    }
"""

# =============================================================================
# OPTIMIZER KERNEL (AdamW)
# =============================================================================

LORA_ADAMW_SOURCE = """
    uint tid = thread_position_in_grid.x;
    uint param_size = const_size[0];
    if (tid >= param_size) return;

    device const float* state = (device const float*)gpu_state;
    float step_val = state[0];
    float b1t = state[1];
    float b2t = state[2];

    float lr = const_lr[0];
    float b1 = const_beta1[0];
    float b2 = const_beta2[0];
    float eps = const_eps[0];
    float wd = const_wd[0];

    float g = grad[tid];
    grad[tid] = 0.0f; // Reset grad

    float m_val = b1 * m[tid] + (1.0f - b1) * g;
    float v_val = b2 * v[tid] + (1.0f - b2) * g * g;
    m[tid] = m_val;
    v[tid] = v_val;

    float m_hat = m_val / (1.0f - b1t);
    float v_hat = v_val / (1.0f - b2t);
    
    float p = float(param[tid]);
    p -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * p);
    param[tid] = T(p);
"""

# =============================================================================
# MAINTENANCE KERNEL
# Updates persistent GPU state (step count, beta powers, etc.)
# =============================================================================

LORA_MAINTENANCE_SOURCE = """
    if (thread_position_in_grid.x != 0) return;
    
    device float* state = (device float*)gpu_state;
    // state[0] = step
    // state[1] = beta1_pow
    // state[2] = beta2_pow
    
    state[0] += 1.0f;
    state[1] *= beta1[0];
    state[2] *= beta2[0];
"""


# =============================================================================
# KERNEL BUILDERS
# KernELS are JIT-compiled on first use and cached globally.
# =============================================================================

_lora_forward_kernel = None
_lora_forward_simd_kernel = None
_lora_backward_kernel = None
_lora_adamw_kernel = None
_lora_maintenance_kernel = None


def _build_lora_maintenance_kernel():
    """Build and cache the maintenance kernel."""
    global _lora_maintenance_kernel
    if _lora_maintenance_kernel is None and mx is not None:
        _lora_maintenance_kernel = mx.fast.metal_kernel(
            name="lora_maintenance",
            input_names=["gpu_state", "beta1", "beta2"],
            output_names=[],
            source=LORA_MAINTENANCE_SOURCE,
            ensure_row_contiguous=True,
        )
    return _lora_maintenance_kernel


def _build_lora_forward_kernel():
    """Build and cache the basic LoRA forward kernel."""
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
            ensure_row_contiguous=True,
        )
    return _lora_forward_kernel


def _build_lora_forward_simd_kernel():
    """Build and cache the SIMD-optimized LoRA forward kernel."""
    global _lora_forward_simd_kernel
    if _lora_forward_simd_kernel is None and mx is not None:
        _lora_forward_simd_kernel = mx.fast.metal_kernel(
            name="lora_forward_simd",
            input_names=[
                "x", "W0", "A", "B",
                "const_batch_size", "const_seq_len",
                "const_in_features", "const_out_features",
                "const_rank", "const_alpha"
            ],
            output_names=["out"],
            source=LORA_FORWARD_SIMD_SOURCE,
            ensure_row_contiguous=True,
        )
    return _lora_forward_simd_kernel


def _build_lora_backward_kernel():
    """Build and cache the LoRA backward kernel."""
    global _lora_backward_kernel
    if _lora_backward_kernel is None and mx is not None:
        _lora_backward_kernel = mx.fast.metal_kernel(
            name="lora_backward_fused",
            input_names=[
                "grad_output", "x", "A", "B",
                "grad_A", "grad_B",
                "const_batch_size", "const_seq_len",
                "const_in_features", "const_out_features",
                "const_rank", "const_alpha", "const_clip"
            ],
            output_names=[], # Updates grad_A/B in-place
            source=LORA_BACKWARD_SOURCE,
            ensure_row_contiguous=True,
        )
    return _lora_backward_kernel


def _build_lora_adamw_kernel():
    """Build and cache the AdamW optimizer kernel."""
    global _lora_adamw_kernel
    if _lora_adamw_kernel is None and mx is not None:
        _lora_adamw_kernel = mx.fast.metal_kernel(
            name="lora_adamw_step",
            input_names=[
                "param", "grad", "m", "v", "gpu_state",
                "const_size", "const_lr", "const_beta1", "const_beta2",
                "const_eps", "const_wd"
            ],
            output_names=[], # Updates param, m, v in-place
            source=LORA_ADAMW_SOURCE,
            ensure_row_contiguous=True,
        )
    return _lora_adamw_kernel


# =============================================================================
# PUBLIC API
# =============================================================================

def persistent_gpu_state_init(beta1: float = 0.9, beta2: float = 0.999):
    """Initialize persistent GPU state for training.
    
    Returns
    -------
    mx.array
        Persistent state buffer: [step, beta1_pow, beta2_pow]
    """
    if not _IS_METAL_AVAILABLE:
        return None
    return mx.array([1.0, beta1, beta2], dtype=mx.float32)


def persistent_gpu_state_step(gpu_state, beta1: float = 0.9, beta2: float = 0.999):
    """Increment step and update beta powers on GPU."""
    if not _IS_METAL_AVAILABLE:
        return
        
    c_b1 = mx.array([beta1], dtype=mx.float32)
    c_b2 = mx.array([beta2], dtype=mx.float32)
    
    kernel = _build_lora_maintenance_kernel()
    kernel(
        inputs=[gpu_state, c_b1, c_b2],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[],
        output_dtypes=[],
    )


def is_metal_available() -> bool:
    """Check if Metal kernels are available."""
    return _IS_METAL_AVAILABLE


def lora_forward_metal(
    x,
    w0,
    a,
    b,
    alpha: float = 16.0,
    batch_size: int | None = None,
    seq_len: int | None = None,
    use_simd: bool | None = None,
):
    """Fused LoRA forward pass."""
    if not _IS_METAL_AVAILABLE:
        raise RuntimeError("Metal kernels not available.")

    out_features, in_features = w0.shape
    rank = a.shape[0]

    original_ndim = x.ndim
    if x.ndim == 2:
        x = x[None, :, :]

    batch_size_actual, seq_len_actual, x_in_features = x.shape

    if use_simd is None:
        use_simd = in_features >= 256 and rank <= 128 and in_features <= 4096

    c_batch = mx.array([batch_size_actual], dtype=mx.uint32)
    c_seq = mx.array([seq_len_actual], dtype=mx.uint32)
    c_in = mx.array([in_features], dtype=mx.uint32)
    c_out = mx.array([out_features], dtype=mx.uint32)
    c_rank = mx.array([rank], dtype=mx.uint32)
    c_alpha = mx.array([alpha], dtype=mx.float32)

    if use_simd:
        kernel = _build_lora_forward_simd_kernel()
        grid = (out_features // 32 + 1, seq_len_actual, batch_size_actual)
        threadgroup = (128, 1, 1)
        
        template = [("T", x.dtype)]
    else:
        kernel = _build_lora_forward_kernel()
        grid = (out_features, seq_len_actual, batch_size_actual)
        threadgroup = (256, 1, 1)
        template = [("T", x.dtype)]

    outputs = kernel(
        inputs=[x, w0, a, b, c_batch, c_seq, c_in, c_out, c_rank, c_alpha],
        template=template,
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(batch_size_actual, seq_len_actual, out_features)],
        output_dtypes=[x.dtype],
    )

    result = outputs[0]
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
    """Fused LoRA backward pass."""
    if not _IS_METAL_AVAILABLE:
        raise RuntimeError("Metal kernels not available.")

    rank, in_features = a.shape
    out_features = b.shape[0]
    batch_size, seq_len, _ = x.shape

    # Accumulators for gradients
    grad_a = mx.zeros_like(a, dtype=mx.float32)
    grad_b = mx.zeros_like(b, dtype=mx.float32)

    c_batch = mx.array([batch_size], dtype=mx.uint32)
    c_seq = mx.array([seq_len], dtype=mx.uint32)
    c_in = mx.array([in_features], dtype=mx.uint32)
    c_out = mx.array([out_features], dtype=mx.uint32)
    c_rank = mx.array([rank], dtype=mx.uint32)
    c_alpha = mx.array([alpha], dtype=mx.float32)
    c_clip = mx.array([clip_value], dtype=mx.float32)

    kernel = _build_lora_backward_kernel()
    grid = (1, seq_len, batch_size)
    threadgroup = (256, 1, 1)

    kernel(
        inputs=[
            grad_output, x, a, b, grad_a, grad_b,
            c_batch, c_seq, c_in, c_out, c_rank, c_alpha, c_clip
        ],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[],
        output_dtypes=[],
    )

    return grad_a.astype(a.dtype), grad_b.astype(b.dtype)


def lora_adamw_step_metal(
    param, grad, m, v, gpu_state,
    lr: float, beta1: float, beta2: float, eps: float, wd: float,
):
    """AdamW optimizer step using Metal kernel."""
    if not _IS_METAL_AVAILABLE:
        raise RuntimeError("Metal kernels not available.")

    size = param.size
    
    c_size = mx.array([size], dtype=mx.uint32)
    c_lr = mx.array([lr], dtype=mx.float32)
    c_b1 = mx.array([beta1], dtype=mx.float32)
    c_b2 = mx.array([beta2], dtype=mx.float32)
    c_eps = mx.array([eps], dtype=mx.float32)
    c_wd = mx.array([wd], dtype=mx.float32)

    kernel = _build_lora_adamw_kernel()
    grid = (size, 1, 1)
    threadgroup = (256, 1, 1)

    kernel(
        inputs=[param, grad, m, v, gpu_state, c_size, c_lr, c_b1, c_b2, c_eps, c_wd],
        template=[("T", param.dtype)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[],
        output_dtypes=[],
    )
