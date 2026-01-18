// ============================================================================
// MetalLoRA - Fused LoRA Backward Pass Kernel
// ============================================================================
// Computes gradients for LoRA matrices A and B in a single fused kernel
//
// Given: ∂L/∂h (gradient of loss w.r.t. output)
// Compute:
//   ∂L/∂A = (Bᵀ @ ∂L/∂h) ⊗ x   (outer product accumulated over batch)
//   ∂L/∂B = ∂L/∂h ⊗ Ax         (outer product accumulated over batch)
//
// Key optimizations:
//   - Fuse both gradient computations to reuse ∂L/∂h loads
//   - Compute Ax once and cache in threadgroup memory
//   - Use atomic accumulation for batch reduction
//   - Optional gradient clipping applied in-kernel
// ============================================================================

#include "lora_common.metal"

// ============================================================================
// GRADIENT COMPUTATION MATHEMATICS
// ============================================================================
//
// Forward:  h = W₀x + (α/r) * BAx
//
// Backward (chain rule):
//   ∂L/∂B = (α/r) * ∂L/∂h @ Axᵀ      [D×R] = [D×1] @ [1×R]
//   ∂L/∂A = (α/r) * Bᵀ @ ∂L/∂h @ xᵀ  [R×K] = [R×D] @ [D×1] @ [1×K]
//
// Which simplifies per-sample to outer products, then sum over batch:
//   dB[d,r] = sum_{batch} (α/r) * dh[d] * Ax[r]
//   dA[r,k] = sum_{batch} (α/r) * (Bᵀ @ dh)[r] * x[k]
//
// Memory access pattern (for batch accumulation):
//   - dh loaded once per threadgroup
//   - x loaded once per threadgroup (cached)
//   - Ax computed and cached
//   - Gradients accumulated via atomics
//
// ============================================================================

// Thread-safe atomic add for gradient accumulation
inline void atomic_add_float(device atomic_float* addr, float val) {
    atomic_fetch_add_explicit(addr, val, memory_order_relaxed);
}

// ============================================================================
// Kernel: lora_backward_fused
// ============================================================================
// Computes gradients for both A and B matrices in a single kernel launch.
//
// Thread Organization:
//   - Grid: (ceil(D/TILE_D) * ceil(K/TILE_K), batch_size * seq_len, 1)
//   - Each threadgroup handles gradients for one (batch, seq) sample
//   - Atomic accumulation handles batch dimension reduction
//
// Memory Layout:
//   grad_h:  [batch_size, seq_len, out_features]  - input gradient
//   x:       [batch_size, seq_len, in_features]   - cached activations
//   A:       [rank, in_features]                   - LoRA down-proj (read)
//   B:       [out_features, rank]                  - LoRA up-proj (read)
//   grad_A:  [rank, in_features]                   - gradient output
//   grad_B:  [out_features, rank]                  - gradient output
// ============================================================================

kernel void lora_backward_fused(
    device const float* grad_h     [[buffer(0)]],  // ∂L/∂h
    device const float* x          [[buffer(1)]],  // Input activations
    device const float* A          [[buffer(2)]],  // LoRA down-projection
    device const float* B          [[buffer(3)]],  // LoRA up-projection
    device atomic_float* grad_A    [[buffer(4)]],  // ∂L/∂A (atomic)
    device atomic_float* grad_B    [[buffer(5)]],  // ∂L/∂B (atomic)
    constant LoRAConfig& config    [[buffer(6)]],  // Configuration
    constant GradConfig& grad_cfg  [[buffer(7)]],  // Gradient config
    
    uint3 tid                      [[thread_position_in_grid]],
    uint3 tgid                     [[threadgroup_position_in_grid]],
    uint lid                       [[thread_index_in_threadgroup]],
    uint simd_lane_id              [[thread_index_in_simdgroup]],
    uint simd_group_id             [[simdgroup_index_in_threadgroup]]
) {
    // ========================================================================
    // DIMENSION EXTRACTION
    // ========================================================================
    const uint batch_size = config.batch_size;
    const uint seq_len = config.seq_len;
    const uint K = config.in_features;
    const uint D = config.out_features;
    const uint R = config.rank;
    const float scale = config.alpha / float(R);
    const float clip_val = grad_cfg.clip_value;
    
    // Decode threadgroup position
    const uint batch_idx = tgid.y / seq_len;
    const uint seq_idx = tgid.y % seq_len;
    
    if (batch_idx >= batch_size) return;
    
    const uint sample_offset = (batch_idx * seq_len + seq_idx);
    const uint x_offset = sample_offset * K;
    const uint dh_offset = sample_offset * D;
    
    // ========================================================================
    // THREADGROUP MEMORY
    // ========================================================================
    threadgroup float tg_x[4096];      // Cached input
    threadgroup float tg_dh[4096];     // Cached grad_h
    threadgroup float tg_Ax[128];      // Cached Ax
    threadgroup float tg_Bt_dh[128];   // Cached Bᵀ @ dh
    
    const uint threads_per_tg = 256;
    
    // ========================================================================
    // PHASE 1: COOPERATIVE LOADING
    // ========================================================================
    
    // Load input x
    for (uint k = lid; k < K; k += threads_per_tg) {
        tg_x[k] = x[x_offset + k];
    }
    
    // Load grad_h
    for (uint d = lid; d < D; d += threads_per_tg) {
        tg_dh[d] = grad_h[dh_offset + d];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // ========================================================================
    // PHASE 2: COMPUTE INTERMEDIATE VALUES
    // ========================================================================
    
    // Compute Ax (needed for grad_B)
    for (uint r = lid; r < R; r += threads_per_tg) {
        float acc = 0.0f;
        for (uint k = 0; k < K; ++k) {
            acc += A[r * K + k] * tg_x[k];
        }
        tg_Ax[r] = acc;
    }
    
    // Compute Bᵀ @ dh (needed for grad_A)
    // Bᵀ[r,d] = B[d,r], so Bᵀ @ dh = sum_d B[d,r] * dh[d]
    for (uint r = lid; r < R; r += threads_per_tg) {
        float acc = 0.0f;
        for (uint d = 0; d < D; ++d) {
            acc += B[d * R + r] * tg_dh[d];
        }
        tg_Bt_dh[r] = acc;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // ========================================================================
    // PHASE 3: COMPUTE AND ACCUMULATE GRADIENTS
    // ========================================================================
    
    // grad_B[d,r] += scale * dh[d] * Ax[r]
    // Each thread handles a subset of (d,r) pairs
    const uint total_B_elements = D * R;
    const uint B_elements_per_thread = (total_B_elements + threads_per_tg - 1) / threads_per_tg;
    
    for (uint i = 0; i < B_elements_per_thread; ++i) {
        uint flat_idx = lid + i * threads_per_tg;
        if (flat_idx < total_B_elements) {
            uint d = flat_idx / R;
            uint r = flat_idx % R;
            
            float grad = scale * tg_dh[d] * tg_Ax[r];
            
            // Gradient clipping
            grad = clamp(grad, -clip_val, clip_val);
            
            atomic_add_float(&grad_B[d * R + r], grad);
        }
    }
    
    // grad_A[r,k] += scale * Bt_dh[r] * x[k]
    const uint total_A_elements = R * K;
    const uint A_elements_per_thread = (total_A_elements + threads_per_tg - 1) / threads_per_tg;
    
    for (uint i = 0; i < A_elements_per_thread; ++i) {
        uint flat_idx = lid + i * threads_per_tg;
        if (flat_idx < total_A_elements) {
            uint r = flat_idx / K;
            uint k = flat_idx % K;
            
            float grad = scale * tg_Bt_dh[r] * tg_x[k];
            
            // Gradient clipping
            grad = clamp(grad, -clip_val, clip_val);
            
            atomic_add_float(&grad_A[r * K + k], grad);
        }
    }
}

// ============================================================================
// Kernel: lora_backward_separate_A
// ============================================================================
// Variant that only computes grad_A (useful for layer-wise gradient comp)
// ============================================================================

kernel void lora_backward_A(
    device const float* grad_h     [[buffer(0)]],
    device const float* x          [[buffer(1)]],
    device const float* B          [[buffer(2)]],
    device atomic_float* grad_A    [[buffer(3)]],
    constant LoRAConfig& config    [[buffer(4)]],
    constant GradConfig& grad_cfg  [[buffer(5)]],
    
    uint3 tid                      [[thread_position_in_grid]],
    uint3 tgid                     [[threadgroup_position_in_grid]],
    uint lid                       [[thread_index_in_threadgroup]]
) {
    const uint batch_size = config.batch_size;
    const uint seq_len = config.seq_len;
    const uint K = config.in_features;
    const uint D = config.out_features;
    const uint R = config.rank;
    const float scale = config.alpha / float(R);
    const float clip_val = grad_cfg.clip_value;
    
    const uint batch_idx = tgid.y / seq_len;
    const uint seq_idx = tgid.y % seq_len;
    
    if (batch_idx >= batch_size) return;
    
    const uint sample_offset = (batch_idx * seq_len + seq_idx);
    
    threadgroup float tg_x[4096];
    threadgroup float tg_Bt_dh[128];
    
    const uint threads_per_tg = 256;
    
    // Load x
    for (uint k = lid; k < K; k += threads_per_tg) {
        tg_x[k] = x[sample_offset * K + k];
    }
    
    // Compute Bᵀ @ dh
    for (uint r = lid; r < R; r += threads_per_tg) {
        float acc = 0.0f;
        for (uint d = 0; d < D; ++d) {
            acc += B[d * R + r] * grad_h[sample_offset * D + d];
        }
        tg_Bt_dh[r] = acc;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute grad_A
    const uint total_A = R * K;
    for (uint flat = lid; flat < total_A; flat += threads_per_tg) {
        uint r = flat / K;
        uint k = flat % K;
        float grad = scale * tg_Bt_dh[r] * tg_x[k];
        grad = clamp(grad, -clip_val, clip_val);
        atomic_add_float(&grad_A[r * K + k], grad);
    }
}

// ============================================================================
// Kernel: lora_backward_separate_B
// ============================================================================
// Variant that only computes grad_B
// ============================================================================

kernel void lora_backward_B(
    device const float* grad_h     [[buffer(0)]],
    device const float* x          [[buffer(1)]],
    device const float* A          [[buffer(2)]],
    device atomic_float* grad_B    [[buffer(3)]],
    constant LoRAConfig& config    [[buffer(4)]],
    constant GradConfig& grad_cfg  [[buffer(5)]],
    
    uint3 tid                      [[thread_position_in_grid]],
    uint3 tgid                     [[threadgroup_position_in_grid]],
    uint lid                       [[thread_index_in_threadgroup]]
) {
    const uint batch_size = config.batch_size;
    const uint seq_len = config.seq_len;
    const uint K = config.in_features;
    const uint D = config.out_features;
    const uint R = config.rank;
    const float scale = config.alpha / float(R);
    const float clip_val = grad_cfg.clip_value;
    
    const uint batch_idx = tgid.y / seq_len;
    const uint seq_idx = tgid.y % seq_len;
    
    if (batch_idx >= batch_size) return;
    
    const uint sample_offset = (batch_idx * seq_len + seq_idx);
    
    threadgroup float tg_Ax[128];
    threadgroup float tg_dh[4096];
    
    const uint threads_per_tg = 256;
    
    // Compute Ax
    for (uint r = lid; r < R; r += threads_per_tg) {
        float acc = 0.0f;
        for (uint k = 0; k < K; ++k) {
            acc += A[r * K + k] * x[sample_offset * K + k];
        }
        tg_Ax[r] = acc;
    }
    
    // Load grad_h
    for (uint d = lid; d < D; d += threads_per_tg) {
        tg_dh[d] = grad_h[sample_offset * D + d];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute grad_B
    const uint total_B = D * R;
    for (uint flat = lid; flat < total_B; flat += threads_per_tg) {
        uint d = flat / R;
        uint r = flat % R;
        float grad = scale * tg_dh[d] * tg_Ax[r];
        grad = clamp(grad, -clip_val, clip_val);
        atomic_add_float(&grad_B[d * R + r], grad);
    }
}

// ============================================================================
// Kernel: lora_backward_with_weight_decay
// ============================================================================
// Fused backward with L2 regularization applied in-kernel
// ============================================================================

kernel void lora_backward_l2(
    device const float* grad_h     [[buffer(0)]],
    device const float* x          [[buffer(1)]],
    device const float* A          [[buffer(2)]],
    device const float* B          [[buffer(3)]],
    device atomic_float* grad_A    [[buffer(4)]],
    device atomic_float* grad_B    [[buffer(5)]],
    constant LoRAConfig& config    [[buffer(6)]],
    constant GradConfig& grad_cfg  [[buffer(7)]],
    
    uint3 tid                      [[thread_position_in_grid]],
    uint3 tgid                     [[threadgroup_position_in_grid]],
    uint lid                       [[thread_index_in_threadgroup]]
) {
    const uint batch_size = config.batch_size;
    const uint seq_len = config.seq_len;
    const uint K = config.in_features;
    const uint D = config.out_features;
    const uint R = config.rank;
    const float scale = config.alpha / float(R);
    const float clip_val = grad_cfg.clip_value;
    const float wd = grad_cfg.weight_decay;
    
    const uint batch_idx = tgid.y / seq_len;
    const uint seq_idx = tgid.y % seq_len;
    
    if (batch_idx >= batch_size) return;
    
    const uint sample_offset = (batch_idx * seq_len + seq_idx);
    
    threadgroup float tg_x[4096];
    threadgroup float tg_dh[4096];
    threadgroup float tg_Ax[128];
    threadgroup float tg_Bt_dh[128];
    
    const uint threads_per_tg = 256;
    
    // Load data
    for (uint k = lid; k < K; k += threads_per_tg) {
        tg_x[k] = x[sample_offset * K + k];
    }
    for (uint d = lid; d < D; d += threads_per_tg) {
        tg_dh[d] = grad_h[sample_offset * D + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute intermediates
    for (uint r = lid; r < R; r += threads_per_tg) {
        float acc = 0.0f;
        for (uint k = 0; k < K; ++k) {
            acc += A[r * K + k] * tg_x[k];
        }
        tg_Ax[r] = acc;
    }
    
    for (uint r = lid; r < R; r += threads_per_tg) {
        float acc = 0.0f;
        for (uint d = 0; d < D; ++d) {
            acc += B[d * R + r] * tg_dh[d];
        }
        tg_Bt_dh[r] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute grad_B with L2 regularization
    const uint total_B = D * R;
    for (uint flat = lid; flat < total_B; flat += threads_per_tg) {
        uint d = flat / R;
        uint r = flat % R;
        float grad = scale * tg_dh[d] * tg_Ax[r];
        grad += wd * B[d * R + r];  // L2 regularization
        grad = clamp(grad, -clip_val, clip_val);
        atomic_add_float(&grad_B[d * R + r], grad);
    }
    
    // Compute grad_A with L2 regularization
    const uint total_A = R * K;
    for (uint flat = lid; flat < total_A; flat += threads_per_tg) {
        uint r = flat / K;
        uint k = flat % K;
        float grad = scale * tg_Bt_dh[r] * tg_x[k];
        grad += wd * A[r * K + k];  // L2 regularization
        grad = clamp(grad, -clip_val, clip_val);
        atomic_add_float(&grad_A[r * K + k], grad);
    }
}
