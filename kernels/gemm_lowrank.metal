// ============================================================================
// MetalLoRA - Rank-Adaptive Low-Rank GEMM Kernels
// ============================================================================
// Specialized GEMM implementations optimized for small inner dimensions
// (typical LoRA ranks: 4, 8, 16, 32, 64)
//
// Key insight: Standard GEMM kernels are optimized for large square matrices.
// For LoRA where one dimension is very small (rank), we need different
// strategies to maximize hardware utilization.
//
// Optimizations:
//   - Tile sizes tuned per-rank to maximize SIMD occupancy
//   - Vectorized loads (float4) for memory bandwidth
//   - Software pipelining to hide memory latency
//   - Specialized paths for common ranks (8, 16, 32)
// ============================================================================

#include "lora_common.metal"

// ============================================================================
// RANK-SPECIFIC TILE SIZE SELECTION
// ============================================================================
//
// Analysis for different ranks:
//
// Rank 4:  4 elements fit in 1 float4, use 8x8 tiles
//          Limited parallelism, prioritize memory coalescence
//
// Rank 8:  8 elements = 2 float4, 8x8 tiles work well
//          Good SIMD utilization with 32-thread groups
//
// Rank 16: 16 elements = 4 float4, can use 16x16 tiles
//          Balanced compute vs memory
//
// Rank 32: 32 elements = 8 float4, 16x16 or 32x8 tiles
//          Full SIMD group can handle one row
//
// Rank 64+: 64+ elements, standard GEMM tiling applies
//           Fall back to larger tiles (32x32)
//
// ============================================================================

// ----------------------------------------------------------------------------
// Kernel: gemm_lowrank_Ax
// ----------------------------------------------------------------------------
// Computes Ax where A is [R x K] and x is [B x S x K]
// Output: [B x S x R]
//
// Optimized for small R (LoRA rank), parallelizes over B and S
// ----------------------------------------------------------------------------

kernel void gemm_lowrank_Ax(
    device const float* A          [[buffer(0)]],  // [R x K]
    device const float* x          [[buffer(1)]],  // [B x S x K]
    device float* Ax               [[buffer(2)]],  // [B x S x R]
    constant uint& R               [[buffer(3)]],  // Rank
    constant uint& K               [[buffer(4)]],  // Input features
    constant uint& B               [[buffer(5)]],  // Batch size
    constant uint& S               [[buffer(6)]],  // Sequence length
    
    uint3 tid                      [[thread_position_in_grid]],
    uint3 tgid                     [[threadgroup_position_in_grid]],
    uint lid                       [[thread_index_in_threadgroup]]
) {
    // Each thread computes one element of Ax
    const uint batch_idx = tid.z;
    const uint seq_idx = tid.y;
    const uint r = tid.x;
    
    if (batch_idx >= B || seq_idx >= S || r >= R) return;
    
    const uint x_offset = (batch_idx * S + seq_idx) * K;
    
    // Vectorized dot product: Ax[r] = sum_k A[r,k] * x[k]
    float acc = 0.0f;
    
    // Process 4 elements at a time
    uint k = 0;
    for (; k + 3 < K; k += 4) {
        float4 a_vec = *reinterpret_cast<device const float4*>(A + r * K + k);
        float4 x_vec = *reinterpret_cast<device const float4*>(x + x_offset + k);
        acc += dot(a_vec, x_vec);
    }
    
    // Handle remaining elements
    for (; k < K; ++k) {
        acc += A[r * K + k] * x[x_offset + k];
    }
    
    Ax[(batch_idx * S + seq_idx) * R + r] = acc;
}

// ----------------------------------------------------------------------------
// Kernel: gemm_lowrank_BAx
// ----------------------------------------------------------------------------
// Computes BAx where B is [D x R] and Ax is [batch x seq x R]
// Output: [batch x seq x D]
//
// Optimized for small R, vectorized over R dimension
// ----------------------------------------------------------------------------

kernel void gemm_lowrank_BAx(
    device const float* B_mat      [[buffer(0)]],  // [D x R]
    device const float* Ax         [[buffer(1)]],  // [batch x seq x R]
    device float* out              [[buffer(2)]],  // [batch x seq x D]
    constant uint& D               [[buffer(3)]],  // Output features
    constant uint& R               [[buffer(4)]],  // Rank
    constant uint& batch_size      [[buffer(5)]],  // Batch size
    constant uint& seq_len         [[buffer(6)]],  // Sequence length
    
    uint3 tid                      [[thread_position_in_grid]],
    uint3 tgid                     [[threadgroup_position_in_grid]],
    uint lid                       [[thread_index_in_threadgroup]]
) {
    const uint batch_idx = tid.z;
    const uint seq_idx = tid.y;
    const uint d = tid.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || d >= D) return;
    
    const uint ax_offset = (batch_idx * seq_len + seq_idx) * R;
    
    // out[d] = sum_r B[d,r] * Ax[r]
    float acc = 0.0f;
    
    // Vectorized for R >= 4
    uint r = 0;
    for (; r + 3 < R; r += 4) {
        float4 b_vec = *reinterpret_cast<device const float4*>(B_mat + d * R + r);
        float4 ax_vec = *reinterpret_cast<device const float4*>(Ax + ax_offset + r);
        acc += dot(b_vec, ax_vec);
    }
    
    // Remaining
    for (; r < R; ++r) {
        acc += B_mat[d * R + r] * Ax[ax_offset + r];
    }
    
    out[(batch_idx * seq_len + seq_idx) * D + d] = acc;
}

// ============================================================================
// SPECIALIZED KERNELS FOR COMMON RANKS
// ============================================================================

// ----------------------------------------------------------------------------
// Kernel: gemm_rank8_Ax - Optimized for rank=8
// ----------------------------------------------------------------------------
// Uses 2x float4 loads for perfect vectorization
// Single simdgroup handles 32 samples in parallel
// ----------------------------------------------------------------------------

kernel void gemm_rank8_Ax(
    device const float* A          [[buffer(0)]],  // [8 x K]
    device const float* x          [[buffer(1)]],  // [B x S x K]
    device float* Ax               [[buffer(2)]],  // [B x S x 8]
    constant uint& K               [[buffer(3)]],  // Input features
    constant uint& B               [[buffer(4)]],  // Batch size
    constant uint& S               [[buffer(5)]],  // Sequence length
    
    uint3 tid                      [[thread_position_in_grid]],
    uint lid                       [[thread_index_in_threadgroup]],
    uint simd_lane_id              [[thread_index_in_simdgroup]]
) {
    const uint sample_idx = tid.y;  // Combined batch & seq
    const uint total_samples = B * S;
    
    if (sample_idx >= total_samples) return;
    
    const uint x_offset = sample_idx * K;
    
    // Each thread in simdgroup computes 8 outputs via reduction
    // Lane handles different samples (for occupancy)
    
    threadgroup float tg_A[8][256];  // Cache A rows in threadgroup
    
    // Load A into threadgroup memory (cooperative)
    if (lid < 8) {
        for (uint k = 0; k < min(K, 256u); k += 32) {
            uint load_k = k + simd_lane_id;
            if (load_k < K) {
                tg_A[lid][load_k] = A[lid * K + load_k];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute 8 outputs
    float results[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    
    for (uint k = 0; k < K; k += 4) {
        float4 x_vec = safe_load4(x + x_offset, k, K);
        
        for (uint r = 0; r < 8; ++r) {
            float4 a_vec = float4(
                k < K ? tg_A[r][k] : 0.0f,
                k+1 < K ? tg_A[r][k+1] : 0.0f,
                k+2 < K ? tg_A[r][k+2] : 0.0f,
                k+3 < K ? tg_A[r][k+3] : 0.0f
            );
            results[r] += dot(a_vec, x_vec);
        }
    }
    
    // Write outputs as float4 pairs for coalesced stores
    float4 out_lo = float4(results[0], results[1], results[2], results[3]);
    float4 out_hi = float4(results[4], results[5], results[6], results[7]);
    
    *reinterpret_cast<device float4*>(Ax + sample_idx * 8) = out_lo;
    *reinterpret_cast<device float4*>(Ax + sample_idx * 8 + 4) = out_hi;
}

// ----------------------------------------------------------------------------
// Kernel: gemm_rank16_Ax - Optimized for rank=16
// ----------------------------------------------------------------------------

kernel void gemm_rank16_Ax(
    device const float* A          [[buffer(0)]],  // [16 x K]
    device const float* x          [[buffer(1)]],  // [B x S x K]
    device float* Ax               [[buffer(2)]],  // [B x S x 16]
    constant uint& K               [[buffer(3)]],
    constant uint& B               [[buffer(4)]],
    constant uint& S               [[buffer(5)]],
    
    uint3 tid                      [[thread_position_in_grid]],
    uint lid                       [[thread_index_in_threadgroup]],
    uint simd_lane_id              [[thread_index_in_simdgroup]],
    uint simd_group_id             [[simdgroup_index_in_threadgroup]]
) {
    const uint sample_idx = tid.y;
    const uint total_samples = B * S;
    
    if (sample_idx >= total_samples) return;
    
    const uint x_offset = sample_idx * K;
    
    // Use simdgroup_matrix for 8x8 tiles
    simdgroup_matrix<float, 8, 8> acc_lo;
    simdgroup_matrix<float, 8, 8> acc_hi;
    
    // Initialize accumulators
    for (uint i = 0; i < 64; ++i) {
        acc_lo.thread_elements()[i] = 0.0f;
        acc_hi.thread_elements()[i] = 0.0f;
    }
    
    // Tile over K dimension in chunks of 8
    for (uint k_tile = 0; k_tile < K; k_tile += 8) {
        // Load A tiles [0:8, k:k+8] and [8:16, k:k+8]
        simdgroup_matrix<float, 8, 8> A_tile_lo;
        simdgroup_matrix<float, 8, 8> A_tile_hi;
        simdgroup_matrix<float, 8, 8> x_tile;
        
        // Each lane loads one element
        uint a_row = simd_lane_id / 8;
        uint a_col = simd_lane_id % 8;
        uint k_idx = k_tile + a_col;
        
        float a_val_lo = (a_row < 8 && k_idx < K) ? A[a_row * K + k_idx] : 0.0f;
        float a_val_hi = (a_row + 8 < 16 && k_idx < K) ? A[(a_row + 8) * K + k_idx] : 0.0f;
        
        simdgroup_load(A_tile_lo, &a_val_lo, 1, 0);
        simdgroup_load(A_tile_hi, &a_val_hi, 1, 0);
        
        // Load x as column vector (replicated across columns)
        float x_val = (k_idx < K) ? x[x_offset + k_idx] : 0.0f;
        simdgroup_load(x_tile, &x_val, 1, 0);
        
        // Multiply-accumulate
        simdgroup_multiply_accumulate(acc_lo, A_tile_lo, x_tile, acc_lo);
        simdgroup_multiply_accumulate(acc_hi, A_tile_hi, x_tile, acc_hi);
    }
    
    // Extract and store results
    // Each row of the accumulator corresponds to one output element
    float results[16];
    for (uint r = 0; r < 8; ++r) {
        results[r] = simd_sum(acc_lo.thread_elements()[r * 8 + simd_lane_id % 8]);
        results[r + 8] = simd_sum(acc_hi.thread_elements()[r * 8 + simd_lane_id % 8]);
    }
    
    // Coalesced store
    if (simd_lane_id < 16) {
        Ax[sample_idx * 16 + simd_lane_id] = results[simd_lane_id];
    }
}

// ----------------------------------------------------------------------------
// Kernel: gemm_rank32_Ax - Optimized for rank=32
// ----------------------------------------------------------------------------

kernel void gemm_rank32_Ax(
    device const float* A          [[buffer(0)]],  // [32 x K]
    device const float* x          [[buffer(1)]],  // [B x S x K]
    device float* Ax               [[buffer(2)]],  // [B x S x 32]
    constant uint& K               [[buffer(3)]],
    constant uint& B               [[buffer(4)]],
    constant uint& S               [[buffer(5)]],
    
    uint3 tid                      [[thread_position_in_grid]],
    uint lid                       [[thread_index_in_threadgroup]],
    uint simd_lane_id              [[thread_index_in_simdgroup]],
    uint simd_group_id             [[simdgroup_index_in_threadgroup]]
) {
    const uint sample_idx = tid.y;
    const uint total_samples = B * S;
    
    if (sample_idx >= total_samples) return;
    
    const uint x_offset = sample_idx * K;
    
    // Each simdgroup handles 8 output elements (32 threads / 4 simd groups = 8 each)
    const uint r_start = simd_group_id * 8;
    
    threadgroup float tg_x[4096];
    
    // Cooperative load of x
    for (uint k = lid; k < K; k += 256) {
        tg_x[k] = x[x_offset + k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Each thread in simdgroup computes partial sum for one r
    // Then reduce across threads
    float partial_sums[8];
    
    for (uint local_r = 0; local_r < 8; ++local_r) {
        uint r = r_start + local_r;
        if (r >= 32) break;
        
        float acc = 0.0f;
        
        // Each lane handles different K elements
        for (uint k = simd_lane_id; k < K; k += 32) {
            acc += A[r * K + k] * tg_x[k];
        }
        
        // Reduce across lanes
        partial_sums[local_r] = simd_sum(acc);
    }
    
    // Write results (only lane 0 of each simdgroup)
    if (simd_lane_id == 0) {
        for (uint local_r = 0; local_r < 8; ++local_r) {
            uint r = r_start + local_r;
            if (r < 32) {
                Ax[sample_idx * 32 + r] = partial_sums[local_r];
            }
        }
    }
}

// ============================================================================
// TRANSPOSED VARIANTS FOR BACKWARD PASS
// ============================================================================

// Computes Bᵀ @ y where B is [D x R], so Bᵀ is [R x D]
// Used in backward pass for computing Bᵀ @ grad_h

kernel void gemm_lowrank_Bt_y(
    device const float* B_mat      [[buffer(0)]],  // [D x R] (will access as Bᵀ)
    device const float* y          [[buffer(1)]],  // [batch x seq x D]
    device float* out              [[buffer(2)]],  // [batch x seq x R]
    constant uint& D               [[buffer(3)]],  // Input to transpose
    constant uint& R               [[buffer(4)]],  // Output (rank)
    constant uint& batch_size      [[buffer(5)]],
    constant uint& seq_len         [[buffer(6)]],
    
    uint3 tid                      [[thread_position_in_grid]]
) {
    const uint batch_idx = tid.z;
    const uint seq_idx = tid.y;
    const uint r = tid.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || r >= R) return;
    
    const uint y_offset = (batch_idx * seq_len + seq_idx) * D;
    
    // out[r] = sum_d Bᵀ[r,d] * y[d] = sum_d B[d,r] * y[d]
    float acc = 0.0f;
    
    for (uint d = 0; d < D; d += 4) {
        float4 y_vec = safe_load4(y + y_offset, d, D);
        float4 b_vec = float4(
            d < D ? B_mat[d * R + r] : 0.0f,
            d+1 < D ? B_mat[(d+1) * R + r] : 0.0f,
            d+2 < D ? B_mat[(d+2) * R + r] : 0.0f,
            d+3 < D ? B_mat[(d+3) * R + r] : 0.0f
        );
        acc += dot(b_vec, y_vec);
    }
    
    out[(batch_idx * seq_len + seq_idx) * R + r] = acc;
}
