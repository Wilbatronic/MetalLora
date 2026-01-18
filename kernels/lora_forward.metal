// ============================================================================
// MetalLoRA - Fused LoRA Forward Pass Kernel
// ============================================================================
// Computes: h = W₀x + (α/r) * BAx in a single optimized kernel
// 
// This kernel fuses what would normally be 4 separate operations:
//   1. Ax = A @ x           (downsample to rank r)
//   2. BAx = B @ Ax         (upsample back to d)
//   3. W0x = W₀ @ x         (base model computation)
//   4. h = W0x + scale*BAx  (combine with LoRA delta)
//
// By fusing these, we:
//   - Eliminate intermediate tensor materialization
//   - Reduce memory bandwidth by ~3x
//   - Enable better instruction-level parallelism
// ============================================================================

#include "lora_common.metal"

// ============================================================================
// ARCHITECTURE DEEP DIVE: FUSED LORA FORWARD
// ============================================================================
//
// Memory Access Pattern Analysis:
//
// Naive approach (4 separate ops):
//   Read x:     B * S * K bytes
//   Read A:     R * K bytes
//   Write Ax:   B * S * R bytes  <- intermediate
//   Read Ax:    B * S * R bytes
//   Read B:     D * R bytes
//   Write BAx:  B * S * D bytes  <- intermediate
//   Read W0:    D * K bytes
//   Read x:     B * S * K bytes  (again!)
//   Write W0x:  B * S * D bytes  <- intermediate
//   Read BAx:   B * S * D bytes
//   Read W0x:   B * S * D bytes
//   Write h:    B * S * D bytes
//
// Fused approach:
//   Read x:     B * S * K bytes  (once, cached in threadgroup)
//   Read A:     R * K bytes
//   Read B:     D * R bytes
//   Read W0:    D * K bytes
//   Write h:    B * S * D bytes
//
// For typical sizes (B=4, S=512, K=D=4096, R=16):
//   Naive:  ~200MB memory traffic
//   Fused:  ~67MB memory traffic
//   Speedup from memory alone: ~3x
//
// ============================================================================

// ----------------------------------------------------------------------------
// Kernel: lora_forward_fused
// ----------------------------------------------------------------------------
// Computes the complete LoRA forward pass for a batch of inputs.
//
// Thread Organization:
//   - Each threadgroup handles one (batch, output_tile) combination
//   - Within threadgroup, threads cooperate on matrix tiles using simdgroups
//   - Output tile size adapts to LoRA rank for optimal occupancy
//
// Memory Layout (all row-major):
//   x:   [batch_size, seq_len, in_features]
//   W0:  [out_features, in_features]
//   A:   [rank, in_features]
//   B:   [out_features, rank]
//   out: [batch_size, seq_len, out_features]
// ----------------------------------------------------------------------------

kernel void lora_forward_fused(
    device const float* x          [[buffer(0)]],  // Input activations
    device const float* W0         [[buffer(1)]],  // Base weight matrix
    device const float* A          [[buffer(2)]],  // LoRA down-projection
    device const float* B          [[buffer(3)]],  // LoRA up-projection
    device float* out              [[buffer(4)]],  // Output
    constant LoRAConfig& config    [[buffer(5)]],  // Configuration
    
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
    const uint K = config.in_features;   // Input dimension
    const uint D = config.out_features;  // Output dimension
    const uint R = config.rank;          // LoRA rank
    const float scale = config.alpha / float(R);  // LoRA scaling factor
    
    // Threadgroup is responsible for one (batch, seq_pos, out_tile)
    const uint batch_idx = tgid.z;
    const uint seq_idx = tgid.y;
    const uint out_tile_idx = tgid.x;
    
    // Early exit for out-of-bounds threadgroups
    if (batch_idx >= batch_size || seq_idx >= seq_len) {
        return;
    }
    
    // Calculate output range for this threadgroup
    const uint TILE_D = 64;  // Output features per threadgroup
    const uint out_start = out_tile_idx * TILE_D;
    const uint out_end = min(out_start + TILE_D, D);
    
    if (out_start >= D) return;
    
    // ========================================================================
    // THREADGROUP MEMORY ALLOCATION
    // ========================================================================
    // We cache:
    //   1. Input x for this (batch, seq_pos): K floats
    //   2. Intermediate Ax result: R floats
    //   3. Tile of A for cooperative loading: R * TILE_K floats
    //   4. Tile of B for cooperative loading: TILE_D * R floats
    // ========================================================================
    
    threadgroup float tg_x[4096];      // Cached input (max 4096 features)
    threadgroup float tg_Ax[128];      // Intermediate Ax (max rank 128)
    threadgroup float tg_A[128 * 64];  // A tile cache
    threadgroup float tg_B[64 * 128];  // B tile cache
    
    // ========================================================================
    // PHASE 1: COOPERATIVE INPUT LOADING
    // ========================================================================
    // All threads in threadgroup cooperate to load input vector x
    // This amortizes memory latency across many threads
    // ========================================================================
    
    const uint x_offset = (batch_idx * seq_len + seq_idx) * K;
    
    // Vectorized cooperative load (float4 per thread)
    const uint threads_per_tg = 256;  // Assumed threadgroup size
    const uint loads_per_thread = (K + threads_per_tg * 4 - 1) / (threads_per_tg * 4);
    
    for (uint i = 0; i < loads_per_thread; ++i) {
        uint load_idx = (lid * loads_per_thread + i) * 4;
        if (load_idx < K) {
            float4 vals = safe_load4(x + x_offset, load_idx, K);
            if (load_idx < K) tg_x[load_idx] = vals.x;
            if (load_idx + 1 < K) tg_x[load_idx + 1] = vals.y;
            if (load_idx + 2 < K) tg_x[load_idx + 2] = vals.z;
            if (load_idx + 3 < K) tg_x[load_idx + 3] = vals.w;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // ========================================================================
    // PHASE 2: COMPUTE Ax (RANK REDUCTION)
    // ========================================================================
    // Each thread computes a subset of the R-dimensional Ax vector
    // For small R (typical 4-64), this fits entirely in threadgroup memory
    // ========================================================================
    
    // Threads cooperate to compute Ax[r] = sum_k A[r,k] * x[k]
    const uint ranks_per_thread = (R + threads_per_tg - 1) / threads_per_tg;
    
    for (uint i = 0; i < ranks_per_thread; ++i) {
        uint r = lid * ranks_per_thread + i;
        if (r < R) {
            float acc = 0.0f;
            
            // Vectorized dot product with A[r, :]
            const uint A_row_offset = r * K;
            
            for (uint k = 0; k < K; k += 4) {
                float4 a_vals = safe_load4(A + A_row_offset, k, K);
                float4 x_vals = float4(
                    tg_x[k],
                    k + 1 < K ? tg_x[k + 1] : 0.0f,
                    k + 2 < K ? tg_x[k + 2] : 0.0f,
                    k + 3 < K ? tg_x[k + 3] : 0.0f
                );
                acc += dot(a_vals, x_vals);
            }
            
            tg_Ax[r] = acc;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // ========================================================================
    // PHASE 3: COMPUTE W₀x + scale * BAx (FUSED OUTPUT)
    // ========================================================================
    // Each thread computes output elements within the output tile
    // The key insight: we compute BAx on-the-fly rather than materializing it
    // ========================================================================
    
    const uint outs_per_thread = (TILE_D + threads_per_tg - 1) / threads_per_tg;
    
    for (uint i = 0; i < outs_per_thread; ++i) {
        uint local_d = lid * outs_per_thread + i;
        uint global_d = out_start + local_d;
        
        if (global_d < D) {
            // Accumulator for final output
            float h = 0.0f;
            
            // -----------------------------------------------------------------
            // Compute W₀x contribution
            // -----------------------------------------------------------------
            const uint W0_row_offset = global_d * K;
            
            for (uint k = 0; k < K; k += 4) {
                float4 w_vals = safe_load4(W0 + W0_row_offset, k, K);
                float4 x_vals = float4(
                    tg_x[k],
                    k + 1 < K ? tg_x[k + 1] : 0.0f,
                    k + 2 < K ? tg_x[k + 2] : 0.0f,
                    k + 3 < K ? tg_x[k + 3] : 0.0f
                );
                h += dot(w_vals, x_vals);
            }
            
            // -----------------------------------------------------------------
            // Compute BAx contribution (using cached Ax)
            // -----------------------------------------------------------------
            // BAx[d] = sum_r B[d,r] * Ax[r]
            const uint B_row_offset = global_d * R;
            float lora_contrib = 0.0f;
            
            for (uint r = 0; r < R; r += 4) {
                float4 b_vals = safe_load4(B + B_row_offset, r, R);
                float4 ax_vals = float4(
                    tg_Ax[r],
                    r + 1 < R ? tg_Ax[r + 1] : 0.0f,
                    r + 2 < R ? tg_Ax[r + 2] : 0.0f,
                    r + 3 < R ? tg_Ax[r + 3] : 0.0f
                );
                lora_contrib += dot(b_vals, ax_vals);
            }
            
            // Combine base output with scaled LoRA contribution
            h += scale * lora_contrib;
            
            // Write final output
            const uint out_idx = (batch_idx * seq_len + seq_idx) * D + global_d;
            out[out_idx] = h;
        }
    }
}

// ============================================================================
// VARIANT: FUSED FORWARD WITH SIMDGROUP MATRIX OPS
// ============================================================================
// Uses Metal's simdgroup_matrix for even higher throughput on larger tiles
// Best for configurations where D and K are multiples of 8
// ============================================================================

kernel void lora_forward_simd(
    device const float* x          [[buffer(0)]],
    device const float* W0         [[buffer(1)]],
    device const float* A          [[buffer(2)]],
    device const float* B          [[buffer(3)]],
    device float* out              [[buffer(4)]],
    constant LoRAConfig& config    [[buffer(5)]],
    
    uint3 tid                      [[thread_position_in_grid]],
    uint3 tgid                     [[threadgroup_position_in_grid]],
    uint lid                       [[thread_index_in_threadgroup]],
    uint simd_lane_id              [[thread_index_in_simdgroup]],
    uint simd_group_id             [[simdgroup_index_in_threadgroup]]
) {
    const uint batch_size = config.batch_size;
    const uint seq_len = config.seq_len;
    const uint K = config.in_features;
    const uint D = config.out_features;
    const uint R = config.rank;
    const float scale = config.alpha / float(R);
    
    const uint batch_idx = tgid.z;
    const uint seq_idx = tgid.y;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    // SIMD group handles 8x8 output tile
    const uint SIMD_TILE = 8;
    const uint simd_out_start = simd_group_id * SIMD_TILE;
    
    if (simd_out_start >= D) return;
    
    // Threadgroup memory for shared data
    threadgroup float tg_x[4096];
    threadgroup float tg_Ax[128];
    
    // Phase 1: Load input (cooperative across all SIMDs)
    const uint x_offset = (batch_idx * seq_len + seq_idx) * K;
    for (uint k = lid; k < K; k += 256) {
        tg_x[k] = x[x_offset + k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Phase 2: Compute Ax using first SIMD group
    if (simd_group_id == 0) {
        for (uint r = simd_lane_id; r < R; r += 32) {
            float acc = 0.0f;
            for (uint k = 0; k < K; ++k) {
                acc += A[r * K + k] * tg_x[k];
            }
            tg_Ax[r] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Phase 3: Compute output using simdgroup_matrix
    simdgroup_matrix<float, 8, 8> acc_mat;
    simdgroup_matrix<float, 8, 8> W0_tile;
    simdgroup_matrix<float, 8, 8> x_tile;
    
    // Initialize accumulator to zero
    for (uint i = 0; i < 8; ++i) {
        for (uint j = 0; j < 8; ++j) {
            acc_mat.thread_elements()[i * 8 + j] = 0.0f;
        }
    }
    
    // Accumulate W0 @ x using tiled GEMM
    for (uint k_tile = 0; k_tile < K; k_tile += 8) {
        // Load W0 tile [simd_out_start:simd_out_start+8, k_tile:k_tile+8]
        uint w0_row = simd_out_start + (simd_lane_id / 8);
        uint w0_col = k_tile + (simd_lane_id % 8);
        float w0_val = (w0_row < D && w0_col < K) ? W0[w0_row * K + w0_col] : 0.0f;
        simdgroup_load(W0_tile, &w0_val, 1, 0);
        
        // Load x tile [k_tile:k_tile+8] as column vector replicated
        float x_val = (k_tile + simd_lane_id % 8 < K) ? tg_x[k_tile + simd_lane_id % 8] : 0.0f;
        simdgroup_load(x_tile, &x_val, 1, 0);
        
        // Multiply-accumulate
        simdgroup_multiply_accumulate(acc_mat, W0_tile, x_tile, acc_mat);
    }
    
    // Add LoRA contribution (B @ Ax)
    for (uint r = 0; r < R; ++r) {
        uint b_row = simd_out_start + (simd_lane_id / 8);
        if (b_row < D) {
            float b_val = B[b_row * R + r];
            float ax_val = tg_Ax[r];
            acc_mat.thread_elements()[simd_lane_id] += scale * b_val * ax_val;
        }
    }
    
    // Store results
    uint out_row = simd_out_start + (simd_lane_id / 8);
    if (out_row < D) {
        uint out_idx = (batch_idx * seq_len + seq_idx) * D + out_row;
        // Reduce across simd lanes for this output row
        float result = simd_sum(acc_mat.thread_elements()[simd_lane_id]);
        if (simd_lane_id % 8 == 0) {
            out[out_idx] = result;
        }
    }
}

// ============================================================================
// VARIANT: FORWARD WITH DROPOUT SUPPORT
// ============================================================================
// Applies dropout to the LoRA contribution (common training technique)
// Uses Philox RNG for reproducible dropout masks
// ============================================================================

// Simple Philox-like RNG for dropout
inline float random_uniform(uint seed, uint idx) {
    uint x = seed ^ idx;
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return float(x) / float(0xFFFFFFFF);
}

kernel void lora_forward_dropout(
    device const float* x          [[buffer(0)]],
    device const float* W0         [[buffer(1)]],
    device const float* A          [[buffer(2)]],
    device const float* B          [[buffer(3)]],
    device float* out              [[buffer(4)]],
    constant LoRAConfig& config    [[buffer(5)]],
    
    uint3 tid                      [[thread_position_in_grid]],
    uint3 tgid                     [[threadgroup_position_in_grid]],
    uint lid                       [[thread_index_in_threadgroup]]
) {
    // Similar to lora_forward_fused but with dropout on LoRA path
    const uint batch_size = config.batch_size;
    const uint seq_len = config.seq_len;
    const uint K = config.in_features;
    const uint D = config.out_features;
    const uint R = config.rank;
    const float scale = config.alpha / float(R);
    const float dropout_prob = config.dropout_prob;
    const float dropout_scale = 1.0f / (1.0f - dropout_prob);
    
    const uint batch_idx = tgid.z;
    const uint seq_idx = tgid.y;
    const uint d = tid.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || d >= D) return;
    
    // Input offset
    const uint x_offset = (batch_idx * seq_len + seq_idx) * K;
    
    // Compute W₀x
    float h = 0.0f;
    for (uint k = 0; k < K; ++k) {
        h += W0[d * K + k] * x[x_offset + k];
    }
    
    // Compute Ax, then BAx with dropout
    float lora_contrib = 0.0f;
    for (uint r = 0; r < R; ++r) {
        // Check dropout for this rank dimension
        uint dropout_idx = batch_idx * seq_len * R + seq_idx * R + r;
        float rand_val = random_uniform(config.seed, dropout_idx);
        
        if (rand_val >= dropout_prob) {
            // Compute Ax[r]
            float ax_r = 0.0f;
            for (uint k = 0; k < K; ++k) {
                ax_r += A[r * K + k] * x[x_offset + k];
            }
            
            // Accumulate B[d,r] * Ax[r] with dropout scaling
            lora_contrib += B[d * R + r] * ax_r * dropout_scale;
        }
    }
    
    h += scale * lora_contrib;
    
    // Write output
    const uint out_idx = (batch_idx * seq_len + seq_idx) * D + d;
    out[out_idx] = h;
}
