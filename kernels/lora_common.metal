// ============================================================================
// MetalLoRA - Common Utilities and Constants
// ============================================================================
// Shared definitions for all LoRA kernels
// ============================================================================

#ifndef LORA_COMMON_METAL
#define LORA_COMMON_METAL

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <metal_simdgroup>

using namespace metal;

// ============================================================================
// HARDWARE ARCHITECTURE NOTES
// ============================================================================
//
// Apple Silicon GPU Architecture:
// - M1: 128 EU, 8 GPU cores, 2.6 TFLOPS FP32
// - M2: 160 EU, 10 GPU cores, 3.6 TFLOPS FP32  
// - M3: 192 EU, 10-40 GPU cores, up to 14.2 TFLOPS FP32
// - M4: ~38 GPU cores, significant Neural Engine improvements
//
// Memory Hierarchy:
// - Registers: Fastest, per-thread
// - Threadgroup Memory: ~32KB per threadgroup, shared across threads
// - Device Memory: Unified with CPU, high bandwidth (~400GB/s on M3 Max)
//
// SIMD Groups:
// - 32 threads per simdgroup on Apple GPUs
// - simdgroup_matrix: 8x8 matrix tiles for efficient GEMM
// - Can perform 8x8x8 matrix multiply-accumulate per instruction
//
// ============================================================================

// Tile sizes optimized for different LoRA ranks
// Lower ranks use smaller tiles to maximize occupancy
// Higher ranks use larger tiles for better SIMD utilization
constant constexpr int TILE_SIZE_RANK_4  = 8;
constant constexpr int TILE_SIZE_RANK_8  = 8;
constant constexpr int TILE_SIZE_RANK_16 = 16;
constant constexpr int TILE_SIZE_RANK_32 = 16;
constant constexpr int TILE_SIZE_RANK_64 = 32;

// Maximum supported LoRA rank
constant constexpr int MAX_LORA_RANK = 128;

// Threadgroup memory size for intermediate storage (in floats)
constant constexpr int TG_MEM_SIZE = 8192;

// Epsilon for numerical stability
constant constexpr float EPSILON = 1e-8f;

// ============================================================================
// NUMERIC STABILITY UTILITIES
// ============================================================================

// Kahan summation for reduced floating-point error accumulation
// Critical for gradient accumulation where small errors compound
struct KahanAccumulator {
    float sum;
    float compensation;
    
    KahanAccumulator() : sum(0.0f), compensation(0.0f) {}
    
    void add(float value) {
        float y = value - compensation;
        float t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    
    float get() const { return sum; }
};

// Safe division with epsilon guard
inline float safe_div(float a, float b) {
    return a / (b + EPSILON);
}

// Fused multiply-add with saturation for gradient clipping
inline float fma_saturate(float a, float b, float c, float clip_val) {
    float result = fma(a, b, c);
    return clamp(result, -clip_val, clip_val);
}

// ============================================================================
// MEMORY ACCESS UTILITIES
// ============================================================================

// Coalesced float4 load with bounds checking
// Returns zero for out-of-bounds access (safe for edge tiles)
inline float4 safe_load4(
    device const float* ptr,
    uint idx,
    uint max_idx
) {
    if (idx + 3 < max_idx) {
        return *reinterpret_cast<device const float4*>(ptr + idx);
    }
    // Partial load for boundary
    float4 result = float4(0.0f);
    for (uint i = 0; i < 4 && idx + i < max_idx; ++i) {
        result[i] = ptr[idx + i];
    }
    return result;
}

// Coalesced float4 store with bounds checking
inline void safe_store4(
    device float* ptr,
    uint idx,
    uint max_idx,
    float4 value
) {
    if (idx + 3 < max_idx) {
        *reinterpret_cast<device float4*>(ptr + idx) = value;
    } else {
        // Partial store for boundary
        for (uint i = 0; i < 4 && idx + i < max_idx; ++i) {
            ptr[idx + i] = value[i];
        }
    }
}

// ============================================================================
// SIMDGROUP MATRIX UTILITIES
// ============================================================================

// Load 8x8 tile from device memory into simdgroup matrix
// Handles boundary conditions by zero-padding
template<typename T>
inline void simdgroup_load_safe(
    simdgroup_matrix<T, 8, 8>& mat,
    device const T* src,
    uint ld,           // leading dimension (stride)
    uint row,          // starting row
    uint col,          // starting column
    uint max_rows,     // bounds for row
    uint max_cols,     // bounds for column
    uint lane_id
) {
    // Each thread in simdgroup loads specific elements
    uint lane_row = lane_id / 8;
    uint lane_col = lane_id % 8;
    
    uint global_row = row + lane_row;
    uint global_col = col + lane_col;
    
    T value = T(0);
    if (global_row < max_rows && global_col < max_cols) {
        value = src[global_row * ld + global_col];
    }
    
    simdgroup_load(mat, &value, 1, 0);
}

// ============================================================================
// INDEX CALCULATIONS
// ============================================================================

// Convert 2D indices to 1D for row-major layout
inline uint idx2d(uint row, uint col, uint num_cols) {
    return row * num_cols + col;
}

// Convert 2D indices to 1D for column-major layout
inline uint idx2d_col(uint row, uint col, uint num_rows) {
    return col * num_rows + row;
}

// Calculate threadgroup-local 2D position
inline uint2 get_tg_pos_2d(uint lid, uint tg_width) {
    return uint2(lid % tg_width, lid / tg_width);
}

// ============================================================================
// BOUNDS CHECKING MACROS
// ============================================================================

#define BOUNDS_CHECK_RETURN(idx, max_val) \
    if ((idx) >= (max_val)) return;

#define BOUNDS_CHECK_CONTINUE(idx, max_val) \
    if ((idx) >= (max_val)) continue;

// ============================================================================
// CONFIGURATION STRUCTURES
// ============================================================================

// LoRA layer configuration passed to kernels
struct LoRAConfig {
    uint batch_size;      // Batch dimension
    uint seq_len;         // Sequence length (tokens)
    uint in_features;     // Input dimension (k)
    uint out_features;    // Output dimension (d)
    uint rank;            // LoRA rank (r)
    float alpha;          // Scaling factor
    float dropout_prob;   // Dropout probability (0 = disabled)
    uint seed;            // Random seed for dropout
};

// Gradient configuration for backward pass
struct GradConfig {
    float learning_rate;  // For fused optimizer step
    float clip_value;     // Gradient clipping threshold
    float weight_decay;   // L2 regularization
    bool use_adam;        // Use Adam moments or just SGD
};

#endif // LORA_COMMON_METAL
