// ============================================================================
// MetalLoRA - Quantized LoRA Kernels (QLoRA)
// ============================================================================
// 
// Implements quantized LoRA for extreme memory efficiency:
//   - 4-bit NormalFloat (NF4) quantization for base weights
//   - 8-bit quantization for LoRA adapters
//   - Double quantization for even smaller memory footprint
//
// Memory savings:
//   - FP16 base: 2 bytes/param
//   - INT8 base: 1 byte/param (50% savings)
//   - INT4 base: 0.5 bytes/param (75% savings)
//
// With QLoRA, a 7B model fits in ~4GB RAM (vs ~14GB FP16)
// ============================================================================

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// ============================================================================
// QUANTIZATION CONSTANTS
// ============================================================================

// NF4 quantization levels (from QLoRA paper)
// These are optimal quantization points for normally distributed weights
constant float NF4_LEVELS[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170635223389f, 0.7229568362236023f, 1.0f
};

// Faster lookup for INT4 (linear quantization)
constant float INT4_SCALE = 1.0f / 7.5f;  // Maps [-7.5, 7.5] to [-1, 1]


// ============================================================================
// DEQUANTIZATION FUNCTIONS
// ============================================================================

// Dequantize INT4 value to float
inline float dequant_int4(uchar packed, bool high_nibble, float scale, float zero_point) {
    int4_t val = high_nibble ? ((packed >> 4) & 0xF) : (packed & 0xF);
    return (float(val) - zero_point) * scale;
}

// Dequantize NF4 value to float
inline float dequant_nf4(uchar packed, bool high_nibble, float absmax) {
    uint idx = high_nibble ? ((packed >> 4) & 0xF) : (packed & 0xF);
    return NF4_LEVELS[idx] * absmax;
}

// Dequantize INT8 value to float
inline float dequant_int8(char val, float scale, float zero_point) {
    return (float(val) - zero_point) * scale;
}


// ============================================================================
// QLORA FORWARD KERNEL (INT4 Base + FP16 LoRA)
// ============================================================================
// Base weights stored as INT4, LoRA adapters as FP16
// This is the "classic" QLoRA configuration
// ============================================================================

kernel void qlora_forward_int4(
    device const uchar* W0_quant    [[buffer(0)]],   // INT4 packed [D, K/2]
    device const float* W0_scales   [[buffer(1)]],   // Per-row scales [D]
    device const float* W0_zeros    [[buffer(2)]],   // Per-row zero points [D]
    device const half* x            [[buffer(3)]],   // Input [B, S, K] FP16
    device const half* A            [[buffer(4)]],   // LoRA A [R, K] FP16
    device const half* B            [[buffer(5)]],   // LoRA B [D, R] FP16
    device half* out                [[buffer(6)]],   // Output [B, S, D] FP16
    constant uint& batch_size       [[buffer(7)]],
    constant uint& seq_len          [[buffer(8)]],
    constant uint& K                [[buffer(9)]],
    constant uint& D                [[buffer(10)]],
    constant uint& R                [[buffer(11)]],
    constant float& alpha           [[buffer(12)]],
    
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint lid                        [[thread_index_in_threadgroup]]
) {
    const float scale = alpha / float(R);
    const uint batch_idx = tgid.z;
    const uint seq_idx = tgid.y;
    const uint d = tgid.x * 256 + lid;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || d >= D) return;
    
    const uint x_offset = (batch_idx * seq_len + seq_idx) * K;
    
    // Get dequantization params for this row
    float w_scale = W0_scales[d];
    float w_zero = W0_zeros[d];
    
    // Compute W0 @ x with on-the-fly dequantization
    float h = 0.0f;
    uint K_packed = K / 2;  // Two INT4 values per byte
    
    for (uint k_packed = 0; k_packed < K_packed; ++k_packed) {
        uchar packed = W0_quant[d * K_packed + k_packed];
        uint k = k_packed * 2;
        
        // Dequantize two weights
        float w0 = dequant_int4(packed, false, w_scale, w_zero);
        float w1 = dequant_int4(packed, true, w_scale, w_zero);
        
        // Multiply with input
        h += w0 * float(x[x_offset + k]);
        if (k + 1 < K) {
            h += w1 * float(x[x_offset + k + 1]);
        }
    }
    
    // LoRA contribution (full precision)
    // First compute Ax
    float Ax_local[64];  // Max rank 64
    for (uint r = 0; r < R; ++r) {
        float acc = 0.0f;
        for (uint k = 0; k < K; ++k) {
            acc += float(A[r * K + k]) * float(x[x_offset + k]);
        }
        Ax_local[r] = acc;
    }
    
    // Then B @ Ax
    float lora = 0.0f;
    for (uint r = 0; r < R; ++r) {
        lora += float(B[d * R + r]) * Ax_local[r];
    }
    
    out[(batch_idx * seq_len + seq_idx) * D + d] = half(h + scale * lora);
}


// ============================================================================
// QLORA FORWARD KERNEL (NF4 Quantization)
// ============================================================================
// Uses NormalFloat4 quantization for better accuracy on normally-distributed
// weights (typical for LLM weights after training)
// ============================================================================

kernel void qlora_forward_nf4(
    device const uchar* W0_quant    [[buffer(0)]],   // NF4 packed [D, K/2]
    device const float* W0_absmax   [[buffer(1)]],   // Per-block absmax [D, K/block]
    device const half* x            [[buffer(2)]],
    device const half* A            [[buffer(3)]],
    device const half* B            [[buffer(4)]],
    device half* out                [[buffer(5)]],
    constant uint& batch_size       [[buffer(6)]],
    constant uint& seq_len          [[buffer(7)]],
    constant uint& K                [[buffer(8)]],
    constant uint& D                [[buffer(9)]],
    constant uint& R                [[buffer(10)]],
    constant float& alpha           [[buffer(11)]],
    constant uint& block_size       [[buffer(12)]],  // Typically 64 or 128
    
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint lid                        [[thread_index_in_threadgroup]]
) {
    const float lora_scale = alpha / float(R);
    const uint batch_idx = tgid.z;
    const uint seq_idx = tgid.y;
    const uint d = tgid.x * 256 + lid;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || d >= D) return;
    
    const uint x_offset = (batch_idx * seq_len + seq_idx) * K;
    const uint blocks_per_row = (K + block_size - 1) / block_size;
    
    // Compute W0 @ x with NF4 dequantization
    float h = 0.0f;
    
    for (uint block = 0; block < blocks_per_row; ++block) {
        float absmax = W0_absmax[d * blocks_per_row + block];
        uint k_start = block * block_size;
        uint k_end = min(k_start + block_size, K);
        
        for (uint k = k_start; k < k_end; k += 2) {
            uint k_packed = k / 2;
            uchar packed = W0_quant[d * (K / 2) + k_packed];
            
            float w0 = dequant_nf4(packed, false, absmax);
            float w1 = (k + 1 < K) ? dequant_nf4(packed, true, absmax) : 0.0f;
            
            h += w0 * float(x[x_offset + k]);
            if (k + 1 < K) {
                h += w1 * float(x[x_offset + k + 1]);
            }
        }
    }
    
    // LoRA (full precision)
    threadgroup float tg_Ax[128];
    
    for (uint r = lid; r < R; r += 256) {
        float acc = 0.0f;
        for (uint k = 0; k < K; ++k) {
            acc += float(A[r * K + k]) * float(x[x_offset + k]);
        }
        tg_Ax[r] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float lora = 0.0f;
    for (uint r = 0; r < R; ++r) {
        lora += float(B[d * R + r]) * tg_Ax[r];
    }
    
    out[(batch_idx * seq_len + seq_idx) * D + d] = half(h + lora_scale * lora);
}


// ============================================================================
// DOUBLE QUANTIZATION KERNEL
// ============================================================================
// From QLoRA paper: quantize the quantization constants too!
// Saves additional ~0.5 bits per parameter
// ============================================================================

kernel void qlora_forward_double_quant(
    device const uchar* W0_quant       [[buffer(0)]],   // NF4 weights [D, K/2]
    device const uchar* scales_quant   [[buffer(1)]],   // Quantized scales
    device const float* scales_scale   [[buffer(2)]],   // Scale of scales
    device const half* x               [[buffer(3)]],
    device const half* A               [[buffer(4)]],
    device const half* B               [[buffer(5)]],
    device half* out                   [[buffer(6)]],
    constant uint& batch_size          [[buffer(7)]],
    constant uint& seq_len             [[buffer(8)]],
    constant uint& K                   [[buffer(9)]],
    constant uint& D                   [[buffer(10)]],
    constant uint& R                   [[buffer(11)]],
    constant float& alpha              [[buffer(12)]],
    constant uint& block_size          [[buffer(13)]],
    
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint lid                           [[thread_index_in_threadgroup]]
) {
    const float lora_scale = alpha / float(R);
    const uint batch_idx = tgid.z;
    const uint seq_idx = tgid.y;
    const uint d = tgid.x * 256 + lid;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || d >= D) return;
    
    const uint x_offset = (batch_idx * seq_len + seq_idx) * K;
    const uint blocks_per_row = (K + block_size - 1) / block_size;
    
    // Dequantize scales for this row
    float row_scale = scales_scale[d / 256];  // Super-block scale
    
    float h = 0.0f;
    
    for (uint block = 0; block < blocks_per_row; ++block) {
        // Double dequant: first dequant the scale, then use it
        uchar scale_quant = scales_quant[d * blocks_per_row + block];
        float absmax = float(scale_quant) * row_scale / 127.0f;
        
        uint k_start = block * block_size;
        uint k_end = min(k_start + block_size, K);
        
        for (uint k = k_start; k < k_end; k += 2) {
            uint k_packed = k / 2;
            uchar packed = W0_quant[d * (K / 2) + k_packed];
            
            float w0 = dequant_nf4(packed, false, absmax);
            float w1 = (k + 1 < K) ? dequant_nf4(packed, true, absmax) : 0.0f;
            
            h += w0 * float(x[x_offset + k]);
            if (k + 1 < K) {
                h += w1 * float(x[x_offset + k + 1]);
            }
        }
    }
    
    // LoRA in FP16
    float lora = 0.0f;
    for (uint r = 0; r < R; ++r) {
        float ax = 0.0f;
        for (uint k = 0; k < K; ++k) {
            ax += float(A[r * K + k]) * float(x[x_offset + k]);
        }
        lora += float(B[d * R + r]) * ax;
    }
    
    out[(batch_idx * seq_len + seq_idx) * D + d] = half(h + lora_scale * lora);
}


// ============================================================================
// INT8 QUANTIZED LORA ADAPTERS
// ============================================================================
// For even smaller adapter files, quantize A and B to INT8
// Particularly useful for serving many adapters simultaneously
// ============================================================================

kernel void qlora_forward_int8_adapters(
    device const half* W0              [[buffer(0)]],
    device const half* x               [[buffer(1)]],
    device const char* A_quant         [[buffer(2)]],   // INT8 A
    device const char* B_quant         [[buffer(3)]],   // INT8 B
    device const float* A_scale        [[buffer(4)]],   // Per-row A scale
    device const float* B_scale        [[buffer(5)]],   // Per-row B scale
    device half* out                   [[buffer(6)]],
    constant uint& batch_size          [[buffer(7)]],
    constant uint& seq_len             [[buffer(8)]],
    constant uint& K                   [[buffer(9)]],
    constant uint& D                   [[buffer(10)]],
    constant uint& R                   [[buffer(11)]],
    constant float& alpha              [[buffer(12)]],
    
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint lid                           [[thread_index_in_threadgroup]]
) {
    const float scale = alpha / float(R);
    const uint batch_idx = tgid.z;
    const uint seq_idx = tgid.y;
    const uint d = tgid.x * 256 + lid;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || d >= D) return;
    
    const uint x_offset = (batch_idx * seq_len + seq_idx) * K;
    
    // W0 @ x (FP16)
    float h = 0.0f;
    for (uint k = 0; k < K; ++k) {
        h += float(W0[d * K + k]) * float(x[x_offset + k]);
    }
    
    // INT8 LoRA with on-the-fly dequantization
    threadgroup float tg_Ax[128];
    
    for (uint r = lid; r < R; r += 256) {
        float a_row_scale = A_scale[r];
        float acc = 0.0f;
        
        for (uint k = 0; k < K; ++k) {
            float a_val = float(A_quant[r * K + k]) * a_row_scale / 127.0f;
            acc += a_val * float(x[x_offset + k]);
        }
        tg_Ax[r] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float lora = 0.0f;
    float b_row_scale = B_scale[d];
    for (uint r = 0; r < R; ++r) {
        float b_val = float(B_quant[d * R + r]) * b_row_scale / 127.0f;
        lora += b_val * tg_Ax[r];
    }
    
    out[(batch_idx * seq_len + seq_idx) * D + d] = half(h + scale * lora);
}
