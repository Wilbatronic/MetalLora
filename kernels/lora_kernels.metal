// MetalLoRA - Production Kernels
// Forward, backward, and utility operations for LoRA inference and training.

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// Configuration structures

struct LoRAConfig {
    uint batch_size;
    uint seq_len;
    uint in_features;
    uint out_features;
    uint rank;
    float alpha;
    float dropout_prob;
    uint seed;
};

struct GradConfig {
    uint batch_size;
    uint seq_len;
    uint in_features;
    uint out_features;
    uint rank;
    float alpha;
    float grad_clip;
};

// Forward pass (FP32)
// h = W0 @ x + (alpha/rank) * B @ A @ x

kernel void lora_forward(
    device const float* x            [[buffer(0)]],
    device const float* W0           [[buffer(1)]],
    device const float* A            [[buffer(2)]],
    device const float* B            [[buffer(3)]],
    device float* output             [[buffer(4)]],
    constant LoRAConfig& cfg         [[buffer(5)]],
    uint3 tgid                       [[threadgroup_position_in_grid]],
    uint lid                         [[thread_index_in_threadgroup]]
) {
    const uint batch_idx = tgid.z;
    const uint seq_idx = tgid.y;
    const uint d = tgid.x * 256 + lid;
    
    if (batch_idx >= cfg.batch_size || seq_idx >= cfg.seq_len || d >= cfg.out_features) return;
    
    const float scale = cfg.alpha / float(cfg.rank);
    const uint x_offset = (batch_idx * cfg.seq_len + seq_idx) * cfg.in_features;
    const uint out_offset = (batch_idx * cfg.seq_len + seq_idx) * cfg.out_features;
    
    threadgroup float tg_x[4096];
    threadgroup float tg_Ax[128];
    
    for (uint k = lid; k < cfg.in_features; k += 256) {
        tg_x[k] = x[x_offset + k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint r = lid; r < cfg.rank; r += 256) {
        float acc = 0.0f;
        for (uint k = 0; k < cfg.in_features; ++k) {
            acc += A[r * cfg.in_features + k] * tg_x[k];
        }
        tg_Ax[r] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float h = 0.0f;
    for (uint k = 0; k < cfg.in_features; k += 4) {
        float4 w = float4(W0[d * cfg.in_features + k], W0[d * cfg.in_features + k + 1],
                          W0[d * cfg.in_features + k + 2], W0[d * cfg.in_features + k + 3]);
        float4 xv = float4(tg_x[k], tg_x[k+1], tg_x[k+2], tg_x[k+3]);
        h += dot(w, xv);
    }
    
    float lora = 0.0f;
    for (uint r = 0; r < cfg.rank; ++r) {
        lora += B[d * cfg.rank + r] * tg_Ax[r];
    }
    
    output[out_offset + d] = h + scale * lora;
}

// Forward pass (FP16)
// Uses FP16 storage with FP32 accumulation.

kernel void lora_forward_fp16(
    device const half* x             [[buffer(0)]],
    device const half* W0            [[buffer(1)]],
    device const half* A             [[buffer(2)]],
    device const half* B             [[buffer(3)]],
    device half* output              [[buffer(4)]],
    constant LoRAConfig& cfg         [[buffer(5)]],
    uint3 tgid                       [[threadgroup_position_in_grid]],
    uint lid                         [[thread_index_in_threadgroup]]
) {
    const uint batch_idx = tgid.z;
    const uint seq_idx = tgid.y;
    const uint d = tgid.x * 256 + lid;
    
    if (batch_idx >= cfg.batch_size || seq_idx >= cfg.seq_len || d >= cfg.out_features) return;
    
    const float scale = cfg.alpha / float(cfg.rank);
    const uint x_offset = (batch_idx * cfg.seq_len + seq_idx) * cfg.in_features;
    const uint out_offset = (batch_idx * cfg.seq_len + seq_idx) * cfg.out_features;
    
    threadgroup half tg_x[4096];
    threadgroup float tg_Ax[128];
    
    for (uint k = lid; k < cfg.in_features; k += 256) {
        tg_x[k] = x[x_offset + k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint r = lid; r < cfg.rank; r += 256) {
        float acc = 0.0f;
        for (uint k = 0; k < cfg.in_features; k += 4) {
            half4 a_vec = half4(A[r * cfg.in_features + k], A[r * cfg.in_features + k + 1],
                                A[r * cfg.in_features + k + 2], A[r * cfg.in_features + k + 3]);
            half4 x_vec = half4(tg_x[k], tg_x[k+1], tg_x[k+2], tg_x[k+3]);
            acc += float(dot(a_vec, x_vec));
        }
        tg_Ax[r] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float h = 0.0f;
    for (uint k = 0; k < cfg.in_features; k += 4) {
        half4 w = half4(W0[d * cfg.in_features + k], W0[d * cfg.in_features + k + 1],
                        W0[d * cfg.in_features + k + 2], W0[d * cfg.in_features + k + 3]);
        half4 xv = half4(tg_x[k], tg_x[k+1], tg_x[k+2], tg_x[k+3]);
        h += float(dot(w, xv));
    }
    
    float lora = 0.0f;
    for (uint r = 0; r < cfg.rank; ++r) {
        lora += float(B[d * cfg.rank + r]) * tg_Ax[r];
    }
    
    output[out_offset + d] = half(h + scale * lora);
}

// Backward pass
// Computes grad_A and grad_B with atomic accumulation.

kernel void lora_backward(
    device const float* grad_output  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device const float* A            [[buffer(2)]],
    device const float* B            [[buffer(3)]],
    device atomic_float* grad_A      [[buffer(4)]],
    device atomic_float* grad_B      [[buffer(5)]],
    constant GradConfig& cfg         [[buffer(6)]],
    uint3 tgid                       [[threadgroup_position_in_grid]],
    uint lid                         [[thread_index_in_threadgroup]]
) {
    const uint batch_idx = tgid.z;
    const uint seq_idx = tgid.y;
    
    if (batch_idx >= cfg.batch_size || seq_idx >= cfg.seq_len) return;
    
    const float scale = cfg.alpha / float(cfg.rank);
    const uint x_offset = (batch_idx * cfg.seq_len + seq_idx) * cfg.in_features;
    const uint grad_offset = (batch_idx * cfg.seq_len + seq_idx) * cfg.out_features;
    
    threadgroup float tg_x[4096];
    threadgroup float tg_grad[4096];
    threadgroup float tg_Ax[128];
    threadgroup float tg_Bt_grad[128];
    
    for (uint k = lid; k < cfg.in_features; k += 256) {
        tg_x[k] = x[x_offset + k];
    }
    for (uint d = lid; d < cfg.out_features; d += 256) {
        tg_grad[d] = grad_output[grad_offset + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint r = lid; r < cfg.rank; r += 256) {
        float acc = 0.0f;
        for (uint k = 0; k < cfg.in_features; ++k) {
            acc += A[r * cfg.in_features + k] * tg_x[k];
        }
        tg_Ax[r] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint r = lid; r < cfg.rank; r += 256) {
        float acc = 0.0f;
        for (uint d = 0; d < cfg.out_features; ++d) {
            acc += B[d * cfg.rank + r] * tg_grad[d];
        }
        tg_Bt_grad[r] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint dr = lid; dr < cfg.out_features * cfg.rank; dr += 256) {
        uint d = dr / cfg.rank;
        uint r = dr % cfg.rank;
        if (d < cfg.out_features && r < cfg.rank) {
            float grad_val = clamp(scale * tg_grad[d] * tg_Ax[r], -cfg.grad_clip, cfg.grad_clip);
            atomic_fetch_add_explicit(&grad_B[d * cfg.rank + r], grad_val, memory_order_relaxed);
        }
    }
    
    for (uint rk = lid; rk < cfg.rank * cfg.in_features; rk += 256) {
        uint r = rk / cfg.in_features;
        uint k = rk % cfg.in_features;
        if (r < cfg.rank && k < cfg.in_features) {
            float grad_val = clamp(scale * tg_Bt_grad[r] * tg_x[k], -cfg.grad_clip, cfg.grad_clip);
            atomic_fetch_add_explicit(&grad_A[r * cfg.in_features + k], grad_val, memory_order_relaxed);
        }
    }
}

// Weight merge
// W_merged = W0 + (alpha/rank) * B @ A

kernel void lora_merge_weights(
    device const float* W0           [[buffer(0)]],
    device const float* A            [[buffer(1)]],
    device const float* B            [[buffer(2)]],
    device float* W_merged           [[buffer(3)]],
    constant uint& D                 [[buffer(4)]],
    constant uint& K                 [[buffer(5)]],
    constant uint& R                 [[buffer(6)]],
    constant float& alpha            [[buffer(7)]],
    uint2 tid                        [[thread_position_in_grid]]
) {
    const uint d = tid.y;
    const uint k = tid.x;
    
    if (d >= D || k >= K) return;
    
    float scale = alpha / float(R);
    float ba = 0.0f;
    for (uint r = 0; r < R; ++r) {
        ba += B[d * R + r] * A[r * K + k];
    }
    
    W_merged[d * K + k] = W0[d * K + k] + scale * ba;
}
