# MetalLoRA Technical Overview

## Problem

LoRA fine-tuning on Apple Silicon is bottlenecked by:
1. Kernel dispatch overhead (each op = new dispatch)
2. Memory bandwidth (moving data between CPU/GPU)
3. Suboptimal use of hardware matrix units

## Solution

MetalLoRA fuses operations and exploits Apple Silicon architecture:

### 1. Operation Fusion

Standard MLX:
```
Ax = A @ x          # dispatch 1
BAx = B @ Ax        # dispatch 2
W0x = W0 @ x        # dispatch 3
out = W0x + BAx     # dispatch 4
```

MetalLoRA:
```
out = fused_lora(x, W0, A, B)  # single dispatch
```

**Benefit:** 3-4x fewer dispatches, data stays in registers.

### 2. simdgroup_matrix (Hardware 8x8 FMA)

Apple GPUs have dedicated matrix units that compute 8×8 matrix multiplies in a single instruction. We use `simdgroup_matrix<float, 8, 8>` for:
- 16x throughput vs scalar operations
- Direct hardware utilization (same as Apple's MPS)

```metal
simdgroup_matrix<float, 8, 8> acc;
simdgroup_multiply_accumulate(acc, W_tile, x_tile, acc);
```

### 3. Tile Memory Persistence

Apple's TBDR (Tile-Based Deferred Rendering) architecture keeps data in fast on-chip tile memory. We structure kernels to:
- Load data once into threadgroup memory
- Perform all operations (A@x, B@Ax, W0@x) without eviction
- Write final result to device memory

**Benefit:** ~3x bandwidth reduction.

### 4. Mixed Precision (FP16 + FP32)

```metal
device const half* x;        // FP16 storage (2x bandwidth)
float acc = 0.0f;            // FP32 accumulation (precision)
output[i] = half(acc);       // FP16 output
```

**Benefit:** 2x memory bandwidth, 2x ALU throughput, no accuracy loss.

### 5. QLoRA (4-bit Quantization)

Base weights stored as 4-bit NF4 (Normal Float 4):
- 16 optimal quantization levels for Gaussian distributions
- 75% memory reduction (28GB → 7GB for 7B model)
- On-the-fly dequantization in kernel

```metal
constant float NF4_LEVELS[16] = {-1.0, -0.696, ..., 1.0};
float w = NF4_LEVELS[packed & 0xF] * absmax;
```

### 6. Multi-Adapter Batching

Serve N adapters from single base model:
- Pack all adapters into batched tensors `[N, R, K]`
- Each sample in batch uses different adapter
- Single kernel dispatch processes all

**Use case:** Multi-tenant serving (100+ users, 1 GPU).

### 7. Speculative Decoding

Small draft model proposes K tokens, target model verifies in parallel:
- Draft: 5 tokens in 5 forward passes
- Verify: 1 forward pass checks all 5
- Accept: 3-4 tokens on average

**Benefit:** 2-5x faster generation.

## Architecture

```
kernels/
├── lora_kernels.metal   # Core + simdgroup + multi-adapter
├── lora_train.metal     # Fused training kernels
└── lora_quantized.metal # INT4/NF4 inference

python/metal_lora/
├── layers.py            # LoRALinear, LoRAEmbedding
├── ops.py               # Low-level operations
├── trainer.py           # Training infrastructure
├── utils.py             # Save/load/apply
└── optimizations.py     # Compression, pooling, speculative
```

## Performance Targets

| Optimization         | Expected Speedup    |
| -------------------- | ------------------- |
| Fused kernels        | 2-3x                |
| simdgroup_matrix     | 2-4x                |
| FP16                 | 2x                  |
| QLoRA                | 4x memory reduction |
| Speculative decoding | 2-5x generation     |

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Apple Metal Best Practices](https://developer.apple.com/metal/)
- [MLX Framework](https://github.com/ml-explore/mlx)
