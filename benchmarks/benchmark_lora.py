"""
MetalLoRA - Benchmark Suite

Compares MetalLoRA performance against baseline MLX LoRA implementation.
Measures forward/backward pass latency and memory usage across various
configurations.
"""

import mlx.core as mx
import mlx.nn as nn
import time
import argparse
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import List, Dict, Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from metal_lora import LoRALinear
from metal_lora.ops import lora_forward, lora_backward_efficient


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    batch_size: int
    seq_len: int
    in_features: int
    out_features: int
    rank: int
    alpha: float = 16.0
    warmup_iters: int = 10
    bench_iters: int = 100


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    config: BenchmarkConfig
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    memory_mb: float
    throughput_samples_per_sec: float


# ============================================================================
# BASELINE IMPLEMENTATION (Standard MLX)
# ============================================================================

class BaselineLoRALinear(nn.Module):
    """Standard LoRA implementation using vanilla MLX ops."""
    
    def __init__(self, in_features, out_features, rank, alpha=16.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        self.W0 = mx.random.normal((out_features, in_features)) * 0.02
        self.A = mx.random.normal((rank, in_features)) * 0.02
        self.B = mx.zeros((out_features, rank))
    
    def __call__(self, x):
        # Standard separate operations
        W0x = mx.matmul(x, self.W0.T)
        Ax = mx.matmul(x, self.A.T)
        BAx = mx.matmul(Ax, self.B.T)
        scale = self.alpha / self.rank
        return W0x + scale * BAx


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def benchmark_forward(layer: nn.Module, x: mx.array, config: BenchmarkConfig) -> float:
    """Benchmark forward pass latency."""
    # Warmup
    for _ in range(config.warmup_iters):
        out = layer(x)
        mx.eval(out)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(config.bench_iters):
        out = layer(x)
        mx.eval(out)
    end = time.perf_counter()
    
    return (end - start) / config.bench_iters * 1000  # ms


def benchmark_backward(
    x: mx.array,
    W0: mx.array,
    A: mx.array,
    B: mx.array,
    config: BenchmarkConfig,
    use_optimized: bool = True,
) -> float:
    """Benchmark backward pass latency."""
    # Create grad_output
    grad_h = mx.random.normal((config.batch_size, config.seq_len, config.out_features))
    
    # Warmup
    for _ in range(config.warmup_iters):
        if use_optimized:
            grad_A, grad_B = lora_backward_efficient(grad_h, x, A, B, config.alpha)
        else:
            # Baseline: separate operations
            scale = config.alpha / config.rank
            Ax = mx.matmul(x, A.T)
            Bt_grad = mx.matmul(grad_h, B)
            grad_B = scale * mx.sum(mx.matmul(grad_h.transpose(0, 2, 1), Ax), axis=0)
            grad_A = scale * mx.sum(mx.matmul(Bt_grad.transpose(0, 2, 1), x), axis=0)
        mx.eval(grad_A, grad_B)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(config.bench_iters):
        if use_optimized:
            grad_A, grad_B = lora_backward_efficient(grad_h, x, A, B, config.alpha)
        else:
            scale = config.alpha / config.rank
            Ax = mx.matmul(x, A.T)
            Bt_grad = mx.matmul(grad_h, B)
            grad_B = scale * mx.sum(mx.matmul(grad_h.transpose(0, 2, 1), Ax), axis=0)
            grad_A = scale * mx.sum(mx.matmul(Bt_grad.transpose(0, 2, 1), x), axis=0)
        mx.eval(grad_A, grad_B)
    end = time.perf_counter()
    
    return (end - start) / config.bench_iters * 1000  # ms


def run_benchmark(config: BenchmarkConfig, compare: bool = True) -> Dict[str, BenchmarkResult]:
    """Run benchmark for both MetalLoRA and baseline."""
    mx.random.seed(42)
    
    # Create input
    x = mx.random.normal((config.batch_size, config.seq_len, config.in_features))
    mx.eval(x)
    
    results = {}
    
    # MetalLoRA
    metal_layer = LoRALinear(
        config.in_features,
        config.out_features,
        rank=config.rank,
        alpha=config.alpha,
    )
    metal_layer.W0 = mx.random.normal((config.out_features, config.in_features)) * 0.02
    mx.eval(metal_layer.W0, metal_layer.A, metal_layer.B)
    
    fwd_time = benchmark_forward(metal_layer, x, config)
    bwd_time = benchmark_backward(x, metal_layer.W0, metal_layer.A, metal_layer.B, config, use_optimized=True)
    
    total_samples = config.batch_size * config.seq_len * config.bench_iters
    throughput = total_samples / ((fwd_time + bwd_time) * config.bench_iters / 1000)
    
    results["metal_lora"] = BenchmarkResult(
        config=config,
        forward_time_ms=fwd_time,
        backward_time_ms=bwd_time,
        total_time_ms=fwd_time + bwd_time,
        memory_mb=0,  # TODO: implement memory tracking
        throughput_samples_per_sec=throughput,
    )
    
    if compare:
        # Baseline
        baseline_layer = BaselineLoRALinear(
            config.in_features,
            config.out_features,
            rank=config.rank,
            alpha=config.alpha,
        )
        mx.eval(baseline_layer.W0, baseline_layer.A, baseline_layer.B)
        
        fwd_time = benchmark_forward(baseline_layer, x, config)
        bwd_time = benchmark_backward(x, baseline_layer.W0, baseline_layer.A, baseline_layer.B, config, use_optimized=False)
        
        throughput = total_samples / ((fwd_time + bwd_time) * config.bench_iters / 1000)
        
        results["baseline"] = BenchmarkResult(
            config=config,
            forward_time_ms=fwd_time,
            backward_time_ms=bwd_time,
            total_time_ms=fwd_time + bwd_time,
            memory_mb=0,
            throughput_samples_per_sec=throughput,
        )
    
    return results


def print_results(results: Dict[str, BenchmarkResult]):
    """Print benchmark results in a formatted table."""
    config = results["metal_lora"].config
    
    print("\n" + "=" * 80)
    print(f"Configuration: batch={config.batch_size}, seq={config.seq_len}, "
          f"dim={config.in_features}, rank={config.rank}")
    print("=" * 80)
    print(f"{'Implementation':<20} {'Forward (ms)':<15} {'Backward (ms)':<15} "
          f"{'Total (ms)':<15} {'Speedup':<10}")
    print("-" * 80)
    
    baseline_total = results.get("baseline", results["metal_lora"]).total_time_ms
    
    for name, result in results.items():
        speedup = baseline_total / result.total_time_ms if result.total_time_ms > 0 else 0
        speedup_str = f"{speedup:.2f}x" if name != "baseline" else "1.00x"
        
        print(f"{name:<20} {result.forward_time_ms:<15.3f} {result.backward_time_ms:<15.3f} "
              f"{result.total_time_ms:<15.3f} {speedup_str:<10}")
    
    print("=" * 80)


def run_benchmark_suite():
    """Run full benchmark suite with various configurations."""
    configs = [
        # Small model (testing)
        BenchmarkConfig(batch_size=1, seq_len=128, in_features=768, out_features=768, rank=8),
        BenchmarkConfig(batch_size=1, seq_len=128, in_features=768, out_features=768, rank=16),
        
        # Medium model (7B scale projections)
        BenchmarkConfig(batch_size=4, seq_len=512, in_features=4096, out_features=4096, rank=8),
        BenchmarkConfig(batch_size=4, seq_len=512, in_features=4096, out_features=4096, rank=16),
        BenchmarkConfig(batch_size=4, seq_len=512, in_features=4096, out_features=4096, rank=32),
        BenchmarkConfig(batch_size=4, seq_len=512, in_features=4096, out_features=4096, rank=64),
        
        # Large batch
        BenchmarkConfig(batch_size=8, seq_len=512, in_features=4096, out_features=4096, rank=16),
        
        # Long sequence
        BenchmarkConfig(batch_size=2, seq_len=2048, in_features=4096, out_features=4096, rank=16),
    ]
    
    print("\n" + "=" * 80)
    print("MetalLoRA Benchmark Suite")
    print("=" * 80)
    
    all_results = []
    
    for config in configs:
        try:
            results = run_benchmark(config, compare=True)
            print_results(results)
            all_results.append(results)
        except Exception as e:
            print(f"Error benchmarking config {config}: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_speedups = []
    for results in all_results:
        if "baseline" in results:
            speedup = results["baseline"].total_time_ms / results["metal_lora"].total_time_ms
            total_speedups.append(speedup)
    
    if total_speedups:
        avg_speedup = sum(total_speedups) / len(total_speedups)
        max_speedup = max(total_speedups)
        min_speedup = min(total_speedups)
        
        print(f"Average Speedup: {avg_speedup:.2f}x")
        print(f"Max Speedup:     {max_speedup:.2f}x")
        print(f"Min Speedup:     {min_speedup:.2f}x")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="MetalLoRA Benchmark Suite")
    parser.add_argument("--compare-baseline", action="store_true", default=True,
                        help="Compare against baseline implementation")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for single benchmark")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length for single benchmark")
    parser.add_argument("--dim", type=int, default=4096,
                        help="Model dimension for single benchmark")
    parser.add_argument("--rank", type=int, default=16,
                        help="LoRA rank for single benchmark")
    parser.add_argument("--full-suite", action="store_true",
                        help="Run full benchmark suite")
    
    args = parser.parse_args()
    
    if args.full_suite:
        run_benchmark_suite()
    else:
        config = BenchmarkConfig(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            in_features=args.dim,
            out_features=args.dim,
            rank=args.rank,
        )
        results = run_benchmark(config, compare=args.compare_baseline)
        print_results(results)


if __name__ == "__main__":
    main()
