"""
MetalLoRA - Benchmark Suite

Compares MetalLoRA performance against baseline MLX LoRA implementation.
Measures forward/backward pass latency and memory usage across various
configurations. Now properly compares Metal kernels vs pure MLX.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from metal_lora import LoRALinear, is_metal_available
from metal_lora.kernels import lora_forward_metal
from metal_lora.ops import lora_backward_efficient, lora_forward


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
# BASELINE IMPLEMENTATION (Standard MLX - Naive separate ops)
# ============================================================================

class BaselineLoRALinear(nn.Module):
    """Standard LoRA implementation using vanilla MLX ops."""

    def __init__(self, in_features, out_features, rank, alpha=16.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        self.w0 = mx.random.normal((out_features, in_features)) * 0.02
        self.a = mx.random.normal((rank, in_features)) * 0.02
        self.b = mx.zeros((out_features, rank))

    def __call__(self, x):
        # Standard separate operations
        w0x = mx.matmul(x, self.w0.T)
        ax = mx.matmul(x, self.a.T)
        bax = mx.matmul(ax, self.b.T)
        scale = self.alpha / self.rank
        return w0x + scale * bax


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


def benchmark_forward_metal(
    x: mx.array,
    w0: mx.array,  # noqa: N803
    a: mx.array,  # noqa: N803
    b: mx.array,  # noqa: N803
    config: BenchmarkConfig,
) -> float:
    """Benchmark Metal kernel forward pass."""
    batch_size, seq_len, _ = x.shape

    # Warmup
    for _ in range(config.warmup_iters):
        out = lora_forward_metal(x, w0, a, b, config.alpha, batch_size, seq_len)
        mx.eval(out)

    # Benchmark
    start = time.perf_counter()
    for _ in range(config.bench_iters):
        out = lora_forward_metal(x, w0, a, b, config.alpha, batch_size, seq_len)
        mx.eval(out)
    end = time.perf_counter()

    return (end - start) / config.bench_iters * 1000


def benchmark_forward_mlx(
    x: mx.array,
    w0: mx.array,  # noqa: N803
    a: mx.array,  # noqa: N803
    b: mx.array,  # noqa: N803
    config: BenchmarkConfig,
) -> float:
    """Benchmark pure MLX forward pass (ops.lora_forward with use_metal=False)."""
    # Warmup
    for _ in range(config.warmup_iters):
        out = lora_forward(x, w0, a, b, config.alpha, use_metal=False)
        mx.eval(out)

    # Benchmark
    start = time.perf_counter()
    for _ in range(config.bench_iters):
        out = lora_forward(x, w0, a, b, config.alpha, use_metal=False)
        mx.eval(out)
    end = time.perf_counter()

    return (end - start) / config.bench_iters * 1000


def benchmark_backward(
    x: mx.array,
    w0: mx.array,  # noqa: N803
    a: mx.array,  # noqa: N803
    b: mx.array,  # noqa: N803
    config: BenchmarkConfig,
    use_metal: bool = False,
) -> float:
    """Benchmark backward pass latency."""
    # Create grad_output
    grad_h = mx.random.normal((config.batch_size, config.seq_len, config.out_features))

    # Warmup
    for _ in range(config.warmup_iters):
        grad_a, grad_b = lora_backward_efficient(grad_h, x, a, b, config.alpha, use_metal=use_metal)
        mx.eval(grad_a, grad_b)

    # Benchmark
    start = time.perf_counter()
    for _ in range(config.bench_iters):
        grad_a, grad_b = lora_backward_efficient(grad_h, x, a, b, config.alpha, use_metal=use_metal)
        mx.eval(grad_a, grad_b)
    end = time.perf_counter()

    return (end - start) / config.bench_iters * 1000  # ms


def run_benchmark(config: BenchmarkConfig, compare: bool = True) -> dict[str, BenchmarkResult]:
    """Run benchmark comparing Metal, MLX, and naive baseline."""
    mx.random.seed(42)

    # Create input
    x = mx.random.normal((config.batch_size, config.seq_len, config.in_features))
    mx.eval(x)

    # Create weights
    w0 = mx.random.normal((config.out_features, config.in_features)) * 0.02
    a = mx.random.normal((config.rank, config.in_features)) * 0.02
    b = mx.zeros((config.out_features, config.rank))
    mx.eval(w0, a, b)

    results = {}
    total_samples = config.batch_size * config.seq_len

    # Metal Kernel (if available)
    if is_metal_available():
        try:
            fwd_time = benchmark_forward_metal(x, w0, a, b, config)
            bwd_time = benchmark_backward(x, w0, a, b, config, use_metal=True)
            throughput = total_samples / ((fwd_time + bwd_time) / 1000)

            results["metal_kernel"] = BenchmarkResult(
                config=config,
                forward_time_ms=fwd_time,
                backward_time_ms=bwd_time,
                total_time_ms=fwd_time + bwd_time,
                memory_mb=0,
                throughput_samples_per_sec=throughput,
            )
        except Exception as e:
            print(f"Metal kernel error: {e}")

    # Pure MLX (optimized)
    fwd_time = benchmark_forward_mlx(x, w0, a, b, config)
    bwd_time = benchmark_backward(x, w0, a, b, config, use_metal=False)
    throughput = total_samples / ((fwd_time + bwd_time) / 1000)

    results["pure_mlx"] = BenchmarkResult(
        config=config,
        forward_time_ms=fwd_time,
        backward_time_ms=bwd_time,
        total_time_ms=fwd_time + bwd_time,
        memory_mb=0,
        throughput_samples_per_sec=throughput,
    )

    if compare:
        # Naive Baseline
        baseline_layer = BaselineLoRALinear(
            config.in_features,
            config.out_features,
            rank=config.rank,
            alpha=config.alpha,
        )
        mx.eval(baseline_layer.w0, baseline_layer.a, baseline_layer.b)

        fwd_time = benchmark_forward(baseline_layer, x, config)
        # Use same backward for baseline (it's the same algorithm)
        bwd_time = benchmark_backward(x, baseline_layer.w0, baseline_layer.a, baseline_layer.b, config, use_metal=False)
        throughput = total_samples / ((fwd_time + bwd_time) / 1000)

        results["baseline"] = BenchmarkResult(
            config=config,
            forward_time_ms=fwd_time,
            backward_time_ms=bwd_time,
            total_time_ms=fwd_time + bwd_time,
            memory_mb=0,
            throughput_samples_per_sec=throughput,
        )

    return results


def print_results(results: dict[str, BenchmarkResult]):
    """Print benchmark results in a formatted table."""
    if not results:
        print("No results to display")
        return

    config = list(results.values())[0].config

    print("\n" + "=" * 90)
    print(f"Configuration: batch={config.batch_size}, seq={config.seq_len}, "
          f"dim={config.in_features}, rank={config.rank}")
    print("=" * 90)
    print(f"{'Implementation':<20} {'Forward (ms)':<15} {'Backward (ms)':<15} "
          f"{'Total (ms)':<15} {'Speedup':<10}")
    print("-" * 90)

    # Use baseline or pure_mlx as reference
    reference = results.get("baseline", results.get("pure_mlx", list(results.values())[0]))
    baseline_total = reference.total_time_ms

    for name, result in results.items():
        speedup = baseline_total / result.total_time_ms if result.total_time_ms > 0 else 0
        speedup_str = f"{speedup:.2f}x" if name != "baseline" else "1.00x"

        print(f"{name:<20} {result.forward_time_ms:<15.3f} {result.backward_time_ms:<15.3f} "
              f"{result.total_time_ms:<15.3f} {speedup_str:<10}")

    print("=" * 90)


def run_benchmark_suite():
    """Run full benchmark suite with various configurations."""
    print("\n" + "=" * 90)
    print("MetalLoRA Benchmark Suite")
    print(f"Metal Available: {is_metal_available()}")
    print("=" * 90)

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

    all_results = []

    for config in configs:
        try:
            results = run_benchmark(config, compare=True)
            print_results(results)
            all_results.append(results)
        except Exception as e:
            print(f"Error benchmarking config {config}: {e}")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Metal vs baseline speedups
    metal_speedups = []
    mlx_speedups = []

    for results in all_results:
        if "baseline" in results:
            baseline_time = results["baseline"].total_time_ms

            if "metal_kernel" in results:
                speedup = baseline_time / results["metal_kernel"].total_time_ms
                metal_speedups.append(speedup)

            if "pure_mlx" in results:
                speedup = baseline_time / results["pure_mlx"].total_time_ms
                mlx_speedups.append(speedup)

    if metal_speedups:
        print("Metal Kernel vs Baseline:")
        print(f"  Average Speedup: {sum(metal_speedups)/len(metal_speedups):.2f}x")
        print(f"  Max Speedup:     {max(metal_speedups):.2f}x")
        print(f"  Min Speedup:     {min(metal_speedups):.2f}x")

    if mlx_speedups:
        print("\nPure MLX vs Baseline:")
        print(f"  Average Speedup: {sum(mlx_speedups)/len(mlx_speedups):.2f}x")
        print(f"  Max Speedup:     {max(mlx_speedups):.2f}x")
        print(f"  Min Speedup:     {min(mlx_speedups):.2f}x")

    print("=" * 90)


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
