"""Benchmark the plain Helion deepseek 1x128 kernel vs torch.compile(reference).

Reports GPU time, achieved GB/s, and % of B200 peak memory bandwidth. The Helion
kernel is autotuned on first call; the reported time excludes that one-time cost.

Methodology mirrors flexquant v1 benchmark.py (do_bench_using_profiling, bytes
moved = in + out + scale, B200 8 TB/s peak).
"""

import torch
from torch._inductor.utils import do_bench_using_profiling

from main import (
    deepseek_quant_1x128,
    deepseek_quant_1x128_reshape,
    deepseek_quant_1x128_reference,
)

B200_PEAK_BW_GBPS = 8000.0  # 8 TB/s HBM3e


def _bytes_moved(x, qdata, scale):
    # bf16 in + fp8 out + fp32 scale (v1 _bytes_moved)
    return (
        x.numel() * x.element_size()
        + qdata.numel() * qdata.element_size()
        + scale.numel() * scale.element_size()
    )


def _measure(run, bytes_per_iter):
    # warm up so compile/autotune/allocator costs don't leak into timing
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    gpu_time_ms = do_bench_using_profiling(run)
    gpu_gbps = bytes_per_iter / (gpu_time_ms * 1e-3) / 1e9
    gpu_pct_peak = gpu_gbps / B200_PEAK_BW_GBPS * 100
    return gpu_time_ms, gpu_gbps, gpu_pct_peak


def main():
    torch.manual_seed(0)
    # M = K = 4096
    M = K = 16384
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    rows = []

    # 1) Helion nested-loop (autotuned). First call triggers autotuning (excluded).
    print("autotuning helion nested-loop kernel (one-time, may take minutes)...", flush=True)
    qdata, scale = deepseek_quant_1x128(x)
    bytes_per_iter = _bytes_moved(x, qdata, scale)
    rows.append(("helion nested_loop", *_measure(lambda: deepseek_quant_1x128(x), bytes_per_iter)))

    # 2) Helion reshape-axis (autotuned). Reduction over a Helion-owned axis, so
    # the autotuner can pick a persistent reduction.
    print("autotuning helion reshape-axis kernel (one-time, may take minutes)...", flush=True)
    deepseek_quant_1x128_reshape(x)
    rows.append(("helion reshape_axis", *_measure(lambda: deepseek_quant_1x128_reshape(x), bytes_per_iter)))

    # 3) torch.compile(reference)
    pt_fn = torch.compile(deepseek_quant_1x128_reference, fullgraph=True)
    rows.append(("torch.compile(ref)", *_measure(lambda: pt_fn(x), bytes_per_iter)))

    print(f"\nshape: ({M}, {K}) bfloat16   bytes/iter: {bytes_per_iter / 1e6:.1f} MB")
    print(f"{'impl':<24} {'gpu_time_ms':>12} {'gpu_gbps':>10} {'gpu_pct_peak':>13}")
    for name, ms, gbps, pct in rows:
        print(f"{name:<24} {ms:>12.4f} {gbps:>10.1f} {pct:>12.1f}%")
    print("\nnote: helion time excludes one-time autotune cost.")


if __name__ == "__main__":
    main()
