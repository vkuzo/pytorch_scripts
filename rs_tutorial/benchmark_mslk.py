"""Benchmark the real MSLK fp4 quantize kernel, with and without stochastic rounding.

`benchmark.py` compares our own minimal SR-to-fp4 kernels. This file instead
benchmarks MSLK's production kernel *as-is* -- imported, not copied -- on the same
16384x16384 bf16 input, so we have a reference point for what a full, optimized
production MXFP4 quantizer achieves. We run MXFP4 both with stochastic casting and
with its default round-to-nearest cast, to see the SR overhead in the production
kernel, plus the NVFP4 quantizer (RTN only -- that entry point has no SR option).

Caveat for apples-to-oranges: MSLK's `triton_quantize_mx4_unpack` is a true MXFP4
quantizer. Beyond the per-element SR-to-fp4 cast that `benchmark.py` measures, it
also computes a shared exponent per group of 32 and writes a separate (swizzled)
scale tensor. So it moves a bit more memory and does more work -- it's not a
like-for-like kernel, just the real thing end to end.

Run:  python benchmark_mslk.py
"""

import torch

import triton

from mslk.quantize.triton.fp4_quantize import (
    triton_quantize_mx4_unpack,
    triton_quantize_nvfp4,
    cal_global_scale_mx4_as_nvfp4,
)

HBM_PEAK_GBPS = 8000.0  # B200 HBM3e peak, for the "% of peak" column


def _bench(run, x):
    """Time `run` (returns (out, scale)) and report bandwidth over bytes moved."""
    out, scale = run()
    ms = triton.testing.do_bench(run, warmup=25, rep=100)

    in_bytes = x.numel() * x.element_size()
    out_bytes = out.numel() * out.element_size()
    scale_bytes = scale.numel() * scale.element_size()
    moved = in_bytes + out_bytes + scale_bytes
    gbps = moved / (ms * 1e-3) / 1e9
    return ms, gbps, gbps / HBM_PEAK_GBPS * 100.0


def _bench_mx4(x, stochastic_casting):
    # stochastic_casting=True selects the per-element stochastic-rounding cast
    # (the path ported in rs.py); False is MSLK's default round-to-nearest cast.
    # The shared-exponent rounding stays at default in both cases.
    return _bench(
        lambda: triton_quantize_mx4_unpack(x, stochastic_casting=stochastic_casting), x
    )


def _bench_nvfp4(x):
    # NVFP4 has no stochastic-rounding option in this entry point -- RTN only.
    # It uses two-level scaling, so compute the per-tensor global scale once,
    # outside the timed region.
    global_scale = cal_global_scale_mx4_as_nvfp4(x)
    return _bench(lambda: triton_quantize_nvfp4(x, global_scale), x)


def main():
    if not torch.cuda.is_available():
        print("No CUDA device available; skipping benchmark.")
        return

    torch.manual_seed(0)
    M = N = 16384
    x = ((torch.rand(M, N, device="cuda") * 12.0) - 6.0).bfloat16()

    rows = [
        ("MSLK MXFP4 (stochastic)", *_bench_mx4(x, stochastic_casting=True)),
        ("MSLK MXFP4 (RTN)", *_bench_mx4(x, stochastic_casting=False)),
        ("MSLK NVFP4 (RTN)", *_bench_nvfp4(x)),
    ]

    print("\nMSLK fp4 quantize, bf16 -> fp4, 16384x16384")
    print(f"{'path':<28}{'time (ms)':>12}{'GB/s':>12}{'% peak':>10}")
    print("-" * 62)
    for name, ms, gbps, pct in rows:
        print(f"{name:<28}{ms:>12.3f}{gbps:>12.1f}{pct:>9.1f}%")


if __name__ == "__main__":
    main()
