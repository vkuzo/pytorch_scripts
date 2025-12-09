"""
Code to test max(tensor) with atomics versus
1. PyTorch eager mode kernel
2. PyTorch torch.compile kernel (using two-stage reduction for medium+ problem sizes)

run with `numerics` to verify numerical correctness of the atomics kernel
run with `performance` to compare performance of all the kernels on a single tensor across various tensor sizes
"""

import fire
import torch
import triton

from torch.profiler import profile, ProfilerActivity

torch.manual_seed(0)

from triton_kernels import (
    max_with_atomics,
)


def test_numerics():
    for dtype in (torch.float32,):
        for size in (256, 1024, 8192):
            for _ in range(10):
                x = torch.randn(size, size, dtype=dtype, device="cuda")
                y_ref = torch.max(x)
                y = max_with_atomics(x)
                torch.testing.assert_close(y_ref, y, rtol=0, atol=0)
    print("numerics match")


@torch.compile
def max_wrapper_for_compile(x):
    return torch.max(x)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        # x_vals=[2**i for i in range(12, 28, 1)],
        x_vals=[2**i for i in range(12, 30, 1)],
        x_log=True,
        line_arg="provider",
        line_vals=["torch_eager", "torch_compile", "triton_atomics"],
        line_names=["torch_eager", "torch_compile", "triton_atomics"],
        ylabel="GB/s",
        plot_name="vector_max_performance",
        args={},
    )
)
def benchmark(size, provider):
    # TODO(future): look at GPU kernel time

    dtype = torch.float32
    x = torch.randn(size, device="cuda", dtype=dtype)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch_eager":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.max(x), quantiles=quantiles
        )
    elif provider == "torch_compile":
        # reset dynamo and warm up, this is needed for accurate performance measurement
        # while inside `triton.testing.perf_report`
        torch._dynamo.reset()
        max_wrapper_for_compile(x)

        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: max_wrapper_for_compile(x), quantiles=quantiles
        )
    elif provider == "triton_atomics":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: max_with_atomics(x), quantiles=quantiles
        )

    # read numel elements, write 1 (ignore the write)
    gbps = lambda ms: x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


def test_performance():
    benchmark.run(
        print_data=True,
        show_plots=False,
        save_path="/home/vasiliy/local/tmp/test_triton_graph",
    )


def profile_performance():
    # TODO(future): look at GPU kernel time

    fname1 = "/home/vasiliy/local/tmp/20240921_torch_compile_trace.json"
    fname2 = "/home/vasiliy/local/tmp/20240921_triton_trace.json"

    size = 8192

    x = torch.randn(size, size, device="cuda", dtype=torch.float32)

    # warm up
    max_wrapper_for_compile(x)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            max_wrapper_for_compile(x)
            torch.cuda.synchronize()
    prof.export_chrome_trace(fname1)

    # warm up
    max_with_atomics(x)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            max_with_atomics(x)
            torch.cuda.synchronize()
    prof.export_chrome_trace(fname2)


def run(mode: str):
    # example input
    # TODO:
    # * ValueError: atomic_max does not support bf16
    # * first make it work in float32, then hack bf16 max (with bit shifts from fp32, or f16, etc)
    # dtype = torch.bfloat16
    # dtype = torch.float
    # size = 1024
    # x = torch.randn(size, size, dtype=dtype, device='cuda')

    if mode == "numerics":
        test_numerics()
    elif mode == "performance":
        test_performance()
    elif mode == "profile":
        profile_performance()

    print("done")


if __name__ == "__main__":
    fire.Fire(run)
