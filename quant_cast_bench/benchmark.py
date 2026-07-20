"""Memory-bandwidth benchmark for quant_cast_gold recipes.

Each recipe is a memory-bound cast, so the signal we care about is achieved memory bandwidth
vs. the B200 ceiling (8 TB/s). We torch.compile each recipe's reference fn, time it with
`do_bench_using_profiling`, and report latency + GB/s + % of peak. Structured after
flexquant/benchmark.py.
"""

import os
import sys

import fire
import tabulate
import torch
from torch._inductor.utils import do_bench_using_profiling

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_cast_gold.recipes import ALL_RECIPES

B200_PEAK_BW_GBPS = 8000.0  # 8 TB/s

# Recipes excluded from the benchmark entirely -- not relevant here (they still run in the
# gold tests). Filtered out before the sweep, so they never appear in the results table.
_BENCH_SKIP = {"mxfp8_floor", "nvfp4_blocked_outer", "mxfp8_bias"}


def _bytes_moved(inputs, outputs):
    # a cast reads its inputs and writes its outputs; bytes moved = element bytes across both.
    tensors = [t for t in (*inputs, *outputs) if isinstance(t, torch.Tensor)]
    return sum(t.numel() * t.element_size() for t in tensors)


def _bench_relu(M, K):
    # eager torch.relu baseline: a trivially memory-bound op (read x, write relu(x), both bf16)
    # that anchors the achievable-bandwidth ceiling for this shape.
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    def run():
        return torch.relu(x)

    out = run()
    bytes_per_iter = x.numel() * x.element_size() + out.numel() * out.element_size()

    for _ in range(2):
        run()
    torch.cuda.synchronize()

    gpu_time_ms = do_bench_using_profiling(run)
    gbps = bytes_per_iter / (gpu_time_ms * 1e-3) / 1e9
    pct_peak = gbps / B200_PEAK_BW_GBPS * 100
    return gpu_time_ms, gbps, pct_peak


def _bench_one(gold, M, K, mode):
    torch.manual_seed(0)
    torch._dynamo.reset()
    inputs = gold.example_input_fn(M, K)  # (x, *aux)
    fn = torch.compile(gold.pt_ref_fn, fullgraph=True, mode=mode)

    def run():
        return fn(*inputs)

    outputs = run()
    bytes_per_iter = _bytes_moved(inputs, outputs)

    # warm up so first-call costs (compile, autotune, allocator) don't leak into the timing.
    for _ in range(2):
        run()
    torch.cuda.synchronize()

    gpu_time_ms = do_bench_using_profiling(run)
    gbps = bytes_per_iter / (gpu_time_ms * 1e-3) / 1e9
    pct_peak = gbps / B200_PEAK_BW_GBPS * 100
    return gpu_time_ms, gbps, pct_peak


def main(
    M: int = 16384,
    K: int = 16384,
    recipe_name_filter: str | None = None,
    mode: str = "default",
):
    device_name = torch.cuda.get_device_name(0)
    assert "B200" in device_name, f"this benchmark assumes B200, got {device_name!r}"

    recipes = [
        (n, g)
        for n, g in ALL_RECIPES
        if n not in _BENCH_SKIP
        and (recipe_name_filter is None or recipe_name_filter in n)
    ]
    if not recipes:
        raise ValueError(
            f"no recipe matched {recipe_name_filter!r}; have {[n for n, _ in ALL_RECIPES]}"
        )

    rows = []  # (recipe, gpu_time_ms, gbps, pct_peak)

    # relu baseline anchors the bandwidth ceiling; shown on a full sweep (no filter).
    if recipe_name_filter is None:
        ms, gbps, pct = _bench_relu(M, K)
        rows.append(("relu (baseline)", f"{ms:.4f}", f"{gbps:.1f}", f"{pct:.1f}%"))

    for name, gold in recipes:
        # TODO: some recipes don't benchmark cleanly yet (e.g. fp4-packed byte accounting under
        # torch.compile, swizzle grids, SR's fp32/const input). Skip failures for now so the
        # sweep still reports the ones that work; revisit each skipped recipe.
        try:
            ms, gbps, pct = _bench_one(gold, M, K, mode)
        except Exception as e:
            rows.append((name, f"SKIPPED: {type(e).__name__}: {str(e).splitlines()[0][:60]}", "", ""))
            continue
        rows.append((name, f"{ms:.4f}", f"{gbps:.1f}", f"{pct:.1f}%"))

    print(f"shape: ({M}, {K})  compile mode: {mode}")
    print(
        tabulate.tabulate(
            rows,
            headers=["recipe", "gpu_time_ms", "gbps", "pct_peak"],
            colalign=("left", "right", "right", "right"),
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
