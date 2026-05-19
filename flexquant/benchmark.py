import fire
import torch
import triton.testing

from api import flex_cast_quant_dense
from recipes import (
    Recipe,
    deepseek_fp8_1_128,
    deepseek_fp8_1_128_dim_m,
    deepseek_fp8_1_128_dim_m_triton,
    deepseek_fp8_1_128_triton,
    deepseek_fp8_128_128,
    deepseek_fp8_128_128_hop,
    deepseek_fp8_128_128_triton,
)

B200_PEAK_BW_GBPS = 8000.0  # 8 TB/s

RECIPES_BY_NAME = {
    r.name: r
    for r in (
        deepseek_fp8_1_128,
        deepseek_fp8_1_128_triton,
        deepseek_fp8_1_128_dim_m,
        deepseek_fp8_1_128_dim_m_triton,
        deepseek_fp8_128_128,
        deepseek_fp8_128_128_triton,
        deepseek_fp8_128_128_hop,
    )
}


def _bytes_moved(x: torch.Tensor, qdata: torch.Tensor, scale: torch.Tensor) -> int:
    return x.numel() * x.element_size() + qdata.numel() * qdata.element_size() + scale.numel() * scale.element_size()


def _bench_one(
    recipe_obj: Recipe, M: int, K: int, warmup_ms: int, rep_ms: int
) -> tuple[float, float, float]:
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    # Triton-backed recipes skip torch.compile — they're already a kernel.
    if recipe_obj.use_triton_kernel:
        fn = flex_cast_quant_dense
    else:
        fn = torch.compile(flex_cast_quant_dense, fullgraph=True)

    def run():
        return fn(
            x,
            block_size=recipe_obj.block_size,
            dim=recipe_obj.dim,
            qdata_dtype=recipe_obj.qdata_dtype,
            scale_dtype=recipe_obj.scale_dtype,
            amax_to_scale_fn=recipe_obj.amax_to_scale_fn,
            cast_to_dtype_fn=recipe_obj.cast_to_dtype_fn,
            use_triton_kernel=recipe_obj.use_triton_kernel,
            amax_to_scale_fn_triton=recipe_obj.amax_to_scale_fn_triton,
            cast_to_dtype_fn_triton=recipe_obj.cast_to_dtype_fn_triton,
            use_hop_path=recipe_obj.use_hop_path,
        )

    qdata, scale = run()
    bytes_per_iter = _bytes_moved(x, qdata, scale)

    avg_ms = triton.testing.do_bench(run, warmup=warmup_ms, rep=rep_ms)
    gbps = bytes_per_iter / (avg_ms * 1e-3) / 1e9
    pct_peak = gbps / B200_PEAK_BW_GBPS * 100
    return avg_ms, gbps, pct_peak


def main(
    recipe_filter: str | None = None,
    M: int = 16384,
    K: int = 16384,
    warmup_ms: int = 25,
    rep_ms: int = 100,
) -> None:
    device_name = torch.cuda.get_device_name(0)
    assert "B200" in device_name, f"this benchmark assumes B200, got {device_name!r}"

    if recipe_filter is not None:
        if recipe_filter not in RECIPES_BY_NAME:
            raise ValueError(
                f"unknown recipe {recipe_filter!r}; available: {list(RECIPES_BY_NAME)}"
            )
        names = [recipe_filter]
    else:
        names = list(RECIPES_BY_NAME)

    print(f"shape: ({M}, {K}) bfloat16")
    print(f"{'recipe':<31} {'time (ms)':>10} {'GB/s':>10} {'% peak':>8}")
    for name in names:
        avg_ms, gbps, pct_peak = _bench_one(
            RECIPES_BY_NAME[name], M, K, warmup_ms, rep_ms
        )
        print(f"{name:<31} {avg_ms:>10.4f} {gbps:>10.1f} {pct_peak:>7.1f}%")


if __name__ == "__main__":
    fire.Fire(main)
