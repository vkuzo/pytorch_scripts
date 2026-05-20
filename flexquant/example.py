"""
An example demonstrating deepseek-style quantization of inputs to an
fp8 gemm.
"""

import torch

from api import flex_cast_quant_dense

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
EPS = 1e-12


def amax_to_scale_fn(amax: torch.Tensor) -> torch.Tensor:
    amax_fp32 = amax.clamp(min=EPS).to(torch.float32)
    return amax_fp32 / FP8_MAX


def cast_to_dtype_fn(tile: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    reciprocal = 1.0 / scale
    y = tile * reciprocal
    return y.to(torch.float8_e4m3fn)


def main() -> None:
    M, K, N = 512, 1024, 2048

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    # compile is opt-in and required for performance
    flex_cast_quant_dense_c = torch.compile(flex_cast_quant_dense)

    # deepseek style fp8_e4m3 1x128 quant along the K dim for activations
    # Note: torchinductor will generate this kernel from scratch. If
    # torch.compile is enabled on the surrounding program, the kernel
    # will (likely, up to the fuser) be fused into the previous op.
    x_q, x_scale = flex_cast_quant_dense_c(
        x,
        block_size=128,
        dim=-1,
        qdata_dtype=torch.float8_e4m3fn,
        scale_dtype=torch.float32,
        amax_to_scale_fn=amax_to_scale_fn,
        cast_to_dtype_fn=cast_to_dtype_fn,
    )
    print(f"x_q:     shape={tuple(x_q.shape)} dtype={x_q.dtype}")
    print(f"x_scale: shape={tuple(x_scale.shape)} dtype={x_scale.dtype}")

    # deepseek style fp8_e4m3 128x128 quant for weights
    # Note: for this kernel, torchinductor will lower the callbacks 
    # onto a handwritten triton template, flex-attention style 
    # (currently faster than generating from scratch)
    w_q, w_scale = flex_cast_quant_dense_c(
        w,
        block_size=(128, 128),
        dim=(-2, -1),
        qdata_dtype=torch.float8_e4m3fn,
        scale_dtype=torch.float32,
        amax_to_scale_fn=amax_to_scale_fn,
        cast_to_dtype_fn=cast_to_dtype_fn,
    )
    print(f"w_q:     shape={tuple(w_q.shape)} dtype={w_q.dtype}")
    print(f"w_scale: shape={tuple(w_scale.shape)} dtype={w_scale.dtype}")

    # lowp gemm here (not shown, since I prototyped this on a B200)
    # TODO(future): hop to an H100 and finish out this example


if __name__ == "__main__":
    main()
