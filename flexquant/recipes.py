from dataclasses import dataclass
from typing import Callable

import torch

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
EPS = 1e-12


@dataclass(frozen=True)
class Recipe:
    name: str
    block_size: int | tuple[int, int]
    dim: int | tuple[int, int]
    qdata_dtype: torch.dtype
    scale_dtype: torch.dtype
    amax_to_scale_fn: Callable
    cast_to_dtype_fn: Callable
    # arguments below are for debugging only
    # Plain PyTorch reference implementation (including the tiling)
    _reference_fn: Callable


def _deepseek_fp8_amax_to_scale_fn(amax: torch.Tensor) -> torch.Tensor:
    amax_fp32 = amax.clamp(min=EPS).to(torch.float32)
    return amax_fp32 / FP8_MAX


def _deepseek_fp8_cast_to_dtype_fn(
    tile: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    # Recover the reciprocal from the forward scale (prod stores forward scale
    # but multiplies by the reciprocal).
    reciprocal = (1.0 / scale)
    y = tile * reciprocal
    return y.to(torch.float8_e4m3fn)


def _deepseek_fp8_1_128_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    *lead, last = x.shape
    n_blocks = last // 128
    x_b = x.reshape(*lead, n_blocks, 128)
    amax = x_b.abs().amax(dim=-1, keepdim=True).clamp(min=EPS).to(torch.float32)
    scale = amax / FP8_MAX  # forward scale (on-disk format)
    scale = scale.to(torch.float32)
    reciprocal = 1.0 / scale
    y = x_b.to(torch.float32) * reciprocal
    qdata = y.to(torch.float8_e4m3fn).reshape(*lead, last)
    return qdata, scale.squeeze(-1)


deepseek_fp8_1_128 = Recipe(
    name="deepseek_fp8_1_128",
    block_size=128,
    dim=-1,
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_deepseek_fp8_amax_to_scale_fn,
    cast_to_dtype_fn=_deepseek_fp8_cast_to_dtype_fn,
    _reference_fn=_deepseek_fp8_1_128_reference,
)

def _deepseek_fp8_1_128_dim_m_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # dim=-2: reduce across M, output in transposed (K, M) layout.
    return _deepseek_fp8_1_128_reference(x.transpose(-2, -1).contiguous())


deepseek_fp8_1_128_dim_m = Recipe(
    name="deepseek_fp8_1_128_dim_m",
    block_size=128,
    dim=-2,
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_deepseek_fp8_amax_to_scale_fn,
    cast_to_dtype_fn=_deepseek_fp8_cast_to_dtype_fn,
    _reference_fn=_deepseek_fp8_1_128_dim_m_reference,
)

def _deepseek_fp8_128_128_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    *lead, D1, D2 = x.shape
    n1, n2 = D1 // 128, D2 // 128
    x_b = (
        x.reshape(*lead, n1, 128, n2, 128)
        .transpose(-3, -2)
        .contiguous()
        .reshape(*lead, n1, n2, 128 * 128)
    )
    amax = x_b.abs().amax(dim=-1, keepdim=True).clamp(min=EPS).to(torch.float32)
    scale = amax / FP8_MAX  # forward scale
    scale = scale.to(torch.float32)
    reciprocal = 1.0 / scale
    y = x_b.to(torch.float32) * reciprocal
    qdata_b = y.to(torch.float8_e4m3fn)
    qdata = (
        qdata_b.reshape(*lead, n1, n2, 128, 128)
        .transpose(-3, -2)
        .contiguous()
        .reshape(*lead, D1, D2)
    )
    return qdata, scale.squeeze(-1)


deepseek_fp8_128_128 = Recipe(
    name="deepseek_fp8_128_128",
    block_size=(128, 128),
    dim=(-2, -1),
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_deepseek_fp8_amax_to_scale_fn,
    cast_to_dtype_fn=_deepseek_fp8_cast_to_dtype_fn,
    _reference_fn=_deepseek_fp8_128_128_reference,
)

