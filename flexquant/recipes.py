from dataclasses import dataclass
from typing import Callable

import torch

from nvfp4_utils import F4_E2M1_MAX, f32_to_f4_unpacked, pack_uint4

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
EPS = 1e-12

F8E4M3_MAX = FP8_MAX
E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny


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


# nvfp4 with single-level scaling and packed fp4 (two fp4 values per byte,
# stored as torch.float4_e2m1fn_x2). Mirrors the `per_tensor_scale is None`
# branch of `nvfp4_quantize` in torchao.

def _nvfp4_amax_to_scale_fn(amax: torch.Tensor) -> torch.Tensor:
    block_scale = amax.to(torch.float32) / F4_E2M1_MAX
    return torch.clamp(block_scale, min=E4M3_EPS, max=F8E4M3_MAX).to(
        torch.float8_e4m3fn
    )


def _nvfp4_cast_to_dtype_fn(
    tile: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    scale_fp32 = scale.to(torch.float32)
    data_scaled = tile.to(torch.float32) * (1.0 / scale_fp32)
    data_scaled = torch.clamp(data_scaled, -F4_E2M1_MAX, F4_E2M1_MAX)
    data_unpacked = f32_to_f4_unpacked(data_scaled)
    return pack_uint4(data_unpacked).view(torch.float4_e2m1fn_x2)


def _nvfp4_no_gs_reference(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    *lead, last = x.shape
    n_blocks = last // 16
    x_b = x.reshape(*lead, n_blocks, 16)
    amax = x_b.abs().amax(dim=-1, keepdim=True)
    scale_e4m3 = _nvfp4_amax_to_scale_fn(amax)
    qdata_b = _nvfp4_cast_to_dtype_fn(x_b, scale_e4m3)
    # qdata_b shape: (*lead, n_blocks, 8) packed -> (*lead, last // 2)
    qdata = qdata_b.reshape(*lead, last // 2)
    return qdata, scale_e4m3.squeeze(-1)


nvfp4_no_gs = Recipe(
    name="nvfp4_no_gs",
    block_size=16,
    dim=-1,
    qdata_dtype=torch.float4_e2m1fn_x2,
    scale_dtype=torch.float8_e4m3fn,
    amax_to_scale_fn=_nvfp4_amax_to_scale_fn,
    cast_to_dtype_fn=_nvfp4_cast_to_dtype_fn,
    _reference_fn=_nvfp4_no_gs_reference,
)

