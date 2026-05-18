from typing import Callable, NamedTuple

import torch


class Recipe(NamedTuple):
    name: str
    block_size: int | tuple[int, int]
    dim: int | tuple[int, int]
    qdata_dtype: torch.dtype
    scale_dtype: torch.dtype
    amax_to_scale_fn: Callable
    cast_to_dtype_fn: Callable
    reference_fn: Callable


def _deepseek_fp8_1_128_amax_to_scale_fn(amax: torch.Tensor) -> torch.Tensor:
    return amax / torch.finfo(torch.float8_e4m3fn).max


def _deepseek_fp8_1_128_cast_to_dtype_fn(tile: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (tile / scale).to(torch.float8_e4m3fn)


def _deepseek_fp8_1_128_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    *lead, last = x.shape
    n_blocks = last // 128
    x_b = x.reshape(*lead, n_blocks, 128)
    scale = x_b.abs().amax(dim=-1, keepdim=True) / torch.finfo(torch.float8_e4m3fn).max
    qdata_b = (x_b / scale).to(torch.float8_e4m3fn)
    qdata = qdata_b.reshape(*lead, last)
    return qdata, scale.squeeze(-1).to(torch.float32)


def _deepseek_fp8_1_128_dim_m_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # dim=-2: reduce across M, output in transposed (K, M) layout.
    return _deepseek_fp8_1_128_reference(x.transpose(-2, -1).contiguous())


def _deepseek_fp8_128_128_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    *lead, D1, D2 = x.shape
    n1, n2 = D1 // 128, D2 // 128
    x_b = (
        x.reshape(*lead, n1, 128, n2, 128)
        .transpose(-3, -2)
        .contiguous()
        .reshape(*lead, n1, n2, 128 * 128)
    )
    scale = x_b.abs().amax(dim=-1, keepdim=True) / fp8_max
    qdata_b = (x_b / scale).to(torch.float8_e4m3fn)
    qdata = (
        qdata_b.reshape(*lead, n1, n2, 128, 128)
        .transpose(-3, -2)
        .contiguous()
        .reshape(*lead, D1, D2)
    )
    return qdata, scale.squeeze(-1).to(torch.float32)


def _rowwise_fp8_amax_to_scale_fn(amax: torch.Tensor) -> torch.Tensor:
    return amax / torch.finfo(torch.float8_e4m3fn).max


def _rowwise_fp8_cast_to_dtype_fn(tile: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    return (tile / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)


def _rowwise_fp8_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = x.abs().amax(dim=-1, keepdim=True) / fp8_max
    qdata = (x / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return qdata, scale.squeeze(-1).to(torch.float32)


def _rowwise_fp8_dim_m_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # dim=-2: reduce across M (becomes "rowwise" of the transposed input).
    return _rowwise_fp8_reference(x.transpose(-2, -1).contiguous())


deepseek_fp8_1_128 = Recipe(
    name="deepseek_fp8_1_128",
    block_size=128,
    dim=-1,
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_deepseek_fp8_1_128_amax_to_scale_fn,
    cast_to_dtype_fn=_deepseek_fp8_1_128_cast_to_dtype_fn,
    reference_fn=_deepseek_fp8_1_128_reference,
)

deepseek_fp8_128_128 = Recipe(
    name="deepseek_fp8_128_128",
    block_size=(128, 128),
    dim=(-2, -1),
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_deepseek_fp8_1_128_amax_to_scale_fn,
    cast_to_dtype_fn=_deepseek_fp8_1_128_cast_to_dtype_fn,
    reference_fn=_deepseek_fp8_128_128_reference,
)

rowwise_fp8 = Recipe(
    name="rowwise_fp8",
    block_size=-1,
    dim=-1,
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_rowwise_fp8_amax_to_scale_fn,
    cast_to_dtype_fn=_rowwise_fp8_cast_to_dtype_fn,
    reference_fn=_rowwise_fp8_reference,
)

deepseek_fp8_1_128_dim_m = Recipe(
    name="deepseek_fp8_1_128_dim_m",
    block_size=128,
    dim=-2,
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_deepseek_fp8_1_128_amax_to_scale_fn,
    cast_to_dtype_fn=_deepseek_fp8_1_128_cast_to_dtype_fn,
    reference_fn=_deepseek_fp8_1_128_dim_m_reference,
)

rowwise_fp8_dim_m = Recipe(
    name="rowwise_fp8_dim_m",
    block_size=-1,
    dim=-2,
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_rowwise_fp8_amax_to_scale_fn,
    cast_to_dtype_fn=_rowwise_fp8_cast_to_dtype_fn,
    reference_fn=_rowwise_fp8_dim_m_reference,
)
