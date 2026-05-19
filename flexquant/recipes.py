from typing import Callable, NamedTuple, Optional

import torch
import triton
import triton.language as tl

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
EPS = 1e-12


class Recipe(NamedTuple):
    name: str
    block_size: int | tuple[int, int]
    dim: int | tuple[int, int]
    qdata_dtype: torch.dtype
    scale_dtype: torch.dtype
    amax_to_scale_fn: Callable
    cast_to_dtype_fn: Callable
    reference_fn: Callable
    # Triton-flavored callbacks for the use_triton_kernel path. Optional —
    # only recipes with a corresponding template need to provide them.
    amax_to_scale_fn_triton: Optional[Callable] = None
    cast_to_dtype_fn_triton: Optional[Callable] = None
    # When True, route this recipe through its hand-written Triton kernel
    # rather than the compile-friendly reference path.
    use_triton_kernel: bool = False


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


@triton.jit
def deepseek_fp8_amax_to_scale_fn_triton(amax):
    EPS: tl.constexpr = 1e-12
    FP8_MAX_C: tl.constexpr = 448.0
    amax_fp32 = tl.maximum(amax.to(tl.float32), EPS)
    return amax_fp32 / FP8_MAX_C


@triton.jit
def deepseek_fp8_cast_to_dtype_fn_triton(tile, scale):
    y = tile * (1.0 / scale)
    return y.to(tl.float8e4nv)


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
    reference_fn=_deepseek_fp8_1_128_reference,
)

deepseek_fp8_1_128_triton = Recipe(
    name="deepseek_fp8_1_128_triton",
    block_size=128,
    dim=-1,
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_deepseek_fp8_amax_to_scale_fn,
    cast_to_dtype_fn=_deepseek_fp8_cast_to_dtype_fn,
    reference_fn=_deepseek_fp8_1_128_reference,
    amax_to_scale_fn_triton=deepseek_fp8_amax_to_scale_fn_triton,
    cast_to_dtype_fn_triton=deepseek_fp8_cast_to_dtype_fn_triton,
    use_triton_kernel=True,
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
    reference_fn=_deepseek_fp8_1_128_dim_m_reference,
)

deepseek_fp8_1_128_dim_m_triton = Recipe(
    name="deepseek_fp8_1_128_dim_m_triton",
    block_size=128,
    dim=-2,
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_deepseek_fp8_amax_to_scale_fn,
    cast_to_dtype_fn=_deepseek_fp8_cast_to_dtype_fn,
    reference_fn=_deepseek_fp8_1_128_dim_m_reference,
    amax_to_scale_fn_triton=deepseek_fp8_amax_to_scale_fn_triton,
    cast_to_dtype_fn_triton=deepseek_fp8_cast_to_dtype_fn_triton,
    use_triton_kernel=True,
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
    reference_fn=_deepseek_fp8_128_128_reference,
)

deepseek_fp8_128_128_triton = Recipe(
    name="deepseek_fp8_128_128_triton",
    block_size=(128, 128),
    dim=(-2, -1),
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_deepseek_fp8_amax_to_scale_fn,
    cast_to_dtype_fn=_deepseek_fp8_cast_to_dtype_fn,
    reference_fn=_deepseek_fp8_128_128_reference,
    amax_to_scale_fn_triton=deepseek_fp8_amax_to_scale_fn_triton,
    cast_to_dtype_fn_triton=deepseek_fp8_cast_to_dtype_fn_triton,
    use_triton_kernel=True,
)


def _rowwise_fp8_amax_to_scale_fn(amax: torch.Tensor) -> torch.Tensor:
    # fp32-throughout (matches the deepseek family).
    return amax.clamp(min=EPS).to(torch.float32) / FP8_MAX


def _rowwise_fp8_cast_to_dtype_fn(
    tile: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    y = tile / scale
    return y.to(torch.float8_e4m3fn)


@triton.jit
def rowwise_fp8_amax_to_scale_fn_triton(amax):
    EPS: tl.constexpr = 1e-12
    FP8_MAX_C: tl.constexpr = 448.0
    amax_fp32 = tl.maximum(amax.to(tl.float32), EPS)
    return amax_fp32 / FP8_MAX_C


@triton.jit
def rowwise_fp8_cast_to_dtype_fn_triton(tile, scale):
    y = tile / scale
    return y.to(tl.float8e4nv)


def _rowwise_fp8_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # fp32-throughout (matches the deepseek family).
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=EPS).to(torch.float32)
    scale = amax / FP8_MAX
    qdata = (x / scale).to(torch.float8_e4m3fn)
    return qdata, scale.squeeze(-1)


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

rowwise_fp8_triton = Recipe(
    name="rowwise_fp8_triton",
    block_size=-1,
    dim=-1,
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_rowwise_fp8_amax_to_scale_fn,
    cast_to_dtype_fn=_rowwise_fp8_cast_to_dtype_fn,
    reference_fn=_rowwise_fp8_reference,
    amax_to_scale_fn_triton=rowwise_fp8_amax_to_scale_fn_triton,
    cast_to_dtype_fn_triton=rowwise_fp8_cast_to_dtype_fn_triton,
    use_triton_kernel=True,
)


def _rowwise_fp8_dim_m_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # dim=-2: reduce across M (becomes "rowwise" of the transposed input).
    return _rowwise_fp8_reference(x.transpose(-2, -1).contiguous())


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

rowwise_fp8_dim_m_triton = Recipe(
    name="rowwise_fp8_dim_m_triton",
    block_size=-1,
    dim=-2,
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_rowwise_fp8_amax_to_scale_fn,
    cast_to_dtype_fn=_rowwise_fp8_cast_to_dtype_fn,
    reference_fn=_rowwise_fp8_dim_m_reference,
    amax_to_scale_fn_triton=rowwise_fp8_amax_to_scale_fn_triton,
    cast_to_dtype_fn_triton=rowwise_fp8_cast_to_dtype_fn_triton,
    use_triton_kernel=True,
)
