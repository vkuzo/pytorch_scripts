from typing import Callable, NamedTuple, Optional

import torch
import triton
import triton.language as tl


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


def _fp8_amax_to_scale_fn(amax: torch.Tensor) -> torch.Tensor:
    # Recipe owns the cast to scale_dtype (torch.float32).
    return (amax / torch.finfo(torch.float8_e4m3fn).max).to(torch.float32)


def _fp8_cast_to_dtype_fn(tile: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # Bring scale back to the tile dtype so the divide happens at tile precision
    # (matches the reference's bf16/bf16 -> bf16 -> fp8 chain).
    return (tile / scale.to(tile.dtype)).to(torch.float8_e4m3fn)


# Triton-flavored versions of the deepseek fp8 callbacks. The kernel template in
# triton_kernels.py owns the layout-sensitive parts (load tile, reduce amax,
# store qdata + scale); these jit functions express the recipe-specific
# pointwise logic. Triton keeps register values in fp32 even when typed as a
# narrower dtype, so the .to(bf16)/.to(fp32) round-trips force actual rounding
# to match the eager reference bit-for-bit.


@triton.jit
def deepseek_fp8_amax_to_scale_fn_triton(amax):
    # Mirrors `_fp8_amax_to_scale_fn` above: divide in the input dtype, then
    # cast to fp32 for storage (recipe owns the scale_dtype cast).
    # TODO(future) fix this: production clamps amax with EPS=1e-12 and computes
    # the scale in fp64. Our reference does neither.
    FP8_MAX: tl.constexpr = 448.0
    fp8_max = tl.full((), FP8_MAX, dtype=amax.dtype)
    scale_in_dtype = (amax / fp8_max).to(amax.dtype).to(tl.float32).to(amax.dtype)
    return scale_in_dtype.to(tl.float32)


@triton.jit
def deepseek_fp8_cast_to_dtype_fn_triton(tile, scale):
    # Mirrors `_fp8_cast_to_dtype_fn` above. Scale arrives in fp32 (the recipe's
    # scale_dtype); cast back to tile.dtype so the divide happens at tile
    # precision, matching the reference's bf16/bf16 -> bf16 -> fp8 chain.
    # TODO(future) fix this: production stores reciprocal scale and multiplies
    # tile by it; production also clamps the result to [-FP8_MAX, FP8_MAX]
    # before the fp8 cast. Our reference does neither.
    scale_in_dtype = scale.to(tile.dtype).to(tl.float32).to(tile.dtype)
    y = (tile / scale_in_dtype).to(tile.dtype).to(tl.float32).to(tile.dtype)
    return y.to(tl.float8e4nv)


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


deepseek_fp8_1_128 = Recipe(
    name="deepseek_fp8_1_128",
    block_size=128,
    dim=-1,
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_fp8_amax_to_scale_fn,
    cast_to_dtype_fn=_fp8_cast_to_dtype_fn,
    reference_fn=_deepseek_fp8_1_128_reference,
)

deepseek_fp8_128_128 = Recipe(
    name="deepseek_fp8_128_128",
    block_size=(128, 128),
    dim=(-2, -1),
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_fp8_amax_to_scale_fn,
    cast_to_dtype_fn=_fp8_cast_to_dtype_fn,
    reference_fn=_deepseek_fp8_128_128_reference,
)

deepseek_fp8_128_128_triton = Recipe(
    name="deepseek_fp8_128_128_triton",
    block_size=(128, 128),
    dim=(-2, -1),
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_fp8_amax_to_scale_fn,
    cast_to_dtype_fn=_fp8_cast_to_dtype_fn,
    reference_fn=_deepseek_fp8_128_128_reference,
    amax_to_scale_fn_triton=deepseek_fp8_amax_to_scale_fn_triton,
    cast_to_dtype_fn_triton=deepseek_fp8_cast_to_dtype_fn_triton,
    use_triton_kernel=True,
)

deepseek_fp8_1_128_dim_m = Recipe(
    name="deepseek_fp8_1_128_dim_m",
    block_size=128,
    dim=-2,
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_fp8_amax_to_scale_fn,
    cast_to_dtype_fn=_fp8_cast_to_dtype_fn,
    reference_fn=_deepseek_fp8_1_128_dim_m_reference,
)


def _rowwise_fp8_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = x.abs().amax(dim=-1, keepdim=True) / fp8_max
    qdata = (x / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return qdata, scale.squeeze(-1).to(torch.float32)


def _rowwise_fp8_dim_m_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # dim=-2: reduce across M (becomes "rowwise" of the transposed input).
    return _rowwise_fp8_reference(x.transpose(-2, -1).contiguous())


rowwise_fp8 = Recipe(
    name="rowwise_fp8",
    block_size=-1,
    dim=-1,
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_fp8_amax_to_scale_fn,
    cast_to_dtype_fn=_fp8_cast_to_dtype_fn,
    reference_fn=_rowwise_fp8_reference,
)

rowwise_fp8_dim_m = Recipe(
    name="rowwise_fp8_dim_m",
    block_size=-1,
    dim=-2,
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_fp8_amax_to_scale_fn,
    cast_to_dtype_fn=_fp8_cast_to_dtype_fn,
    reference_fn=_rowwise_fp8_dim_m_reference,
)
