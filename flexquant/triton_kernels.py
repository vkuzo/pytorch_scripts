# Kernel + wrapper copied from torchao:
#   /home/dev/ao/torchao/prototype/blockwise_fp8_training/kernels.py
#   commit 9058b58a4f0ac5ea3c0fca4cc8f4f225d79f7137
# Numerics have been adjusted to match this prototype's reference implementation
# (`_deepseek_fp8_128_128_reference` in recipes.py). Each divergence from the
# original ao kernel is tagged with `TODO(future) fix this` so we can re-align
# to production numerics later.

from typing import Callable, Tuple

import torch
import triton
import triton.language as tl

# Quantization kernels autotuner configs (copied from ao kernels.py:457-465)
quant_kernel_configs = [
    triton.Config(
        {},
        num_warps=warps,
        num_stages=stages,
    )
    for warps in [4, 8]
    for stages in [2, 4]
]


@triton.autotune(configs=quant_kernel_configs, key=["M", "N"])
@triton.jit
def triton_fp8_blockwise_weight_quant_rhs_kernel(
    x_ptr,
    x_stride_dim_0,
    x_stride_dim_1,
    y_ptr,
    y_stride_dim_0,
    y_stride_dim_1,
    s_ptr,
    s_stride_dim_0,
    s_stride_dim_1,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    amax_to_scale_fn: tl.constexpr,
    cast_to_dtype_fn: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load (block_size x block_size) block of x, where input is row major
    x_offs = offs_m[:, None] * x_stride_dim_0 + offs_n[None, :] * x_stride_dim_1
    x_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    # Reduce amax; recipe callbacks own all numerics (clamp/eps, dtype
    # promotion, clamp before cast).
    amax = tl.max(tl.abs(x))
    scale = amax_to_scale_fn(amax)
    y = cast_to_dtype_fn(x, scale)

    # Store output
    y_offs = offs_m[:, None] * y_stride_dim_0 + offs_n[None, :] * y_stride_dim_1
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptr + y_offs, y, mask=y_mask)

    # Write scale (scalar value); the callback already produced scale_dtype.
    scale_m_off = pid_m * s_stride_dim_0
    scale_n_off = pid_n * s_stride_dim_1
    tl.store(s_ptr + scale_m_off + scale_n_off, scale)


def triton_fp8_blockwise_weight_quant_128_128(
    x: torch.Tensor,
    amax_to_scale_fn: Callable,
    cast_to_dtype_fn: Callable,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.dim() == 2, "Input tensor must have 2 dimensions"
    block_size = 128
    dtype = torch.float8_e4m3fn
    assert x.size(0) % block_size == 0 and x.size(1) % block_size == 0, (
        f"Both dimensions of x must be divisible by block_size (block_size={block_size})"
    )
    M, N = x.size()
    y = torch.empty_like(x, dtype=dtype)  # row-major (M, N)
    M_blocks, N_blocks = triton.cdiv(M, block_size), triton.cdiv(N, block_size)
    s = x.new_empty(M_blocks, N_blocks, dtype=torch.float32)  # row-major

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_SIZE"]),
            triton.cdiv(N, meta["BLOCK_SIZE"]),
        )

    triton_fp8_blockwise_weight_quant_rhs_kernel[grid](
        x,
        x.stride(0),
        x.stride(1),
        y,
        y.stride(0),
        y.stride(1),
        s,
        s.stride(0),
        s.stride(1),
        M,
        N,
        BLOCK_SIZE=block_size,
        amax_to_scale_fn=amax_to_scale_fn,
        cast_to_dtype_fn=cast_to_dtype_fn,
    )
    return y, s
