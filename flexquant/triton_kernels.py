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


# 1x128 (and transposed) variants need NUM_GROUPS for grid sizing
# (copied from ao kernels.py:467-475)
quant_kernel_configs_with_groups = [
    triton.Config(
        {"NUM_GROUPS": groups},
        num_warps=warps,
        num_stages=stages,
    )
    for groups in [2, 16, 32, 64, 128]
    for warps in [2, 4, 8]
    for stages in [2, 4, 6]
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


# ----------------------------------------------------------------------------
# 1x128 act-quant kernels — adapted from ao kernels.py:481-569 (lhs) and
# 668-768 (transposed_lhs). Recipe-specific numerics live in the callbacks.
# ----------------------------------------------------------------------------


@triton.autotune(configs=quant_kernel_configs_with_groups, key=["K"])
@triton.jit
def triton_fp8_blockwise_act_quant_lhs_kernel(
    x_ptr,
    x_stride_dim_0,
    x_stride_dim_1,
    y_ptr,
    y_stride_dim_0,
    y_stride_dim_1,
    s_ptr,
    s_stride_dim_0,
    s_stride_dim_1,
    M,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    amax_to_scale_fn: tl.constexpr,
    cast_to_dtype_fn: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    # Load (NUM_GROUPS x BLOCK_SIZE) tile of x, row major.
    m_offs = pid_m * NUM_GROUPS + tl.arange(0, NUM_GROUPS)
    k_offs = pid_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_offs = m_offs[:, None] * x_stride_dim_0 + k_offs[None, :] * x_stride_dim_1
    x_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
    x = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

    # One scale per row in the tile, shape (NUM_GROUPS, 1).
    amax = tl.max(tl.abs(x), axis=1)[:, None]
    scale = amax_to_scale_fn(amax)
    y = cast_to_dtype_fn(x, scale)

    y_offs = m_offs[:, None] * y_stride_dim_0 + k_offs[None, :] * y_stride_dim_1
    y_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
    tl.store(y_ptr + y_offs, y, mask=y_mask)

    # Scales: shape (M, K // BLOCK_SIZE), row-major.
    scale_offs = m_offs[:, None] * s_stride_dim_0 + pid_k * s_stride_dim_1
    scale_mask = m_offs[:, None] < M
    tl.store(s_ptr + scale_offs, scale, mask=scale_mask)


def triton_fp8_blockwise_act_quant_lhs(
    x: torch.Tensor,
    amax_to_scale_fn: Callable,
    cast_to_dtype_fn: Callable,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.dim() == 2, "Input tensor must have 2 dimensions"
    block_size = 128
    dtype = torch.float8_e4m3fn
    M, K = x.size()
    assert K % block_size == 0, (
        f"K={K} must be divisible by block_size={block_size}"
    )
    y = torch.empty_like(x, dtype=dtype)  # row-major (M, K)
    s = x.new_empty(M, K // block_size, dtype=torch.float32)  # row-major

    def grid(meta):
        return (
            triton.cdiv(M, meta["NUM_GROUPS"]),
            triton.cdiv(K, meta["BLOCK_SIZE"]),
        )

    triton_fp8_blockwise_act_quant_lhs_kernel[grid](
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
        K=K,
        BLOCK_SIZE=block_size,
        amax_to_scale_fn=amax_to_scale_fn,
        cast_to_dtype_fn=cast_to_dtype_fn,
    )
    return y, s


@triton.autotune(configs=quant_kernel_configs_with_groups, key=["K"])
@triton.jit
def triton_fp8_blockwise_act_quant_transposed_lhs_kernel(
    x_ptr,
    x_stride_dim_0,
    x_stride_dim_1,
    y_ptr,
    y_stride_dim_0,
    y_stride_dim_1,
    s_ptr,
    s_stride_dim_0,
    s_stride_dim_1,
    M,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    amax_to_scale_fn: tl.constexpr,
    cast_to_dtype_fn: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    # Load (BLOCK_SIZE x NUM_GROUPS) block of input, row-major.
    m_offs = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    k_offs = pid_k * NUM_GROUPS + tl.arange(0, NUM_GROUPS)
    x_offs = m_offs[:, None] * x_stride_dim_0 + k_offs[None, :] * x_stride_dim_1
    x_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    # Column-wise amax (one scale per column = per K element), shape (NUM_GROUPS,).
    amax = tl.max(tl.abs(x), axis=0)
    scale = amax_to_scale_fn(amax)
    y = cast_to_dtype_fn(x, scale[None, :])

    # Store output transposed: (K, M) row-major.
    y_offs = k_offs[:, None] * y_stride_dim_0 + m_offs[None, :] * y_stride_dim_1
    y_mask = (k_offs[:, None] < K) & (m_offs[None, :] < M)
    tl.store(y_ptr + y_offs, y.trans(1, 0), mask=y_mask)

    # Scale tensor shape (K, M // BLOCK_SIZE), row-major.
    scale_m_off = pid_m
    scale_offs = k_offs * s_stride_dim_0 + scale_m_off * s_stride_dim_1
    scale_mask = (k_offs < K) & (scale_m_off < M // BLOCK_SIZE)
    tl.store(s_ptr + scale_offs, scale, mask=scale_mask)


# ----------------------------------------------------------------------------
# Rowwise quant kernel. Single-pass: each program loads BLOCK_M whole rows
# into registers, reduces to a per-row amax, and writes qdata in the same
# pass. Autotuned over BLOCK_M, num_warps, num_stages.
# ----------------------------------------------------------------------------


rowwise_quant_configs = [
    triton.Config({"BLOCK_M": bm}, num_warps=w, num_stages=s)
    for bm in [1, 2, 4, 8]
    for w in [4, 8, 16]
    for s in [2, 3, 4]
]


@triton.autotune(configs=rowwise_quant_configs, key=["K"])
@triton.jit
def triton_fp8_rowwise_quant_kernel(
    x_ptr,
    x_stride_dim_0,
    x_stride_dim_1,
    y_ptr,
    y_stride_dim_0,
    y_stride_dim_1,
    s_ptr,
    M,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    amax_to_scale_fn: tl.constexpr,
    cast_to_dtype_fn: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offs = tl.arange(0, K)
    m_mask = m_offs[:, None] < M

    # Load BLOCK_M whole rows in one shot.
    x_offs = m_offs[:, None] * x_stride_dim_0 + k_offs[None, :] * x_stride_dim_1
    x = tl.load(x_ptr + x_offs, mask=m_mask, other=0.0)

    # Per-row amax.
    amax = tl.max(tl.abs(x), axis=1)
    scale = amax_to_scale_fn(amax)

    tl.store(s_ptr + m_offs, scale, mask=m_offs < M)

    y = cast_to_dtype_fn(x, scale[:, None])
    y_offs = m_offs[:, None] * y_stride_dim_0 + k_offs[None, :] * y_stride_dim_1
    tl.store(y_ptr + y_offs, y, mask=m_mask)


def triton_fp8_rowwise_quant(
    x: torch.Tensor,
    amax_to_scale_fn: Callable,
    cast_to_dtype_fn: Callable,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.dim() == 2, "Input tensor must have 2 dimensions"
    dtype = torch.float8_e4m3fn
    M, K = x.size()
    y = torch.empty_like(x, dtype=dtype)
    s = x.new_empty(M, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    triton_fp8_rowwise_quant_kernel[grid](
        x,
        x.stride(0),
        x.stride(1),
        y,
        y.stride(0),
        y.stride(1),
        s,
        M,
        K=K,
        amax_to_scale_fn=amax_to_scale_fn,
        cast_to_dtype_fn=cast_to_dtype_fn,
    )
    return y, s


# ----------------------------------------------------------------------------
# Rowwise dim_m quant — output is (K, M) row-major, one fp32 scale per
# column of the input. Two-kernel design so that all global memory access
# is row-major-coalesced:
#
#   Pass 1 (amax): tile (BLOCK_M, BLOCK_K) row-major, reduce over M with a
#                  fp32 buffer of column amaxes via atomic max.
#   Pass 2 (cast): tile (BLOCK_M, BLOCK_K) row-major from x; in-kernel
#                  amax→scale via the recipe callback; transpose, write
#                  (BLOCK_K, BLOCK_M) row-major into the (K, M) output. Also
#                  writes the per-column scale (only from pid_m == 0).
# ----------------------------------------------------------------------------


rowwise_quant_dim_m_amax_configs = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_K": bk},
        num_warps=w,
        num_stages=s,
    )
    for bm in [512, 1024]
    for bk in [128, 256]
    for w in [4]
    for s in [2, 3]
]


@triton.autotune(configs=rowwise_quant_dim_m_amax_configs, key=["M", "K"])
@triton.jit
def _rowwise_quant_dim_m_amax_kernel(
    x_ptr,
    x_stride_dim_0,
    x_stride_dim_1,
    amax_ptr,  # fp32 (K,) output, pre-zeroed
    M,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)

    x_offs = m_offs[:, None] * x_stride_dim_0 + k_offs[None, :] * x_stride_dim_1
    x = tl.load(x_ptr + x_offs, mask=mask, other=0.0)

    # Per-column amax over the BLOCK_M rows in this tile.
    tile_amax = tl.max(tl.abs(x), axis=0).to(tl.float32)

    k_mask = k_offs < K
    tl.atomic_max(amax_ptr + k_offs, tile_amax, mask=k_mask)


rowwise_quant_dim_m_cast_configs = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_K": bk},
        num_warps=w,
        num_stages=s,
    )
    for bm in [64, 128]
    for bk in [128, 256]
    for w in [8]
    for s in [2, 3]
]


@triton.autotune(configs=rowwise_quant_dim_m_cast_configs, key=["M", "K"])
@triton.jit
def _rowwise_quant_dim_m_cast_kernel(
    x_ptr,
    x_stride_dim_0,
    x_stride_dim_1,
    y_ptr,  # (K, M) row-major
    y_stride_dim_0,
    y_stride_dim_1,
    amax_ptr,  # (K,) fp32, populated by the amax kernel
    s_ptr,  # (K,) fp32 forward scale (output)
    M,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    amax_to_scale_fn: tl.constexpr,
    cast_to_dtype_fn: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offs < K

    # Load the per-column amax produced by pass 1, then derive the scale
    # via the recipe's amax→scale callback. Every program along the M axis
    # recomputes the same scale (cheap, in-register), but only pid_m == 0
    # writes it to the s_ptr output.
    amax = tl.load(amax_ptr + k_offs, mask=k_mask, other=0.0)
    scale = amax_to_scale_fn(amax)
    if pid_m == 0:
        tl.store(s_ptr + k_offs, scale, mask=k_mask)

    in_mask = (m_offs[:, None] < M) & k_mask[None, :]
    x_offs = m_offs[:, None] * x_stride_dim_0 + k_offs[None, :] * x_stride_dim_1
    x = tl.load(x_ptr + x_offs, mask=in_mask, other=0.0)

    y = cast_to_dtype_fn(x, scale[None, :])

    # Output is (K, M) row-major: row k_offs, col m_offs.
    y_offs = k_offs[:, None] * y_stride_dim_0 + m_offs[None, :] * y_stride_dim_1
    out_mask = k_mask[:, None] & (m_offs[None, :] < M)
    tl.store(y_ptr + y_offs, tl.trans(y, 1, 0), mask=out_mask)


def triton_fp8_rowwise_quant_dim_m(
    x: torch.Tensor,
    amax_to_scale_fn: Callable,
    cast_to_dtype_fn: Callable,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.dim() == 2, "Input tensor must have 2 dimensions"
    dtype = torch.float8_e4m3fn
    M, K = x.size()
    y = torch.empty(K, M, dtype=dtype, device=x.device)  # row-major (K, M)
    s = x.new_empty(K, dtype=torch.float32)

    # Pre-zeroed amax buffer; atomic_max will accumulate column maxima.
    amax_buf = torch.zeros(K, dtype=torch.float32, device=x.device)

    def amax_grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(K, meta["BLOCK_K"]))

    _rowwise_quant_dim_m_amax_kernel[amax_grid](
        x,
        x.stride(0),
        x.stride(1),
        amax_buf,
        M,
        K,
    )

    def cast_grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(K, meta["BLOCK_K"]))

    _rowwise_quant_dim_m_cast_kernel[cast_grid](
        x,
        x.stride(0),
        x.stride(1),
        y,
        y.stride(0),
        y.stride(1),
        amax_buf,
        s,
        M,
        K,
        amax_to_scale_fn=amax_to_scale_fn,
        cast_to_dtype_fn=cast_to_dtype_fn,
    )
    return y, s


def triton_fp8_blockwise_act_quant_transposed_lhs(
    x: torch.Tensor,
    amax_to_scale_fn: Callable,
    cast_to_dtype_fn: Callable,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.dim() == 2, "Input tensor must have 2 dimensions"
    block_size = 128
    dtype = torch.float8_e4m3fn
    M, K = x.size()
    assert M % block_size == 0, (
        f"M={M} must be divisible by block_size={block_size}"
    )
    y = torch.empty(K, M, dtype=dtype, device=x.device)  # row-major (K, M)
    s = x.new_empty(K, M // block_size, dtype=torch.float32)  # row-major

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_SIZE"]),
            triton.cdiv(K, meta["NUM_GROUPS"]),
        )

    triton_fp8_blockwise_act_quant_transposed_lhs_kernel[grid](
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
        K=K,
        BLOCK_SIZE=block_size,
        amax_to_scale_fn=amax_to_scale_fn,
        cast_to_dtype_fn=cast_to_dtype_fn,
    )
    return y, s
