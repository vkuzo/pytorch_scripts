from typing import Callable, Tuple, Union

import torch


def flex_cast_quant_dense(
    # input tensor
    input: torch.Tensor,
    *,
    # block_size and dim define the scaling
    block_size: Union[int, Tuple[int, int]],
    dim: Union[int, Tuple[int, int]],
    # statically known output dtypes are nice
    qdata_dtype: torch.dtype,
    scale_dtype: torch.dtype,
    # user defined function to go from amax of tile to single scale
    amax_to_scale_fn: Callable,
    # user defined function to go from data tile + scale to value
    cast_to_dtype_fn: Callable,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input.ndim == 2
    assert input.is_contiguous()

    block_size_t = (block_size,) if isinstance(block_size, int) else tuple(block_size)
    dim_t = (dim,) if isinstance(dim, int) else tuple(dim)
    assert len(block_size_t) == len(dim_t)

    n_block_dims = len(block_size_t)
    normalized_dim_t = tuple(d if d >= 0 else d + input.ndim for d in dim_t)
    M, K = input.shape

    # TODO(future): more interesting cases, such as:
    # 1. scale swizzling
    # 2. grouped variants with offsets
    # 3. padding and other edge cases
    # 4. actual lowering to templates, to beat the compiler

    if n_block_dims == 1 and block_size_t[0] > 0:
        # 1D blocked scaling
        block_size_int = block_size_t[0]
        if normalized_dim_t == (1,):
            # dim=-1: reduce across K; output qdata in (M, K), scale in (M, n_blocks)
            assert K % block_size_int == 0, (
                f"input.shape[-1]={K} must be divisible by block_size={block_size_int}"
            )
            n_blocks = K // block_size_int
            x_b = input.reshape(M, n_blocks, block_size_int)
            amax = x_b.abs().amax(dim=-1, keepdim=True)  # (M, n_blocks, 1)
            scale_bc = amax_to_scale_fn(amax)
            qdata_b = cast_to_dtype_fn(x_b, scale_bc)
            qdata = qdata_b.reshape(M, K)
            scale = scale_bc.squeeze(-1).to(scale_dtype)
        elif normalized_dim_t == (0,):
            # dim=-2: reduce across M; output qdata/scale row-major in (K, M) layout
            assert M % block_size_int == 0, (
                f"input.shape[-2]={M} must be divisible by block_size={block_size_int}"
            )
            n_blocks = M // block_size_int
            x_b = input.reshape(n_blocks, block_size_int, K)
            amax = x_b.abs().amax(dim=-2, keepdim=True)  # (n_blocks, 1, K)
            scale_bc = amax_to_scale_fn(amax)
            qdata_b = cast_to_dtype_fn(x_b, scale_bc)
            qdata = qdata_b.reshape(M, K).transpose(-2, -1).contiguous()
            scale = (
                scale_bc.squeeze(-2).transpose(-2, -1).contiguous().to(scale_dtype)
            )
        else:
            raise AssertionError(f"unsupported dim={dim} for 1D blocks")

    elif n_block_dims == 1 and block_size_t == (-1,):
        # rowwise scaling (entire row is one block)
        if normalized_dim_t == (1,):
            # dim=-1: reduce across K; output qdata in (M, K), scale shape (M,)
            amax = input.abs().amax(dim=-1, keepdim=True)  # (M, 1)
            scale_bc = amax_to_scale_fn(amax)
            qdata = cast_to_dtype_fn(input, scale_bc)
            scale = scale_bc.squeeze(-1).to(scale_dtype)
        elif normalized_dim_t == (0,):
            # dim=-2: reduce across M; output qdata in (K, M), scale shape (K,)
            amax = input.abs().amax(dim=-2, keepdim=True)  # (1, K)
            scale_bc = amax_to_scale_fn(amax)
            qdata = cast_to_dtype_fn(input, scale_bc).transpose(-2, -1).contiguous()
            scale = scale_bc.squeeze(-2).to(scale_dtype)
        else:
            raise AssertionError(f"unsupported dim={dim} for rowwise scaling")

    elif n_block_dims == 2:
        # 2D blocked scaling; tile is flattened to a single trailing dim so callbacks
        # designed for 1D blocks work as-is.
        # Note: likely to be slow with the current reference implementation
        if normalized_dim_t == (0, 1):
            B1, B2 = block_size_t
            assert B1 > 0 and B2 > 0
            assert M % B1 == 0 and K % B2 == 0, (
                f"input trailing dims {(M, K)} must be divisible by block_size={(B1, B2)}"
            )
            n1, n2 = M // B1, K // B2

            # (M, K) -> (n1, B1, n2, B2) -> (n1, n2, B1, B2) -> (n1, n2, B1*B2)
            x_b = (
                input.reshape(n1, B1, n2, B2)
                .transpose(-3, -2)
                .contiguous()
                .reshape(n1, n2, B1 * B2)
            )
            amax = x_b.abs().amax(dim=-1, keepdim=True)
            scale_bc = amax_to_scale_fn(amax)
            qdata_b = cast_to_dtype_fn(x_b, scale_bc)
            qdata = (
                qdata_b.reshape(n1, n2, B1, B2)
                .transpose(-3, -2)
                .contiguous()
                .reshape(M, K)
            )
            scale = scale_bc.squeeze(-1).to(scale_dtype)
        else:
            raise AssertionError(f"unsupported dim={dim} for 2D blocks")

    else:
        raise AssertionError(f"unsupported block_size rank: {n_block_dims}")

    return qdata, scale
