from typing import Callable, Tuple, Union

import torch


def flex_quant_dense(
    # input tensor
    input: torch.Tensor,
    *,
    # block_size and dim define the scaling
    block_size: Union[int, Tuple[int, int]],
    dim: Union[int, Tuple[int, int]],
    # statically known output dtypes are nice
    qdata_dtype: torch.dtype,
    scale_dtype: torch.dtype,
    # user defined function to go from scaling tile to single scale
    calc_scale_fn: Callable,
    # user defined function to go from data tile + scale to value
    cast_to_dtype_fn: Callable,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input.ndim >= 1

    block_size_t = (block_size,) if isinstance(block_size, int) else tuple(block_size)
    dim_t = (dim,) if isinstance(dim, int) else tuple(dim)
    assert len(block_size_t) == len(dim_t)

    n_block_dims = len(block_size_t)
    expected_dims = tuple(range(input.ndim - n_block_dims, input.ndim))
    normalized_dim_t = tuple(d if d >= 0 else d + input.ndim for d in dim_t)
    assert normalized_dim_t == expected_dims, (
        f"only trailing dims supported, got dim={dim} for ndim={input.ndim}"
    )

    if block_size_t == (-1,):
        # rowwise scaling

        # entire row is one block; the trailing block dim equals input.shape[-1]
        scale = calc_scale_fn(input)
        qdata = cast_to_dtype_fn(input, scale)

    elif n_block_dims == 1:
        # 1D scaling with `block_size` across the last dim

        block_size_int = block_size_t[0]
        assert block_size_int > 0
        last = input.shape[-1]
        assert last % block_size_int == 0, (
            f"input.shape[-1]={last} must be divisible by block_size={block_size_int}"
        )

        lead = input.shape[:-1]
        n_blocks = last // block_size_int
        x_blocked = input.reshape(*lead, n_blocks, block_size_int)

        scale = calc_scale_fn(x_blocked)
        qdata_blocked = cast_to_dtype_fn(x_blocked, scale)

        qdata = qdata_blocked.reshape(*lead, last)

    elif n_block_dims == 2:
        # 2D scaling with `block_size` across the last 2 dims; tile is flattened
        # to a single trailing dim so callbacks designed for 1D blocks work as-is.
        # Note: likely to be slow with the current reference implementation

        B1, B2 = block_size_t
        assert B1 > 0 and B2 > 0
        *lead, D1, D2 = input.shape
        assert D1 % B1 == 0 and D2 % B2 == 0, (
            f"input trailing dims {(D1, D2)} must be divisible by block_size={(B1, B2)}"
        )
        n1, n2 = D1 // B1, D2 // B2

        # (..., D1, D2) -> (..., n1, B1, n2, B2) -> (..., n1, n2, B1, B2) -> (..., n1, n2, B1*B2)
        x_blocked = (
            input.reshape(*lead, n1, B1, n2, B2)
            .transpose(-3, -2)
            .contiguous()
            .reshape(*lead, n1, n2, B1 * B2)
        )

        scale = calc_scale_fn(x_blocked)
        qdata_blocked = cast_to_dtype_fn(x_blocked, scale)

        qdata = (
            qdata_blocked.reshape(*lead, n1, n2, B1, B2)
            .transpose(-3, -2)
            .contiguous()
            .reshape(*lead, D1, D2)
        )

    else:
        raise AssertionError(f"unsupported block_size rank: {n_block_dims}")

    # TODO(future): more interesting cases, such as:
    # 1. casting across dim1 (for backward)
    # 2. scale swizzling
    # 3. grouped variants with offsets
    # 4. padding and other edge cases
    # 5. actual lowering to templates, to beat the compiler

    scale = scale.squeeze(-1).to(scale_dtype)
    return qdata, scale
