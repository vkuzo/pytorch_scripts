from typing import Callable, Tuple, Union

import torch

from triton_kernels import (
    triton_fp8_blockwise_act_quant_transposed_lhs,
    triton_fp8_blockwise_weight_quant_128_128,
)
# Side-effect import: registers the FlexQuant HOP, its Dynamo variable,
# and its Inductor lowering. Imported eagerly so the registrations are in
# place before the user's first torch.compile call.
from hop import flex_quant


def flex_cast_quant_dense(
    input: torch.Tensor,
    *,
    block_size: Union[int, Tuple[int, int]],
    dim: Union[int, Tuple[int, int]],
    qdata_dtype: torch.dtype,
    scale_dtype: torch.dtype,
    amax_to_scale_fn: Callable,
    cast_to_dtype_fn: Callable,
    # the arguments below are temporary and should be removed from the final
    # version of this API
    use_triton_kernel: bool = False,
    amax_to_scale_fn_triton: Callable | None = None,
    cast_to_dtype_fn_triton: Callable | None = None,
    # if True, route through the FlexQuant HigherOrderOperator. Under
    # torch.compile this lets Inductor codegen the user's PyTorch callbacks
    # into a hand-written Triton template; in eager mode it falls back to the
    # same body as the compile path.
    use_hop_path: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor with user-defined per-tile scaling.

    The framework owns the layout-sensitive parts of quantization (which
    elements form a tile, how the tile reduction is computed, and how qdata
    and scale outputs are laid out in memory). The user owns two pointwise
    callbacks: how to turn a tile's amax into a scale, and how to cast a tile
    given its scale.

    Tiles are defined by ``block_size`` and ``dim``. Supported shapes:

    - **1D blocked**: ``block_size: int``, ``dim: int``
      - dim=-1: output qdata is `(M, K)`, scale is `(M, K // block_size)`
      - dim=-2: output qdata is `(K, M)`, scale is `(K, M // block_size)`

    - **2D blocked** ``block_size: tuple[int, int]``, ``dim=(-2, -1)``
      - qdata is `(M, K)`, scale is `(M // B1, K // B2)`.

    Args:
        input: 2D contiguous tensor.
        block_size: tile size along ``dim``.
        dim: dimension(s) the tile spans.
        qdata_dtype: dtype of returned qdata. Statically known to enable
            template specialization.
        scale_dtype: dtype of returned scale.
        amax_to_scale_fn: ``(amax) -> scale``. Called by the framework on the
            per-tile amax it computed. Must be a pure pointwise op.
        cast_to_dtype_fn: ``(tile, scale) -> qdata``. Called per-tile to
            produce the quantized values. Must be a pure pointwise op.
        use_triton_kernel: if True, route to a hand-written Triton kernel
            instead of the compile-friendly path. Currently only the 128x128
            deepseek fp8 recipe has a Triton template; other recipes assert.
        amax_to_scale_fn_triton: `@triton.jit` version of amax_to_scale_fn, this
            is temporary to avoid dealing with HOPs for now
        cast_to_dtype_fn_triton: `@triton.jit` version of cast_to_dtype_fn, this
            is temporary to avoid dealing with HOPs for now

    Returns:
        ``(qdata, scale)`` with dtypes ``qdata_dtype`` / ``scale_dtype`` and
        the layouts described above.
    """
    assert input.ndim == 2
    assert input.is_contiguous()

    block_size_t = (block_size,) if isinstance(block_size, int) else tuple(block_size)
    dim_t = (dim,) if isinstance(dim, int) else tuple(dim)
    assert len(block_size_t) == len(dim_t)

    n_block_dims = len(block_size_t)
    # normalized_dim_t can be either (0,), (1,) or (0, 1)
    normalized_dim_t = tuple(d if d >= 0 else d + input.ndim for d in dim_t)
    M, K = input.shape

    # TODO(future):
    # * scale swizzling
    # * padding and other edge cases

    if n_block_dims == 1 and block_size_t[0] > 0:
        # 1D blocked scaling

        block_size_int = block_size_t[0]
        if normalized_dim_t == (1,):
            # dim=-1: reduce across K; output qdata in (M, K), scale in (M, n_blocks)
            assert not use_hop_path, (
                "use_hop_path for 1D blocks is only supported for dim=-2"
            )
            assert not use_triton_kernel, (
                "use_triton_kernel is not supported for 1D blocks dim=-1"
            )
            assert K % block_size_int == 0, (
                f"input.shape[-1]={K} must be divisible by block_size={block_size_int}"
            )
            n_blocks = K // block_size_int
            x_b = input.reshape(M, n_blocks, block_size_int)
            amax = x_b.abs().amax(dim=-1, keepdim=True)  # (M, n_blocks, 1)
            scale_bc = amax_to_scale_fn(amax)
            qdata_b = cast_to_dtype_fn(x_b, scale_bc)
            qdata = qdata_b.reshape(M, K)
            scale = scale_bc.squeeze(-1)
        elif normalized_dim_t == (0,):
            # dim=-2: reduce across M; output qdata/scale row-major in (K, M) layout
            assert M % block_size_int == 0, (
                f"input.shape[-2]={M} must be divisible by block_size={block_size_int}"
            )
            if (
                use_hop_path
                and block_size_int == 128
                and qdata_dtype == torch.float8_e4m3fn
                and scale_dtype == torch.float32
            ):
                qdata, scale = flex_quant(
                    input,
                    amax_to_scale_fn,
                    cast_to_dtype_fn,
                    block_size_int,
                    -2,
                    qdata_dtype,
                    scale_dtype,
                )
            elif (
                use_triton_kernel
                and block_size_int == 128
                and qdata_dtype == torch.float8_e4m3fn
                and scale_dtype == torch.float32
            ):
                assert amax_to_scale_fn_triton is not None
                assert cast_to_dtype_fn_triton is not None
                qdata, scale = triton_fp8_blockwise_act_quant_transposed_lhs(
                    input, amax_to_scale_fn_triton, cast_to_dtype_fn_triton
                )
            else:
                assert not use_triton_kernel, (
                    "use_triton_kernel for 1D blocks dim=-2 only supports 1x128 deepseek"
                )
                assert not use_hop_path, (
                    "use_hop_path for 1D blocks dim=-2 only supports 1x128 deepseek"
                )
                n_blocks = M // block_size_int
                x_b = input.reshape(n_blocks, block_size_int, K)
                amax = x_b.abs().amax(dim=-2, keepdim=True)  # (n_blocks, 1, K)
                scale_bc = amax_to_scale_fn(amax)
                qdata_b = cast_to_dtype_fn(x_b, scale_bc)
                qdata = qdata_b.reshape(M, K).transpose(-2, -1).contiguous()
                scale = scale_bc.squeeze(-2).transpose(-2, -1).contiguous()
        else:
            raise AssertionError(f"unsupported dim={dim} for 1D blocks")

    elif n_block_dims == 2:
        # 2D blocked scaling; tile is flattened to a single trailing dim so callbacks
        # designed for 1D blocks work as-is.

        if normalized_dim_t == (0, 1):
            B1, B2 = block_size_t
            assert B1 > 0 and B2 > 0
            assert M % B1 == 0 and K % B2 == 0, (
                f"input trailing dims {(M, K)} must be divisible by block_size={(B1, B2)}"
            )
            if (
                use_hop_path
                and (B1, B2) == (128, 128)
                and qdata_dtype == torch.float8_e4m3fn
                and scale_dtype == torch.float32
            ):
                qdata, scale = flex_quant(
                    input,
                    amax_to_scale_fn,
                    cast_to_dtype_fn,
                    (B1, B2),
                    (-2, -1),
                    qdata_dtype,
                    scale_dtype,
                )
            elif (
                use_triton_kernel
                and (B1, B2) == (128, 128)
                and qdata_dtype == torch.float8_e4m3fn
                and scale_dtype == torch.float32
            ):
                assert amax_to_scale_fn_triton is not None
                assert cast_to_dtype_fn_triton is not None
                qdata, scale = triton_fp8_blockwise_weight_quant_128_128(
                    input, amax_to_scale_fn_triton, cast_to_dtype_fn_triton
                )
            else:
                assert not use_triton_kernel, (
                    "use_triton_kernel only supports 128x128 deepseek for 2D blocks"
                )
                assert not use_hop_path, (
                    "use_hop_path only supports 128x128 deepseek for 2D blocks"
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
                scale = scale_bc.squeeze(-1)

        else:
            raise AssertionError(f"unsupported dim={dim} for 2D blocks")

    else:
        raise AssertionError(f"unsupported block_size rank: {n_block_dims}")

    return qdata, scale
