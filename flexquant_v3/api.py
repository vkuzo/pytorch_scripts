import enum
from typing import Callable

import torch


class FlexCastQuantBackend(enum.Enum):
    # for debugging, just runs the callback on the entire tensor
    REFERENCE = "reference"
    # for debugging, manually tiles and runs the callback on each
    # Note: only works with aux_inputs which are replicated and not tiled
    # TODO: design how tiling for aux_inputs works here
    MANUAL_TILE = "manual_tile"
    # TODO(future): actual backend


def flex_cast_quant(
    input: torch.Tensor,
    f: Callable,  # tile-invariant fn: input -> (out, *aux_out)
    *,
    # these are needed for production quant, but final design TBD
    # TODO(future): we also need a way for a single fused kernel to write
    # out dim0 and dim1 quant at the same time
    _swap_input_axes: bool = False,
    # kwargs below are only for debugging
    _backend: FlexCastQuantBackend = FlexCastQuantBackend.REFERENCE,
) -> tuple[torch.Tensor, ...]:
    """Single-kernel quantization API.

    `f` must be *tile-invariant*: the same pointwise/blockwise computation applied
    independently to every tile of `input`. It returns `(out, *aux_out)` -- one primary
    output plus zero or more auxiliary outputs (e.g. a scale; a pure transform has none).
    Recipes needing multiple kernels call this for one kernel and call something else for
    the rest. This API is consumer-agnostic (no notion of weight/activation/KV/gradient) --
    `f` owns all format knowledge.

    TODO(future work): inspect `f` and lower to an efficient (Helion/Triton) kernel that
    exploits tile-invariance (cf. flexquant_v2 tile_map).
    """

    assert len(input.shape) == 2, "only input of rank 2 is supported"

    if _swap_input_axes:
        input = input.t().contiguous()

    if _backend is FlexCastQuantBackend.REFERENCE:
        outs = f(input)
    elif _backend is FlexCastQuantBackend.MANUAL_TILE:
        outs = _manual_tile(input, f)
    else:
        raise AssertionError(f"unknown {_backend=}")

    return outs


def _manual_tile(
    input: torch.Tensor,
    f: Callable,
) -> tuple[torch.Tensor, ...]:
    """Tile `input` into 4 quadrants, run `f` on each, and recompose (debug backend).

    Splits the (M, N) input at (M // 2, N // 2) into upper-left, upper-right, lower-left,
    lower-right, calls `f` per quadrant, then glues each output back with a 2x2 concat.
    Because `f` is tile-invariant, this equals the REFERENCE result exactly -- provided
    the split points land on tile boundaries so no tile is severed (e.g. for a recipe
    that reduces over 1x128 blocks, M // 2 and N // 2 must be multiples of 128).

    Recomposing by 2x2 concat works uniformly for every output regardless of its grid:
    an output that is coarser than `input` by a per-dim factor (e.g. a (M, N // 128)
    scale) simply has its quadrants split at (M // 2, N // 2) // factor, so the same
    concat reassembles it.
    """
    assert input.ndim == 2, f"MANUAL_TILE expects a 2D input, got {input.ndim}D"
    M, N = input.shape
    mid_m, mid_n = M // 2, N // 2

    ul = f(input[:mid_m, :mid_n])
    ur = f(input[:mid_m, mid_n:])
    ll = f(input[mid_m:, :mid_n])
    lr = f(input[mid_m:, mid_n:])

    def _cat(tensors, dim):
        # torch.cat is not implemented for sub-byte float4_e2m1fn_x2; cat via uint8 view.
        if tensors[0].dtype == torch.float4_e2m1fn_x2:
            return torch.cat([t.view(torch.uint8) for t in tensors], dim).view(
                torch.float4_e2m1fn_x2
            )
        return torch.cat(tensors, dim)

    def _compose(i: int) -> torch.Tensor:
        top = _cat([ul[i], ur[i]], dim=-1)
        bottom = _cat([ll[i], lr[i]], dim=-1)
        return _cat([top, bottom], dim=-2)

    # compose every output f produced (out + 0 or more aux), all in the same 2x2 grid.
    return tuple(_compose(i) for i in range(len(ul)))
