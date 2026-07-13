import enum
from typing import Callable, Tuple

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
    # restrictions on tiling
    _tile_multiple_of: Tuple[int, int] | None = None,
    _inner_tile_multiple_of: Tuple[int, int] | None = None,
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

        # verify that tile constraints are respected, if specified
        if _tile_multiple_of is not None:
            M, K = input.shape
            assert M % _tile_multiple_of[0] == 0
            assert K % _tile_multiple_of[1] == 0

        # _inner_tile_multiple_of is satisfied automatically as there is only
        # one tile

        outs = f(input)

    elif _backend is FlexCastQuantBackend.MANUAL_TILE:
        outs = _manual_tile(input, f, _tile_multiple_of, _inner_tile_multiple_of)

    else:
        raise AssertionError(f"unknown {_backend=}")

    return outs


def _manual_tile(
    input: torch.Tensor,
    f: Callable,
    _tile_multiple_of: Tuple[int, int] | None = None,
    _inner_tile_multiple_of: Tuple[int, int] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Tile `input` into 256x256 tiles, run `f` on each, and recompose (debug backend).
    """
    assert input.ndim == 2, f"MANUAL_TILE expects a 2D input, got {input.ndim}D"

    # choose tile size if not specified
    # TODO(future): make this configurable
    tile_size_0, tile_size_1 = 256, 256

    # verify tiling constraints are honored
    if _tile_multiple_of is not None:
        assert tile_size_0 % _tile_multiple_of[0] == 0
        assert tile_size_1 % _tile_multiple_of[1] == 0
        assert input.shape[0] % _tile_multiple_of[0] == 0
        assert input.shape[1] % _tile_multiple_of[1] == 0
    if _inner_tile_multiple_of is not None:
        assert tile_size_0 % _inner_tile_multiple_of[0] == 0
        assert tile_size_1 % _inner_tile_multiple_of[1] == 0

    # manually tile the tensor
    t0_start = list(range(0, input.shape[0], tile_size_0))
    t1_start = list(range(0, input.shape[1], tile_size_1))

    # [
    #   [(o0_0_0, o0_0_1, ...), ...],
    #   [(o1_0_0, o1_0_1, ...), ...],
    # ]
    outs_outer = []

    for t0_start_idx in t0_start:

        # [(o0_0_0, o0_0_1, ...), (o0_1_0, o0_1_1, ...), ...]
        outs_inner = []

        t0_end_idx = t0_start_idx + tile_size_0
        for t1_start_idx in t1_start:
            t1_end_idx = t1_start_idx + tile_size_1

            input_tile = input[t0_start_idx:t0_end_idx, t1_start_idx:t1_end_idx]
            # TODO(future): properly broadcast aux_inputs, right now this
            # will silently fail as the captured aux inputs are always replicated

            outs = f(input_tile)
            
            # combine the outputs
            outs_inner.append(outs)

        outs_outer.append(outs_inner)

    # stitch the output tensors to get final result
    # TODO(future): handle `f` returning either tuple of multiple tensors
    # or a single tensor
    num_outs = len(outs_outer[0][0])
    final_outs = []
    for out_idx in range(num_outs):
        
        # first, extract 
        # [
        #   [o0_0_idx, o0_1_idx, ...], 
        #   [o1_0_idx, o1_1_idx, ...],
        # ]
        extracted_out = [
            [col[out_idx] for col in row]
            for row in outs_outer
        ]

        # then, get
        # [
        #   tensor(*o0_0_idx...*o0_1_idx...], 
        #   [o1_0_idx, o1_1_idx, ...],
        # ]
        extracted_out = [
            torch.cat(row, dim=1) for row in extracted_out
        ]

        # then, get final tensor
        extracted_out = torch.cat(extracted_out, dim=0)
        final_outs.append(extracted_out)

    return tuple(final_outs)
