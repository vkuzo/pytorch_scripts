import enum
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode
from utils import _pad_to_multiple


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
    # input padding
    _pad_input_to_multiple_of: Tuple[int, int] | None = None,
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

        # pad input (before the constraint asserts, so they validate the padded shape)
        if _pad_input_to_multiple_of is not None:
            input = _pad_to_multiple(input, _pad_input_to_multiple_of)

        # verify that tile constraints are respected, if specified
        if _tile_multiple_of is not None:
            M, K = input.shape
            assert M % _tile_multiple_of[0] == 0
            assert K % _tile_multiple_of[1] == 0

        # _inner_tile_multiple_of is satisfied automatically as there is only
        # one tile

        outs = f(input)

    elif _backend is FlexCastQuantBackend.MANUAL_TILE:
        outs = _manual_tile(
            input, 
            f, 
            _pad_input_to_multiple_of,
            _tile_multiple_of, 
            _inner_tile_multiple_of,
        )

    else:
        raise AssertionError(f"unknown {_backend=}")

    return outs


def _manual_tile(
    input: torch.Tensor,
    f: Callable,
    _pad_input_to_multiple_of: Tuple[int, int] | None = None,
    _tile_multiple_of: Tuple[int, int] | None = None,
    _inner_tile_multiple_of: Tuple[int, int] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Tile `input` into 256x256 tiles, run `f` on each, and recompose (debug backend).
    """
    assert input.ndim == 2, f"MANUAL_TILE expects a 2D input, got {input.ndim}D"

    # pad input first, so the whole pipeline (constraint asserts, fake-probe output shapes,
    # preallocation, tiling offsets) all key off the padded shape.
    if _pad_input_to_multiple_of is not None:
        input = _pad_to_multiple(input, _pad_input_to_multiple_of)

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

    # Preallocate-then-scatter: infer each output's global shape/dtype by running `f` on a
    # FakeTensor of the full input, preallocate the outputs, then write each tile's local
    # output into its slice. This models a real backend (write into a buffer at computed
    # offsets) rather than concatenating per-tile results.

    # 1. infer output shapes/dtypes by running `f` on a full-shape fake input.
    # allow_non_fake_inputs=True is needed for recipes that close over real cuda tensors
    # (nvfp4 outer_scale, RHT matrix); meta inputs don't work (device mismatch / no RNG).
    with FakeTensorMode(allow_non_fake_inputs=True):
        fake_in = torch.empty(input.shape, dtype=input.dtype, device=input.device)
        fake_outs = f(fake_in)

    # 2. preallocate the final outputs on the real device.
    final_outs = [
        torch.empty(fo.shape, dtype=fo.dtype, device=input.device) for fo in fake_outs
    ]

    # 3. per-output (dim0, dim1) divisors: the input->output tiling ratio, constant across
    # tiles, so a tile at input offset (r, c) writes to output offset (r//div0, c//div1).
    # Only dims 0,1 are tiled; trailing dims (e.g. the swizzle grid's 32,16) are copied whole.
    divisors = []
    for fo in fake_outs:
        # scalar/replicated outputs (e.g. tensorwise per-tensor scale) are not supported
        # under MANUAL_TILE: every output must be tiled along dims 0 and 1.
        assert fo.ndim >= 2, (
            f"MANUAL_TILE requires outputs tiled on dims 0,1, got shape {tuple(fo.shape)}"
        )
        d = []
        for dim in range(2):
            assert input.shape[dim] % fo.shape[dim] == 0
            d.append(input.shape[dim] // fo.shape[dim])
        divisors.append(d)

    # 4. tile the input and scatter each tile's outputs into the preallocated buffers.
    t0_start = list(range(0, input.shape[0], tile_size_0))
    t1_start = list(range(0, input.shape[1], tile_size_1))

    for r in t0_start:
        for c in t1_start:
            input_tile = input[r : r + tile_size_0, c : c + tile_size_1]
            # TODO(future): properly broadcast aux_inputs, right now this
            # will silently fail as the captured aux inputs are always replicated
            outs = f(input_tile)

            for i, local in enumerate(outs):
                div0, div1 = divisors[i]
                # extent = this tile's own output shape, so ragged last tiles work.
                dst = (
                    slice(r // div0, r // div0 + local.shape[0]),
                    slice(c // div1, c // div1 + local.shape[1]),
                )  # trailing dims (e.g. 32,16) left fully selected
                o = final_outs[i]
                if o.dtype == torch.float4_e2m1fn_x2:
                    # torch.cat/fill/copy aren't implemented for sub-byte float4; assign
                    # through the uint8 view of both sides (1 byte/elem preserves the slice).
                    o.view(torch.uint8)[dst] = local.view(torch.uint8)
                else:
                    o[dst] = local

    return tuple(final_outs)
