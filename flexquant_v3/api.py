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
    # Note: only supports AuxKind.REPLICATE aux_inputs so far (handed whole to every tile);
    # TILE/ROW/COL slicing is defined in AuxKind but not yet implemented.
    MANUAL_TILE = "manual_tile"
    # TODO(future): actual backend


class GlobalInputTransform(enum.Enum):
    # no transform
    NONE = "none"
    # swap the axes before doing quantization
    SWAP_0_AND_1_AXES = "swap_0_and_1_axes"
    # both none and swap_0_and_1_axes
    BOTH_NONE_AND_SWAP_0_AND_1_AXES = "both_none_and_swap_0_and_1_axes"


class AuxKind(enum.Enum):
    """How a captured auxiliary input is presented to `f` per tile.

    Aux tensors are lifted out of `f`'s closure into explicit `aux_inputs` (cf. flex_gemm's
    epilogue arg kinds), and each carries a kind saying how the framework hands it to each tile.
    Only REPLICATE is implemented so far; TILE/ROW/COL are defined but raise NotImplementedError.
    """

    # whole tensor handed to every tile (e.g. a per-tensor scale). The only kind implemented.
    REPLICATE = "replicate"
    # leading dims match the input (M, N) grid; slice the matching sub-tile per tile.
    TILE = "tile"
    # (1, N): broadcast down the row (M) tiles.
    ROW = "row"
    # (M, 1): broadcast across the col (N) tiles.
    COL = "col"


def _resolve_aux_kinds(
    aux_inputs: Tuple[torch.Tensor, ...],
    aux_kinds: Tuple["AuxKind", ...] | None,
) -> Tuple["AuxKind", ...]:
    """Validate `aux_kinds` against `aux_inputs`, defaulting to REPLICATE, and reject
    kinds that are not yet implemented."""
    if aux_kinds is None:
        aux_kinds = (AuxKind.REPLICATE,) * len(aux_inputs)
    assert len(aux_kinds) == len(aux_inputs), (
        f"aux_kinds ({len(aux_kinds)}) must match aux_inputs ({len(aux_inputs)})"
    )
    for kind in aux_kinds:
        if kind is not AuxKind.REPLICATE:
            raise NotImplementedError(f"aux kind {kind} is not yet implemented")
    return aux_kinds


def _aux_for_tile(aux, kind):
    """Present a single aux tensor to one tile according to its `kind`.

    The single place per-tile aux slicing lives; today only REPLICATE (whole tensor to every
    tile) is implemented. TILE/ROW/COL slicing will slot in here, and will need to account for
    _global_input_transform=SWAP_0_AND_1_AXES (row<->col swap, cf. flex_gemm _SWAPPED_ARG_KIND).
    """
    if kind is AuxKind.REPLICATE:
        return aux
    raise NotImplementedError(f"aux kind {kind} is not yet implemented")


def flex_cast_quant(
    input: torch.Tensor,
    f: Callable,  # tile-invariant fn: (input, *aux_inputs) -> (out, *aux_out)
    *,
    aux_inputs: Tuple[torch.Tensor, ...] = (),  # auxiliary inputs
    aux_kinds: Tuple[AuxKind, ...] | None = None,  # for each aux input, specify how to broadcast it
    # these are needed for production quant, but final design TBD
    # TODO(future): we also need a way for a single fused kernel to write
    # out dim0 and dim1 quant at the same time
    _global_input_transform: GlobalInputTransform = GlobalInputTransform.NONE,
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
    independently to every tile of `input`. It has signature `(input, *aux_inputs) ->
    (out, *aux_out)` -- one primary output plus zero or more auxiliary outputs (e.g. a scale;
    a pure transform has none). Any tensors `f` needs beyond `input` (e.g. a per-tensor scale,
    an RHT matrix) are passed explicitly via `aux_inputs` rather than captured in a closure, so
    the framework can present each one per tile according to its `AuxKind`. Recipes needing
    multiple kernels call this for one kernel and call something else for the rest. This API is
    consumer-agnostic (no notion of weight/activation/KV/gradient) -- `f` owns all format
    knowledge.

    TODO(future work): inspect `f` and lower to an efficient (Helion/Triton) kernel that
    exploits tile-invariance (cf. flexquant_v2 tile_map).
    """

    assert len(input.shape) == 2, "only input of rank 2 is supported"

    assert (
        _global_input_transform is not GlobalInputTransform.BOTH_NONE_AND_SWAP_0_AND_1_AXES
    ), "GlobalInputTransform.BOTH_NONE_AND_SWAP_0_AND_1_AXES is not yet implemented"

    aux_kinds = _resolve_aux_kinds(aux_inputs, aux_kinds)

    if _global_input_transform is GlobalInputTransform.SWAP_0_AND_1_AXES:
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

        # one tile == whole tensor, so every aux is presented whole (REPLICATE).
        outs = f(input, *aux_inputs)

    elif _backend is FlexCastQuantBackend.MANUAL_TILE:
        outs = _manual_tile(
            input,
            f,
            aux_inputs,
            aux_kinds,
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
    aux_inputs: Tuple[torch.Tensor, ...] = (),
    aux_kinds: Tuple[AuxKind, ...] = (),
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
    # allow_non_fake_inputs=True is needed because aux_inputs are real cuda tensors (nvfp4
    # outer_scale, RHT matrix); meta inputs don't work (device mismatch / no RNG). The probe is
    # whole-tensor, so aux is presented whole (REPLICATE) here regardless of kind.
    with FakeTensorMode(allow_non_fake_inputs=True):
        fake_in = torch.empty(input.shape, dtype=input.dtype, device=input.device)
        fake_outs = f(fake_in, *aux_inputs)

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
            # present each aux to this tile per its kind (only REPLICATE implemented: the whole
            # aux tensor is handed to every tile).
            aux_tiles = [
                _aux_for_tile(aux, kind) for aux, kind in zip(aux_inputs, aux_kinds)
            ]
            outs = f(input_tile, *aux_tiles)

            for i, local in enumerate(outs):
                div0, div1 = divisors[i]
                # extent = this tile's own output shape, so ragged last tiles work.
                dst = (
                    slice(r // div0, r // div0 + local.shape[0]),
                    slice(c // div1, c // div1 + local.shape[1]),
                )  # trailing dims (e.g. 32,16) left fully selected
                final_outs[i][dst] = local

    return tuple(final_outs)
