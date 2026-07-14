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
    # Note: supports AuxKind.REPLICATE (whole aux to every tile) and AuxKind.TILE (per-tile
    # sub-region); ROW/COL are defined in AuxKind but not yet implemented.
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
    REPLICATE and TILE are implemented; ROW/COL are defined but raise NotImplementedError.
    """

    # whole tensor handed to every tile (e.g. a per-tensor scale).
    REPLICATE = "replicate"
    # leading dims match the input (M, N) grid; slice the matching sub-tile per tile
    # (e.g. a 128x128-blocked scale, or a same-shape bias). `f` block-broadcasts it.
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
        if kind not in (AuxKind.REPLICATE, AuxKind.TILE):
            raise NotImplementedError(f"aux kind {kind} is not yet implemented")
    return aux_kinds


def _aux_for_tile(aux, kind, r, c, tile_shape, input_shape):
    """Present a single aux tensor to one tile according to its `kind`.

    The single place per-tile aux slicing lives. REPLICATE hands the whole aux to every tile;
    TILE slices the matching sub-region (the aux's leading two dims map to the input (M, N) grid
    at a per-dim ratio inferred from the shapes). ROW/COL are not implemented yet.

    TODO(future): TILE + global_input_transform=SWAP_0_AND_1_AXES needs the aux transposed on
    load / row<->col swap (cf. flex_gemm _SWAPPED_ARG_KIND); not handled yet.
    """
    if kind is AuxKind.REPLICATE:
        return aux
    if kind is AuxKind.TILE:
        bm, bn = tile_shape
        assert input_shape[0] % aux.shape[0] == 0 and input_shape[1] % aux.shape[1] == 0, (
            f"TILE aux dims {tuple(aux.shape[:2])} must evenly divide input dims "
            f"{tuple(input_shape)}"
        )
        # input-elements-per-aux-element per dim (e.g. 1 for a same-shape bias, 128 for a
        # 128x128-blocked scale). The tile at input offset (r, c) reads the sub-region
        # aux[r//div0 : (r+bm)//div0, ...]; `f` block-broadcasts it back over the data.
        div0 = input_shape[0] // aux.shape[0]
        div1 = input_shape[1] // aux.shape[1]
        assert r % div0 == 0 and bm % div0 == 0 and c % div1 == 0 and bn % div1 == 0, (
            f"tile ({r}:{r+bm}, {c}:{c+bn}) must align to TILE aux block ({div0}, {div1})"
        )
        return aux[r // div0 : (r + bm) // div0, c // div1 : (c + bn) // div1]
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
    global_input_transform: GlobalInputTransform = GlobalInputTransform.NONE,
    # input padding
    pad_input_to_multiple_of: Tuple[int, int] | None = None,
    # restrictions on tiling
    tile_multiple_of: Tuple[int, int] | None = None,
    full_tile_multiple_of: Tuple[int, int] | None = None,
    # kwargs below are only for debugging
    _backend: FlexCastQuantBackend = FlexCastQuantBackend.REFERENCE,
) -> tuple[torch.Tensor, ...]:
    """Single-kernel quantization API.

    `f` must be *tile-invariant*: the same pointwise/blockwise computation applied
    independently to every tile of `input`. It has signature `(input, *aux_inputs, global_row,
    global_col, num_col) -> (out, *aux_out)` -- one primary output plus zero or more auxiliary
    outputs (e.g. a scale; a pure transform has none). Any tensors `f` needs beyond `input`
    (e.g. a per-tensor scale, an RHT matrix) are passed explicitly via `aux_inputs` rather than
    captured in a closure, so the framework can present each one per tile according to its
    `AuxKind`. The framework ALWAYS passes the tile's global position as keyword args --
    `global_row`/`global_col` (the tile's origin in the full tensor) and `num_col` (the full row
    stride) -- so a recipe can key per-element behavior on global position (e.g. tiling-invariant
    stochastic rounding); recipes that don't need position absorb them with `**kwargs`. Recipes
    needing multiple kernels call this for one kernel and call something else for the rest. This
    API is consumer-agnostic (no notion of weight/activation/KV/gradient) -- `f` owns all format
    knowledge.

    TODO(future work): inspect `f` and lower to an efficient (Helion/Triton) kernel that
    exploits tile-invariance (cf. flexquant_v2 tile_map).
    """

    assert len(input.shape) == 2, "only input of rank 2 is supported"

    assert (
        global_input_transform is not GlobalInputTransform.BOTH_NONE_AND_SWAP_0_AND_1_AXES
    ), "GlobalInputTransform.BOTH_NONE_AND_SWAP_0_AND_1_AXES is not yet implemented"

    aux_kinds = _resolve_aux_kinds(aux_inputs, aux_kinds)

    if global_input_transform is GlobalInputTransform.SWAP_0_AND_1_AXES:
        input = input.t().contiguous()

    if _backend is FlexCastQuantBackend.REFERENCE:

        # pad input (before the constraint asserts, so they validate the padded shape)
        if pad_input_to_multiple_of is not None:
            input = _pad_to_multiple(input, pad_input_to_multiple_of)

        # verify that tile constraints are respected, if specified
        if tile_multiple_of is not None:
            M, K = input.shape
            assert M % tile_multiple_of[0] == 0
            assert K % tile_multiple_of[1] == 0

        # full_tile_multiple_of is satisfied automatically as there is only
        # one tile

        # one tile == whole tensor, so every aux is presented whole (REPLICATE) and the tile
        # origin is (0, 0). num_col is the (post-swap, post-pad) row stride.
        outs = f(input, *aux_inputs, global_row=0, global_col=0, num_col=input.shape[1])

    elif _backend is FlexCastQuantBackend.MANUAL_TILE:
        outs = _manual_tile(
            input,
            f,
            aux_inputs,
            aux_kinds,
            pad_input_to_multiple_of,
            tile_multiple_of,
            full_tile_multiple_of,
        )

    else:
        raise AssertionError(f"unknown {_backend=}")

    return outs


def _manual_tile(
    input: torch.Tensor,
    f: Callable,
    aux_inputs: Tuple[torch.Tensor, ...] = (),
    aux_kinds: Tuple[AuxKind, ...] = (),
    pad_input_to_multiple_of: Tuple[int, int] | None = None,
    tile_multiple_of: Tuple[int, int] | None = None,
    full_tile_multiple_of: Tuple[int, int] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Tile `input` into 256x256 tiles, run `f` on each, and recompose (debug backend).
    """
    assert input.ndim == 2, f"MANUAL_TILE expects a 2D input, got {input.ndim}D"

    # pad input first, so the whole pipeline (constraint asserts, fake-probe output shapes,
    # preallocation, tiling offsets) all key off the padded shape.
    if pad_input_to_multiple_of is not None:
        input = _pad_to_multiple(input, pad_input_to_multiple_of)

    # choose tile size if not specified
    # TODO(future): make this configurable
    tile_size_0, tile_size_1 = 256, 256

    # verify tiling constraints are honored
    if tile_multiple_of is not None:
        assert tile_size_0 % tile_multiple_of[0] == 0
        assert tile_size_1 % tile_multiple_of[1] == 0
        assert input.shape[0] % tile_multiple_of[0] == 0
        assert input.shape[1] % tile_multiple_of[1] == 0
    if full_tile_multiple_of is not None:
        assert tile_size_0 % full_tile_multiple_of[0] == 0
        assert tile_size_1 % full_tile_multiple_of[1] == 0

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
        # whole-shape probe: position doesn't affect output shape, use the origin.
        fake_outs = f(fake_in, *aux_inputs, global_row=0, global_col=0, num_col=input.shape[1])

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
            # present each aux to this tile per its kind: REPLICATE hands the whole tensor,
            # TILE slices the matching sub-region (input_tile.shape gives ragged-correct extent).
            aux_tiles = [
                _aux_for_tile(aux, kind, r, c, input_tile.shape, input.shape)
                for aux, kind in zip(aux_inputs, aux_kinds)
            ]
            # pass the tile's global origin (r, c) and the full row stride, so a recipe can key
            # per-element randomness on global position (tiling-invariant). num_col is the FULL
            # (post-swap) width -- consecutive rows are num_col apart, not tile-width apart.
            outs = f(input_tile, *aux_tiles, global_row=r, global_col=c, num_col=input.shape[1])

            for i, local in enumerate(outs):
                div0, div1 = divisors[i]
                # extent = this tile's own output shape, so ragged last tiles work.
                dst = (
                    slice(r // div0, r // div0 + local.shape[0]),
                    slice(c // div1, c // div1 + local.shape[1]),
                )  # trailing dims (e.g. 32,16) left fully selected
                final_outs[i][dst] = local

    return tuple(final_outs)
