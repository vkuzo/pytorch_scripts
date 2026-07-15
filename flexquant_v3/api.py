import enum
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode
from utils import _pad_to_multiple


class FlexTileMapBackend(enum.Enum):
    # for debugging, runs the callback on the entire tensor
    REFERENCE = "reference"
    # for debugging, manually tiles in 256x256 tiles and runs the callback on each
    MANUAL_TILE = "manual_tile"
    # TODO(future): actual backend


class OutputKind(enum.Enum):
    """How the framework places one of `f`'s outputs into the final tensor when tiling.

    NORMAL: a tile computed at grid position [m, n] writes to output grid position [m, n].
    SWAP_TILE_INDEX: writes to output grid position [n, m] instead -- a grid transpose of the
      tile-index only; the tile's CONTENTS are written as-is (NOT element-transposed).
    """

    NORMAL = "normal"
    SWAP_TILE_INDEX = "swap_tile_index"


class TileMustSpanDim(enum.Enum):
    """Which dim, if any, a tile must fully span.

    NONE: default 2D tiles (no dim must span).
    DIM0: tiles span all of dim0 (enables colwise: one scale per column,
      reduced over all rows).
    DIM1: tiles span all of dim1 (enables rowwise: one scale per row,
      reduced over all columns).
    """

    NONE = "none"
    DIM0 = "dim0"
    DIM1 = "dim1"


class AuxKind(enum.Enum):
    """How a captured auxiliary input is presented to `f` per tile.

    REPLICATE: pass aux_input as-is
    TILE: tile aux_input consistently with the main input
      - Example 1 (matching shapes)
        - input [M, N], aux_input [M, N]
        - tiling 2x2
        - tiled input [M // 2, N // 2], tiled aux_input [M // 2, N // 2]
      - Example 2 (shape of input is a multiple of the shape of aux_input)
        - input [M, N], aux_input [M // 2, N // 2]
        - tiling 2x2
        - tiled input [M // 2, N // 2], tiled aux_input [M // 4, N // 4]
    ROW: a per-row aux of shape exactly (M, 1); each tile gets its rows sliced (aux[r:r+bm, :])
      and the single column broadcasts across the tile's columns (e.g. a precalculated rowwise scale)
    COL: a per-column aux of shape exactly (1, N); each tile gets its columns sliced
      (aux[:, c:c+bn]) and the single row broadcasts across the tile's rows (e.g. a
      precalculated colwise scale)
    """

    REPLICATE = "replicate"
    TILE = "tile"
    ROW = "row"
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
        if kind not in (AuxKind.REPLICATE, AuxKind.TILE, AuxKind.ROW, AuxKind.COL):
            raise NotImplementedError(f"aux kind {kind} is not yet implemented")
    return aux_kinds


def _resolve_output_kinds(
    num_outputs: int,
    output_kinds: Tuple["OutputKind", ...] | None,
) -> Tuple["OutputKind", ...]:
    """Validate `output_kinds` against the number of outputs `f` returns, defaulting to NORMAL."""
    if output_kinds is None:
        output_kinds = (OutputKind.NORMAL,) * num_outputs
    assert len(output_kinds) == num_outputs, (
        f"output_kinds ({len(output_kinds)}) must match number of outputs ({num_outputs})"
    )
    return output_kinds


def _aux_for_tile(aux, kind, r, c, tile_shape, input_shape):
    """Present a single aux tensor to one tile according to its `kind`.

    The single place per-tile aux slicing lives. REPLICATE hands the whole aux to every tile;
    TILE slices the matching sub-region (the aux's leading two dims map to the input (M, N) grid
    at a per-dim ratio inferred from the shapes); ROW slices a (M, 1) per-row aux along dim0 and
    lets it broadcast across the tile's columns; COL slices a (1, N) per-column aux along dim1 and
    lets it broadcast across the tile's rows.

    TODO(future): a TILE aux combined with a SWAP_TILE_INDEX output would need the aux's
    row<->col roles swapped too (cf. flex_gemm _SWAPPED_ARG_KIND); not handled yet.
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
    if kind is AuxKind.ROW:
        # a per-row aux: shape (M, 1), exactly one value per input row, broadcast across columns.
        # Slice the tile's rows and leave the size-1 col dim to broadcast.
        bm, bn = tile_shape
        assert aux.shape == (input_shape[0], 1), (
            f"ROW aux must be exactly (M, 1) matching input rows, got {tuple(aux.shape)} "
            f"for input {tuple(input_shape)}"
        )
        return aux[r : r + bm, :]
    if kind is AuxKind.COL:
        # a per-column aux: shape (1, N), exactly one value per input column, broadcast across
        # rows. Slice the tile's columns and leave the size-1 row dim to broadcast.
        bm, bn = tile_shape
        assert aux.shape == (1, input_shape[1]), (
            f"COL aux must be exactly (1, N) matching input cols, got {tuple(aux.shape)} "
            f"for input {tuple(input_shape)}"
        )
        return aux[:, c : c + bn]
    raise NotImplementedError(f"aux kind {kind} is not yet implemented")


def flex_tile_map(
    input: torch.Tensor,
    f: Callable,
    *,
    aux_inputs: Tuple[torch.Tensor, ...] = (),
    aux_kinds: Tuple[AuxKind, ...] | None = None,
    output_kinds: Tuple[OutputKind, ...] | None = None,
    pad_input_to_multiple_of: Tuple[int, int] | None = None,
    tile_must_span_dim: TileMustSpanDim = TileMustSpanDim.NONE,
    tile_multiple_of: Tuple[int, int] | None = None,
    full_tile_multiple_of: Tuple[int, int] | None = None,
    _backend: FlexTileMapBackend = FlexTileMapBackend.REFERENCE,
) -> tuple[torch.Tensor, ...]:
    """Executes a user specified `f(input, *aux_inputs, **kwargs) -> (outputs)` in a 
    single kernel, tiled for efficient execution (the "single-kernel" part is not
    implemented yet as we only have debug backends).

    `f` must be *near-tile-invariant*. The *near* caveats are:
    1. `f` must place restrictions on tile size to ensure quantization 
        validity and correctness. Examples:
        a. `pad_input_to_multiple_of=(a, b)` for aligning input size with any static
            quantization block size
        b. `tile_must_span_dim=TileMustSpanDim.DIM1` for rowwise quantization in
            a single kernel. Note that this is incompatible with efficient gemm epilogue.
        c. `tile_multiple_of=(1, 32)` to ensure that a 1x32 quantization block size
            reduction does not span multiple tiles
        d. `full_tile_multiple_of=(128, 128)` to ensure that a mxfp8|nvfp4 scale
            swizzle does not span multiple tiles
    2. `f` can optionally take a tile's position in the parent tensor
        with `global_row, global_col, num_col` kwargs to implement tile-invariant 
        per-element randomness for stochastic rounding.
       

    Args:
        input: the 2D tensor to cast. Rank 2 only.
        f: the tile-invariant function described above.
        aux_inputs: extra tensors `f` needs beyond `input` (e.g. a per-tensor scale, an RHT
            matrix, a per-element bias)
        aux_kinds: control how each tensor in `aux_inputs` is passed to a tile:
            - AuxKind.REPLICATE
            - AuxKind.TILE
            - AuxKind.ROW
            - AuxKind.COL
        output_kinds: control how each output tile is written to the respective output tensor
            - OutputKind.NORMAL - written as-is
            - OutputKind.SWAP_TILE_INDEX - 2D transpose the tiles of the output tensor on write.
                This allows the user to express a global 2D transpose (`tensor.t()`) with a 
                composition of
                    (i) a tile-local transpose (`tile.t()` inside of `f`), and 
                    (ii) a transpose of each tile as a whole with respect to the parent tensor:
                    given output shape [M, N], output tile shape [m, n] and output tile 
                    global index [m_start, n_start], write the [m, n] tensor to the
                    2d region starting in [n_start, m_start].
        pad_input_to_multiple_of: if given, `(mult0, mult1)` -- zero-pad each `input` dim up to a
            multiple of `mult{0,1}` on load (for ragged shapes, e.g. LLM decode/prefill token
            dims). Outputs come back at the padded shape. `None` (default) does no padding.
        tile_must_span_dim: which dim, if any, a tile must fully span
            - TileMustSpanDim.NONE (default) - no restrictions
            - TileMustSpanDim.DIM0 - given input shape [M, N], each tile must be shaped [M, {anything}]
            - TileMustSpanDim.DIM1 - given input shape [M, N], each tile must be shaped [{anything}, N]
        tile_multiple_of: if given, `(mult0, mult1)` -- assert every tile's size is a multiple of
            `mult{0,1}` (a per-dim block/reduction granularity `f` requires). `None` skips the check.
        full_tile_multiple_of: like `tile_multiple_of` but applies only to full (non-edge) tiles
            (e.g. the 128x128 swizzle atom, which edge/remainder tiles are exempt from).
        _backend: debug backend. 
            REFERENCE runs `f` on the whole tensor without tiling
            MANUAL_TILE is a debug backend that tiles the input in tiles of 256x256
            We don't have a real backend yet

    Returns:
        The tuple `f` produced (`(out, *aux_out)`), assembled over the whole tensor.

    Example:
        deepseek fp8 1x128 (reduce over 128-element groups along the last dim, one fp8 tensor
        + one fp32 scale per group). `f` is written for a single tile; the framework applies it
        to every tile.

        >>> def deepseek_1x128_f(x, **kwargs):
        ...     fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
        ...     *lead, last = x.shape
        ...     x_b = x.reshape(*lead, last // 128, 128)
        ...     amax = x_b.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12).to(torch.float32)
        ...     scale = amax / fp8_max
        ...     qdata = (x_b.to(torch.float32) / scale).to(torch.float8_e4m3fn)
        ...     return qdata.reshape(*lead, last), scale.squeeze(-1)
        >>>
        >>> x = torch.randn(1024, 512, dtype=torch.bfloat16, device="cuda")
        >>> # tile_multiple_of=(1, 128): keep each 1x128 reduction group inside one tile.
        >>> qdata, scale = flex_tile_map(x, deepseek_1x128_f, tile_multiple_of=(1, 128))
        >>> qdata.shape, scale.shape
        (torch.Size([1024, 512]), torch.Size([1024, 4]))

    TODO(future work): verify we want this, align with flex_gemm|flex_ep|flex_moe, build a backend
    """

    assert len(input.shape) == 2, "only input of rank 2 is supported"

    aux_kinds = _resolve_aux_kinds(aux_inputs, aux_kinds)

    if _backend is FlexTileMapBackend.REFERENCE:

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

    elif _backend is FlexTileMapBackend.MANUAL_TILE:
        outs = _manual_tile(
            input,
            f,
            aux_inputs,
            aux_kinds,
            output_kinds,
            pad_input_to_multiple_of,
            tile_must_span_dim,
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
    output_kinds: Tuple[OutputKind, ...] | None = None,
    pad_input_to_multiple_of: Tuple[int, int] | None = None,
    tile_must_span_dim: TileMustSpanDim = TileMustSpanDim.NONE,
    tile_multiple_of: Tuple[int, int] | None = None,
    full_tile_multiple_of: Tuple[int, int] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Tile `input`, run `f` on each tile, and recompose (debug backend). Uses 2D tiles by
    default; `tile_must_span_dim` can make a dim span the tensor (for rowwise/colwise reductions).
    """
    assert input.ndim == 2, f"MANUAL_TILE expects a 2D input, got {input.ndim}D"

    # pad input first, so the whole pipeline (constraint asserts, fake-probe output shapes,
    # preallocation, tiling offsets) all key off the padded shape.
    # TODO(future): move this inside the tiling logic
    if pad_input_to_multiple_of is not None:
        input = _pad_to_multiple(input, pad_input_to_multiple_of)

    # choose tile size: 256 per dim, except a dim that must span the tensor uses the full
    # (post-pad) extent, so `f`'s reduction over that whole dim is never severed by tiling.
    # TODO(future): make the 256 configurable.
    tile_size_0 = input.shape[0] if tile_must_span_dim is TileMustSpanDim.DIM0 else 256
    tile_size_1 = input.shape[1] if tile_must_span_dim is TileMustSpanDim.DIM1 else 256

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

    output_kinds = _resolve_output_kinds(len(fake_outs), output_kinds)

    # 3. per-output (dim0, dim1) divisors: the input->output tiling ratio, constant across
    # tiles, so a tile at input offset (r, c) writes to output offset (r//div0, c//div1).
    # Only dims 0,1 are tiled; trailing dims (e.g. the swizzle grid's 32,16) are copied whole.
    # For a SWAP_TILE_INDEX output the grid is transposed -- output dim0 corresponds to the
    # INPUT's dim1 and vice versa -- so the divisor is taken against the swapped input dims.
    divisors = []
    for fo, okind in zip(fake_outs, output_kinds):
        # scalar/replicated outputs (e.g. tensorwise per-tensor scale) are not supported
        # under MANUAL_TILE: every output must be tiled along dims 0 and 1.
        assert fo.ndim >= 2, (
            f"MANUAL_TILE requires outputs tiled on dims 0,1, got shape {tuple(fo.shape)}"
        )
        # input dim that maps to each output dim (swapped for SWAP_TILE_INDEX).
        in_dims = (1, 0) if okind is OutputKind.SWAP_TILE_INDEX else (0, 1)
        d = []
        for out_dim in range(2):
            in_dim = in_dims[out_dim]
            assert input.shape[in_dim] % fo.shape[out_dim] == 0
            d.append(input.shape[in_dim] // fo.shape[out_dim])
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
                # tile-grid offset per output dim: NORMAL -> (r, c); SWAP_TILE_INDEX writes tile
                # grid [m,n] to output grid [n,m], so the input row/col roles swap -> (c, r).
                # extent = this tile's own output shape, so ragged last tiles work.
                off0, off1 = (c, r) if output_kinds[i] is OutputKind.SWAP_TILE_INDEX else (r, c)
                dst = (
                    slice(off0 // div0, off0 // div0 + local.shape[0]),
                    slice(off1 // div1, off1 // div1 + local.shape[1]),
                )  # trailing dims (e.g. 32,16) left fully selected
                final_outs[i][dst] = local

    return tuple(final_outs)
