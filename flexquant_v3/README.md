# flexquant_v3

A single API for one-kernel, tile-invariant tensor casts (quantization and friends).

## The idea

```python
# auxiliary inputs (a global scale, an RHT matrix) are passed explicitly, not closed over
out, *aux = flex_quant_cast(input, f, aux_inputs=(outer_scale,), aux_kinds=(AuxKind.REPLICATE,))
```

`f` is a **tile-invariant** function — the same per-tile computation applied independently
to every tile of `input`. It returns one primary output plus zero or more auxiliary outputs
(e.g. a scale). `f` owns all the format knowledge.

This API is for a single kernel. User is responsible for composing multiple kernels into
a quant recipe that requires multiple kernels (global outer scale, etc).

`f` has the signature:

```python
def f(tile: torch.Tensor, *aux_inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    # (out,)            -- a plain transform (e.g. Hadamard, stochastic rounding)
    # (out, scale)      -- a quant cast (e.g. deepseek fp8, mxfp8, nvfp4)
    ...
```

It takes the tile (or the whole input in the `REFERENCE` backend) followed by any auxiliary
input tensors, and returns a tuple: the primary output first, then zero or more auxiliary
outputs. Tensors computed outside the kernel (a global scale, an RHT matrix, a PRNG key) are
**lifted to explicit `aux_inputs`** rather than closed over, so the framework can present each
one to every tile according to its `AuxKind` (see below). E.g. stochastic rounding takes a
`torch.func._random` key (`prng.key(seed)`) as a REPLICATE aux input.

Requirements on all outputs of `f`: must be at least 2d, and the first two dimensions
must directly correspond to the two input dimensions.

`flex_quant_cast` has the signature:

```python
def flex_quant_cast(
    input: torch.Tensor,
    f: Callable,
    *,
    aux_inputs: Tuple[torch.Tensor, ...] = (),
    aux_kinds: Tuple[AuxKind, ...] | None = None,              # per-aux broadcast kind (None => REPLICATE)
    output_kinds: Tuple[OutputKind, ...] | None = None,        # per-output placement (None => all NORMAL)
    pad_input_to_multiple_of: Tuple[int, int] | None = None,  # zero-pad ragged dims up on load
    tile_must_span_dim: TileMustSpanDim = TileMustSpanDim.NONE,             # or DIM0/DIM1 (colwise/rowwise)
    tile_multiple_of: Tuple[int, int] | None = None,          # tile-size divisibility constraint
    full_tile_multiple_of: Tuple[int, int] | None = None,    # swizzle-atom constraint
    _backend: FlexQuantCastBackend = FlexQuantCastBackend.REFERENCE,
) -> tuple[torch.Tensor, ...]:                               # (out, *aux) from `f`
    ...
```

`aux_inputs` are tensors `f` needs beyond `input` (a global scale, an RHT matrix, a
128x128-blocked scale, a per-element bias), passed positionally to `f` after `input`. `aux_kinds`
tags each with an `AuxKind` saying how the framework presents it per tile: `REPLICATE` (hand the
whole tensor to every tile), or `TILE` (the aux's leading two dims map to the input (M, N) grid at
a ratio inferred from the shapes; the framework slices the matching sub-region per tile and `f`
block-broadcasts it — divisor 1 = a per-element bias, divisor 128 = a 128x128-blocked scale;
tiles must align to the aux block). `ROW`/`COL` are defined but not yet implemented.
`aux_kinds=None` defaults every aux to `REPLICATE`.

`output_kinds` tags each output of `f` with an `OutputKind` saying how the framework places it into
the final tensor when tiling: `NORMAL` (tile grid `[m,n]` -> output grid `[m,n]`) or
`SWAP_TILE_INDEX` (tile grid `[m,n]` -> output grid `[n,m]`, a grid-index transpose only — the
tile's contents are written as-is, NOT element-transposed). `SWAP_TILE_INDEX` is how orientation is
expressed: `f` does the within-tile transpose (e.g. dim-M recipes transpose their tile then reduce
the last dim) and the flag does the grid transpose, since `full_transpose = grid_transpose o
within_tile_transpose`. `output_kinds=None` defaults every output to `NORMAL`.

`tile_must_span_dim` (a `TileMustSpanDim`) controls which dim (if any) a tile must fully span:
`NONE` (default, 2D tiles), `DIM0` (dim0 spans — for colwise reductions), or `DIM1` (dim1 spans
— for rowwise reductions). A recipe whose reduction covers a whole dimension (rowwise/colwise) is
only tile-invariant under the matching spanning mode, so no tile ever severs that reduction.

It applies `f` to `input` under the chosen backend and returns whatever tuple `f`
produced. The `_`-prefixed args are for debugging/reference; the rest (aux, output placement,
input padding, tiling mode/constraints) default to the plain reference path.

## Example recipes (see `recipes.py`)

- **deepseek fp8** — 1x128, 128x128, and 1x128 dim-M (`f` transposes its tile + `output_kinds=SWAP_TILE_INDEX`).
- **mxfp8 FLOOR** — 1x32 blocks, e8m0 power-of-two scale; plain and swizzled (NVIDIA
  32x4x4 blocked scale layout). The swizzled scale is emitted as a 4D block grid
  `(n_row_blocks, n_col_blocks, 32, 16)` (see below).
- **float8 tensorwise** — global per-tensor scale computed outside, passed as a REPLICATE aux input.
- **nvfp4 with global scale** — two-level (per-tensor fp32 outer + per-16 e4m3 inner),
  fp4-packed, swizzled inner scale (same 4D block grid).
- **rowwise / colwise fp8** — one scale per row (`tile_must_span_dim=DIM1`) or per column
  (`DIM0`); the spanning tile keeps the full-dim reduction intact.
- **randomized Hadamard (RHT)** — a non-quant transform (bf16 in, bf16 out, no scale).
- **stochastic rounding fp32 -> bf16** — unbiased rounding; not tile-invariant by design.

## Design questions to work through (in flux)

1. **(resolved)** to properly implement stochastic rounding, we need a per-element random number
keyed on the element's GLOBAL position (so draws don't shift with tiling). The framework now
ALWAYS passes each tile's global position to `f` as keyword args -- `global_row`, `global_col`
(the tile's origin in the full tensor) and `num_col` (the full row stride) -- and `sr_bf16_global_f`
uses them to build a per-element Philox key `[seed, (global_row+i)*num_col + (global_col+j)]`,
making it tiling-invariant (REFERENCE == MANUAL_TILE bit-for-bit). The original `sr_bf16_f` is kept
as the tile-local, NOT-invariant counterexample. Recipes that don't need position absorb the kwargs
via `**kwargs`.

## Missing pieces

* aux input broadcasting along rows or columns (AuxKind.ROW/COL; REPLICATE and TILE are done)
* AuxKind.TILE combined with an OutputKind.SWAP_TILE_INDEX output (row<->col swap of the aux)
* quantizing both dim0 and dim1 from one call (expressible via per-output SWAP_TILE_INDEX on the
  dim0 outputs + possibly a second `f`, but not implemented yet)
* a real backend, for not we just have reference backends
* more edge case testing

## Files

| File | Contents |
|------|----------|
| `api.py` | `flex_quant_cast` + `FlexQuantCastBackend` (`REFERENCE`, `MANUAL_TILE`). |
| `recipes.py` | The `Recipe(quant, dequant)` dataclass and all example `f` recipes. |
| `utils.py` | Sub-byte fp4 (e2m1) conversion + 4-bit packing helpers. |
| `test.py` | Numerical tests: reference-vs-backend, SQNR, and per-recipe properties. |

## Running the tests

```bash
cd flexquant_v3 && python -m pytest test.py -v   # requires CUDA
```
