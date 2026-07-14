# flexquant_v3

A single API for one-kernel, tile-invariant tensor casts (quantization and friends).

## The idea

```python
# auxiliary inputs (a global scale, an RHT matrix) are passed explicitly, not closed over
out, *aux = flex_cast_quant(input, f, aux_inputs=(outer_scale,), aux_kinds=(AuxKind.REPLICATE,))
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

`flex_cast_quant` has the signature:

```python
def flex_cast_quant(
    input: torch.Tensor,
    f: Callable,
    *,
    aux_inputs: Tuple[torch.Tensor, ...] = (),
    aux_kinds: Tuple[AuxKind, ...] | None = None,              # per-aux broadcast kind (None => REPLICATE)
    _global_input_transform: GlobalInputTransform = GlobalInputTransform.NONE,  # e.g. transpose on load
    _pad_input_to_multiple_of: Tuple[int, int] | None = None,  # zero-pad ragged dims up on load
    _tile_multiple_of: Tuple[int, int] | None = None,          # tile-size divisibility constraint
    _inner_tile_multiple_of: Tuple[int, int] | None = None,    # swizzle-atom constraint
    _backend: FlexCastQuantBackend = FlexCastQuantBackend.REFERENCE,
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

`_global_input_transform` selects a global, pre-tiling load transform (a `GlobalInputTransform`
enum): `NONE` (default) or `SWAP_0_AND_1_AXES` (transpose the input on load, for dim-M recipes).
`BOTH_NONE_AND_SWAP_0_AND_1_AXES` is reserved for writing both dim0 and dim1 casts from one kernel
and is not yet implemented.

It applies `f` to `input` under the chosen backend and returns whatever tuple `f`
produced. The `_`-prefixed args are for debugging/reference (backend selection, dim-M axis
swap via `_global_input_transform`, input padding, tiling constraints) and default to the
plain reference path.

## Example recipes (see `recipes.py`)

- **deepseek fp8** — 1x128, 128x128, and 1x128 dim-M (via `_global_input_transform=SWAP_0_AND_1_AXES`).
- **mxfp8 FLOOR** — 1x32 blocks, e8m0 power-of-two scale; plain and swizzled (NVIDIA
  32x4x4 blocked scale layout). The swizzled scale is emitted as a 4D block grid
  `(n_row_blocks, n_col_blocks, 32, 16)` (see below).
- **float8 tensorwise** — global per-tensor scale computed outside, passed as a REPLICATE aux input.
- **nvfp4 with global scale** — two-level (per-tensor fp32 outer + per-16 e4m3 inner),
  fp4-packed, swizzled inner scale (same 4D block grid).
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
* AuxKind.TILE combined with GlobalInputTransform.SWAP_0_AND_1_AXES (row<->col swap of the aux)
* GlobalInputTransform.BOTH_NONE_AND_SWAP_0_AND_1_AXES is not implemented yet
* a real backend, for not we just have reference backends
* proper testing (currently not many edge cases are tested)

## Out of scope

* dynamic rowwise scaling (this needs 1d tiles to efficiently implement, not worth
  the complexity to start).

## Files

| File | Contents |
|------|----------|
| `api.py` | `flex_cast_quant` + `FlexCastQuantBackend` (`REFERENCE`, `MANUAL_TILE`). |
| `recipes.py` | The `Recipe(quant, dequant)` dataclass and all example `f` recipes. |
| `utils.py` | Sub-byte fp4 (e2m1) conversion + 4-bit packing helpers. |
| `test.py` | Numerical tests: reference-vs-backend, SQNR, and per-recipe properties. |

## Backends

- **`REFERENCE`** — runs `f` on the whole tensor. The correctness oracle.
- **`MANUAL_TILE`** — splits the input into 256x256 tiles, runs `f` per tile, and scatters
  each tile's outputs into preallocated buffers. For a tile-invariant `f` it must match
  `REFERENCE` bit-for-bit; it's how we *check* tile-invariance.
  `_global_input_transform=SWAP_0_AND_1_AXES` transposes the input on load (for dim-M recipes).

Both are debug/reference backends. `TODO`: lower `f` to a non-reference backend.

## Running the tests

```bash
cd flexquant_v3 && python -m pytest test.py -v   # requires CUDA
```
