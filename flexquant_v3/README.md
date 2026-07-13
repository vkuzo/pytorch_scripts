# flexquant_v3

A single API for one-kernel, tile-invariant tensor casts (quantization and friends).

## The idea

```python
# f can close over auxiliary inputs
out, *aux = flex_cast_quant(input, f)
```

`f` is a **tile-invariant** function — the same per-tile computation applied independently
to every tile of `input`. It returns one primary output plus zero or more auxiliary outputs
(e.g. a scale). `f` owns all the format knowledge.

This API is for a single kernel. User is responsible for composing multiple kernels into
a quant recipe that requires multiple kernels (global outer scale, etc).

`f` has the signature:

```python
def f(tile: torch.Tensor) -> tuple[torch.Tensor, ...]:
    # (out,)            -- a plain transform (e.g. Hadamard, stochastic rounding)
    # (out, scale)      -- a quant cast (e.g. deepseek fp8, mxfp8, nvfp4)
    ...
```

It takes one tensor (a tile, or the whole input in the `REFERENCE` backend) and returns a
tuple: the primary output first, then zero or more auxiliary outputs. To depend on values
computed outside the kernel (a global scale, a sign vector, an RNG seed), build `f` from a
factory that closes over them — e.g. `make_nvfp4_gs_swizzle_recipe(outer_scale)`.

Requirements on all outputs of `f`: must be at least 2d, and the first two dimensions
must directly correspond to the two input dimensions.

`flex_cast_quant` has the signature:

```python
def flex_cast_quant(
    input: torch.Tensor,
    f: Callable,
    *,
    _global_input_transform: GlobalInputTransform = GlobalInputTransform.NONE,  # e.g. transpose on load
    _pad_input_to_multiple_of: Tuple[int, int] | None = None,  # zero-pad ragged dims up on load
    _tile_multiple_of: Tuple[int, int] | None = None,          # tile-size divisibility constraint
    _inner_tile_multiple_of: Tuple[int, int] | None = None,    # swizzle-atom constraint
    _backend: FlexCastQuantBackend = FlexCastQuantBackend.REFERENCE,
) -> tuple[torch.Tensor, ...]:                               # (out, *aux) from `f`
    ...
```

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
- **float8 tensorwise** — global per-tensor scale computed outside, cast bound via factory.
- **nvfp4 with global scale** — two-level (per-tensor fp32 outer + per-16 e4m3 inner),
  fp4-packed, swizzled inner scale (same 4D block grid).
- **randomized Hadamard (RHT)** — a non-quant transform (bf16 in, bf16 out, no scale).
- **stochastic rounding fp32 -> bf16** — unbiased rounding; not tile-invariant by design.

## Design questions to work through (in flux)

1. to properly implement stochastic rounding, we need a per-element random number. This is
usually created using a per-tensor seed + per-tile offset. We need to expose the per-tile offset
(or equivalent information) to `f` to properly implement this inside of `f`. The current
stochastic rounding example punts this to a TODO: it is unbiased (E[SR(x)] ~= x), but the
tile-local offset repeats per tile, so draws are correlated across tiles -- not
statistically sound under tiling.
2. for training, we often need to do "x.t().contiguous().quant_with_recipe(...)". This is not
expressible as a tile invariant function of `x`, as `x.t().contiguous()` is a global transform.
Therefore, we add a global input transformation option, the `_global_input_transform`
(`GlobalInputTransform`) enum: `NONE` or `SWAP_0_AND_1_AXES`. The third option,
`BOTH_NONE_AND_SWAP_0_AND_1_AXES` -- a single kernel writing casts in both directions (dim0 and
dim1) -- is defined but not yet implemented (asserts out).
3. rowwise scaling is not currently in here. We could either leave it out of scope or
add a concept of "tile that fully spans a dim".
4. we need to design how to configure replicate vs broadcast aux inputs, as this 
is not always recoverable from just `f`. For example, imagine a `[2, 2]` aux_input and a `[4, 4]`
tensor. Both "replicate" and "scatter across tiles" behavior makes sense and is semantically
different.
5. **swizzled scale layout (partially resolved).** The NVIDIA blocked scale layout is a
global, grid-shape-dependent *serialization*: laying the 128x4-scale atoms out into the
flat buffer `_scaled_mm` consumes bakes the (row-block, col-block) walk order into the
result, so composing it inside a tile-invariant `f` breaks under a column split (and under
any non-128-aligned tile split). Fix: `f` now emits the swizzle as a 4D block grid
`(n_row_blocks, n_col_blocks, 32, 16)` -- the per-atom swizzle is local/tile-invariant and
the two block axes stay separate, so MANUAL_TILE reassembles a column tile with `cat(dim=1)`
and a row tile with `cat(dim=0)`, bit-exact. The grid `.reshape(-1)` still equals torchao's
`to_blocked` buffer; that final serialization is a global step done ONCE outside `f`, after
tiles are reassembled. STILL OPEN: tiles must be whole 128x4 atoms (rows a multiple of 128,
cols a multiple of the block width) -- a non-atom-aligned split still pads partial blocks
independently and diverges. That alignment contract is unenforced (see the two 384-row
tests in test.py: aligned split is invariant, the default quadrant split at 192 is not).

## Missing pieces

* a real backend, for not we just have reference backends.
* proper testing (currently no edge cases are tested)

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
