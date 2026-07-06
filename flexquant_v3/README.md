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

`flex_cast_quant` has the signature:

```python
def flex_cast_quant(
    input: torch.Tensor,
    f: Callable,
    *,
    _swap_input_axes: bool = False,                          # transpose input on load
    _backend: FlexCastQuantBackend = FlexCastQuantBackend.REFERENCE,
) -> tuple[torch.Tensor, ...]:                               # (out, *aux) from `f`
    ...
```

It applies `f` to `input` under the chosen backend and returns whatever tuple `f`
produced. The `_`-prefixed args are for debugging/reference (backend selection, dim-M axis
swap) and default to the plain reference path.

## Example recipes (see `recipes.py`)

- **deepseek fp8** — 1x128, 128x128, and 1x128 dim-M (via `_swap_input_axes`).
- **mxfp8 FLOOR** — 1x32 blocks, e8m0 power-of-two scale; plain and swizzled (NVIDIA
  32x4x4 blocked scale layout).
- **float8 tensorwise** — global per-tensor scale computed outside, cast bound via factory.
- **nvfp4 with global scale** — two-level (per-tensor fp32 outer + per-16 e4m3 inner),
  fp4-packed, swizzled inner scale.
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
Therefore, we add a global input transformation option, currently a bool named `_swap_input_axes`.
This needs a better name, and we also would need support a third option for a single 
kernel to write out casts in both directions (dim0 and dim1).
3. rowwise scaling is not currently in here. We could either leave it out of scope or
add a concept of "tile that fully spans a dim".
4. we need to design how to configure replicate vs broadcast aux inputs, as this 
is not always recoverable from just `f`. For example, imagine a `[2, 2]` aux_input and a `[4, 4]`
tensor. Both "replicate" and "scatter across tiles" behavior makes sense and is semantically
different.

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
- **`MANUAL_TILE`** — splits the input into quadrants, runs `f` per tile, and recomposes.
  For a tile-invariant `f` it must match `REFERENCE` bit-for-bit; it's how we *check*
  tile-invariance. `_swap_input_axes=True` transposes the input on load (for dim-M recipes).

Both are debug/reference backends. `TODO`: lower `f` to a non-reference backend.

## Running the tests

```bash
cd flexquant_v3 && python -m pytest test.py -v   # requires CUDA
```
