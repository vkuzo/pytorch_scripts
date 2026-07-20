"""Quant recipes for flex_tile_map, bundled as `Recipe` dataclasses.

Each recipe pairs a plain-PyTorch quant kernel `quant(x) -> (qdata, aux_out)` (v1
`_reference_fn` style) with its `dequant(qdata, scale) -> fp32` inverse. The `RECIPES`
table registers them (plus per-recipe test metadata) for the tests in test.py. Math
mirrors flexquant v1 recipes.py.
"""

import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import torch
import torch.func._random as prng

from api import AuxKind, OutputKind

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_cast_gold.recipes import (
    colwise_precalc_scale,
    ColwiseFp8Gold,
    ColwisePrecalcGold,
    Deepseek1x128DimMGold,
    Deepseek1x128Gold,
    Deepseek128x128Gold,
    float8_tensorwise_scale,
    Float8TensorwiseGold,
    Mxfp8FloorGold,
    Mxfp8FloorSwizzleGold,
    Mxfp8BiasGold,
    nvfp4_blocked_outer_scale,
    nvfp4_gs_scale,
    Nvfp4BlockedOuterGold,
    Nvfp4GsSwizzleGold,
    QuantCastSingleKernelGold,
    rowwise_precalc_scale,
    RowwiseFp8Gold,
    RowwisePrecalcGold,
)


@dataclass(frozen=True)
class RecipeV2(QuantCastSingleKernelGold):
    """A flexquant_v3 recipe backed directly by a quant_cast_gold golden recipe.

    Inherits `pt_ref_fn`/`correctness_fn` from `QuantCastSingleKernelGold` unchanged --
    flexquant_v3 adds the things gold doesn't know about: the flex_tile_map tiling
    constraint, and (for recipes whose `pt_ref_fn` takes precomputed aux args beyond `x`,
    e.g. a precalculated per-row/per-col scale) how to build and pass those aux inputs.
    Recipes migrate here incrementally (see quant_cast_gold/recipes.py); `Recipe` above
    remains for not-yet-migrated recipes.
    """

    valid_tile_size_fn: Callable[
        [Tuple[int, int], Tuple[int, int], Tuple[int, int]], bool
    ] | None = None
    # (x) -> tuple of precomputed aux tensors (e.g. a precalculated scale), computed OUTSIDE
    # flex_tile_map and passed as `aux_inputs`. None => pt_ref_fn takes only `x`.
    aux_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, ...]] | None = None
    aux_kinds: Tuple[Any, ...] | None = None  # AuxKind per aux_fn() output
    output_kinds: Tuple[Any, ...] | None = None  # OutputKind per pt_ref_fn() output

    @classmethod
    def from_gold(
        cls,
        gold: QuantCastSingleKernelGold,
        valid_tile_size_fn=None,
        aux_fn=None,
        aux_kinds=None,
        output_kinds=None,
    ) -> "RecipeV2":
        """Build a RecipeV2 from a gold recipe, adding the tiling constraint and (if the
        recipe needs it) how to build its precomputed aux inputs."""
        return cls(
            pt_ref_fn=gold.pt_ref_fn,
            correctness_fn=gold.correctness_fn,
            valid_tile_size_fn=valid_tile_size_fn,
            aux_fn=aux_fn,
            aux_kinds=aux_kinds,
            output_kinds=output_kinds,
        )


# All single-kernel quant recipes have migrated to quant_cast_gold (see
# quant_cast_gold/recipes.py); their Gold objects (and precompute helpers like
# `nvfp4_blocked_outer_scale`) are imported above. Only the non-quant examples (RHT,
# stochastic rounding) remain defined locally below, alongside the RecipeV2 constructions.


# Reduction constraints check `actual` (the real, possibly-ragged tile extent) so a severed
# reduction block is rejected at the edge; swizzle-atom constraints check `padded` (the nominal
# tile size, so ragged edge tiles are exempt -- recovers the old full_tile_multiple_of semantics).
# Migrated to quant_cast_gold: pt_ref_fn/correctness_fn come straight from the Gold object,
# RecipeV2 adds only the tiling constraint.
DEEPSEEK_1X128 = RecipeV2.from_gold(
    Deepseek1x128Gold,
    valid_tile_size_fn=lambda ts, a, p: a[1] % 128 == 0,
)
DEEPSEEK_128X128 = RecipeV2.from_gold(
    Deepseek128x128Gold,
    valid_tile_size_fn=lambda ts, a, p: a[0] % 128 == 0 and a[1] % 128 == 0,
)
# dim-M: `f` transposes the tile + reduces last dim; caller pairs it with
# output_kinds=SWAP_TILE_INDEX for the grid transpose. correctness_fn works in the (K, M)
# transposed frame and transposes back before comparing to `x`. After the within-tile
# transpose the reduced (last) dim is the tile's original ROWS, so rows must be a
# 128-multiple (checked on `actual`).
# TODO(future): the recipe framework needs a new field to test correctness for both
# the reference function as well as the backend specified function, right now this
# is not intuitive.
DEEPSEEK_1X128_DIM_M = RecipeV2.from_gold(
    Deepseek1x128DimMGold,
    valid_tile_size_fn=lambda ts, a, p: a[0] % 128 == 0,
    # `f` writes both outputs transposed locally; the grid swap yields the (K, M) layout.
    output_kinds=(OutputKind.SWAP_TILE_INDEX, OutputKind.SWAP_TILE_INDEX),
)
# rowwise / colwise: the tile must span the reduced dim (predicate forces it to equal the tensor
# extent), so the framework's tile-size search selects a full-span tile.
ROWWISE_FP8 = RecipeV2.from_gold(
    RowwiseFp8Gold,
    valid_tile_size_fn=lambda ts, a, p: a[1] == ts[1],  # span all columns
)
COLWISE_FP8 = RecipeV2.from_gold(
    ColwiseFp8Gold,
    valid_tile_size_fn=lambda ts, a, p: a[0] == ts[0],  # span all rows
    # `f` writes both outputs transposed locally; the grid swap yields the (N, M) layout.
    output_kinds=(OutputKind.SWAP_TILE_INDEX, OutputKind.SWAP_TILE_INDEX),
)
# rowwise with a precalculated (M, 1) scale passed as an AuxKind.ROW aux input; the divide is
# tile-invariant under plain 2D tiling (no tiling constraint needed). aux_fn computes the scale
# (a global row-reduction, outside flex_tile_map) that pt_ref_fn takes as its second arg.
ROWWISE_PRECALC = RecipeV2.from_gold(
    RowwisePrecalcGold,
    aux_fn=lambda x: (rowwise_precalc_scale(x),),
    aux_kinds=(AuxKind.ROW,),
)
# colwise with a precalculated (1, N) scale (AuxKind.COL) + transposed-contiguous output (pair
# with output_kinds=SWAP_TILE_INDEX); tile-invariant under plain 2D tiling.
COLWISE_PRECALC = RecipeV2.from_gold(
    ColwisePrecalcGold,
    aux_fn=lambda x: (colwise_precalc_scale(x),),
    aux_kinds=(AuxKind.COL,),
    output_kinds=(OutputKind.SWAP_TILE_INDEX,),
)
# reduction (1x32) checked on `actual`.
MXFP8_FLOOR = RecipeV2.from_gold(
    Mxfp8FloorGold,
    valid_tile_size_fn=lambda ts, a, p: a[1] % 32 == 0,
)
# reduction (1x32) checked on `actual`; swizzle atom (128x128) checked on `padded` (edge-exempt).
MXFP8_FLOOR_SWIZZLE = RecipeV2.from_gold(
    Mxfp8FloorSwizzleGold,
    valid_tile_size_fn=lambda ts, a, p: a[1] % 32 == 0 and p[0] % 128 == 0 and p[1] % 128 == 0,
)
# Tensorwise recipe: the per-tensor scale is computed outside (via float8_tensorwise_scale)
# and passed to pt_ref_fn as an explicit aux input (AuxKind.REPLICATE).
FLOAT8_TENSORWISE = RecipeV2.from_gold(
    Float8TensorwiseGold,
    aux_fn=lambda x: (float8_tensorwise_scale(x),),
    aux_kinds=(AuxKind.REPLICATE,),
)
# nvfp4 recipe: the per-tensor outer scale is computed outside (via nvfp4_gs_scale) and passed
# to pt_ref_fn as an explicit aux input (AuxKind.REPLICATE). reduction (1x16 inner) on `actual`;
# swizzle atom (128x64) on `padded` (edge-exempt).
NVFP4_GS_SWIZZLE = RecipeV2.from_gold(
    Nvfp4GsSwizzleGold,
    valid_tile_size_fn=lambda ts, a, p: a[1] % 16 == 0 and p[0] % 128 == 0 and p[1] % 64 == 0,
    aux_fn=lambda x: (nvfp4_gs_scale(x),),
    aux_kinds=(AuxKind.REPLICATE,),
)
# nvfp4 with a 128x128-blocked outer scale (computed via nvfp4_blocked_outer_scale) passed as an
# AuxKind.TILE aux. Same swizzle-atom constraints as NVFP4_GS_SWIZZLE; the 128x128 outer block is
# coarser than the (128, 64) atom so it adds no new alignment constraint at 128-aligned tiles.
NVFP4_BLOCKED_OUTER = RecipeV2.from_gold(
    Nvfp4BlockedOuterGold,
    valid_tile_size_fn=lambda ts, a, p: a[1] % 16 == 0 and p[0] % 128 == 0 and p[1] % 64 == 0,
    aux_fn=lambda x: (nvfp4_blocked_outer_scale(x),),
    aux_kinds=(AuxKind.TILE,),
)
# mxfp8 FLOOR with an elementwise bias (same shape as input) added before quant, passed as an
# AuxKind.TILE aux with divisor (1, 1). aux_fn uses a fixed ones-tensor for the generic gold
# test -- bias isn't derived from `x` (it's an arbitrary same-shape input), so there's no
# canonical non-trivial choice; Mxfp8BiasGold's correctness_fn only checks shape/dtype.
MXFP8_BIAS = RecipeV2.from_gold(
    Mxfp8BiasGold,
    valid_tile_size_fn=lambda ts, a, p: a[1] % 32 == 0,
    aux_fn=lambda x: (torch.ones_like(x),),
    aux_kinds=(AuxKind.TILE,),
)
RECIPES_V2 = [
    ("deepseek_1x128", DEEPSEEK_1X128),
    ("deepseek_128x128", DEEPSEEK_128X128),
    ("deepseek_1x128_dim_m", DEEPSEEK_1X128_DIM_M),
    ("rowwise_fp8", ROWWISE_FP8),
    ("colwise_fp8", COLWISE_FP8),
    ("rowwise_precalc", ROWWISE_PRECALC),
    ("colwise_precalc", COLWISE_PRECALC),
    ("mxfp8_floor", MXFP8_FLOOR),
    ("mxfp8_floor_swizzle", MXFP8_FLOOR_SWIZZLE),
    ("float8_tensorwise", FLOAT8_TENSORWISE),
    ("nvfp4_gs_swizzle", NVFP4_GS_SWIZZLE),
    ("nvfp4_blocked_outer", NVFP4_BLOCKED_OUTER),
    ("mxfp8_bias", MXFP8_BIAS),
]


# ---------------------------------------------------------------------------
# A non-quant example: the 16x16 randomized Hadamard transform (RHT). bf16 in, bf16 out,
# NO scale/aux -- `f` returns a 1-tuple `(out,)`. This is the building block for torchao's
# RHT-fused nvfp4 kernels (moe_training/nvfp4_training). RHT = diag(sign) @ H, where H is
# the 16x16 Sylvester-Walsh matrix / sqrt(16); mirrors torchao get_rht_matrix. RHT is
# orthogonal (its inverse is its transpose) but, with a sign vector, not an involution.
# ---------------------------------------------------------------------------
# 16x16 Sylvester-Walsh Hadamard values (torchao hadamard_utils.py get_hadamard_matrix).
_HADAMARD_16 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
    [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
    [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
    [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
    [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1],
    [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1],
    [1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1],
    [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1],
    [1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
    [1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1],
    [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1],
    [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1],
    [1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1],
    [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1],
]


def _hadamard_16_matrix(device, dtype):
    """16x16 Sylvester-Walsh Hadamard matrix scaled by 1/sqrt(16) (orthonormal)."""
    return torch.tensor(_HADAMARD_16, dtype=dtype, device=device) / (16**0.5)


def hadamard_rht_matrix(sign_vector, device, dtype):
    """RHT = diag(sign) @ H (torchao get_rht_matrix). `sign_vector` is a length-16 tensor."""
    H = _hadamard_16_matrix(device, dtype)
    return torch.diag(sign_vector.to(device=device, dtype=dtype)) @ H


def hadamard_rht_f(x, rht, **kwargs):
    """Tile-invariant `f` applying the 16x16 RHT along the last dim.

    `rht` is the RHT matrix (built via `hadamard_rht_matrix`), passed as an explicit aux input
    (REPLICATE: the same matrix is used for every tile). Returns a 1-tuple `(out,)` -- no scale.
    """
    *lead, last = x.shape
    out = (x.reshape(*lead, last // 16, 16) @ rht).reshape(*lead, last)
    return (out,)


# ---------------------------------------------------------------------------
# A non-quant example: stochastic rounding (SR) fp32 -> bf16. bf16 shares fp32's 8-bit
# exponent, so this is the simplest SR target -- no exponent rebias, no packing, no scale,
# no subnormal edge case: just dither the 16 discarded mantissa bits, then truncate. `f`
# returns a 1-tuple `(out,)`. Building block for the SR fp4 path (torchao _pack_fp4 +
# cvt.rs) and mirrors rs_tutorial/rs.py, specialized fp32 -> bf16.
#
# SR is unbiased: a value between two bf16 grid points rounds up with probability
# p_up = (x-lo)/(hi-lo). Adding a uniform 16-bit int to the fp32 bit pattern then
# truncating carries into the kept bits with exactly that probability, so E[SR(x)] = x.
# ---------------------------------------------------------------------------
def _sr_bf16_dither(x, rand16):
    """Apply a uniform 16-bit dither `rand16` to `x` (fp32) then truncate to bf16.

    dither the 16 mantissa bits fp32->bf16 drops, then truncate them (mask off the low 16
    bits). -65536 == 0xFFFF0000 as int32; .to(bfloat16) is exact since the low bits are zero.
    """
    xi = x.contiguous().view(torch.int32) + rand16
    xi = xi & -65536
    return xi.view(torch.float32).to(torch.bfloat16)


def sr_bf16_f(x, key, **kwargs):
    """`f` doing fp32 -> bf16 stochastic rounding, keyed on the TILE-LOCAL element layout.

    `key` is a torch.func._random (stateless counter-based Philox) PRNG key, passed as an
    explicit aux input (AuxKind.REPLICATE: the same key is handed to every tile) rather than
    built from a closed-over seed. One uniform is drawn per element in tile-local order, so
    offsets repeat across tiles and tiling CHANGES the rounding -- NOT tile-invariant, kept as
    the counterexample. `sr_bf16_global_f` is the tiling-invariant version. Returns `(out,)`.
    """
    assert x.dtype == torch.float32, f"SR bf16 expects fp32 input, got {x.dtype}"
    # uniform [0, 1) per element from the Philox key, scaled to a uniform 16-bit dither.
    # TODO(future): expose random integer generation directly in PyTorch
    # instead of having to do a multiply here
    u = prng.uniform(key, tuple(x.shape))
    rand16 = (u * (1 << 16)).to(torch.int32)  # uniform int in [0, 2**16)
    return (_sr_bf16_dither(x, rand16),)


def sr_bf16_global_f(x, key, **kwargs):
    """Tiling-invariant fp32 -> bf16 stochastic rounding: keys the dither on each element's
    GLOBAL position in the parent tensor, so the draws don't shift with tiling.

    The framework supplies the tile's global origin and row stride via kwargs, read here as
    `global_row`, `global_col`, `num_col`. Each element's global flat index is
    `(global_row + i) * num_col + (global_col + j)`; we build a per-element Philox key
    `[seed, global_index]` (vectorized, no host sync) and draw one uniform each. Because the
    index is global, element (i, j) gets the same draw regardless of which tile it lands in, so
    REFERENCE == MANUAL_TILE bit-for-bit. Returns `(out,)`.
    """
    assert x.dtype == torch.float32, f"SR bf16 expects fp32 input, got {x.dtype}"
    global_row = kwargs["global_row"]
    global_col = kwargs["global_col"]
    num_col = kwargs["num_col"]
    M, N = x.shape
    # per-element global flat index (int64 arithmetic; uint64 mul is unsupported on cuda).
    i = (global_row + torch.arange(M, device=x.device)).view(-1, 1)
    j = (global_col + torch.arange(N, device=x.device)).view(1, -1)
    gidx = (i * num_col + j).reshape(-1).to(torch.int64)
    # per-element Philox key [seed, global_index]; seed = key[0:1] (a slice, not .item(), so this
    # stays traceable / survives the FakeTensor shape-probe).
    seed = key[0:1].to(torch.int64).expand(gidx.numel())
    keys = torch.stack([seed, gidx], dim=-1).to(torch.uint64)
    u = prng.uniform(keys, (gidx.numel(),)).reshape(M, N)
    rand16 = (u * (1 << 16)).to(torch.int32)
    return (_sr_bf16_dither(x, rand16),)
