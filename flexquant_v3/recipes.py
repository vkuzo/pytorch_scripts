"""Quant recipes for flex_cast_quant, bundled as `Recipe` dataclasses.

Each recipe pairs a plain-PyTorch quant kernel `quant(x) -> (qdata, aux_out)` (v1
`_reference_fn` style) with its `dequant(qdata, scale) -> fp32` inverse. The `RECIPES`
table registers them (plus per-recipe test metadata) for the tests in test.py. Math
mirrors flexquant v1 recipes.py.
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
import torch.func._random as prng

from utils import f32_to_f4_unpacked, f4_unpacked_to_f32, pack_uint4, unpack_uint4


@dataclass(frozen=True)
class Recipe:
    """A single-kernel quant recipe: the quant kernel and its dequant.

    `quant(x) -> (qdata, aux_out)` is the `f` passed to `flex_cast_quant`; it is always
    tile-invariant (the same computation applied independently to every tile) -- that is a
    precondition of the API, so it isn't a per-recipe field. `dequant(qdata, scale) -> fp32`
    inverts it (for SQNR checks). Single-kernel only for now: recipes that need a separate
    kernel (e.g. tensorwise's global scale reduction) keep that step outside the Recipe.
    """

    quant: Callable
    dequant: Callable
    _tile_multiple_of: Tuple[int, int] | None = None
    _inner_tile_multiple_of: Tuple[int, int] | None = None


# ---------------------------------------------------------------------------
# The recipe: deepseek fp8 1x128, expressed as a tile-invariant `f`.
#
# Whole-tensor + block-aware form (v1 _reference_fn style). `flex_cast_quant` runs this
# as a passthrough today; the reshape->amax->scale->cast is the same computation applied
# independently to every 1x128 group, i.e. tile-invariant. Math mirrors v1
# _deepseek_fp8_1_128_reference (recipes.py:60-70). Constants are inlined per recipe
# (block size 128, eps 1e-12, fp8 e4m3 max 448.0) rather than shared as globals, since
# other recipes with different constants will be added alongside this one.
# ---------------------------------------------------------------------------
def deepseek_1x128_f(x, **kwargs):  # kwargs: framework-supplied global_row/global_col/num_col (unused)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    *lead, last = x.shape
    x_b = x.reshape(*lead, last // 128, 128)
    amax = x_b.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12).to(torch.float32)
    scale = (amax / fp8_max).to(torch.float32)  # forward scale
    qdata = (x_b.to(torch.float32) * (1.0 / scale)).to(torch.float8_e4m3fn)
    return qdata.reshape(*lead, last), scale.squeeze(-1)


# ---------------------------------------------------------------------------
# The recipe: deepseek fp8 128x128, expressed as a tile-invariant `f`.
#
# 2D block variant: amax over a full 128x128 tile, one scale per tile. Math mirrors v1
# _deepseek_fp8_128_128_reference (recipes.py:100-121); the reshape/transpose gymnastics
# gather each 128x128 tile into a contiguous group before reducing. Constants inlined
# per recipe, as above.
# ---------------------------------------------------------------------------
def deepseek_128x128_f(x, **kwargs):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    *lead, d1, d2 = x.shape
    n1, n2 = d1 // 128, d2 // 128
    x_b = (
        x.reshape(*lead, n1, 128, n2, 128)
        .transpose(-3, -2)
        .contiguous()
        .reshape(*lead, n1, n2, 128 * 128)
    )
    amax = x_b.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12).to(torch.float32)
    scale = (amax / fp8_max).to(torch.float32)  # forward scale
    qdata_b = (x_b.to(torch.float32) * (1.0 / scale)).to(torch.float8_e4m3fn)
    qdata = (
        qdata_b.reshape(*lead, n1, n2, 128, 128)
        .transpose(-3, -2)
        .contiguous()
        .reshape(*lead, d1, d2)
    )
    return qdata, scale.squeeze(-1)


# ---------------------------------------------------------------------------
# The recipe: mxfp8 with FLOOR rounding (1x32 blocks, e8m0 power-of-two scale).
#
# Tile-invariant like deepseek_1x128 (reduce over 32-element groups along N, no
# transpose), but the scale is an e8m0 (float8_e8m0fnu) power-of-two rather than fp32.
# Math mirrors v1 _mxfp8_floor_reference + its amax_to_scale/cast helpers
# (recipes.py:321-362): the e8m0 scale is derived by extracting amax's fp32 exponent via
# integer bit-ops (FLOOR, no log2), and the cast reconstructs the pow2 factor by shifting
# the biased exponent back into the fp32 exponent field. Constants (block 32, e8m0/fp32
# exponent bias 127, 23 fp32 mantissa bits, e4m3 max pow2 8) inlined per recipe.
# ---------------------------------------------------------------------------
def mxfp8_floor_f(x, **kwargs):
    e8m0_exponent_bias = 127
    f32_exp_bias = 127
    mbits_f32 = 23
    f8e4m3_max_pow2 = 8
    f32_min_normal = 2.0**-126
    e8m0_nan = 255

    *lead, last = x.shape
    x_b = x.reshape(*lead, last // 32, 32)
    amax = x_b.abs().amax(dim=-1, keepdim=True)

    # amax -> e8m0 block scale (FLOOR): extract fp32 exponent by integer bit-ops.
    max_abs = amax.to(torch.float32)
    max_abs_int32 = max_abs.view(torch.int32)
    extracted_pow2 = ((max_abs_int32 >> mbits_f32) & 0xFF) - f32_exp_bias
    scale_unbiased = extracted_pow2 - f8e4m3_max_pow2
    scale_unbiased = torch.clamp(
        scale_unbiased, -e8m0_exponent_bias, e8m0_exponent_bias + 1
    )
    scale_biased = (scale_unbiased + e8m0_exponent_bias).to(torch.uint8)
    scale_biased = torch.where(
        torch.isnan(max_abs), torch.full_like(scale_biased, e8m0_nan), scale_biased
    )
    scale_e8m0 = scale_biased.view(torch.float8_e8m0fnu)

    # cast: reconstruct the fp32 pow2 factor from the e8m0 biased exponent, then divide.
    biased_i32 = scale_e8m0.view(torch.uint8).to(torch.int32)
    scale_fp32 = (biased_i32 << mbits_f32).view(torch.float32)
    scale_fp32 = torch.clamp(scale_fp32, min=f32_min_normal)
    qdata = (x_b.to(torch.float32) / scale_fp32).to(torch.float8_e4m3fn)

    return qdata.reshape(*lead, last), scale_e8m0.squeeze(-1)


def _to_blocked_4d(scale):
    """Swizzle a row-major block-scale (H, W) into NVIDIA's blocked layout, kept as an
    explicit 4D block grid `(n_row_blocks, n_col_blocks, 32, 16)`.

    Ported from flexquant v1 swizzle.py:11-46 (itself a port of torchao's `to_blocked`,
    torchao/prototype/mx_formats/utils.py), but the final `(n_row_blocks*32,
    n_col_blocks*16)` reshape is NOT applied here. Serializing to that 2D buffer folds the
    (row-block, col-block) walk order into the axes, which makes the result depend on the
    GLOBAL grid shape -- a column split then reorders the buffer, so `f` composed with the
    2D swizzle is not tile-invariant. Keeping the two block axes separate makes the swizzle
    tile-invariant: a column tile concatenates on `dim=1` (n_col_blocks), a row tile on
    `dim=0` (n_row_blocks), and `.reshape(-1)` still equals torchao's `to_blocked` buffer
    (do that serialization once, outside `f`, after tiles are reassembled).

    Each 128x4 scale block swizzles independently into a 32x16 block, so this is a LOCAL
    (per-atom) transform: valid only when tiles are whole 128x4 scale atoms.
    """
    def _ceil_div(a, b):
        return (a + b - 1) // b

    rows, cols = scale.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = scale
    if torch.compiler.is_compiling() or (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols), device=scale.device, dtype=scale.dtype
        )
        padded[:rows, :cols] = scale

    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    # (n_row_blocks, n_col_blocks, 128, 4) -> (n_row_blocks, n_col_blocks, 32, 16), keeping
    # the two block axes intact (no reshape across them).
    rearranged = blocks.reshape(n_row_blocks, n_col_blocks, 4, 32, 4).transpose(-3, -2)
    return rearranged.reshape(n_row_blocks, n_col_blocks, 32, 16)


def _from_blocked_4d(blocked, rows, cols):
    """Inverse of `_to_blocked_4d` for the exact case rows % 128 == 0, cols % 4 == 0.

    `blocked` is the 4D block grid `(n_row_blocks, n_col_blocks, 32, 16)`.
    """
    nrb, ncb = rows // 128, cols // 4
    x = blocked.reshape(nrb, ncb, 32, 4, 4).transpose(-3, -2)
    x = x.reshape(nrb, ncb, 128, 4).permute(0, 2, 1, 3)
    return x.reshape(rows, cols)


# nvfp4 format constants (fp4 e2m1 max value + e4m3 scale range).
F4_E2M1_MAX = 6.0
F8E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny


# ---------------------------------------------------------------------------
# The recipe: mxfp8 FLOOR with swizzled (NVIDIA 32x4x4 blocked) scale.
#
# Same quantization as mxfp8_floor_f, but the e8m0 scale is emitted in the blocked layout
# `_scaled_mm` consumes (v1 mxfp8_floor_swizzle, recipes.py:401-423). The swizzle is a
# LOCAL, tile-invariant transform when tiles are whole 128x128 hp units (= 128x4 e8m0
# tiles): each 128x4 scale tile swizzles independently into a 32x16 block. The scale is
# returned as the 4D block grid `(n_row_blocks, n_col_blocks, 32, 16)` (see
# `_to_blocked_4d`): keeping the block axes separate makes it tile-invariant under BOTH
# row and column splits -- MANUAL_TILE reassembles a column tile with cat(dim=1) and a row
# tile with cat(dim=0). The final serialization to the flat `_scaled_mm` buffer (a global,
# grid-shape-dependent step) is `.reshape(-1)`, done once outside `f` after reassembly.
# ---------------------------------------------------------------------------
def mxfp8_floor_swizzle_f(x, **kwargs):
    qdata, scale_e8m0 = mxfp8_floor_f(x)
    return qdata, _to_blocked_4d(scale_e8m0)


# ---------------------------------------------------------------------------
# The recipe: deepseek fp8 1x128, reduced across M (128x1 blocks), transposed output.
#
# This is plain `deepseek_1x128_f` composed with
# `flex_cast_quant(_global_input_transform=SWAP_0_AND_1_AXES)`:
# the framework swaps the input axes on load (transpose-first), so `f` sees the (K, M)
# orientation and reduces the correct axis, yielding (K, M) qdata and (K, M//128) scale --
# equivalent to v1 _deepseek_fp8_1_128_dim_m_reference (recipes.py:84-86). Because the
# transpose happens BEFORE tiling (swap-on-load), the per-tile work stays tile-invariant,
# so this now runs under MANUAL_TILE too -- no dedicated transposing `f` needed.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The recipe: float8 tensorwise (per-tensor) scaling.
#
# Unlike the block recipes, the scale is a single per-tensor value that needs a GLOBAL
# reduction over the whole tensor -- that reduction is NOT tile-invariant, so it lives
# OUTSIDE flex_cast_quant (`float8_tensorwise_scale`; the "call something else" kernel
# from api.py's docstring). Given that precomputed scale, quantization is just dividing
# every element by one fixed scalar -- identical across tiles, hence tile-invariant -- so
# it runs INSIDE flex_cast_quant via an `f` that takes the scale as an explicit REPLICATE
# aux input (handed whole to every tile).
# ---------------------------------------------------------------------------
def float8_tensorwise_scale(x):
    """Per-tensor scale (global reduction; computed outside flex_cast_quant)."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    amax = x.abs().amax().clamp(min=1e-12).to(torch.float32)
    return (amax / fp8_max).to(torch.float32)  # scalar forward scale


def float8_tensorwise_f(x, scale, **kwargs):
    """Tile-invariant `f` taking the precomputed per-tensor `scale` as an explicit aux input
    (REPLICATE: the same scalar scale is used for every tile)."""
    qdata = (x.to(torch.float32) * (1.0 / scale)).to(torch.float8_e4m3fn)
    return qdata, scale


# ---------------------------------------------------------------------------
# The recipe: nvfp4 with global scale (two-level) + swizzled inner scale.
#
# Two-level scaling like v1 nvfp4_with_gs (recipes.py:255-320): a per-tensor fp32 OUTER
# scale plus a per-16-element e4m3 INNER scale, with fp4-packed qdata. The outer scale is
# a GLOBAL amax reduction -- not tile-invariant -- so (like tensorwise) it is computed
# OUTSIDE flex_cast_quant (`nvfp4_gs_scale`) and passed in as a REPLICATE aux input. The inner scale is
# additionally swizzled into the NVIDIA blocked layout and returned as the 4D block grid
# `(n_row_blocks, n_col_blocks, 32, 16)`, so MANUAL_TILE's cat recomposition reproduces the
# full-tensor swizzle under both row and column splits (cf. mxfp8 swizzle).
# ---------------------------------------------------------------------------
def nvfp4_gs_scale(x):
    """Per-tensor fp32 outer scale (global reduction; computed outside flex_cast_quant)."""
    outer_amax = x.abs().to(torch.float32).amax()
    return outer_amax / (F8E4M3_MAX * F4_E2M1_MAX)


def nvfp4_gs_swizzle_f(x, outer_scale, **kwargs):
    """Tile-invariant `f` taking the precomputed per-tensor `outer_scale` as an explicit aux
    input (REPLICATE: the same scalar outer scale is used for every tile)."""
    *lead, last = x.shape
    x_b = x.reshape(*lead, last // 16, 16)
    local_amax = x_b.abs().amax(dim=-1, keepdim=True)
    # inner e4m3 block scale, relative to the outer scale.
    inner = torch.clamp(
        (local_amax.to(torch.float32) / F4_E2M1_MAX) / outer_scale,
        min=E4M3_EPS, max=F8E4M3_MAX,
    ).to(torch.float8_e4m3fn)
    # cast: divide by (outer * inner), clamp to fp4 range, pack two per byte.
    reciprocal = (1.0 / outer_scale) / inner.to(torch.float32)
    data_scaled = torch.clamp(x_b.to(torch.float32) * reciprocal, -F4_E2M1_MAX, F4_E2M1_MAX)
    qdata_b = pack_uint4(f32_to_f4_unpacked(data_scaled)).view(torch.float4_e2m1fn_x2)
    qdata = qdata_b.reshape(*lead, last // 2)
    inner_swizzled = _to_blocked_4d(inner.squeeze(-1))
    return qdata, inner_swizzled


# ---------------------------------------------------------------------------
# Dequant: recover fp32 from (qdata, scale) by broadcasting the scale back over qdata.
# The swizzle recipe un-swizzles the scale first; the dim-M recipe reuses the plain 1x128
# dequant and the test transposes the result back to (M, N).
# ---------------------------------------------------------------------------
def _e8m0_to_fp32(scale):
    # inverse of mxfp8_floor_f's cast: e8m0 biased exponent -> fp32 pow2 factor.
    biased_i32 = scale.contiguous().view(torch.uint8).to(torch.int32)
    scale_fp32 = (biased_i32 << 23).view(torch.float32)
    return torch.clamp(scale_fp32, min=2.0**-126)


def deepseek_1x128_dq_f(q, scale):
    M, N = q.shape
    nb = N // 128
    return (q.float().reshape(M, nb, 128) * scale.reshape(M, nb, 1)).reshape(M, N)


def deepseek_128x128_dq_f(q, scale):
    M, N = q.shape
    n1, n2 = M // 128, N // 128
    return (q.float().reshape(n1, 128, n2, 128) * scale.reshape(n1, 1, n2, 1)).reshape(M, N)


def mxfp8_floor_dq_f(q, scale):
    M, N = q.shape
    nb = N // 32
    s = _e8m0_to_fp32(scale).reshape(M, nb, 1)
    return (q.float().reshape(M, nb, 32) * s).reshape(M, N)


def dq_tensorwise(q, scale):
    return q.float() * scale


def mxfp8_floor_swizzle_dq_f(q, scale):
    # un-swizzle the 4D block grid back to (M, N//32) e8m0, then dequant as mxfp8.
    M, N = q.shape
    rows, cols = M, N // 32
    scale_e8m0 = _from_blocked_4d(scale, rows, cols)
    return mxfp8_floor_dq_f(q, scale_e8m0)


def nvfp4_gs_swizzle_dq_f(q, inner_swizzled, outer_scale):
    """Dequant taking the per-tensor `outer_scale` as an explicit arg (symmetric with the
    lifted quant `nvfp4_gs_swizzle_f`)."""
    # q: (M, N//2) packed fp4; N = 2 * packed cols.
    M, half = q.shape
    N = half * 2
    cols = N // 16  # number of 16-blocks per row (== inner scale cols)
    # unpack fp4 -> fp32 in (M, N).
    unpacked = f4_unpacked_to_f32(unpack_uint4(q.view(torch.uint8))).reshape(M, N)
    # un-swizzle inner scale (4D block grid) back to (M, cols) e4m3 -> fp32.
    inner = _from_blocked_4d(inner_swizzled, M, cols)
    inner_fp32 = inner.to(torch.float32).reshape(M, cols, 1)
    return (unpacked.reshape(M, cols, 16) * inner_fp32 * outer_scale).reshape(M, N)


# ---------------------------------------------------------------------------
# The recipe: nvfp4 with a 128x128-BLOCKED outer scale (instead of a global scalar).
#
# Same two-level nvfp4 as nvfp4_gs_swizzle_f, but the outer scale is one value per 128x128
# block -- shape (M//128, N//128) -- computed outside and passed as an AuxKind.TILE aux. The
# framework hands `f` the sub-block of the outer scale covering the current tile; `f`
# block-broadcasts it to per-element (option-4 pattern: expand+reshape, no materialized (M,N)
# scale). Because 128 is a multiple of the 16-element inner block, the outer is constant within
# each 16-group, so one representative per 16-group aligns with the inner scale.
# ---------------------------------------------------------------------------
def nvfp4_blocked_outer_scale(x, blk=128):
    """Per-128x128-block fp32 outer scale (block reduction; computed outside flex_cast_quant).
    Returns shape (M//blk, N//blk)."""
    Mb, Nb = x.shape[0] // blk, x.shape[1] // blk
    block_amax = x.abs().to(torch.float32).reshape(Mb, blk, Nb, blk).amax(dim=(1, 3))
    return block_amax / (F8E4M3_MAX * F4_E2M1_MAX)  # (Mb, Nb)


def nvfp4_blocked_outer_f(x, outer_blocked, **kwargs):
    """Tile-invariant `f`: nvfp4 cast with a 128x128-blocked outer scale (AuxKind.TILE).

    Instead of expanding the outer scale to per-element, reshape `x` so the outer block grid is
    explicit -- (Mb, rows_per_block, Nb, n16_per_block, 16) -- and let `outer_blocked` broadcast
    against it via size-1 axes. Each outer element then maps directly to its block slice of the
    input; the full (M, N) outer scale is never materialized.
    """
    M, N = x.shape
    Mb, Nb = outer_blocked.shape
    rpb, cpb = M // Mb, N // Nb          # rows / cols per outer block (e.g. 128, 128)
    n16 = cpb // 16                      # inner 16-groups per outer block along N
    # block-grid view: last dim is the 16-element inner block.
    x_b = x.reshape(Mb, rpb, Nb, n16, 16)
    outer_b = outer_blocked[:, None, :, None, None]     # (Mb, 1, Nb, 1, 1), broadcasts
    local_amax = x_b.abs().amax(dim=-1, keepdim=True)   # (Mb, rpb, Nb, n16, 1)
    inner = torch.clamp(
        (local_amax.to(torch.float32) / F4_E2M1_MAX) / outer_b,
        min=E4M3_EPS, max=F8E4M3_MAX,
    ).to(torch.float8_e4m3fn)
    reciprocal = (1.0 / outer_b) / inner.to(torch.float32)
    data_scaled = torch.clamp(x_b.to(torch.float32) * reciprocal, -F4_E2M1_MAX, F4_E2M1_MAX)
    qdata = pack_uint4(f32_to_f4_unpacked(data_scaled)).view(torch.float4_e2m1fn_x2).reshape(M, N // 2)
    # inner scale back to (M, N//16) row-major, then swizzle.
    inner_swizzled = _to_blocked_4d(inner.squeeze(-1).reshape(M, N // 16))
    return qdata, inner_swizzled


def nvfp4_blocked_outer_dq_f(q, inner_swizzled, outer_blocked):
    """Dequant for nvfp4_blocked_outer_f. Reshapes onto the outer block grid so `outer_blocked`
    broadcasts via size-1 axes (no materialized (M, N) outer scale), mirroring the quant."""
    M, half = q.shape
    N = half * 2
    Mb, Nb = outer_blocked.shape
    rpb, cpb = M // Mb, N // Nb
    n16 = cpb // 16
    unpacked = f4_unpacked_to_f32(unpack_uint4(q.view(torch.uint8))).reshape(M, N)
    inner = _from_blocked_4d(inner_swizzled, M, N // 16)
    # block-grid view: (Mb, rpb, Nb, n16, 16); outer broadcasts on the block axes.
    data = unpacked.reshape(Mb, rpb, Nb, n16, 16)
    inner_b = inner.to(torch.float32).reshape(Mb, rpb, Nb, n16, 1)
    outer_b = outer_blocked[:, None, :, None, None]
    return (data * inner_b * outer_b).reshape(M, N)


# ---------------------------------------------------------------------------
# The recipe: mxfp8 FLOOR with an elementwise bias added before quant.
#
# `bias` is the same shape as the input -> AuxKind.TILE with divisor (1, 1): the framework
# partitions it exactly like the input (one bias element per input element). `f` just adds it
# and runs the existing mxfp8 cast; dequant is the plain mxfp8 dequant (the bias is folded in).
# ---------------------------------------------------------------------------
def mxfp8_bias_f(x, bias, **kwargs):
    """Tile-invariant `f`: add an elementwise `bias` (AuxKind.TILE, per-element) then mxfp8."""
    return mxfp8_floor_f(x + bias.to(x.dtype))


DEEPSEEK_1X128 = Recipe(
    quant=deepseek_1x128_f,
    dequant=deepseek_1x128_dq_f,
    _tile_multiple_of=(1, 128),
)
DEEPSEEK_128X128 = Recipe(
    quant=deepseek_128x128_f, 
    dequant=deepseek_128x128_dq_f,
    _tile_multiple_of=(128, 128),
)
# dim-M reuses deepseek_1x128_f entirely: the (K, M) orientation comes from swap_axes
# below, and dequant is the plain 1x128 dequant (in (K, M) space). The test transposes
# the dequant result back to (M, N) when swap_axes is set.
DEEPSEEK_1X128_DIM_M = Recipe(
    quant=deepseek_1x128_f, 
    dequant=deepseek_1x128_dq_f,
    _tile_multiple_of=(1, 128),
)
MXFP8_FLOOR = Recipe(
    quant=mxfp8_floor_f, 
    dequant=mxfp8_floor_dq_f,
    _tile_multiple_of=(1, 32),
)
MXFP8_FLOOR_SWIZZLE = Recipe(
    quant=mxfp8_floor_swizzle_f, 
    dequant=mxfp8_floor_swizzle_dq_f,
    _tile_multiple_of=(1, 32),
    _inner_tile_multiple_of=(128, 128),  # 128x128 to enforce that each scale swizzle does not cross tile boundaries
)


# Tensorwise recipe: the per-tensor scale is computed outside (via float8_tensorwise_scale)
# and passed to flex_cast_quant as an explicit aux input (AuxKind.REPLICATE), not bound here.
FLOAT8_TENSORWISE = Recipe(
    quant=float8_tensorwise_f,
    dequant=dq_tensorwise,
)


# nvfp4 recipe: the per-tensor outer scale is computed outside (via nvfp4_gs_scale) and passed
# to flex_cast_quant / dequant as an explicit aux input (AuxKind.REPLICATE), not bound here.
NVFP4_GS_SWIZZLE = Recipe(
    quant=nvfp4_gs_swizzle_f,
    dequant=nvfp4_gs_swizzle_dq_f,
    _tile_multiple_of=(1, 16),
    _inner_tile_multiple_of=(128, 64),  # 128x64 to enforce that each scale swizzle does not cross tile boundaries
)


# nvfp4 with a 128x128-blocked outer scale (computed via nvfp4_blocked_outer_scale) passed as an
# AuxKind.TILE aux. Same swizzle-atom constraints as NVFP4_GS_SWIZZLE; the 128x128 outer block is
# coarser than the (128, 64) atom so it adds no new alignment constraint at 128-aligned tiles.
NVFP4_BLOCKED_OUTER = Recipe(
    quant=nvfp4_blocked_outer_f,
    dequant=nvfp4_blocked_outer_dq_f,
    _tile_multiple_of=(1, 16),
    _inner_tile_multiple_of=(128, 64),
)


# mxfp8 FLOOR with an elementwise bias (same shape as input) added before quant, passed as an
# AuxKind.TILE aux with divisor (1, 1). Dequant is the plain mxfp8 dequant.
MXFP8_BIAS = Recipe(
    quant=mxfp8_bias_f,
    dequant=mxfp8_floor_dq_f,
    _tile_multiple_of=(1, 32),
)


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
