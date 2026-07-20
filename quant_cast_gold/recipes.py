"""Golden single-kernel quant-cast reference recipes, decoupled from flexquant_v3.

Each `QuantCastSingleKernelGold` pairs a plain-PyTorch reference kernel (`pt_ref_fn`,
the function that would be handed to flex_tile_map as `f`) with a `correctness_fn`
that asserts a candidate set of outputs is "close enough" to `pt_ref_fn`'s own
semantics. This package is intentionally independent of flexquant_v3 -- it must not
import from it, since it exists to grade it. Migrated incrementally, recipe by
recipe (see `RECIPES` below for the full list; flexquant_v3/recipes.py holds the
recipes not yet migrated).
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import torch

from quant_cast_gold.utils import f32_to_f4_unpacked, f4_unpacked_to_f32, pack_uint4, unpack_uint4


@dataclass(frozen=True)
class QuantCastSingleKernelGold:
    """A golden single-kernel quant-cast reference.

    `pt_ref_fn(*inputs, **kwargs) -> outputs` is a plain-PyTorch reference function 

    `correctness_fn(inputs, outputs) -> None`
      - inputs - inputs to pt_ref_fn
      - outputs - outputs from an implementation of pt_ref_fn

      The function checks that the outputs are valid, and asserts with an error
      message if they are not. For example, if `pt_ref_fn` quantizes a tensor,
      `correctness_fn` could check SQNR between ref and quantized outputs.
    """

    pt_ref_fn: Callable
    correctness_fn: Callable


def _compute_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # torchao's `compute_error` (quantization/utils.py:63) -- SQNR in dB. Duplicated
    # locally (not imported from flexquant_v3/test.py) so gold has no dependency on the
    # thing it grades.
    return 20 * torch.log10(
        torch.linalg.vector_norm(x) / torch.linalg.vector_norm(x - y)
    )


# ---------------------------------------------------------------------------
# Golden recipe: deepseek fp8 1x128.
# ---------------------------------------------------------------------------
def deepseek_1x128_f(x, **kwargs):  # kwargs: framework-supplied global_row/global_col/num_col (unused)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    *lead, last = x.shape
    x_b = x.reshape(*lead, last // 128, 128)
    amax = x_b.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12).to(torch.float32)
    scale = (amax / fp8_max).to(torch.float32)  # forward scale
    qdata = (x_b.to(torch.float32) * (1.0 / scale)).to(torch.float8_e4m3fn)
    return qdata.reshape(*lead, last), scale.squeeze(-1)


def deepseek_1x128_dq_f(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _deepseek_1x128_correctness, and importable
    # directly by consumers (e.g. flexquant_v3's own Recipe.dequant) that need the inverse.
    M, N = q.shape
    nb = N // 128
    return (q.float().reshape(M, nb, 128) * scale.reshape(M, nb, 1)).reshape(M, N)


def _deepseek_1x128_correctness(
    inputs: Tuple[torch.Tensor, ...], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Assert dequant(outputs) recovers `x` with SQNR above threshold."""
    (x,) = inputs
    qdata, scale = outputs
    x_hat = deepseek_1x128_dq_f(qdata, scale)
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 20.0
    assert sqnr > threshold, f"deepseek_1x128: sqnr={sqnr.item():.2f} dB below {threshold} dB"


Deepseek1x128Gold = QuantCastSingleKernelGold(
    pt_ref_fn=deepseek_1x128_f,
    correctness_fn=_deepseek_1x128_correctness,
)


# ---------------------------------------------------------------------------
# Golden recipe: deepseek fp8 128x128.
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


def deepseek_128x128_dq_f(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _deepseek_128x128_correctness, and importable
    # directly by consumers (e.g. flexquant_v3's own Recipe.dequant) that need the inverse.
    M, N = q.shape
    n1, n2 = M // 128, N // 128
    return (q.float().reshape(n1, 128, n2, 128) * scale.reshape(n1, 1, n2, 1)).reshape(M, N)


def _deepseek_128x128_correctness(
    inputs: Tuple[torch.Tensor, ...], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Assert dequant(outputs) recovers `x` with SQNR above threshold."""
    (x,) = inputs
    qdata, scale = outputs
    x_hat = deepseek_128x128_dq_f(qdata, scale)
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 20.0
    assert sqnr > threshold, f"deepseek_128x128: sqnr={sqnr.item():.2f} dB below {threshold} dB"


Deepseek128x128Gold = QuantCastSingleKernelGold(
    pt_ref_fn=deepseek_128x128_f,
    correctness_fn=_deepseek_128x128_correctness,
)


# ---------------------------------------------------------------------------
# Golden recipe: deepseek fp8 1x128, reduced across M (128x1 blocks), transposed output.
# ---------------------------------------------------------------------------
def deepseek_1x128_dim_m_f(x, **kwargs):
    """dim-M deepseek: reduce over dim0 (128x1 blocks along rows), then write the tile's outputs
    TRANSPOSED locally. Pair with OutputKind.SWAP_TILE_INDEX on both outputs so the framework
    places each transposed tile at the swapped grid position -> the full (K, M) layout.

    Inlined from deepseek_1x128_f but reducing the other axis: reshape rows into 128-blocks and
    amax over dim1 (the 128 within-block dim), giving a (M//128, N) scale; transpose both outputs
    to (N, M) / (N, M//128) so a tile computed at grid [m, n] carries (bn, bm)-shaped data.
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    M, N = x.shape
    x_b = x.reshape(M // 128, 128, N)
    amax = x_b.abs().amax(dim=1, keepdim=True).clamp(min=1e-12).to(torch.float32)
    scale = (amax / fp8_max).to(torch.float32)  # forward scale, (M//128, 1, N)
    qdata = (x_b.to(torch.float32) * (1.0 / scale)).to(torch.float8_e4m3fn).reshape(M, N)
    # write outputs transposed locally; the framework's SWAP_TILE_INDEX handles the grid swap.
    return qdata.t().contiguous(), scale.squeeze(1).t().contiguous()


def _deepseek_1x128_dim_m_correctness(
    inputs: Tuple[torch.Tensor, ...], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Assert dequant(outputs), transposed back to (M, N), recovers `x` with SQNR above
    threshold. Reuses deepseek_1x128_dq_f -- it works in the (K, M) transposed frame."""
    (x,) = inputs
    qdata, scale = outputs
    x_hat = deepseek_1x128_dq_f(qdata, scale).t()
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 20.0
    assert sqnr > threshold, f"deepseek_1x128_dim_m: sqnr={sqnr.item():.2f} dB below {threshold} dB"


Deepseek1x128DimMGold = QuantCastSingleKernelGold(
    pt_ref_fn=deepseek_1x128_dim_m_f,
    correctness_fn=_deepseek_1x128_dim_m_correctness,
)


# ---------------------------------------------------------------------------
# Golden recipe: rowwise fp8 (one scale per row, amax over all columns).
# ---------------------------------------------------------------------------
def rowwise_fp8_f(x, **kwargs):
    """Rowwise fp8: one fp32 scale per row (amax over all columns). Tile must span all columns."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-12).to(torch.float32)  # (M, 1)
    scale = (amax / fp8_max).to(torch.float32)
    qdata = (x.to(torch.float32) * (1.0 / scale)).to(torch.float8_e4m3fn)
    return qdata, scale  # scale shape (M, 1)


def rowwise_fp8_dq_f(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _rowwise_fp8_correctness, and importable
    # directly by consumers that need the inverse.
    return q.float() * scale  # scale (M, 1) broadcasts over columns


def _rowwise_fp8_correctness(
    inputs: Tuple[torch.Tensor, ...], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Assert dequant(outputs) recovers `x` with SQNR above threshold."""
    (x,) = inputs
    qdata, scale = outputs
    x_hat = rowwise_fp8_dq_f(qdata, scale)
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 20.0
    assert sqnr > threshold, f"rowwise_fp8: sqnr={sqnr.item():.2f} dB below {threshold} dB"


RowwiseFp8Gold = QuantCastSingleKernelGold(
    pt_ref_fn=rowwise_fp8_f,
    correctness_fn=_rowwise_fp8_correctness,
)


# ---------------------------------------------------------------------------
# Golden recipe: colwise fp8 (one scale per column, amax over all rows).
# ---------------------------------------------------------------------------
def colwise_fp8_f(x, **kwargs):
    """Colwise fp8: one fp32 scale per column (amax over all rows). Tile must span all rows.

    Writes both outputs transposed locally (q -> (N, M), scale -> (N, 1)); pair with
    output_kinds=SWAP_TILE_INDEX so the framework's grid swap yields the transposed (N, M) layout.
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    amax = x.abs().amax(dim=0, keepdim=True).clamp(min=1e-12).to(torch.float32)  # (1, N)
    scale = (amax / fp8_max).to(torch.float32)
    qdata = (x.to(torch.float32) * (1.0 / scale)).to(torch.float8_e4m3fn)
    return qdata.t().contiguous(), scale.t().contiguous()  # (N, M), (N, 1)


def colwise_fp8_dq_f(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _colwise_fp8_correctness, and importable
    # directly by consumers that need the inverse. q is (N, M) transposed frame; scale
    # (N, 1) broadcasts over q's columns (the original rows).
    return q.float() * scale


def _colwise_fp8_correctness(
    inputs: Tuple[torch.Tensor, ...], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Assert dequant(outputs), transposed back to (M, N), recovers `x` with SQNR above
    threshold. outputs are in the transposed (N, M) frame."""
    (x,) = inputs
    qdata, scale = outputs
    x_hat = colwise_fp8_dq_f(qdata, scale).t()
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 20.0
    assert sqnr > threshold, f"colwise_fp8: sqnr={sqnr.item():.2f} dB below {threshold} dB"


ColwiseFp8Gold = QuantCastSingleKernelGold(
    pt_ref_fn=colwise_fp8_f,
    correctness_fn=_colwise_fp8_correctness,
)


# ---------------------------------------------------------------------------
# Golden recipe: rowwise fp8 with a PRECALCULATED per-row scale (AuxKind.ROW).
#
# Unlike rowwise_fp8_f (which reduces the row itself), here the (M, 1) per-row scale is
# computed OUTSIDE and passed as an explicit second positional arg (an AuxKind.ROW aux
# input under flex_tile_map). `pt_ref_fn` only does the divide; `rowwise_precalc_scale`
# is the row-reduction that lives outside flex_tile_map, so `inputs = (x, scale)` carries
# the precomputed scale through to correctness_fn (outputs has no scale to recover it from).
# ---------------------------------------------------------------------------
def rowwise_precalc_scale(x):
    """Per-row fp32 scale (row reduction; computed outside flex_tile_map). Returns (M, 1)."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-12).to(torch.float32)
    return (amax / fp8_max).to(torch.float32)  # (M, 1)


def rowwise_precalc_f(x, scale, **kwargs):
    """Rowwise fp8 cast given a precalculated (M, 1) per-row `scale` (AuxKind.ROW aux input)."""
    qdata = (x.to(torch.float32) / scale).to(torch.float8_e4m3fn)
    return (qdata,)  # scale is an input, not a returned output


def rowwise_precalc_dq_f(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _rowwise_precalc_correctness, and importable
    # directly by consumers that need the inverse.
    return q.float() * scale  # (M, 1) broadcasts over columns


def _rowwise_precalc_correctness(
    inputs: Tuple[torch.Tensor, torch.Tensor], outputs: Tuple[torch.Tensor]
) -> None:
    """Assert dequant(outputs, using the precalculated scale from `inputs`) recovers `x`
    with SQNR above threshold."""
    x, scale = inputs
    (qdata,) = outputs
    x_hat = rowwise_precalc_dq_f(qdata, scale)
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 20.0
    assert sqnr > threshold, f"rowwise_precalc: sqnr={sqnr.item():.2f} dB below {threshold} dB"


RowwisePrecalcGold = QuantCastSingleKernelGold(
    pt_ref_fn=rowwise_precalc_f,
    correctness_fn=_rowwise_precalc_correctness,
)


# ---------------------------------------------------------------------------
# Golden recipe: colwise fp8 with a PRECALCULATED per-column scale (AuxKind.COL), transposed
# output. The symmetric partner of the ROW precalc recipe above.
# ---------------------------------------------------------------------------
def colwise_precalc_scale(x):
    """Per-column fp32 scale (col reduction; computed outside flex_tile_map). Returns (1, N)."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    amax = x.abs().amax(dim=0, keepdim=True).clamp(min=1e-12).to(torch.float32)
    return (amax / fp8_max).to(torch.float32)  # (1, N)


def colwise_precalc_f(x, scale, **kwargs):
    """Colwise fp8 cast given a precalculated (1, N) per-column `scale` (AuxKind.COL aux input);
    writes the tile output transposed-contiguous (pair with output_kinds=SWAP_TILE_INDEX)."""
    qdata = (x.to(torch.float32) / scale).to(torch.float8_e4m3fn)
    return (qdata.t().contiguous(),)  # (Ntile, Mtile); scale is an input, not a returned output


def colwise_precalc_dq_f(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _colwise_precalc_correctness, and importable
    # directly by consumers that need the inverse. q is (N, M) transposed frame; scale
    # (1, N) -> transpose to (N, 1) to broadcast over q's cols.
    return q.float() * scale.t()


def _colwise_precalc_correctness(
    inputs: Tuple[torch.Tensor, torch.Tensor], outputs: Tuple[torch.Tensor]
) -> None:
    """Assert dequant(outputs, using the precalculated scale from `inputs`), transposed back
    to (M, N), recovers `x` with SQNR above threshold."""
    x, scale = inputs
    (qdata,) = outputs
    x_hat = colwise_precalc_dq_f(qdata, scale).t()
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 20.0
    assert sqnr > threshold, f"colwise_precalc: sqnr={sqnr.item():.2f} dB below {threshold} dB"


ColwisePrecalcGold = QuantCastSingleKernelGold(
    pt_ref_fn=colwise_precalc_f,
    correctness_fn=_colwise_precalc_correctness,
)


# ---------------------------------------------------------------------------
# Golden recipe: mxfp8 with FLOOR rounding (1x32 blocks, e8m0 power-of-two scale).
#
# Tile-invariant like deepseek_1x128 (reduce over 32-element groups along N, no transpose),
# but the scale is an e8m0 (float8_e8m0fnu) power-of-two rather than fp32: derived by
# extracting amax's fp32 exponent via integer bit-ops (FLOOR, no log2), and the cast
# reconstructs the pow2 factor by shifting the biased exponent back into the fp32 exponent
# field.
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


def _e8m0_to_fp32(scale):
    # inverse of mxfp8_floor_f's cast: e8m0 biased exponent -> fp32 pow2 factor.
    biased_i32 = scale.contiguous().view(torch.uint8).to(torch.int32)
    scale_fp32 = (biased_i32 << 23).view(torch.float32)
    return torch.clamp(scale_fp32, min=2.0**-126)


def mxfp8_floor_dq_f(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _mxfp8_floor_correctness, and importable
    # directly by consumers (e.g. flexquant_v3's mxfp8_floor_swizzle_dq_f/MXFP8_BIAS) that
    # need the inverse.
    M, N = q.shape
    nb = N // 32
    s = _e8m0_to_fp32(scale).reshape(M, nb, 1)
    return (q.float().reshape(M, nb, 32) * s).reshape(M, N)


def _mxfp8_floor_correctness(
    inputs: Tuple[torch.Tensor, ...], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Assert dequant(outputs) recovers `x` with SQNR above threshold. The e8m0 pow2 scale
    is coarser than fp32, so the threshold is lower than the fp8 recipes' (20 dB)."""
    (x,) = inputs
    qdata, scale = outputs
    x_hat = mxfp8_floor_dq_f(qdata, scale)
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 15.0
    assert sqnr > threshold, f"mxfp8_floor: sqnr={sqnr.item():.2f} dB below {threshold} dB"


Mxfp8FloorGold = QuantCastSingleKernelGold(
    pt_ref_fn=mxfp8_floor_f,
    correctness_fn=_mxfp8_floor_correctness,
)


# ---------------------------------------------------------------------------
# Golden recipe: mxfp8 FLOOR with swizzled (NVIDIA 32x4x4 blocked) scale.
#
# Same quantization as mxfp8_floor_f, but the e8m0 scale is emitted in the blocked layout
# `_scaled_mm` consumes. The swizzle is a LOCAL, tile-invariant transform when tiles are
# whole 128x128 hp units (= 128x4 e8m0 tiles): each 128x4 scale tile swizzles independently
# into a 32x16 block. The scale is returned as the 4D block grid `(n_row_blocks,
# n_col_blocks, 32, 16)` (see `_to_blocked_4d`): keeping the block axes separate makes it
# tile-invariant under BOTH row and column splits -- MANUAL_TILE reassembles a column tile
# with cat(dim=1) and a row tile with cat(dim=0). The final serialization to the flat
# `_scaled_mm` buffer (a global, grid-shape-dependent step) is `.reshape(-1)`, done once
# outside `f` after reassembly.
# ---------------------------------------------------------------------------
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


def mxfp8_floor_swizzle_f(x, **kwargs):
    qdata, scale_e8m0 = mxfp8_floor_f(x)
    return qdata, _to_blocked_4d(scale_e8m0)


def mxfp8_floor_swizzle_dq_f(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _mxfp8_floor_swizzle_correctness, and importable
    # directly by consumers that need the inverse. Un-swizzle the 4D block grid back to
    # (M, N//32) e8m0, then dequant as mxfp8.
    M, N = q.shape
    rows, cols = M, N // 32
    scale_e8m0 = _from_blocked_4d(scale, rows, cols)
    return mxfp8_floor_dq_f(q, scale_e8m0)


def _mxfp8_floor_swizzle_correctness(
    inputs: Tuple[torch.Tensor, ...], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Assert dequant(outputs) recovers `x` with SQNR above threshold."""
    (x,) = inputs
    qdata, scale = outputs
    x_hat = mxfp8_floor_swizzle_dq_f(qdata, scale)
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 15.0
    assert sqnr > threshold, f"mxfp8_floor_swizzle: sqnr={sqnr.item():.2f} dB below {threshold} dB"


Mxfp8FloorSwizzleGold = QuantCastSingleKernelGold(
    pt_ref_fn=mxfp8_floor_swizzle_f,
    correctness_fn=_mxfp8_floor_swizzle_correctness,
)


# ---------------------------------------------------------------------------
# Golden recipe: float8 tensorwise (per-tensor) scaling.
#
# Unlike the block recipes, the scale is a single per-tensor value that needs a GLOBAL
# reduction over the whole tensor -- that reduction is NOT tile-invariant, so it lives
# OUTSIDE flex_tile_map (`float8_tensorwise_scale`). Given that precomputed scale,
# quantization is just dividing every element by one fixed scalar -- identical across
# tiles, hence tile-invariant -- so it runs INSIDE flex_tile_map via an `f` that takes the
# scale as an explicit REPLICATE aux input (handed whole to every tile).
# ---------------------------------------------------------------------------
def float8_tensorwise_scale(x):
    """Per-tensor scale (global reduction; computed outside flex_tile_map)."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    amax = x.abs().amax().clamp(min=1e-12).to(torch.float32)
    return (amax / fp8_max).to(torch.float32)  # scalar forward scale


def float8_tensorwise_f(x, scale, **kwargs):
    """Tile-invariant `f` taking the precomputed per-tensor `scale` as an explicit aux input
    (REPLICATE: the same scalar scale is used for every tile). `scale` is an input, not a
    returned output, so `f` returns a 1-tuple `(qdata,)`."""
    qdata = (x.to(torch.float32) * (1.0 / scale)).to(torch.float8_e4m3fn)
    return (qdata,)


def dq_tensorwise(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _float8_tensorwise_correctness, and importable
    # directly by consumers that need the inverse.
    return q.float() * scale


def _float8_tensorwise_correctness(
    inputs: Tuple[torch.Tensor, torch.Tensor], outputs: Tuple[torch.Tensor]
) -> None:
    """Assert dequant(outputs, using the precalculated scale from `inputs`) recovers `x`
    with SQNR above threshold."""
    x, scale = inputs
    (qdata,) = outputs
    x_hat = dq_tensorwise(qdata, scale)
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 20.0
    assert sqnr > threshold, f"float8_tensorwise: sqnr={sqnr.item():.2f} dB below {threshold} dB"


Float8TensorwiseGold = QuantCastSingleKernelGold(
    pt_ref_fn=float8_tensorwise_f,
    correctness_fn=_float8_tensorwise_correctness,
)


# nvfp4 format constants (fp4 e2m1 max value + e4m3 scale range).
F4_E2M1_MAX = 6.0
F8E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny


# ---------------------------------------------------------------------------
# Golden recipe: nvfp4 with global scale (two-level) + swizzled inner scale.
#
# A per-tensor fp32 OUTER scale plus a per-16-element e4m3 INNER scale, with fp4-packed
# qdata. The outer scale is a GLOBAL amax reduction -- not tile-invariant -- so it is
# computed OUTSIDE flex_tile_map (`nvfp4_gs_scale`) and passed in as a REPLICATE aux
# input. The inner scale is additionally swizzled into the NVIDIA blocked layout and
# returned as the 4D block grid `(n_row_blocks, n_col_blocks, 32, 16)`, so MANUAL_TILE's
# cat recomposition reproduces the full-tensor swizzle under both row and column splits
# (cf. mxfp8 swizzle).
# ---------------------------------------------------------------------------
def nvfp4_gs_scale(x):
    """Per-tensor fp32 outer scale (global reduction; computed outside flex_tile_map)."""
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


def nvfp4_gs_swizzle_dq_f(q: torch.Tensor, inner_swizzled: torch.Tensor, outer_scale: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _nvfp4_gs_swizzle_correctness, and importable
    # directly by consumers that need the inverse. Takes the per-tensor `outer_scale` as an
    # explicit arg (symmetric with the lifted quant `nvfp4_gs_swizzle_f`).
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


def _nvfp4_gs_swizzle_correctness(
    inputs: Tuple[torch.Tensor, torch.Tensor], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Assert dequant(outputs, using the precalculated outer scale from `inputs`) recovers
    `x` with SQNR above threshold. nvfp4 is 4-bit, coarser than fp8/mxfp8, so a lower floor."""
    x, outer_scale = inputs
    qdata, inner_swizzled = outputs
    x_hat = nvfp4_gs_swizzle_dq_f(qdata, inner_swizzled, outer_scale)
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 12.0
    assert sqnr > threshold, f"nvfp4_gs_swizzle: sqnr={sqnr.item():.2f} dB below {threshold} dB"


Nvfp4GsSwizzleGold = QuantCastSingleKernelGold(
    pt_ref_fn=nvfp4_gs_swizzle_f,
    correctness_fn=_nvfp4_gs_swizzle_correctness,
)


# ---------------------------------------------------------------------------
# Golden recipe: nvfp4 with a 128x128-BLOCKED outer scale (instead of a global scalar).
#
# Same two-level nvfp4 as nvfp4_gs_swizzle_f, but the outer scale is one value per 128x128
# block -- shape (M//128, N//128) -- computed outside and passed as an AuxKind.TILE aux. The
# framework hands `f` the sub-block of the outer scale covering the current tile; `f`
# block-broadcasts it to per-element (option-4 pattern: expand+reshape, no materialized (M,N)
# scale). Because 128 is a multiple of the 16-element inner block, the outer is constant within
# each 16-group, so one representative per 16-group aligns with the inner scale.
# ---------------------------------------------------------------------------
def nvfp4_blocked_outer_scale(x, blk=128):
    """Per-128x128-block fp32 outer scale (block reduction; computed outside flex_tile_map).
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


def nvfp4_blocked_outer_dq_f(q: torch.Tensor, inner_swizzled: torch.Tensor, outer_blocked: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _nvfp4_blocked_outer_correctness, and importable
    # directly by consumers that need the inverse. Reshapes onto the outer block grid so
    # `outer_blocked` broadcasts via size-1 axes (no materialized (M, N) outer scale),
    # mirroring the quant.
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


def _nvfp4_blocked_outer_correctness(
    inputs: Tuple[torch.Tensor, torch.Tensor], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Assert dequant(outputs, using the precalculated blocked outer scale from `inputs`)
    recovers `x` with SQNR above threshold."""
    x, outer_blocked = inputs
    qdata, inner_swizzled = outputs
    x_hat = nvfp4_blocked_outer_dq_f(qdata, inner_swizzled, outer_blocked)
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 12.0
    assert sqnr > threshold, f"nvfp4_blocked_outer: sqnr={sqnr.item():.2f} dB below {threshold} dB"


Nvfp4BlockedOuterGold = QuantCastSingleKernelGold(
    pt_ref_fn=nvfp4_blocked_outer_f,
    correctness_fn=_nvfp4_blocked_outer_correctness,
)


# ---------------------------------------------------------------------------
# Golden recipe: mxfp8 FLOOR with an elementwise bias added before quant.
#
# `bias` is the same shape as the input -> AuxKind.TILE with divisor (1, 1): the framework
# partitions it exactly like the input (one bias element per input element). `f` just adds it
# and runs the existing mxfp8 cast; dequant is the plain mxfp8 dequant (the bias is folded in).
# ---------------------------------------------------------------------------
def mxfp8_bias_f(x, bias, **kwargs):
    """Tile-invariant `f`: add an elementwise `bias` (AuxKind.TILE, per-element) then mxfp8."""
    return mxfp8_floor_f(x + bias.to(x.dtype))


def _mxfp8_bias_correctness(
    inputs: Tuple[torch.Tensor, torch.Tensor], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Assert `outputs` has the expected shape/dtype for `x`. `bias` is a fixed ones-tensor
    (not derived from `x`), so an SQNR-against-`x` check doesn't make sense here -- adding 1
    to every element is not a quantization error, it's the recipe's definition. Shape/dtype is
    the only invariant this recipe promises."""
    x, _bias = inputs
    qdata, scale = outputs
    assert qdata.shape == x.shape, f"mxfp8_bias: qdata shape {qdata.shape} != x shape {x.shape}"
    assert qdata.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float8_e8m0fnu


Mxfp8BiasGold = QuantCastSingleKernelGold(
    pt_ref_fn=mxfp8_bias_f,
    correctness_fn=_mxfp8_bias_correctness,
)


# (recipe_name, gold) -- an index of migrated recipes for discoverability.
RECIPES = [
    ("deepseek_1x128", Deepseek1x128Gold),
    ("deepseek_128x128", Deepseek128x128Gold),
    ("deepseek_1x128_dim_m", Deepseek1x128DimMGold),
    ("rowwise_fp8", RowwiseFp8Gold),
    ("colwise_fp8", ColwiseFp8Gold),
    ("rowwise_precalc", RowwisePrecalcGold),
    ("colwise_precalc", ColwisePrecalcGold),
    ("mxfp8_floor", Mxfp8FloorGold),
    ("mxfp8_floor_swizzle", Mxfp8FloorSwizzleGold),
    ("float8_tensorwise", Float8TensorwiseGold),
    ("nvfp4_gs_swizzle", Nvfp4GsSwizzleGold),
    ("nvfp4_blocked_outer", Nvfp4BlockedOuterGold),
    ("mxfp8_bias", Mxfp8BiasGold),
]
