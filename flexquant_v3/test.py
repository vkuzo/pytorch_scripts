"""Battle-test flex_tile_map against a plain-PyTorch reference, recipe by recipe.

Comparison discipline mirrors flexquant v1/v2 test.py: bit-exact `torch.equal` on both
qdata (compared as fp32) and scale. Recipes live in recipes.py.
"""

import pytest
import torch
import torch.func._random as prng
import torch.nn.functional as F

from api import AuxKind, FlexTileMapBackend, OutputKind, flex_tile_map
from recipes import (
    DEEPSEEK_1X128,
    DEEPSEEK_1X128_DIM_M,
    DEEPSEEK_128X128,
    FLOAT8_TENSORWISE,
    MXFP8_BIAS,
    MXFP8_FLOOR,
    MXFP8_FLOOR_SWIZZLE,
    COLWISE_FP8,
    COLWISE_PRECALC,
    NVFP4_BLOCKED_OUTER,
    NVFP4_GS_SWIZZLE,
    RECIPES_V2,
    ROWWISE_FP8,
    ROWWISE_PRECALC,
    RecipeV2,
    float8_tensorwise_scale,
    hadamard_rht_f,
    hadamard_rht_matrix,
    nvfp4_blocked_outer_scale,
    colwise_precalc_scale,
    nvfp4_gs_scale,
    rowwise_precalc_scale,
    sr_bf16_f,
    sr_bf16_global_f,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


# (recipe_name, recipe, scale shape+dtype, qdata_dtype, flat_compare, sqnr_min).
# These recipes are all NORMAL orientation (no tile-index swap). The dim-M recipe, which uses
# OutputKind.SWAP_TILE_INDEX, has its own dedicated tests (test_deepseek_dim_m_*).
# qdata_dtype: fp4 packs two values per byte (float4_e2m1fn_x2) and is compared via its
# uint8 view; everything else compares as fp32 (see _qdata_equal).
# flat_compare: retained per recipe but now always False -- the swizzled scale is a 4D
# block grid (n_row_blocks, n_col_blocks, 32, 16), which is tile-invariant, so REFERENCE
# and MANUAL_TILE produce the SAME shape and compare bit-exact (no flatten needed).
# sqnr_min: fp8 e4m3 recipes clear ~20 dB; the mxfp8 e8m0 pow2 scale is coarser (15 dB).
# mxfp8_floor_swizzle scale: (256, 8) block scale -> nrb=2, ncb=2 -> (2, 2, 32, 16).
RECIPES = [
    ("deepseek_1x128", DEEPSEEK_1X128, (512, 512 // 128), torch.float32, torch.float8_e4m3fn, False, 20.0),
    ("deepseek_128x128", DEEPSEEK_128X128, (512 // 128, 512 // 128), torch.float32, torch.float8_e4m3fn, False, 20.0),
    ("mxfp8_floor", MXFP8_FLOOR, (512, 512 // 32), torch.float8_e8m0fnu, torch.float8_e4m3fn, False, 15.0),
    ("mxfp8_floor_swizzle", MXFP8_FLOOR_SWIZZLE, (4, 4, 32, 16), torch.float8_e8m0fnu, torch.float8_e4m3fn, False, 15.0),
]


def _compute_error(x, y):
    # torchao's `compute_error` (quantization/utils.py:63) -- SQNR in dB.
    return 20 * torch.log10(
        torch.linalg.vector_norm(x) / torch.linalg.vector_norm(x - y)
    )


def _qdata_equal(a, b):
    # dtype-aware bit-exact qdata compare: packed fp4 (float4_e2m1fn_x2) has no float cast,
    # so compare its raw bytes via the uint8 view; everything else compares as fp32.
    if a.dtype == torch.float4_e2m1fn_x2:
        return torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    return torch.equal(a.to(torch.float32), b.to(torch.float32))


def _recipe_kernel(recipe):
    """The flex_tile_map kernel `f` for either a v1 Recipe (`.quant`) or a migrated
    RecipeV2 (`.pt_ref_fn`) -- the RECIPES table mixes both while migration is ongoing."""
    return recipe.pt_ref_fn if isinstance(recipe, RecipeV2) else recipe.quant


def test_float8_tensorwise_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")

    kernel = FLOAT8_TENSORWISE.pt_ref_fn
    # scale computed outside flex_tile_map, passed in as a REPLICATE aux input. `f` returns
    # only qdata (the scale is an input, not a returned output).
    scale = float8_tensorwise_scale(x)
    (qdata,) = flex_tile_map(
        x,
        kernel,
        aux_inputs=(scale,),
        aux_kinds=(AuxKind.REPLICATE,),
        valid_tile_size_fn=FLOAT8_TENSORWISE.valid_tile_size_fn,
    )
    (qdata_ref,) = kernel(x, scale)

    # shapes / dtypes
    assert qdata.shape == (512, 512)
    assert qdata.dtype == torch.float8_e4m3fn

    # bit-exact vs reference (matches v1/v2 discipline)
    assert torch.equal(qdata.to(torch.float32), qdata_ref.to(torch.float32))


# nvfp4 with global scale needs the runtime outer scale (a REPLICATE aux input), so -- like
# tensorwise -- it lives in dedicated tests rather than the static RECIPES table.
def test_nvfp4_gs_swizzle_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")

    kernel = NVFP4_GS_SWIZZLE.pt_ref_fn
    # outer scale computed outside, passed in as a REPLICATE aux input.
    outer = nvfp4_gs_scale(x)
    qdata, scale = flex_tile_map(
        x,
        kernel,
        aux_inputs=(outer,),
        aux_kinds=(AuxKind.REPLICATE,),
        valid_tile_size_fn=NVFP4_GS_SWIZZLE.valid_tile_size_fn,
    )
    qdata_ref, scale_ref = kernel(x, outer)

    # shapes / dtypes: packed fp4 qdata + swizzled e4m3 inner scale as a 4D block grid.
    # inner scale is (256, 256//16) = (256, 16) -> nrb=2, ncb=4 -> (2, 4, 32, 16).
    assert qdata.shape == (512, 256)
    assert qdata.dtype == torch.float4_e2m1fn_x2
    assert scale.shape == (4, 8, 32, 16)
    assert scale.dtype == torch.float8_e4m3fn

    assert _qdata_equal(qdata, qdata_ref)
    assert torch.equal(scale, scale_ref)


def test_nvfp4_gs_swizzle_backends_match():
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")

    kernel = NVFP4_GS_SWIZZLE.pt_ref_fn
    outer = nvfp4_gs_scale(x)
    kw = dict(
        aux_inputs=(outer,),
        aux_kinds=(AuxKind.REPLICATE,),
        valid_tile_size_fn=NVFP4_GS_SWIZZLE.valid_tile_size_fn,
    )
    qdata_ref, scale_ref = flex_tile_map(
        x, kernel, _backend=FlexTileMapBackend.REFERENCE, **kw
    )
    qdata_tile, scale_tile = flex_tile_map(
        x, kernel, _backend=FlexTileMapBackend.MANUAL_TILE, **kw
    )

    # exercises the _manual_tile packed-fp4 cat (via uint8 view).
    assert _qdata_equal(qdata_tile, qdata_ref)
    # swizzled scale is a 4D block grid: tile-invariant, so same shape AND bit-exact.
    assert scale_tile.shape == scale_ref.shape
    assert torch.equal(scale_tile, scale_ref)


# randomized Hadamard transform (RHT): a non-quant example. `f` returns a 1-tuple `(out,)`
# (no scale) and takes the RHT matrix as a REPLICATE aux input, identical across backends.
def _rht_sign_vector():
    torch.manual_seed(0)
    return torch.randint(0, 2, (16,), device="cuda") * 2 - 1  # length-16 +/-1


def _rht_matrix():
    return hadamard_rht_matrix(_rht_sign_vector(), "cuda", torch.bfloat16)


def test_hadamard_rht_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")

    rht = _rht_matrix()
    (out,) = flex_tile_map(x, hadamard_rht_f, aux_inputs=(rht,), aux_kinds=(AuxKind.REPLICATE,))
    (out_ref,) = hadamard_rht_f(x, rht)

    assert out.shape == (512, 512)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, out_ref)


def test_hadamard_rht_backends_match():
    # single-output f: 256 // 2 == 128 is a multiple of 16, so quadrants don't sever a
    # 16-group -> tile invariant. Exercises the generalized single-output _manual_tile.
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")

    rht = _rht_matrix()
    kw = dict(aux_inputs=(rht,), aux_kinds=(AuxKind.REPLICATE,))
    (out_ref,) = flex_tile_map(x, hadamard_rht_f, _backend=FlexTileMapBackend.REFERENCE, **kw)
    (out_tile,) = flex_tile_map(x, hadamard_rht_f, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)

    assert torch.equal(out_tile, out_ref)


def test_hadamard_rht_roundtrip_sqnr():
    # RHT is orthogonal, so its inverse is its transpose (NOT applying it twice).
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")

    rht = _rht_matrix()
    (y,) = flex_tile_map(x, hadamard_rht_f, aux_inputs=(rht,), aux_kinds=(AuxKind.REPLICATE,))

    M, N = x.shape
    x_rec = (y.reshape(M, N // 16, 16) @ rht.t()).reshape(M, N)
    assert _compute_error(x.float(), x_rec.float()) > 25.0


# stochastic rounding fp32 -> bf16: non-quant, single-output `(out,)`, and by design NOT
# tile-invariant (tile-local RNG offsets repeat across tiles, so tiling changes rounding).
# The PRNG key is a tensor passed as a REPLICATE aux input (built via prng.key(seed)).
def test_sr_bf16_matches_reference():
    # determinism: same key -> bit-exact; different key -> differs.
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.float32, device="cuda")

    key0 = prng.key(0, device="cuda")
    (out,) = flex_tile_map(x, sr_bf16_f, aux_inputs=(key0,), aux_kinds=(AuxKind.REPLICATE,))
    (out_ref,) = sr_bf16_f(x, key0)

    assert out.shape == (512, 512)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, out_ref)

    (out_other,) = sr_bf16_f(x, prng.key(1, device="cuda"))
    assert not torch.equal(out, out_other)


def test_sr_bf16_unbiased():
    # the defining SR property: outputs land on the two bracketing bf16 grid points and
    # E[SR(x)] ~= x. Pick a value strictly between two bf16 values (spacing near 1.0 is
    # 2**-7 ~= 0.0078).
    v = 1.0 + 0.003
    x = torch.full((1024, 1024), v, dtype=torch.float32, device="cuda")

    key0 = prng.key(0, device="cuda")
    (out,) = flex_tile_map(x, sr_bf16_f, aux_inputs=(key0,), aux_kinds=(AuxKind.REPLICATE,))

    lo = torch.tensor(v, dtype=torch.bfloat16).float().item()  # RTN neighbor (round down)
    hi = torch.tensor(v + 2**-7, dtype=torch.bfloat16).float().item()
    assert lo < v < hi
    uniq = set(out.float().unique().tolist())
    assert uniq <= {lo, hi}, f"unexpected values {uniq - {lo, hi}}"
    assert abs(out.float().mean().item() - v) < 1e-3


def test_sr_bf16_tiling_changes_rounding():
    # documents the accepted non-invariance: REFERENCE vs MANUAL_TILE differ bit-for-bit
    # (tile-local offsets repeat), yet both stay unbiased (mean ~= input).
    v = 1.0 + 0.003
    x = torch.full((512, 512), v, dtype=torch.float32, device="cuda")

    key0 = prng.key(0, device="cuda")
    kw = dict(aux_inputs=(key0,), aux_kinds=(AuxKind.REPLICATE,))
    (out_ref,) = flex_tile_map(x, sr_bf16_f, _backend=FlexTileMapBackend.REFERENCE, **kw)
    (out_tile,) = flex_tile_map(x, sr_bf16_f, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)

    assert not torch.equal(out_ref, out_tile)
    assert abs(out_ref.float().mean().item() - v) < 2e-3
    assert abs(out_tile.float().mean().item() - v) < 2e-3


def test_sr_bf16_global_tiling_invariant():
    # the tiling-invariant SR: keyed on GLOBAL element position, so REFERENCE == MANUAL_TILE
    # bit-for-bit (contrast test_sr_bf16_tiling_changes_rounding, which uses the tile-local key).
    v = 1.0 + 0.003
    x = torch.full((512, 512), v, dtype=torch.float32, device="cuda")

    key0 = prng.key(0, device="cuda")
    kw = dict(aux_inputs=(key0,), aux_kinds=(AuxKind.REPLICATE,))
    (out_ref,) = flex_tile_map(x, sr_bf16_global_f, _backend=FlexTileMapBackend.REFERENCE, **kw)
    (out_tile,) = flex_tile_map(x, sr_bf16_global_f, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)

    assert torch.equal(out_ref, out_tile)  # global-position keying is tiling-invariant
    assert abs(out_ref.float().mean().item() - v) < 2e-3  # still unbiased


@pytest.mark.parametrize(
    "recipe, scale_shape, scale_dtype, qdata_dtype",
    [(r, scale_shape, scale_dtype, qdata_dtype) for _, r, scale_shape, scale_dtype, qdata_dtype, _, _ in RECIPES],
    ids=[name for name, *_ in RECIPES],
)
def test_matches_reference(recipe, scale_shape, scale_dtype, qdata_dtype):
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
    kernel = _recipe_kernel(recipe)

    qdata, scale = flex_tile_map(
        x,
        kernel,
        valid_tile_size_fn=recipe.valid_tile_size_fn,
    )
    qdata_ref, scale_ref = kernel(x)

    # shapes / dtypes
    assert qdata.dtype == qdata_dtype
    assert scale.shape == scale_shape
    assert scale.dtype == scale_dtype

    # bit-exact vs reference (matches v1/v2 discipline)
    assert _qdata_equal(qdata, qdata_ref)
    assert torch.equal(scale, scale_ref)


@pytest.mark.parametrize(
    "recipe, flat_compare",
    [(r, flat_compare) for _, r, _, _, _, flat_compare, _ in RECIPES],
    ids=[name for name, *_ in RECIPES],
)
def test_backends_match(recipe, flat_compare):
    # every recipe is tile-invariant, so the MANUAL_TILE backend must match REFERENCE
    # exactly. 256 // 2 == 128 keeps the quadrant split on a 128x128 tile boundary.
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
    kernel = _recipe_kernel(recipe)

    qdata_ref, scale_ref = flex_tile_map(
        x,
        kernel,
        _backend=FlexTileMapBackend.REFERENCE,
        valid_tile_size_fn=recipe.valid_tile_size_fn,
    )
    qdata_tile, scale_tile = flex_tile_map(
        x,
        kernel,
        _backend=FlexTileMapBackend.MANUAL_TILE,
        valid_tile_size_fn=recipe.valid_tile_size_fn,
    )

    assert _qdata_equal(qdata_tile, qdata_ref)
    # every scale (incl. the 4D swizzled block grid) is tile-invariant: same shape, bit-exact.
    del flat_compare  # retained in RECIPES for column layout; no longer needed here
    assert scale_tile.shape == scale_ref.shape
    assert torch.equal(scale_tile, scale_ref)


@pytest.mark.parametrize(
    "name, recipe",
    RECIPES_V2,
    ids=[name for name, _ in RECIPES_V2],
)
def test_gold_correctness(name, recipe):
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")

    # aux_fn (if any) precomputes extra positional args pt_ref_fn takes beyond `x` (e.g. a
    # precalculated scale); inputs is the full tuple correctness_fn compares outputs against.
    aux = recipe.aux_fn(x) if recipe.aux_fn is not None else ()
    inputs = (x, *aux)

    outputs = flex_tile_map(
        x,
        recipe.pt_ref_fn,
        aux_inputs=aux,
        aux_kinds=recipe.aux_kinds,
        output_kinds=recipe.output_kinds,
        valid_tile_size_fn=recipe.valid_tile_size_fn,
    )
    recipe.correctness_fn(inputs, outputs)  # raises AssertionError on failure


# dim-M deepseek: `f` transposes the tile + reduces last dim, and OutputKind.SWAP_TILE_INDEX
# grid-transposes the placement. Together they reproduce deepseek_1x128_f(x.t()) -- the dim-M
# layout that used to be expressed by the removed global_input_transform=SWAP_0_AND_1_AXES.
_DIM_M_SWAP = (OutputKind.SWAP_TILE_INDEX, OutputKind.SWAP_TILE_INDEX)


def test_deepseek_dim_m_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
    kernel = DEEPSEEK_1X128_DIM_M.pt_ref_fn

    qdata, scale = flex_tile_map(
        x,
        kernel,
        output_kinds=_DIM_M_SWAP,
        valid_tile_size_fn=DEEPSEEK_1X128_DIM_M.valid_tile_size_fn,
    )
    # reference: the old SWAP_0_AND_1_AXES layout == plain 1x128 on the transposed input.
    qdata_ref, scale_ref = kernel(x)
    assert qdata.shape == (512, 512)
    assert scale.shape == (512, 512 // 128)
    assert _qdata_equal(qdata, qdata_ref)
    assert torch.equal(scale, scale_ref)


def test_deepseek_dim_m_backends_match():
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")

    kernel = DEEPSEEK_1X128_DIM_M.pt_ref_fn
    kw = dict(output_kinds=_DIM_M_SWAP, valid_tile_size_fn=DEEPSEEK_1X128_DIM_M.valid_tile_size_fn)
    qr, sr = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.REFERENCE, **kw)
    qt, st = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)
    assert _qdata_equal(qt, qr)
    assert st.shape == sr.shape
    assert torch.equal(st, sr)


def test_deepseek_dim_m_non_square():
    # non-square input exercises the grid-transpose (P != Q): a 384x512 input produces a
    # (512, 384) qdata / (512, 3) scale swapped-grid output; REFERENCE == MANUAL_TILE bit-exact.
    torch.manual_seed(0)
    x = torch.randn(384, 512, dtype=torch.bfloat16, device="cuda")

    kernel = DEEPSEEK_1X128_DIM_M.pt_ref_fn
    kw = dict(output_kinds=_DIM_M_SWAP, valid_tile_size_fn=DEEPSEEK_1X128_DIM_M.valid_tile_size_fn)
    qr, sr = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.REFERENCE, **kw)
    qt, st = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)
    assert qr.shape == (512, 384)  # grid-transposed
    assert sr.shape == (512, 384 // 128)
    assert _qdata_equal(qt, qr)
    assert torch.equal(st, sr)


# rowwise / colwise fp8: the scale reduces over a whole row/column, so tiling must span that dim
# (valid_tile_size_fn forces the tile to span all columns / all rows). REFERENCE ==
# MANUAL_TILE bit-exact proves the spanning tile keeps the full-dim reduction intact.
def test_rowwise_fp8_backends_match():
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")

    kernel = ROWWISE_FP8.pt_ref_fn
    kw = dict(valid_tile_size_fn=ROWWISE_FP8.valid_tile_size_fn)
    qr, sr = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.REFERENCE, **kw)
    qt, st = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)
    assert sr.shape == (512, 1)
    assert _qdata_equal(qt, qr)
    assert torch.equal(st, sr)


# rowwise with a PRECALCULATED (M, 1) scale passed as an AuxKind.ROW aux input. The divide is
# tile-invariant under plain 2D tiling (each tile gets its rows' slice of the scale, broadcast
# across columns), so -- unlike ROWWISE_FP8 -- no tiling constraint is needed.
def test_rowwise_precalc_row_aux_backends_match():
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")

    kernel = ROWWISE_PRECALC.pt_ref_fn
    scale = rowwise_precalc_scale(x)  # (512, 1)
    assert scale.shape == (512, 1)
    kw = dict(aux_inputs=(scale,), aux_kinds=(AuxKind.ROW,))  # 2D tiling (default)
    (qr,) = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.REFERENCE, **kw)
    (qt,) = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)

    # matches applying the precalc scale directly, and REFERENCE == MANUAL_TILE bit-exact.
    (q_direct,) = kernel(x, scale)
    assert _qdata_equal(qr, q_direct)
    assert _qdata_equal(qt, qr)


# colwise with a PRECALCULATED (1, N) scale passed as an AuxKind.COL aux input, writing the
# output transposed-contiguous + OutputKind.SWAP_TILE_INDEX. Tile-invariant under plain 2D tiling
# (each tile gets its columns' slice of the scale, broadcast across rows); non-square input makes
# the transposed (N, M) layout observable.
def test_colwise_precalc_col_aux_backends_match():
    torch.manual_seed(0)
    x = torch.randn(512, 384, dtype=torch.bfloat16, device="cuda")

    kernel = COLWISE_PRECALC.pt_ref_fn
    scale = colwise_precalc_scale(x)  # (1, 384)
    assert scale.shape == (1, 384)
    kw = dict(aux_inputs=(scale,), aux_kinds=(AuxKind.COL,), output_kinds=(OutputKind.SWAP_TILE_INDEX,))
    (qr,) = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.REFERENCE, **kw)
    (qt,) = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)

    assert qr.shape == (384, 512)  # transposed (N, M)
    (q_direct,) = kernel(x, scale)
    assert _qdata_equal(qr, q_direct)
    assert _qdata_equal(qt, qr)


# colwise transposes its outputs locally + SWAP_TILE_INDEX, so the emitted layout is the
# transposed (N, M) qdata / (N, 1) scale. Use a non-square input so the transpose is observable.
_COLWISE_SWAP = (OutputKind.SWAP_TILE_INDEX, OutputKind.SWAP_TILE_INDEX)


def test_colwise_fp8_backends_match():
    torch.manual_seed(0)
    x = torch.randn(512, 384, dtype=torch.bfloat16, device="cuda")

    kernel = COLWISE_FP8.pt_ref_fn
    kw = dict(valid_tile_size_fn=COLWISE_FP8.valid_tile_size_fn, output_kinds=_COLWISE_SWAP)
    qr, sr = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.REFERENCE, **kw)
    qt, st = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)
    assert qr.shape == (384, 512)  # transposed (N, M)
    assert sr.shape == (384, 1)  # transposed (N, 1)
    assert _qdata_equal(qt, qr)
    assert torch.equal(st, sr)


# input padding (`pad_input_to_multiple_of`): a ragged input (e.g. LLM decode/prefill token
# dim) is zero-padded up to a multiple so the tile-invariant recipe sees an aligned shape.
# Outputs are returned at the PADDED shape (the swizzle scale grid is 128-row-atom-structured
# and can't be sliced back to an arbitrary original M). Pad multiples are chosen to satisfy
# each recipe's block/atom so the padded shape passes the existing constraint asserts.
def _ceil_to(v, m):
    return ((v + m - 1) // m) * m


def test_valid_tile_size_fn_unsatisfiable_raises_then_pad_fixes():
    # deepseek's predicate (actual[1] % 128 == 0) can't be satisfied on a ragged 512x300 (the
    # 44-wide edge fails, and spanning 300 fails too) -> the tile-size search raises. Padding the
    # columns up to a multiple of 128 makes it satisfiable.
    torch.manual_seed(0)
    x = torch.randn(512, 300, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(ValueError):
        flex_tile_map(
            x,
            DEEPSEEK_1X128.pt_ref_fn,
            valid_tile_size_fn=DEEPSEEK_1X128.valid_tile_size_fn,
            _backend=FlexTileMapBackend.MANUAL_TILE,
        )

    # pad N 300 -> 384 (multiple of 128); now every tile's column extent is 128-aligned.
    qdata, scale = flex_tile_map(
        x,
        DEEPSEEK_1X128.pt_ref_fn,
        valid_tile_size_fn=DEEPSEEK_1X128.valid_tile_size_fn,
        pad_input_to_multiple_of=(1, 128),
        _backend=FlexTileMapBackend.MANUAL_TILE,
    )
    assert qdata.shape == (512, 384)  # returned at the padded shape


def test_pad_ref_shapes_swizzle():
    # ragged 200x300 padded to (128,128)-multiple -> (256, 384); swizzle grid nrb=2, ncb=3.
    torch.manual_seed(0)
    x = torch.randn(200, 300, dtype=torch.bfloat16, device="cuda")
    qdata, scale = flex_tile_map(
        x,
        MXFP8_FLOOR_SWIZZLE.pt_ref_fn,
        pad_input_to_multiple_of=(128, 128),
        valid_tile_size_fn=MXFP8_FLOOR_SWIZZLE.valid_tile_size_fn,
    )
    assert qdata.shape == (256, 384)
    assert scale.shape == (2, 3, 32, 16)


@pytest.mark.parametrize(
    "recipe, pad_to",
    [
        (MXFP8_FLOOR, (1, 32)),
        (MXFP8_FLOOR_SWIZZLE, (128, 128)),
        (DEEPSEEK_1X128, (1, 128)),
    ],
    ids=["mxfp8_floor", "mxfp8_floor_swizzle", "deepseek_1x128"],
)
def test_pad_backends_match(recipe, pad_to):
    # padded ragged input: MANUAL_TILE must match REFERENCE bit-exact (padding happens before
    # tiling in both paths, so the two backends see the identical padded tensor).
    torch.manual_seed(0)
    x = torch.randn(200, 300, dtype=torch.bfloat16, device="cuda")
    kernel = _recipe_kernel(recipe)
    kw = dict(
        pad_input_to_multiple_of=pad_to,
        valid_tile_size_fn=recipe.valid_tile_size_fn,
    )
    qdata_ref, scale_ref = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.REFERENCE, **kw)
    qdata_tile, scale_tile = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)
    assert _qdata_equal(qdata_tile, qdata_ref)
    assert scale_tile.shape == scale_ref.shape
    assert torch.equal(scale_tile, scale_ref)


def test_pad_matches_manual_pad():
    # padding inside the API == padding the input outside it, then running the recipe.
    torch.manual_seed(0)
    x = torch.randn(200, 300, dtype=torch.bfloat16, device="cuda")
    kernel = MXFP8_FLOOR.pt_ref_fn
    qdata, scale = flex_tile_map(
        x,
        kernel,
        pad_input_to_multiple_of=(1, 32),
        valid_tile_size_fn=MXFP8_FLOOR.valid_tile_size_fn,
    )
    # manual pad: 200 stays (mult of 1), 300 -> 320 (mult of 32); high-edge zero pad.
    x_padded = F.pad(x, (0, _ceil_to(300, 32) - 300, 0, 0))
    qdata_ref, scale_ref = kernel(x_padded)
    assert _qdata_equal(qdata, qdata_ref)
    assert torch.equal(scale, scale_ref)


# AuxKind.TILE: nvfp4 with a 128x128-blocked outer scale. The framework slices the outer scale
# (2, 2) to the sub-block covering each tile; `f` block-broadcasts it. Divisor = 256//2 = 128.
def test_nvfp4_blocked_outer_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    kernel = NVFP4_BLOCKED_OUTER.pt_ref_fn
    outer = nvfp4_blocked_outer_scale(x)  # (2, 2)
    assert outer.shape == (2, 2)
    qdata, scale = flex_tile_map(
        x,
        kernel,
        aux_inputs=(outer,),
        aux_kinds=(AuxKind.TILE,),
        valid_tile_size_fn=NVFP4_BLOCKED_OUTER.valid_tile_size_fn,
    )
    qdata_ref, scale_ref = kernel(x, outer)

    assert qdata.shape == (256, 128)
    assert qdata.dtype == torch.float4_e2m1fn_x2
    assert scale.shape == (2, 4, 32, 16)
    assert _qdata_equal(qdata, qdata_ref)
    assert torch.equal(scale, scale_ref)


def test_nvfp4_blocked_outer_backends_match():
    # the real proof TILE slicing + f's block-broadcast compose: REFERENCE == MANUAL_TILE.
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    kernel = NVFP4_BLOCKED_OUTER.pt_ref_fn
    outer = nvfp4_blocked_outer_scale(x)
    kw = dict(
        aux_inputs=(outer,),
        aux_kinds=(AuxKind.TILE,),
        valid_tile_size_fn=NVFP4_BLOCKED_OUTER.valid_tile_size_fn,
    )
    qdata_ref, scale_ref = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.REFERENCE, **kw)
    qdata_tile, scale_tile = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)

    assert _qdata_equal(qdata_tile, qdata_ref)
    assert scale_tile.shape == scale_ref.shape
    assert torch.equal(scale_tile, scale_ref)


# test_nvfp4_blocked_outer_sqnr_vs_high_precision has migrated to quant_cast_gold (see
# quant_cast_gold/recipes.py) and is now covered by the generic test_gold_correctness above.


# AuxKind.TILE with divisor (1, 1): an elementwise bias (same shape as input) added before mxfp8.
def test_mxfp8_bias_backends_match():
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    kernel = MXFP8_BIAS.pt_ref_fn
    kw = dict(aux_inputs=(bias,), aux_kinds=(AuxKind.TILE,), valid_tile_size_fn=MXFP8_BIAS.valid_tile_size_fn)
    qdata_ref, scale_ref = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.REFERENCE, **kw)
    qdata_tile, scale_tile = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)

    # matches a direct (whole-tensor) bias-add + quant, and REFERENCE == MANUAL_TILE.
    qdata_direct, scale_direct = kernel(x, bias)
    assert _qdata_equal(qdata_ref, qdata_direct)
    assert torch.equal(scale_ref, scale_direct)
    assert _qdata_equal(qdata_tile, qdata_ref)
    assert torch.equal(scale_tile, scale_ref)
