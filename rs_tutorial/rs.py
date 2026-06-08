"""Educational stochastic rounding (SR) from float32 to fp4_e2m1fn.

This is a from-scratch, readability-first implementation meant for *understanding*
stochastic rounding, not for speed. A production kernel would do this with bit
manipulation (see ``flexquant/nvfp4_utils.py`` in this repo for a round-to-nearest
version); here we keep everything explicit.

--------------------------------------------------------------------------------
What is stochastic rounding?
--------------------------------------------------------------------------------
When we quantize a float ``x`` to a coarse grid, it usually falls *between* two
representable values ``lo <= x <= hi``. Two ways to pick which one to keep:

  * round-to-nearest: always snap to whichever of lo/hi is closer. Deterministic,
    but biased for a stream of values -- e.g. if every x is slightly above lo, they
    all round down to lo and the small positive remainders are lost forever. This
    bias is what kills low-precision training.

  * stochastic rounding: round *up* with probability proportional to how far x has
    travelled from lo toward hi:

        p_up = (x - lo) / (hi - lo)

    Then round up if a random draw r ~ Uniform[0, 1) satisfies  r < p_up.

    The point: in expectation the rounded value equals x exactly --

        E[round(x)] = lo * (1 - p_up) + hi * p_up
                    = lo + p_up * (hi - lo)
                    = lo + (x - lo)
                    = x

    So SR is *unbiased*. Errors are random rather than systematic, so they tend to
    cancel when accumulated instead of compounding. That's why SR matters for
    low-precision training and gradient accumulation.

--------------------------------------------------------------------------------
fp4_e2m1fn
--------------------------------------------------------------------------------
fp4 e2m1 = 4 bits: 1 sign, 2 exponent, 1 mantissa. "fn" = finite (no inf/NaN
encodings reserved). That leaves 16 codes mapping to just 16 distinct values
(8 magnitudes, each with a sign). We store one fp4 code in the low 4 bits of a
uint8 ("unpacked" -- packing two codes into one byte is a separate concern).
"""

import torch
from torch import Tensor

# The 16 fp4 e2m1 values, indexed by their 4-bit code (sign-magnitude):
#   bit 3 = sign, bits 2:0 = magnitude index.
# Copied from `flexquant/nvfp4_utils.py` (`_FP4_E2M1_VALS`) so this file is
# self-contained.
FP4_E2M1_VALS = torch.tensor(
    [
         0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)

# The 8 representable magnitudes, ascending. Index i here is exactly the low-3-bit
# magnitude code, which is convenient for reassembling the final code below.
FP4_E2M1_MAGS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)

F4_E2M1_MAX = 6.0  # largest representable magnitude

# --- fp32 / fp4 bit-layout constants, used by the bits-based SR below ---------
# fp32: 1 sign | 8 exponent | 23 mantissa.  fp4 e2m1: 1 sign | 2 exponent | 1
# mantissa.  These mirror the constants in MSLK's fp4 Triton cast
# (`/home/dev/MSLK/mslk/quantize/triton/fp4_quantize.py`), specialised here from
# fp16 to fp32 source bit widths.
MBITS_F32, EBITS_F32 = 23, 8
F32_EXP_BIAS = 127
F32_EXP_OFFSET = MBITS_F32                  # exponent starts at bit 23
F32_EXP_MASK = 0x7F800000                   # bits 30:23
F32_MANTISSA_MASK = 0x007FFFFF              # bits 22:0
F32_IMPLIED_1 = 1 << MBITS_F32              # the hidden leading 1 for normals

EBITS_F4, MBITS_F4 = 2, 1
F4_EXP_BIAS = 1
MBITS_F4_IMPLICIT = MBITS_F4 + 1            # 1 stored + 1 implied = 2 sig bits
MANTISSA_OVERFLOW = (1 << MBITS_F4_IMPLICIT) - 1   # 0b11 = 3
EXPONENT_OVERFLOW = (1 << EBITS_F4) - 1            # 0b11 = 3
IMPLICIT_1_MASK_F4 = (1 << (MBITS_F4_IMPLICIT - 1)) - 1  # keep only stored mantissa bit -> 0b1

# Stochastic-rounding random mask, for a fp4 *normal-range* target. fp32 keeps 23
# mantissa bits; an fp4 normal keeps 1; so 22 bits are discarded, and a uniform
# random integer over exactly those 22 bits turns truncation into stochastic
# rounding (see `_sr_to_fp4_with_bits`).
#
# IMPORTANT: 22 is only the right count when the fp4 target is *normal* (|value| in
# [1.0, 6.0]). For an fp4 *subnormal* target (|value| < 1.0, the grid {0, 0.5}) the
# cast drops MORE than 22 bits, and the count varies with magnitude -- see the long
# comment in `_sr_to_fp4_with_bits` step 6. That function therefore sizes its dither
# mask per element and does NOT use this constant directly. We keep RAND_MASK as the
# canonical "normal-range" value because it's the constant MSLK's kernel uses (a
# compile-time mask is cheaper than a per-element one); the tradeoff is explained at
# the call site.
RAND_MASK = (1 << (MBITS_F32 - MBITS_F4)) - 1   # (1 << 22) - 1 = 0x3FFFFF


def stochastic_round_to_fp4_e2m1(x: Tensor, rand: Tensor) -> Tensor:
    """Stochastically round float32 tensor ``x`` to fp4 e2m1 codes (uint8).

    Args:
        x:    float32 tensor of any shape.
        rand: float32 tensor, same shape as ``x``, values in [0, 1). One random
              draw per element decides that element's rounding direction.

    Returns:
        uint8 tensor, same shape as ``x``. Each element holds a 4-bit fp4 e2m1
        code in its low 4 bits (bit 3 = sign, bits 2:0 = magnitude index).
    """
    assert x.dtype == torch.float32, f"x must be float32, got {x.dtype}"
    assert rand.shape == x.shape, f"rand shape {rand.shape} != x shape {x.shape}"

    mags = FP4_E2M1_MAGS.to(x.device)

    # 1. Split sign from magnitude. We round on the magnitude against a monotonic
    #    grid, then reattach the sign. (signbit so that -0.0 keeps its sign.)
    sign = x.signbit()           # bool: True where x is negative
    mag = x.abs()

    # 2. Clamp to the representable range. fp4 e2m1fn has no inf/NaN, so anything
    #    above the max magnitude saturates to 6.0 (sign preserved).
    mag = mag.clamp(max=F4_E2M1_MAX)

    # 3. Bracket each magnitude between two grid points: lo <= mag <= hi.
    #    searchsorted(mags, mag, right=True) returns the index of the first grid
    #    point strictly greater than mag -> that's our hi index. lo = hi - 1.
    hi_idx = torch.searchsorted(mags, mag, right=True)
    hi_idx = hi_idx.clamp(max=len(mags) - 1)   # mag == 6.0 would land out of range
    lo_idx = (hi_idx - 1).clamp(min=0)

    lo = mags[lo_idx]
    hi = mags[hi_idx]

    # 4. Probability of rounding up = fractional position of mag within [lo, hi].
    #    Guard the lo == hi case (mag is exactly representable, or sits below the
    #    smallest gap): no rounding needed, p_up = 0.
    gap = hi - lo
    p_up = torch.where(gap > 0, (mag - lo) / gap, torch.zeros_like(mag))

    # 5. The stochastic decision: round up when the random draw falls below p_up.
    round_up = rand < p_up
    mag_idx = lo_idx + round_up.to(lo_idx.dtype)   # chosen magnitude's grid index

    # 6. Reassemble the 4-bit code: bits 2:0 = magnitude index, bit 3 = sign.
    code = mag_idx.to(torch.uint8) | (sign.to(torch.uint8) << 3)
    return code


def _sr_to_fp4_with_bits(x: Tensor, rand_bits: Tensor) -> Tensor:
    """Core of the production-style (bit-trick) stochastic rounding.

    This is how real low-precision kernels do stochastic rounding -- with integer
    arithmetic on the float's bit pattern, no float division and no lookup table.
    Ported from MSLK's fp4 Triton cast
    (`/home/dev/MSLK/mslk/quantize/triton/fp4_quantize.py`, the value-level path),
    specialised to an fp32 source.

    The whole idea in one line (step 6 below):

        significand + (uniform random over the bits being dropped)  -->  truncate

    To reach fp4 we keep only the top 2 significand bits and drop the low `k` bits
    (k = 22 for a value in fp4's normal range, more for subnormals). Write the true
    value as  grid_lo + dropped/2^k  where the `dropped` low bits encode the fraction.
    Truncation alone always floors -> biased. But if we first ADD a uniform random
    integer r over those bits, the sum carries up into the kept bit with probability
    ~ dropped/2^k, which is precisely the stochastic-rounding probability the
    float-based `stochastic_round_to_fp4_e2m1` computes as `p_up = (x-lo)/(hi-lo)`.
    Same math, but for free as a carry out of an integer add.

    IMPORTANT -- faithful to MSLK, including its bias: we dither with MSLK's FIXED
    22-bit mask `RAND_MASK`, not a per-element mask. That makes SR exact/unbiased for
    fp4-NORMAL targets (k == 22) but biased TOWARD ZERO for fp4-SUBNORMAL targets
    (|x| < 1.0, k > 22), where the top dropped bits are never dithered and just get
    floored. See step 6 for the full explanation. (MSLK lives with this because it's
    MXFP4: block-scaling sends the dominant elements into the normal range.)

    Args:
        x:         float32 tensor of any shape.
        rand_bits: int32 tensor, same shape, of random bits (one independent draw
                   per element). Only the low 22 bits (`RAND_MASK`) are used.

    Returns:
        uint8 fp4 e2m1 codes in the low 4 bits, same contract as
        :func:`stochastic_round_to_fp4_e2m1`.
    """
    assert x.dtype == torch.float32, f"x must be float32, got {x.dtype}"
    assert rand_bits.shape == x.shape, "rand_bits must match x shape"

    # Work in int32: PyTorch has no uint32, and several bit ops don't accept it,
    # so we stay in signed int32 and are careful with the sign bit (same approach
    # as `flexquant/nvfp4_utils.py`).
    xi = x.contiguous().view(torch.int32)
    rand_bits = rand_bits.to(torch.int32)

    # 1. Peel off the sign bit (bit 31); round the magnitude, reattach sign later.
    # 1s:8e:23m -> s
    sign = (xi >> 31) & 0x1
    # 1s:8e:23m -> 0:8e:23m
    mag = xi & 0x7FFFFFFF

    # 2. Split the magnitude into fp32 exponent + mantissa fields. (The
    #    stochastic-rounding dither is applied in step 6, once we know exactly how
    #    many low bits the truncation will drop -- see the note there.)
    # 0:8e:23m -> 8e
    biased_exp = (mag & F32_EXP_MASK) >> F32_EXP_OFFSET
    # 0:8e:23m -> 23m
    trailing_mantissa = mag & F32_MANTISSA_MASK
    is_subnorm_src = biased_exp == 0

    # 3. Rebias the exponent from fp32 to fp4's range.
    new_biased_exp = biased_exp - F32_EXP_BIAS + F4_EXP_BIAS

    # 4. When the target exponent underflows fp4's normal range (<= 0), the value
    #    must become an fp4 *subnormal*: we keep fewer mantissa bits and shift the
    #    implied 1 down. `exp_diff` is how many extra bits to shift for that, clamped
    #    so we never shift by more than the mantissa has.
    exp_diff = torch.where(
        new_biased_exp <= 0, 1 - new_biased_exp, torch.zeros_like(new_biased_exp)
    )
    exp_diff = exp_diff.clamp(max=MBITS_F32 + 1)

    # 5. Build the significand: prepend fp32's implied leading 1 for normal inputs.
    mantissa = torch.where(
        is_subnorm_src, trailing_mantissa, trailing_mantissa + F32_IMPLIED_1
    )
    # Number of significant bits above the binary point: 24 for normals (1 implied
    # + 23 stored), 23 for subnormals (no implied 1).
    sig_bits = torch.where(
        is_subnorm_src,
        torch.full_like(mantissa, MBITS_F32),
        torch.full_like(mantissa, MBITS_F32 + 1),
    )

    # 6. THE stochastic-rounding step, then truncate.
    #
    #    `shift` is exactly how many low bits of the significand get dropped to reach
    #    fp4 width:
    #        shift = sig_bits + exp_diff - MBITS_F4_IMPLICIT
    #    For a fp4 *normal* target exp_diff == 0, so shift == 22 (drop 22 of fp32's
    #    23 mantissa bits, keep 1). For a fp4 *subnormal* target exp_diff > 0, so
    #    shift == 22 + exp_diff -- and the smaller the value, the larger exp_diff,
    #    the more bits dropped.
    #
    #    To round stochastically we add a uniform random integer over the dropped bits,
    #    then floor. Write the value as  grid_lo + dropped/2^shift  with dropped in
    #    [0, 2^shift). Adding a uniform r carries into the kept bit iff dropped + r
    #    spills past 2^shift, i.e. with probability ~ dropped/2^shift -- which is
    #    `p_up = (x - lo)/(hi - lo)` from the float-based function. For that carry
    #    probability to be exact, r must span ALL `shift` dropped bits.
    #
    #    We deliberately match MSLK here and use its FIXED compile-time mask
    #    `RAND_MASK = (1<<22)-1` rather than a per-element `(1<<shift)-1`. The
    #    consequences:
    #      * fp4-NORMAL target (shift == 22): the mask spans every dropped bit, so SR
    #        is exact / unbiased. This is the common case in MSLK because it's MXFP4 --
    #        a block of values shares one power-of-two scale taken from the block max,
    #        so the largest elements normalize into fp4's normal range.
    #      * fp4-SUBNORMAL target (shift > 22): the 22-bit mask dithers only the low 22
    #        of the `shift` dropped bits; the top `exp_diff` bits above the mask window
    #        get floored deterministically, with no chance to round up. The result is
    #        biased TOWARD ZERO, by up to ~half an fp4-ulp, and worse the smaller the
    #        value (e.g. x=0.3 -> mean ~0.0 instead of 0.3). In MXFP4 these are the
    #        smallest, least significant elements of each block, and a constant mask is
    #        one branch-free op vs. a per-element variable shift -- so MSLK accepts it.
    #
    #    (A per-element `(1<<shift)-1` mask would dither every dropped bit and stay
    #    unbiased on subnormals too; we don't do that, on purpose, to mirror MSLK.)
    shift = sig_bits + exp_diff - MBITS_F4_IMPLICIT
    shift = shift.clamp(min=0)
    mantissa = mantissa + (rand_bits & RAND_MASK)        # FIXED 22-bit mask (MSLK)
    mantissa = mantissa >> shift

    # 7. Handle mantissa overflow (e.g. 1.111.. rounded up ticks the exponent).
    overflow = mantissa > MANTISSA_OVERFLOW
    mantissa = torch.where(
        overflow & (~is_subnorm_src), mantissa >> 1, mantissa
    )
    # A subnormal whose mantissa rounded up to the implied-1 position becomes the
    # smallest normal (exp = 1).
    new_biased_exp = torch.where(
        (new_biased_exp <= 0) & (mantissa == 2),
        torch.ones_like(new_biased_exp),
        new_biased_exp,
    )
    # Drop the implied 1, keeping only fp4's stored mantissa bit.
    mantissa = mantissa & IMPLICIT_1_MASK_F4
    new_biased_exp = torch.where(overflow, new_biased_exp + 1, new_biased_exp)
    # Saturate: anything past the top fp4 exponent clamps to the max magnitude (6.0).
    mantissa = torch.where(
        new_biased_exp > EXPONENT_OVERFLOW, torch.ones_like(mantissa), mantissa
    )
    new_biased_exp = new_biased_exp.clamp(min=0, max=EXPONENT_OVERFLOW)

    # 8. Assemble the 4-bit e2m1 code: [sign(1) | exp(2) | mantissa(1)].
    code = (new_biased_exp << MBITS_F4) | mantissa
    code = (sign << (EBITS_F4 + MBITS_F4)) | code
    return code.to(torch.uint8)


def stochastic_round_to_fp4_e2m1_bits(x: Tensor, seed: int) -> Tensor:
    """Production-style stochastic rounding of float32 -> fp4 e2m1, from a seed.

    Uses the integer bit trick that real kernels use (see
    :func:`_sr_to_fp4_with_bits`) and takes a single integer ``seed`` instead of a
    float random tensor. It matches the float-based
    :func:`stochastic_round_to_fp4_e2m1` distribution for fp4-NORMAL targets, but --
    faithful to MSLK -- is biased toward zero for fp4-SUBNORMAL targets (|x| < 1.0);
    see :func:`_sr_to_fp4_with_bits`.

    From that one seed we generate an *independent random integer per element*. This
    mirrors how MSLK's kernel works: the host draws one seed per group, and the
    kernel expands it into per-element random bits with a counter-based RNG (Philox,
    keyed on each element's offset). We don't have Triton's `randint4x` here, so a
    seeded `torch.Generator` stands in for that expansion.

    Args:
        x:    float32 tensor of any shape.
        seed: int seed; same seed + same shape -> same rounding (reproducible).

    Returns:
        uint8 tensor of fp4 e2m1 codes (low 4 bits), same shape as ``x``.
    """
    assert x.dtype == torch.float32, f"x must be float32, got {x.dtype}"
    gen = torch.Generator(device=x.device).manual_seed(seed)
    # One uniform 31-bit random integer per element (low 22 bits are what matter).
    rand_bits = torch.randint(
        0, 2**31, x.shape, generator=gen, dtype=torch.int32, device=x.device
    )
    return _sr_to_fp4_with_bits(x, rand_bits)


def dequantize_fp4_e2m1(codes: Tensor) -> Tensor:
    """Decode fp4 e2m1 codes (uint8) back to float32 values, via table lookup.

    Inverse of :func:`stochastic_round_to_fp4_e2m1` (lossy: the rounding already
    happened). Handy for inspecting/verifying what the codes represent.
    """
    assert codes.dtype == torch.uint8, f"codes must be uint8, got {codes.dtype}"
    vals = FP4_E2M1_VALS.to(codes.device)
    return vals[codes.long()]
