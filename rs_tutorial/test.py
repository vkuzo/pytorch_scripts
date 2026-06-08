"""Tests for the stochastic rounding implementation in rs.py.

Run with:  pytest test.py

We pick x = 2.6, which sits between the fp4 grid points 2.0 and 3.0, so:

    p_up = (2.6 - 2.0) / (3.0 - 2.0) = 0.6

i.e. with a fair random draw we'd round to 3.0 about 60% of the time and to 2.0
about 40% of the time.
"""

import pytest
import torch

from rs import (
    stochastic_round_to_fp4_e2m1,
    stochastic_round_to_fp4_e2m1_bits,
    _sr_to_fp4_with_bits,
    dequantize_fp4_e2m1,
    RAND_MASK,
)

torch.manual_seed(0)

# The benchmark kernels need CUDA + triton; the hardware path additionally needs
# SM100 (Blackwell) for the cvt.rs fp4 intrinsic. Guard the GPU tests accordingly.
GPU = torch.cuda.is_available()
SM100 = GPU and torch.cuda.get_device_capability()[0] >= 10
requires_hw = pytest.mark.skipif(
    not SM100, reason="needs CUDA + SM100 (Blackwell) for the fp4 cvt.rs intrinsic"
)


def test_single_input_rounds_up():
    # x = 2.6 brackets 2.0 / 3.0, so p_up = 0.6. A draw of 0.5 < 0.6 rounds up.
    x = torch.tensor([2.6], dtype=torch.float32)
    rand = torch.tensor([0.5], dtype=torch.float32)

    code = stochastic_round_to_fp4_e2m1(x, rand)
    deq = dequantize_fp4_e2m1(code)

    assert code.item() == 5      # magnitude index 5 (-> 3.0), sign bit 0
    assert deq.item() == 3.0


def test_single_input_rounds_down():
    # Same x, but a draw of 0.8 >= 0.6 rounds down to 2.0.
    x = torch.tensor([2.6], dtype=torch.float32)
    rand = torch.tensor([0.8], dtype=torch.float32)

    deq = dequantize_fp4_e2m1(stochastic_round_to_fp4_e2m1(x, rand))

    assert deq.item() == 2.0


def test_rounding_up_probability():
    # Draw many random values for a fixed x and check that the fraction rounded
    # up matches p_up = (2.6 - 2.0) / (3.0 - 2.0) = 0.6. This is the empirical
    # demonstration that stochastic rounding is unbiased.
    n = 100
    x = torch.full((n,), 2.6, dtype=torch.float32)
    rand = torch.rand(n, dtype=torch.float32)

    deq = dequantize_fp4_e2m1(stochastic_round_to_fp4_e2m1(x, rand))

    expected_p_up = 0.6
    actual_p_up = (deq == 3.0).float().mean().item()

    print(f"\nexpected P(round up): {expected_p_up}")
    print(f"actual   P(round up): {actual_p_up}  ({n} draws)")

    # Loose bound -- with only 100 draws there's real sampling noise.
    assert abs(actual_p_up - expected_p_up) < 0.15


# --------------------------------------------------------------------------------
# Tests for the production-style, random-BITS implementation.
# These drive the private `_sr_to_fp4_with_bits(x, rand_bits)` helper for the
# deterministic edge cases, and the public seed-based API for the probability test.
# --------------------------------------------------------------------------------


def _deq_with_bits(x, rand_bits_value):
    """Round x using a constant rand_bits value for every element, then dequantize."""
    rand_bits = torch.full_like(x, rand_bits_value, dtype=torch.int32)
    return dequantize_fp4_e2m1(_sr_to_fp4_with_bits(x, rand_bits))


def test_bits_extremes_normal_range():
    # x = 2.6 brackets 2.0 / 3.0. rand_bits = 0 adds no dither -> truncate -> 2.0.
    # rand_bits = RAND_MASK is the max dither -> always carries up -> 3.0.
    x = torch.tensor([2.6], dtype=torch.float32)

    assert _deq_with_bits(x, 0).item() == 2.0
    assert _deq_with_bits(x, RAND_MASK).item() == 3.0


def test_bits_subnormal_ceil_is_unreachable_with_fixed_mask():
    # Below 1.0, fp4's grid is {0, 0.5, 1.0} (the subnormal region), where the cast
    # drops MORE than RAND_MASK's 22 bits. Because we deliberately mirror MSLK's
    # FIXED 22-bit mask (not a per-element one), the top dropped bits are never
    # dithered: they are always floored. So even the maximum dither (all 31 random
    # bits set) cannot push 0.3 up to 0.5 here -- it stays floored at 0.0. This is
    # the visible symptom of the subnormal approximation documented in rs.py.
    x = torch.tensor([0.3, 0.7], dtype=torch.float32)

    assert _deq_with_bits(x, 0).tolist() == [0.0, 0.5]            # truncate toward 0
    assert _deq_with_bits(x, 0x7FFFFFFF).tolist() == [0.0, 0.5]   # same! (fixed mask)


def test_bits_rounding_up_probability():
    # Same demonstration as test_rounding_up_probability, but through the public
    # seed-based bit-trick API. p_up = (2.6 - 2.0) / (3.0 - 2.0) = 0.6.
    n = 100
    x = torch.full((n,), 2.6, dtype=torch.float32)

    deq = dequantize_fp4_e2m1(stochastic_round_to_fp4_e2m1_bits(x, seed=0))

    expected_p_up = 0.6
    actual_p_up = (deq == 3.0).float().mean().item()

    print(f"\n[bits] expected P(round up): {expected_p_up}")
    print(f"[bits] actual   P(round up): {actual_p_up}  ({n} draws)")

    assert abs(actual_p_up - expected_p_up) < 0.15


def test_bits_unbiased_in_normal_range():
    # The defining property of stochastic rounding: E[round(x)] == x. With MSLK's
    # fixed 22-bit mask this holds EXACTLY for fp4-normal targets (|x| >= 1.0), where
    # the cast drops exactly 22 bits and the mask dithers all of them.
    xs = [2.6, 1.2, 4.9, 5.5, 1.7, 3.5]
    n = 200_000
    for xv in xs:
        x = torch.full((n,), xv, dtype=torch.float32)
        deq = dequantize_fp4_e2m1(stochastic_round_to_fp4_e2m1_bits(x, seed=0))
        assert abs(deq.mean().item() - xv) < 0.005 * xv


def test_bits_subnormal_is_biased_toward_zero():
    # Counterpart to the test above: for fp4-SUBNORMAL targets (|x| < 1.0) the cast
    # drops more than 22 bits, so the fixed 22-bit mask leaves the top bits undithered
    # -> they are floored -> the mean is biased TOWARD ZERO. This documents MSLK's
    # approximation (see the long comment in rs.py step 6). The per-element-mask
    # variant would instead be unbiased here.
    #
    # Expected means measured empirically: 0.7 -> ~0.5, 0.3 -> ~0.0, 0.9 -> ~0.8.
    n = 200_000
    cases = [(0.7, 0.50), (0.3, 0.00), (0.9, 0.80)]
    for xv, expected_mean in cases:
        x = torch.full((n,), xv, dtype=torch.float32)
        deq = dequantize_fp4_e2m1(stochastic_round_to_fp4_e2m1_bits(x, seed=0))
        mean = deq.mean().item()
        # It is biased away from the input...
        assert mean < xv - 0.05, f"x={xv}: expected downward bias, got mean {mean}"
        # ...and lands near the empirically-known biased mean.
        assert abs(mean - expected_mean) < 0.02, f"x={xv}: mean {mean} != {expected_mean}"


def test_bits_seed_is_reproducible():
    # Same seed + same shape must give identical codes.
    x = torch.full((64,), 2.6, dtype=torch.float32)
    assert torch.equal(
        stochastic_round_to_fp4_e2m1_bits(x, seed=123),
        stochastic_round_to_fp4_e2m1_bits(x, seed=123),
    )


# --------------------------------------------------------------------------------
# GPU tests: the two benchmark kernels (software bit-trick vs hardware cvt.rs
# intrinsic) should agree statistically. They use the same RNG but the hardware
# packs nibbles/values in its own order and rounds in hardware, so they are NOT
# elementwise-equal -- we compare order-invariant statistics: the histogram over
# the 16 fp4 codes and the mean of dequantized values.
#
# Comparisons use input restricted to the fp4 NORMAL range |x| in [1, 6], where
# both paths are unbiased (rs.py's fixed 22-bit mask biases subnormals toward
# zero; the hardware unit need not share that bias).
# --------------------------------------------------------------------------------


def _unpack_fp4(packed):
    """Unpack uint8 [M, N//2] (2 fp4 codes/byte) to fp4 codes [M, N]."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    # Order within the byte doesn't matter for a histogram or a mean.
    return torch.stack([low, high], dim=-1).reshape(packed.shape[0], -1)


def _normal_range_input(m, n, seed):
    """bf16 tensor with |x| in [1, 6] (fp4 normal range), random sign."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    mag = torch.rand(m, n, generator=g, device="cuda") * 5.0 + 1.0
    sign = torch.where(
        torch.rand(m, n, generator=g, device="cuda") < 0.5, -1.0, 1.0
    )
    return (mag * sign).bfloat16()


def _code_histogram(deq):
    """Normalized histogram of dequantized values over the 16 fp4 magnitudes."""
    grid = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=deq.device,
    )
    # Map each value to its grid index, then bincount.
    idx = (deq.reshape(-1, 1) == grid.reshape(1, -1)).float().argmax(dim=1)
    return torch.bincount(idx, minlength=grid.numel()).float() / idx.numel()


@requires_hw
def test_hw_sw_histogram_match():
    from benchmark import sr_fp4_software, sr_fp4_hardware

    x = _normal_range_input(4096, 4096, seed=0)
    sw = dequantize_fp4_e2m1(_unpack_fp4(sr_fp4_software(x, 0)).to(torch.uint8))
    hw = dequantize_fp4_e2m1(_unpack_fp4(sr_fp4_hardware(x, 0)).to(torch.uint8))

    h_sw = _code_histogram(sw)
    h_hw = _code_histogram(hw)
    max_diff = (h_sw - h_hw).abs().max().item()
    assert max_diff < 0.01, f"histograms differ by {max_diff}"


@requires_hw
def test_hw_sw_mean_match():
    from benchmark import sr_fp4_software, sr_fp4_hardware

    x = _normal_range_input(4096, 4096, seed=0)
    sw = dequantize_fp4_e2m1(_unpack_fp4(sr_fp4_software(x, 0)).to(torch.uint8))
    hw = dequantize_fp4_e2m1(_unpack_fp4(sr_fp4_hardware(x, 0)).to(torch.uint8))

    in_mean = x.float().mean().item()
    # The two paths agree with each other...
    assert abs(sw.mean().item() - hw.mean().item()) < 1e-3
    # ...and both are unbiased (track the input mean) in the normal range.
    assert abs(sw.mean().item() - in_mean) < 5e-3
    assert abs(hw.mean().item() - in_mean) < 5e-3


@requires_hw
def test_sw_triton_matches_cpu_reference():
    # The Triton software kernel must reproduce the CPU reference
    # `rs.stochastic_round_to_fp4_e2m1_bits` distribution (same bit-trick math).
    from benchmark import sr_fp4_software

    x = _normal_range_input(2048, 2048, seed=0)
    triton_deq = dequantize_fp4_e2m1(
        _unpack_fp4(sr_fp4_software(x, 0)).to(torch.uint8)
    )
    cpu_codes = stochastic_round_to_fp4_e2m1_bits(x.float().cpu(), seed=0)
    cpu_deq = dequantize_fp4_e2m1(cpu_codes)

    h_triton = _code_histogram(triton_deq).cpu()
    h_cpu = _code_histogram(cpu_deq.cuda()).cpu()
    max_diff = (h_triton - h_cpu).abs().max().item()
    assert max_diff < 0.01, f"triton vs cpu histograms differ by {max_diff}"


@requires_hw
def test_hw_tma_matches_hw():
    # The TMA variant uses a different RNG offset scheme (anchored to 2D position)
    # and different tiling, so it won't match the flat hardware path elementwise --
    # but the distribution and mean must agree.
    from benchmark import sr_fp4_hardware, sr_fp4_hardware_tma

    x = _normal_range_input(4096, 4096, seed=0)
    flat = dequantize_fp4_e2m1(_unpack_fp4(sr_fp4_hardware(x, 0)).to(torch.uint8))
    tma = dequantize_fp4_e2m1(_unpack_fp4(sr_fp4_hardware_tma(x, 0)).to(torch.uint8))

    max_diff = (_code_histogram(flat) - _code_histogram(tma)).abs().max().item()
    assert max_diff < 0.01, f"flat vs TMA histograms differ by {max_diff}"

    in_mean = x.float().mean().item()
    assert abs(tma.mean().item() - in_mean) < 5e-3
