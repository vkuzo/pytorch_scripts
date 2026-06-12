"""Benchmark: software vs hardware-intrinsic stochastic rounding to fp4 e2m1.

Two Triton kernels do the SAME job -- stochastically round a bfloat16 tensor to
fp4 e2m1, packed 2 values per byte -- and we compare the HBM bandwidth each
achieves on a 16384x16384 tensor.

  PATH 1 (software): the integer bit-trick from `rs._sr_to_fp4_with_bits`, ported
      to Triton. Dither the mantissa with random bits, then truncate.
  PATH 2 (hardware): NVIDIA's `cvt.rs.satfinite.e2m1x4.f32` PTX intrinsic, which
      does fp4 stochastic rounding + packing in one instruction. Requires SM100
      (Blackwell, e.g. B200).

The two paths are apples-to-apples: same bf16 input, same RNG. Both generate one
32-bit random per OUTPUT BYTE via `tl.randint(seed, byte_offset)` with an identical
offset scheme; the ONLY difference is whether those bits are applied by software
bit-manipulation or fed to the hardware intrinsic.

Why bf16 in but f32 math: the intrinsic is f32-only, so both kernels load bf16 and
upcast to f32 in-register (bf16->f32 is exact). The software path then runs the
exact same fp32 bit-trick as rs.py.

Run:  python benchmark.py
"""

import torch
from torch import Tensor

import triton
import triton.language as tl

# Reuse the bit-layout constants from the reference implementation so the Triton
# port can't drift from `rs._sr_to_fp4_with_bits`.
from rs import (
    F32_EXP_MASK,
    F32_EXP_OFFSET,
    F32_EXP_BIAS,
    F32_MANTISSA_MASK,
    F32_IMPLIED_1,
    MBITS_F32,
    F4_EXP_BIAS,
    MBITS_F4,
    EBITS_F4,
    MBITS_F4_IMPLICIT,
    MANTISSA_OVERFLOW,
    EXPONENT_OVERFLOW,
    IMPLICIT_1_MASK_F4,
    RAND_MASK,
)

# Triton @jit functions can only read globals that are `tl.constexpr`, so re-bind
# the rs.py int constants (imported above to avoid drift) into constexprs.
_F32_EXP_MASK = tl.constexpr(F32_EXP_MASK)
_F32_EXP_OFFSET = tl.constexpr(F32_EXP_OFFSET)
_F32_EXP_BIAS = tl.constexpr(F32_EXP_BIAS)
_F32_MANTISSA_MASK = tl.constexpr(F32_MANTISSA_MASK)
_F32_IMPLIED_1 = tl.constexpr(F32_IMPLIED_1)
_MBITS_F32 = tl.constexpr(MBITS_F32)
_F4_EXP_BIAS = tl.constexpr(F4_EXP_BIAS)
_MBITS_F4 = tl.constexpr(MBITS_F4)
_EBITS_F4 = tl.constexpr(EBITS_F4)
_MBITS_F4_IMPLICIT = tl.constexpr(MBITS_F4_IMPLICIT)
_MANTISSA_OVERFLOW = tl.constexpr(MANTISSA_OVERFLOW)
_EXPONENT_OVERFLOW = tl.constexpr(EXPONENT_OVERFLOW)
_IMPLICIT_1_MASK_F4 = tl.constexpr(IMPLICIT_1_MASK_F4)
_RAND_MASK = tl.constexpr(RAND_MASK)

HAS_SM100 = (
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
)

# Autotuning configs, shared by all three kernels. We sweep the number of bf16
# elements per program (BLOCK) and the warp count. BLOCK must be a power of 2
# (Triton requirement) and a multiple of 8 so HALF = BLOCK//2 stays divisible by
# 4 for the randint4x 4-stream interleave. Autotuning is only safe because the
# output is BLOCK-invariant (the pid*QUARTER counter in `_per_byte_rbits`), so the
# result is identical regardless of which config wins.
_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": block}, num_warps=warps, num_stages=stages)
    for block in (512, 1024, 2048, 4096, 8192, 16384)
    for warps in (1, 2, 4, 8)
    for stages in (1, 2, 4)
]
# Autotune keys on the element count so a re-tune triggers only when size changes.
_AUTOTUNE_KEY = ["n_elements"]

# B200 HBM3e peak bandwidth, for the "% of peak" column.
HBM_PEAK_GBPS = 8000.0


# ---------------------------------------------------------------------------
# Hardware intrinsic helpers, copied VERBATIM from torchao
# (`/home/dev/ao/torchao/prototype/moe_training/nvfp4_training/hadamard_utils.py`).
# Both take 8 fp32 values (as two lanes of 4) and emit 4 packed fp4 bytes; the
# `_rs` variant additionally takes a random int per byte and uses the `cvt.rs`
# (round-stochastic) PTX instruction, while the plain variant uses `cvt.rn`
# (round-to-nearest-even) and needs no randomness.
# ---------------------------------------------------------------------------
@triton.jit
def convert_8xfp32_to_4xfp4_packed(x_pairs):
    """Convert 8 FP32 values to 4 packed FP4 bytes using round-to-nearest(-even).
    Calls four cvt.rn instructions, each packing two FP32 values into one packed int8."""
    x_fp4x2 = tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b8 byte0, byte1, byte2, byte3;
        cvt.rn.satfinite.e2m1x2.f32 byte0, $5, $1;
        cvt.rn.satfinite.e2m1x2.f32 byte1, $6, $2;
        cvt.rn.satfinite.e2m1x2.f32 byte2, $7, $3;
        cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $4;
        mov.b32 $0, {byte0, byte1, byte2, byte3};
        }
        """,
        constraints=("=r,r,r,r,r,r,r,r,r"),
        args=x_pairs,
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )
    return x_fp4x2


@triton.jit
def convert_8xfp32_to_4xfp4_packed_rs(x_pairs, rbits):
    """Convert 8 FP32 values to 4 packed FP4 bytes using stochastic rounding.
    Calls two cvt.rs instructions, each packing four FP32 values into one packed int8."""
    x_fp4x2 = tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b16 half0, half1;
        cvt.rs.satfinite.e2m1x4.f32 half0, {$6, $2, $5, $1}, $9;
        cvt.rs.satfinite.e2m1x4.f32 half1, {$8, $4, $7, $3}, $10;
        mov.b32 $0, {half0, half1};
        }
        """,
        constraints=("=r,r,r,r,r,r,r,r,r,r,r,r,r"),
        args=[x_pairs[0], x_pairs[1], rbits],
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )
    return x_fp4x2


# ---------------------------------------------------------------------------
# Shared RNG: one random int32 per output byte, via tl.randint4x.
#
# tl.randint4x is Philox's most efficient entry point -- one call yields FOUR
# independent int32 streams. We generate over a quarter of the byte indices and
# join the four streams back to the full [HALF] shape, so every output byte still
# gets its own independent random int (the same per-byte granularity the hardware
# cvt.rs intrinsic consumes). Both SR kernels call this so their randomness is
# identical; the only difference between them is how the bits are applied.
# ---------------------------------------------------------------------------
@triton.jit
def _per_byte_rbits(seed, pid, HALF: tl.constexpr):
    QUARTER: tl.constexpr = HALF // 4
    # Base the Philox counter on pid*QUARTER (not pid*HALF) so the counter for an
    # absolute byte b works out to exactly b // 4, with stream = b % 4. This makes
    # the random for each byte a pure function of its ABSOLUTE position, so the
    # output is reproducible even if BLOCK (the tiling) changes -- not just across
    # tile order. (BLOCK is a power of 2 in Triton, so HALF is always divisible by
    # 4 and the 4-stream interleave tiles cleanly.)
    quarter_off = pid * QUARTER + tl.arange(0, QUARTER)
    r0, r1, r2, r3 = tl.randint4x(seed, quarter_off)
    # Interleave the four streams back into one [HALF] block.
    return tl.join(tl.join(r0, r1), tl.join(r2, r3)).reshape(HALF)


# ---------------------------------------------------------------------------
# PATH 1: software bit-trick.
# ---------------------------------------------------------------------------
@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=_AUTOTUNE_KEY)
@triton.jit
def _sr_fp4_software_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    seed,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    # This program owns BLOCK contiguous bf16 elements -> BLOCK // 2 output bytes.
    HALF: tl.constexpr = BLOCK // 2
    # Contiguous element load (coalesced), then split into low/high lanes of each
    # output byte in-register -- a[0::2] are low lanes, a[1::2] are high lanes.
    elem_off = pid * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(x_ptr + elem_off, mask=elem_off < n_elements).to(tl.float32)
    a_lo, a_hi = a.reshape(HALF, 2).split()

    # Per-output-byte index for the packed store.
    byte_off = pid * HALF + tl.arange(0, HALF)
    byte_mask = byte_off < (n_elements // 2)

    # One random int per output byte -- IDENTICAL scheme to the hardware kernel.
    # Both lanes of the byte dither with the same value, mirroring what the
    # hardware intrinsic receives (one rbits lane per output byte).
    rbits = _per_byte_rbits(seed, pid, HALF)

    lo_code = _sr_fp4_one(a_lo, rbits)
    hi_code = _sr_fp4_one(a_hi, rbits)

    # Pack two fp4 codes into a byte: high nibble = high lane (MSLK convention).
    packed = ((hi_code << 4) | (lo_code & 0xF)).to(tl.uint8)
    tl.store(out_ptr + byte_off, packed, mask=byte_mask)


@triton.jit
def _sr_fp4_one(a, rbits):
    """Stochastically round one fp32 lane to a 4-bit fp4 e2m1 code (int32 0..15).

    Line-by-line port of `rs._sr_to_fp4_with_bits` (the fixed-RAND_MASK, no-`+1`
    variant). `a` is fp32, `rbits` is the per-byte random int.
    """
    xi = a.to(tl.int32, bitcast=True)

    sign = (xi >> 31) & 0x1
    mag = xi & 0x7FFFFFFF

    biased_exp = (mag & _F32_EXP_MASK) >> _F32_EXP_OFFSET
    trailing = mag & _F32_MANTISSA_MASK
    is_subn = biased_exp == 0

    new_exp = biased_exp - _F32_EXP_BIAS + _F4_EXP_BIAS

    exp_diff = tl.where(new_exp <= 0, 1 - new_exp, 0)
    exp_diff = tl.minimum(exp_diff, _MBITS_F32 + 1)

    mantissa = tl.where(is_subn, trailing, trailing + _F32_IMPLIED_1)
    sig_bits = tl.where(is_subn, _MBITS_F32, _MBITS_F32 + 1)

    # The stochastic-rounding step: dither with the fixed 22-bit mask, then floor.
    shift = sig_bits + exp_diff - _MBITS_F4_IMPLICIT
    shift = tl.maximum(shift, 0)
    mantissa = mantissa + (rbits & _RAND_MASK)
    mantissa = mantissa >> shift

    overflow = mantissa > _MANTISSA_OVERFLOW
    mantissa = tl.where(overflow & (biased_exp != 0), mantissa >> 1, mantissa)
    new_exp = tl.where((new_exp <= 0) & (mantissa == 2), 1, new_exp)
    mantissa = mantissa & _IMPLICIT_1_MASK_F4
    new_exp = tl.where(overflow, new_exp + 1, new_exp)
    mantissa = tl.where(new_exp > _EXPONENT_OVERFLOW, 1, mantissa)
    new_exp = tl.minimum(tl.maximum(new_exp, 0), _EXPONENT_OVERFLOW)

    code = (new_exp << _MBITS_F4) | mantissa
    code = (sign << (_EBITS_F4 + _MBITS_F4)) | code
    return code


# ---------------------------------------------------------------------------
# PATH 2: hardware intrinsic (stochastic rounding).
# ---------------------------------------------------------------------------
@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=_AUTOTUNE_KEY)
@triton.jit
def _sr_fp4_hardware_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    seed,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    HALF: tl.constexpr = BLOCK // 2

    # Same coalesced load + in-register pair split as the software kernel.
    elem_off = pid * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(x_ptr + elem_off, mask=elem_off < n_elements).to(tl.float32)
    a_lo, a_hi = a.reshape(HALF, 2).split()

    byte_off = pid * HALF + tl.arange(0, HALF)
    byte_mask = byte_off < (n_elements // 2)

    # Same RNG as the software kernel.
    rbits = _per_byte_rbits(seed, pid, HALF)

    # The intrinsic helper consumes the pair (lo, hi) lanes and one rbits per byte,
    # and returns the packed byte (it saturates to +-6 and does SR + packing).
    packed = convert_8xfp32_to_4xfp4_packed_rs([a_lo, a_hi], rbits)
    tl.store(out_ptr + byte_off, packed, mask=byte_mask)


# ---------------------------------------------------------------------------
# PATH 2b: hardware intrinsic via TMA (Tensor Memory Accelerator).
#
# Same cvt.rs stochastic-rounding cast as PATH 2, but loads/stores 2D tiles
# through `tl.make_tensor_descriptor` (TMA async bulk copy) instead of flat masked
# loads. The question this answers: does TMA overlap the loads with the cvt.rs +
# RNG compute better than PATH 2's flat loads? (Spoiler from measurement: no -- it
# is slower here; see the benchmark output / the note in the host wrapper.)
#
# Autotuning over BLOCK_M/BLOCK_N is intentionally omitted to keep the TMA
# descriptor shapes static; a fixed sane tile is used (see the host wrapper).
# ---------------------------------------------------------------------------
@triton.jit
def _sr_fp4_hardware_tma_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    seed,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Device-side TMA descriptors: bf16 input [M, N], packed uint8 output [M, N//2].
    a_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N]
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[M, N // 2],
        strides=[N // 2, 1],
        block_shape=[BLOCK_M, BLOCK_N // 2],
    )

    num_pid_n = tl.cdiv(N, BLOCK_N)
    HALF_N: tl.constexpr = BLOCK_N // 2

    # Non-persistent grid: one program per tile (grid = num_tiles), so there are
    # far more programs than SMs -> plenty of memory-level parallelism for latency
    # hiding. (A persistent NUM_SMS-program grid-stride loop was measured ~3x
    # slower here, same lesson as MXFP4's flat kernel: too few in-flight programs.)
    tile_id = tl.program_id(0)
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    row0 = pid_m * BLOCK_M
    col0 = pid_n * BLOCK_N

    a = a_desc.load([row0, col0]).to(tl.float32)              # [BLOCK_M, BLOCK_N]
    a_lo, a_hi = a.reshape(BLOCK_M, HALF_N, 2).split()        # two [BLOCK_M, HALF_N]

    # Per-output-byte random, anchored to ABSOLUTE byte position so the result
    # is independent of tiling. The output byte at (row, bcol) has absolute
    # index row*(N//2) + bcol.
    rows = (row0 + tl.arange(0, BLOCK_M))[:, None]
    bcols = (pid_n * HALF_N + tl.arange(0, HALF_N))[None, :]
    byte_idx = rows * (N // 2) + bcols
    rbits = tl.randint(seed, byte_idx)

    packed = convert_8xfp32_to_4xfp4_packed_rs([a_lo, a_hi], rbits)
    out_desc.store([row0, col0 // 2], packed)


# ---------------------------------------------------------------------------
# PATH 2c: TMA on a PERSISTENT grid-stride loop with `num_stages`.
#
# This is the configuration where Triton's software pipeliner can actually overlap
# the TMA load of tile N+1 with the cvt.rs + RNG compute of tile N (multi-stage
# SMEM buffering across loop iterations). The non-persistent kernel above has no
# loop, so nothing to pipeline. A coworker hit RTNE-level perf with a hand-written
# TMA pipeline in CuTe DSL; this tests how close Triton's *automatic* pipelining
# gets via num_stages.
# ---------------------------------------------------------------------------
@triton.jit
def _sr_fp4_hardware_tma_pipe_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    seed,
    NUM_SMS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    a_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N]
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[M, N // 2],
        strides=[N // 2, 1],
        block_shape=[BLOCK_M, BLOCK_N // 2],
    )

    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = tl.cdiv(M, BLOCK_M) * num_pid_n
    HALF_N: tl.constexpr = BLOCK_N // 2

    start_pid = tl.program_id(0)
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, num_stages=NUM_STAGES):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        row0 = pid_m * BLOCK_M
        col0 = pid_n * BLOCK_N

        a = a_desc.load([row0, col0]).to(tl.float32)
        a_lo, a_hi = a.reshape(BLOCK_M, HALF_N, 2).split()

        rows = (row0 + tl.arange(0, BLOCK_M))[:, None]
        bcols = (pid_n * HALF_N + tl.arange(0, HALF_N))[None, :]
        byte_idx = rows * (N // 2) + bcols
        rbits = tl.randint(seed, byte_idx)

        packed = convert_8xfp32_to_4xfp4_packed_rs([a_lo, a_hi], rbits)
        out_desc.store([row0, col0 // 2], packed)


# NOTE on warp specialization (the "Option B" we tried and dropped):
# Adding `warp_specialize=True` to a persistent grid-stride loop -- the pattern
# torchao uses for its amax/rht kernels -- fails to compile here. Triton 3.7's
# warp-specialization pass on SM100 cannot partition a pure load -> elementwise ->
# store loop (the MLIR verifier aborts: an arith.constant lacks the ttg.partition
# attribute). It needs a partitionable compute op in the loop, i.e. a tl.dot / MMA,
# which is exactly what every torchao WS kernel contains. A pure cast has no such
# op, so producer/consumer warp specialization is not available for this kernel.
# (Verified independently on a minimal `a*2.0` TMA loop -- same failure.)


# ---------------------------------------------------------------------------
# PATH 3: hardware intrinsic, round-to-nearest-even (NO stochastic rounding).
# A baseline to see how much the SR machinery (RNG + the cvt.rs variant) costs
# over a plain deterministic cast.
# ---------------------------------------------------------------------------
@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=_AUTOTUNE_KEY)
@triton.jit
def _rtne_fp4_hardware_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    seed,  # unused; kept so the launch signature matches the other kernels
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    HALF: tl.constexpr = BLOCK // 2

    elem_off = pid * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(x_ptr + elem_off, mask=elem_off < n_elements).to(tl.float32)
    a_lo, a_hi = a.reshape(HALF, 2).split()

    byte_off = pid * HALF + tl.arange(0, HALF)
    byte_mask = byte_off < (n_elements // 2)

    # No random bits: cvt.rn does deterministic round-to-nearest-even + packing.
    packed = convert_8xfp32_to_4xfp4_packed([a_lo, a_hi])
    tl.store(out_ptr + byte_off, packed, mask=byte_mask)


# ---------------------------------------------------------------------------
# Host wrappers (public entry points).
# ---------------------------------------------------------------------------
def _check_input(x: Tensor):
    assert x.dtype == torch.bfloat16, f"x must be bfloat16, got {x.dtype}"
    assert x.is_cuda, "x must be on CUDA"
    assert x.is_contiguous(), "x must be contiguous"
    assert x.numel() % 2 == 0, "need an even number of elements to pack 2/byte"


def _launch(kernel, x: Tensor, seed: int) -> Tensor:
    n = x.numel()
    out = torch.empty(x.shape[0], x.shape[1] // 2, dtype=torch.uint8, device=x.device)
    # BLOCK is chosen by @triton.autotune, so the grid is a function of the winning
    # config's meta["BLOCK"] rather than a fixed value.
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    kernel[grid](x, out, n, seed)
    return out


def sr_fp4_software(x: Tensor, seed: int) -> Tensor:
    """Stochastically round bf16 `x` to packed fp4 (uint8) via the software bit-trick."""
    _check_input(x)
    return _launch(_sr_fp4_software_kernel, x, seed)


def sr_fp4_hardware(x: Tensor, seed: int) -> Tensor:
    """Stochastically round bf16 `x` to packed fp4 (uint8) via the cvt.rs intrinsic."""
    _check_input(x)
    assert HAS_SM100, "hardware fp4 cvt.rs intrinsic requires SM100 (Blackwell)"
    return _launch(_sr_fp4_hardware_kernel, x, seed)


# Best tile from a small sweep (BM x BN); larger tiles spill shared memory and
# tank to a few % of peak. Even the best is well below the flat PATH 2.
_TMA_BLOCK_M = 64
_TMA_BLOCK_N = 256


def sr_fp4_hardware_tma(x: Tensor, seed: int) -> Tensor:
    """Stochastically round bf16 `x` to packed fp4 (uint8) via cvt.rs, loading and
    storing through TMA (2D-tiled, one program per tile)."""
    _check_input(x)
    assert HAS_SM100, "hardware fp4 cvt.rs intrinsic requires SM100 (Blackwell)"
    assert x.ndim == 2, "TMA path expects a 2D tensor"
    M, N = x.shape
    out = torch.empty(M, N // 2, dtype=torch.uint8, device=x.device)
    grid = (triton.cdiv(M, _TMA_BLOCK_M) * triton.cdiv(N, _TMA_BLOCK_N),)

    # TMA needs an allocator for its descriptor workspace; set it around the launch.
    triton.set_allocator(
        lambda size, align, stream: torch.empty(size, dtype=torch.int8, device=x.device)
    )
    try:
        _sr_fp4_hardware_tma_kernel[grid](
            x, out, M, N, seed, BLOCK_M=_TMA_BLOCK_M, BLOCK_N=_TMA_BLOCK_N
        )
    finally:
        triton.set_allocator(None)
    return out


def rtne_fp4_hardware(x: Tensor, seed: int = 0) -> Tensor:
    """Round bf16 `x` to packed fp4 (uint8) via the cvt.rn intrinsic (no SR).

    Deterministic round-to-nearest-even baseline; `seed` is ignored.
    """
    _check_input(x)
    assert HAS_SM100, "hardware fp4 cvt.rn intrinsic requires SM100 (Blackwell)"
    return _launch(_rtne_fp4_hardware_kernel, x, seed)


# ---------------------------------------------------------------------------
# Benchmark harness.
# ---------------------------------------------------------------------------
def _bench(fn, x: Tensor, seed: int):
    """Return (median_ms, gbps, pct_peak) for `fn`."""
    ms = triton.testing.do_bench(lambda: fn(x, seed), warmup=25, rep=100)
    n = x.numel()
    moved_bytes = n * 2 + n // 2  # bf16 in + uint8-packed out
    gbps = moved_bytes / (ms * 1e-3) / 1e9
    return ms, gbps, gbps / HBM_PEAK_GBPS * 100.0


def main():
    if not torch.cuda.is_available():
        print("No CUDA device available; skipping benchmark.")
        return

    torch.manual_seed(0)
    M = N = 16384
    x = ((torch.rand(M, N, device="cuda") * 12.0) - 6.0).bfloat16()

    rows = []
    rows.append(("SR software (bit-trick)", *_bench(sr_fp4_software, x, 0)))
    if HAS_SM100:
        rows.append(("SR hardware (cvt.rs)", *_bench(sr_fp4_hardware, x, 0)))
        rows.append(("SR hardware (cvt.rs, TMA)", *_bench(sr_fp4_hardware_tma, x, 0)))
        rows.append(("RTNE hardware (cvt.rn)", *_bench(rtne_fp4_hardware, x, 0)))
    else:
        print("(SM100 not detected -- skipping the hardware-intrinsic paths)")

    print(f"\nbf16 -> fp4 e2m1, {M}x{N}")
    print(f"{'path':<26}{'time (ms)':>12}{'GB/s':>12}{'% peak':>10}")
    print("-" * 60)
    for name, ms, gbps, pct in rows:
        print(f"{name:<26}{ms:>12.3f}{gbps:>12.1f}{pct:>9.1f}%")


if __name__ == "__main__":
    main()
