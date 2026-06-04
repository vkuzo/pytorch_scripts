"""Simple roofline model for an EP experts layer (dispatch -> experts -> combine).

Implements the formulas from roofline.md:
  * Per-GPU GEMM FLOPs (fwd+bwd) for the routed grouped MMs.
  * Per-GPU elements/bytes over the wire for dispatch+combine (fwd+bwd).

It then converts each into a time (ms) against B200 constants, keeping the GEMM
(compute) time and the all-to-all (network) time SEPARATE -- no arithmetic-intensity
/ roofline-crossover analysis here, just the two times side by side.

Reports gemm/a2a/ratio columns for each quant dtype (bf16/fp8/fp4). The dtype affects
both the a2a bytes (only x and dy are quantized, y and dx stay bf16) and the GEMM
compute throughput (fp8 = 2x bf16, fp4 = 4x bf16).

Usage:
    python roofline.py                      # defaults: T=8192, ep=64, rdma
    python roofline.py --tokens 4096 --ep 8 --interconnect nvlink
"""

import fire


# --- formulas (per roofline.md) ---------------------------------------------


def per_gpu_gemm_flops(T: int, H: int, I: int, k: int, ep: int) -> float:
    """Per-GPU GEMM FLOPs for the routed experts, fwd+bwd.

    Total (fwd+bwd) = 18 * T * k * H * I   (6 fwd + 12 bwd, SwiGLU fused gate+up).
    EP distributes the work, so per-GPU divides by ep (balanced routing).
    """
    return 18.0 * T * k * H * I / ep


def per_gpu_wire_elements(T: int, H: int, k: int, ep: int) -> float:
    """Per-GPU elements over the wire for dispatch+combine, fwd+bwd (large ep).

    total = 4 * T * k * H / ep elements
        (dispatch fwd+bwd = 2*T*k*H/ep, combine fwd+bwd = 2*T*k*H/ep).
    """
    return 4.0 * T * k * H / ep


# Effective bytes per communicated element by wire dtype (roofline.md "bytes over the
# wire"). The doc gives totals of bf16=8, fp8=6, fp4=5 (per T*k*H/ep) because only x and
# dy are quantized while y and dx stay high precision. Dividing by the 4*T*k*H/ep
# element count gives the effective bytes/elem below: {2.0, 1.5, 1.25}.
COMMS_EFFECTIVE_BYTES_PER_ELEM = {
    "bf16": 2.0,
    "fp8": 1.5,
    "fp4": 1.25,
}


def per_gpu_wire_bytes(T: int, H: int, k: int, ep: int, dtype: str) -> float:
    """Per-GPU bytes over the wire for the given wire dtype.

    Only x and dy are quantized (y and dx stay bf16), so the effective per-element
    cost is dtype-dependent; see COMMS_EFFECTIVE_BYTES_PER_ELEM.
    """
    return per_gpu_wire_elements(T, H, k, ep) * COMMS_EFFECTIVE_BYTES_PER_ELEM[dtype]


# --- hardware constants (B200, from roofline.md) ----------------------------

# Peak dense bf16 throughput (the 4.5 PFLOP/s headline is 2:4 sparse; dense is half).
B200_BF16_FLOPS = 2.25e15  # FLOP/s

# Compute speedup vs bf16 by tensor-core dtype on B200: fp8 is 2x bf16, fp4 is 4x.
# (The FLOP count is unchanged; the effective throughput scales.)
COMPUTE_SPEEDUP = {
    "bf16": 1.0,
    "fp8": 2.0,
    "fp4": 4.0,
}

# Interconnect bandwidth per direction, by tier (bytes/s).
INTERCONNECT_BW = {
    "nvlink": 0.9e12,  # NVLink 5: 1.8 TB/s bidirectional -> ~0.9 TB/s each direction
    "rdma": 50e9,      # inter-node RDMA: ~400 Gb/s = 50 GB/s per direction per NIC
}


# --- per-model constants (from roofline.md table) ---------------------------
# H, I, k, E from each model's HF config.json. I is the expert intermediate size.

MODELS = {
    "Mixtral-8x22B": dict(H=6144, I=16384, k=2, E=8),
    "DeepSeek-V3.2-Exp": dict(H=7168, I=2048, k=8, E=256),
    "Kimi-K2": dict(H=7168, I=2048, k=8, E=384),
    "Qwen3-Next-80B-A3B": dict(H=2048, I=512, k=10, E=512),
}


DTYPES = ["bf16", "fp8", "fp4"]


def main(
    tokens: int = 8192,
    ep: int = 64,
    interconnect: str = "rdma",
):
    """Compute per-GPU GEMM-time and a2a-time (ms) for one experts block per model.

    Reports a set of columns (gemm ms, a2a ms, and their ratio) for each quantization
    dtype in {bf16, fp8, fp4}. The dtype affects both the a2a bytes (only x and dy are
    quantized, y and dx stay bf16) and the GEMM compute throughput (fp8 = 2x bf16,
    fp4 = 4x bf16 on B200). The ratio = gemm_ms / a2a_ms (>1 compute-bound,
    <1 comms-bound).

    Args:
        tokens: T, tokens per microbatch (across all EP ranks for this layer).
        ep: EP degree (experts sharded across this many ranks).
        interconnect: bandwidth tier for the a2a, one of {nvlink, rdma}.
    """
    assert interconnect in INTERCONNECT_BW, (
        f"interconnect must be one of {list(INTERCONNECT_BW)}"
    )

    bw = INTERCONNECT_BW[interconnect]

    print(
        f"Config: T={tokens}, ep={ep}, interconnect={interconnect} "
        f"({bw / 1e9:.0f} GB/s), B200 dense bf16={B200_BF16_FLOPS / 1e12:.0f} TFLOP/s "
        f"(fp8 2x, fp4 4x)\n"
    )

    # Header: model constants, then a (gemm, a2a, ratio) triple per dtype.
    cols = f"{'model':<22} {'H':>5} {'I':>6} {'k':>3} {'E':>4} {'GFLOP/gpu':>10}"
    for dt in DTYPES:
        cols += f" {'gemm_' + dt + '_ms':>12} {'a2a_' + dt + '_ms':>12} {'ratio_' + dt:>9}"
    print(cols)
    print("-" * len(cols))

    for name, c in MODELS.items():
        H, I, k, E = c["H"], c["I"], c["k"], c["E"]
        flops = per_gpu_gemm_flops(tokens, H, I, k, ep)

        row = f"{name:<22} {H:>5} {I:>6} {k:>3} {E:>4} {flops / 1e9:>10.2f}"
        for dt in DTYPES:
            gemm_ms = flops / (B200_BF16_FLOPS * COMPUTE_SPEEDUP[dt]) * 1e3
            a2a_ms = per_gpu_wire_bytes(tokens, H, k, ep, dt) / bw * 1e3
            ratio = gemm_ms / a2a_ms
            row += f" {gemm_ms:>12.4f} {a2a_ms:>12.4f} {ratio:>9.3f}"
        print(row)


if __name__ == "__main__":
    fire.Fire(main)
