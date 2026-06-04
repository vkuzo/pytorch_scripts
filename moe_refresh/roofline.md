# very simple moe layer roofline

1. count # of flops in grouped gemms (for bf16, fp8, fp4, for all gemms), convert to ms at roofline peak tensor core ops/second
2. count bytes over the wire in a2a (for bf16, fp8, fp4, and assume y and dx are always bf16), convert to ms at roofline peak inter-node bandwidth
3. match that vs modeling dims in real models to see what total time on gemms and comms looks like

This completely ignores:
a. overlapping of any kind (to keep it simple)
b. GPU memory bandwidth (this does matter, just not in here yet)
c. extra padding for quantization (this also matters, just not in here yet)

## assumptions

* T - total tokens in the batch (across all EP ranks for this layer)
* H - hidden / model dim (SonicMoE's d)
* I - expert intermediate size (SonicMoE's n)
* k - top-k (experts per token)
* E - total experts
* ep - EP degree (E_local = E/ep)

## fwd+bwd FLOPs

  Building block: a matmul [M,K]@[K,N] costs 2·M·K·N FLOPs (the 2 = one multiply + one add).
  One token through one expert (SwiGLU, fused gate+up):
  - up/gate proj: H → 2I, cost 2·H·(2I) = 4HI
  - down proj: I → H, cost 2·I·H = 2HI
  - forward per token-expert = 6HI (SwiGLU elementwise is negligible)

  Number of token-expert pairs = T·k (each token hits k experts; dropless).

  So:

  Forward FLOPs   = 6 · T · k · H · I

  Backward uses the standard rule: for every GEMM, backward = 2× forward (one dgrad GEMM + one wgrad GEMM, each the same cost as fprop). So:

  Backward FLOPs  = 12 · T · k · H · I
  Total (fwd+bwd) = 18 · T · k · H · I

  This is exactly SonicMoE's (6+12)·T·K·n·d formula — with their K=k, n=I, d=H. Nice cross-check.

  Per-GPU (balanced routing): divide by ep, since EP just distributes the same total work:
  Per-GPU FLOPs ≈ 18 · T · k · H · I / ep

## bytes over the wire (dispatch + combine, fwd + bwd)

  The unit moved is an H-vector (one token's hidden state, or one expert's output for a token), = H elements = 2H bytes in bf16. Routing indices/metadata are tiny — ignore them.

  Forward:
  - dispatch: each of the T·k (token, expert) pairs sends an H-vector to the expert's owner → T·k·H elements
  - combine: each expert output H-vector comes back → T·k·H elements
  - forward total = 2·T·k·H elements

  Backward is the mirror image (we saw this in the code: dispatch's backward is a combine, combine's backward is a dispatch). Same volume of gradient H-vectors moves:
  - backward total = 2·T·k·H elements

  So:

  Total a2a volume = 4 · T · k · H  elements
                   = 8 · T · k · H  bytes   (bf16, ×2 B/elem)

  Per-GPU + "actually over the wire": This is the part that matters for a roofline against NIC/NVLink bandwidth. Two adjustments:

  1. Divide by ep — each GPU originates T/ep tokens, so injects ≈ 8·(T/ep)·k·H bytes of traffic across the 4 phases.
  2. Local fraction stays off the wire — a token whose expert is on its own rank isn't sent. That's a (ep−1)/ep factor. For large ep it's ≈1 (almost everything crosses the network).

  Per-GPU wire bytes ≈ 8 · (T/ep) · k · H · (ep−1)/ep   ≈ 8 · (T/ep) · k · H   (large ep)

  Simplifying the above, for large ep
  * dispatch fwd + bwd: (2 * T * k * H / ep) elements
  * combine fwd + bwd: (2 * T * k * H / ep) elements
  * total: (4 * T * k * H / ep) elements

  Bf16: 2 bytes per elem:
    - (8 * T * k * H / ep) bytes
  Fp8: 1 byte per elem, ignore scales, only quantize `x` and `dy` (y and dx in high precision), so we get
    - (6 * T * k * H / ep) bytes
  Fp4: similar to fp8 but 0.5 byte per elem
    - (5 * T * k * H / ep) bytes

## constants

* NVIDIA B200 (Blackwell)
  - peak bf16 FLOPS/s: 2.25e15 (2.25 PFLOP/s) dense
    - note: NVIDIA datasheet headline is 4.5 PFLOP/s, but that is the 2:4 *sparse*
      number; dense (what our FLOP count assumes) is half → 2.25 PFLOP/s.
  - peak interconnect bandwidth: depends on tier (the roofline's two-tier caveat)
    - intra-node NVLink 5: 1.8 TB/s per GPU, bidirectional (≈0.9 TB/s each direction)
    - inter-node RDMA (per ConnectX-7/BlueField NIC): ~400 Gb/s = 50 GB/s per direction
    - use NVLink BW for an NVL72-domain roofline; use the RDMA BW when the a2a
      crosses nodes (it is ~36x slower, so inter-node traffic usually dominates).

## example values of T, H, I, k, E, ep

H, I, k, E are read from each model's HF config.json. I is the *expert* intermediate
size (moe_intermediate_size for DeepSeek-style; intermediate_size for Mixtral, which has
no separate MoE size). T (tokens/batch) and ep (EP degree) are deployment choices, not
model constants — pick per run.

| model                       |    H |     I |  k |   E | shared | G = H/I | ρ = k/E |
|-----------------------------|-----:|------:|---:|----:|-------:|--------:|--------:|
| Mixtral 8x22B               | 6144 | 16384 |  2 |   8 |      0 |   0.375 |  0.250  |
| DeepSeek V3.2-Exp           | 7168 |  2048 |  8 | 256 |      1 |   3.5   |  0.031  |
| Kimi K2 (K2.5 same arch)    | 7168 |  2048 |  8 | 384 |      1 |   3.5   |  0.021  |
| Qwen3-Next-80B-A3B-Instruct | 2048 |   512 | 10 | 512 |      1 |   4.0   |  0.020  |

Notes:
* Mixtral is *coarse-grained* (G < 1: each expert is wider than the model) and *dense*
  (ρ = 0.25) — the old-style MoE. The newer models are *fine-grained* (G ≈ 3.5–4, many
  small experts) and *sparse* (ρ ≈ 0.02–0.03) — exactly the regime where the a2a
  dominates and the FLOPs/byte intensity (≈ 2.25·I for bf16) is low.
* "shared" = number of shared experts (DeepSeek-style; run locally, no a2a, excluded
  from the routed-expert FLOP/byte counts above).
* T, ep examples for plugging into the formulas: e.g. T = 8192 tokens/microbatch,
  ep = 64 (DeepSeek-V3 trained with large EP across nodes); both scale the per-GPU
  numbers as shown (FLOPs ∝ T/ep, wire bytes ∝ T/ep).
