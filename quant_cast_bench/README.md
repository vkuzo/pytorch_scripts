# quant_cast_bench

Memory-bandwidth benchmark for the `quant_cast_gold` recipes. Each recipe is a memory-bound
cast, so the signal is achieved memory bandwidth vs. the B200 ceiling (8 TB/s). Per `mode`, the
benchmark either `torch.compile`s each gold recipe's reference fn (`compile`, the default) or
runs its hand-written kernels (`triton` / `cute`), times each with `do_bench_using_profiling`,
and reports latency + GB/s + % of peak. The `relu (baseline)` row anchors the achievable ceiling
for the shape.

## torchinductor gaps vs triton

* square quant block sizes (32x32, 128x128, etc)
  - For example, on fp8_deepseek_128x128, inductor 44.3% peak mem -> triton 77.1% peak mem
* reductions across M-dim, or K-dim and M-dim in the same kernel
  - For example, on mxfp8_floor_dim_m, inductor 17.5% peak mem -> triton 57.8% peak mem
* fp4
  - nvfp4_swizzle with plain pytorch ops: inductor 20.7% peak mem -> triton with inline asm 62.6% peak mem

## triton gaps vs SOL (CUDA / CUTLASS / cute)

* reductions across M-dim, or K-dim and M-dim in the same kernel
  - For example, on mxfp8_floor_dim_m, triton 57.8% peak mem -> CUDA 67.7% peak mem

    - The CUDA kernel writes quantized values directly into a transposed smem layout (out_colwise_sh[col][row]) and TMA-stores that smem tile. Triton has no
      user-facing __shared__ + __syncthreads(), so every transpose goes through the compiler's tl.trans — which either (a) produces the uncoalesced 21-sectors/request store, or (b) if you TMA-store it, pays a register→smem
      transpose tax. This is the crux: CUDA decouples "coalesced transposed store" from "small register footprint," and Triton cannot.
    - Efficient TMA transfers and high occupancy at the same time. CUDA gets both because it manages smem by hand and uses tiny 64-thread CTAs. In Triton, TMA transfer size and occupancy are both governed by the tile size, so
      you're forced to choose — big tiles (efficient TMA, low occupancy) or small tiles (high occupancy, tiny inefficient TMA transfers).

## Repro

```bash
cd /home/dev/pytorch_scripts

# torch.compile the gold reference fns (default mode)
python quant_cast_bench/benchmark.py --mode compile

# hand-written Triton kernels
python quant_cast_bench/benchmark.py --mode triton

# optional: single shape / single recipe
python quant_cast_bench/benchmark.py --mode triton --M 16384 --K 16384
python quant_cast_bench/benchmark.py --mode triton --recipe_name_filter mxfp8_floor_dim_m
```

Default shape is `(M, K) = (16384, 16384)`. Assumes a B200 (peak 8 TB/s).

## Output

### `--mode compile`

```
shape: (16384, 16384)  mode: compile
recipe                            gpu_time_ms    gbps    pct_peak  perf_description
------------------------------  -------------  ------  ----------  ---------------------------------
relu (baseline)                        0.1783  6020.5       75.3%
fp8_tensorwise_precalc_scale            0.143  5630.3       70.4%  elementwise
mxfp8_floor_swizzle                    0.1301  6256.5       78.2%  (1,32) block, swizzle
mxfp8_floor_dim_m                      0.5818  1398.5       17.5%  (32,1) block, t-contig
mxfp8_32x32_floor                      0.3793  2123.6       26.5%  (32,32) block
fp8_deepseek_1x128                     0.1311  6204.5       77.6%  (1,128) block
fp8_deepseek_1x128_dim_m               0.2466  3299.6       41.2%  (128,1) block, t-contig
fp8_deepseek_128x128                    0.227  3548.3       44.4%  (128,128) block
fp8_rowwise                            0.1225    6577       82.2%  (1,-1) block
fp8_colwise                            0.3873  2079.6       26.0%  (-1,1) block, t-contig
nvfp4_swizzle                          0.4151  1657.2       20.7%  (1,16) block, fp4 qdata, swizzle
bf16_rht                               0.4595  2336.5       29.2%  elementwise RHT
fp32_to_bf16_sr                        1.0288  1565.6       19.6%
fp32_to_bf16_sr_global_offsets         3.2413   496.9        6.2%  elementwise SR with stateless RNG
```

### `--mode triton`

```
shape: (16384, 16384)  mode: triton
recipe                          gpu_time_ms    gbps    pct_peak  perf_description
----------------------------  -------------  ------  ----------  --------------------------------
relu (baseline)                      0.1784  6019.7       75.2%
fp8_tensorwise_precalc_scale         0.1422    5663       70.8%  elementwise
mxfp8_floor_swizzle                  0.1248  6519.5       81.5%  (1,32) block, swizzle
mxfp8_floor_dim_m                    0.1758  4628.5       57.9%  (32,1) block, t-contig
mxfp8_32x32_floor                    0.1287  6258.3       78.2%  (32,32) block
fp8_deepseek_1x128                   0.1341  6066.1       75.8%  (1,128) block
fp8_deepseek_1x128_dim_m             0.1428  5696.8       71.2%  (128,1) block, t-contig
fp8_deepseek_128x128                 0.1306  6167.0       77.1%  (128,128) block
fp8_rowwise                          0.1293  6227.6       77.8%  (1,-1) block
fp8_colwise                          0.2844  2832.2       35.4%  (-1,1) block, t-contig
nvfp4_swizzle                        0.1373  5011.5       62.6%  (1,16) block, fp4 qdata, swizzle
```

## Known issues

* `fp32_to_bf16_sr` (compile) reports only ~19.6% peak, but this understates the real bandwidth.
  The stochastic-rounding uniform is drawn via `torch.func._random.uniform` → `aten._philox_uniform`,
  which inductor treats as an opaque extern op rather than a fusible in-kernel RNG. So it runs as
  two DRAM passes: kernel 1 materializes a full-size fp32 random tensor (~1.07 GB write, ~63% of
  the runtime), kernel 2 reads it back alongside `x` to dither+truncate. Real traffic is ~3.76 GB
  (write u + read x + read u + write out) ≈ 46% of peak; the benchmark only counts input+output
  (~1.61 GB), so the wasted RNG round-trip shows up as the low 19.6%. Fix: fuse the Philox RNG into
  the dither kernel (generate uniforms in-register, never materialize) — as inductor already does
  for `torch.rand`/dropout — which would cut traffic to ~1.61 GB and approach the relu ceiling
  (~2–3× speedup).

* `bf16_rht` (compile) runs at only ~29% peak, and here the traffic is not wasted (the whole 1.07 GB
  is useful read x + write out) — it's GEMM-kernel inefficiency. The 16×16 RHT `x.reshape(..., 16) @ rht`
  is lowered to a single cuBLAS GEMM via `extern_kernels.mm`, shape `(M·N/16, 16) @ (16, 16)` — i.e.
  `K=16, N=16`. That GEMM is ~99.5% of the runtime. The op is really memory-bound (~4 flop/byte), but
  cuBLAS runs it as a compute-oriented matmul, and the skinny `K=N=16` shape tiles terribly (N-tiling
  wasted, no K-reuse to amortize), so it fails to saturate DRAM — 29% vs the ~75% relu ceiling for the
  same 1.07 GB (~2.6× slower than bandwidth-bound). Fix direction: a fused kernel that loads a 16-vector,
  applies the transform in registers, and writes 16 (or a Triton matmul template tuned for the skinny
  shape) would approach the relu ceiling.

* `fp32_to_bf16_sr_global_offsets` (compile) runs at only ~6.2% peak — ~3.2× slower than
  `fp32_to_bf16_sr` (19.6%) for identical dithering math. The difference is how the Philox draw is
  keyed. Both use `torch.func._random.uniform` (experimental stateless Philox → unfused
  `aten._philox_uniform`, so both share the same materialized-`u` ~46%-real-BW ceiling). The plain
  variant keys on tile-LOCAL position (one shared key, counter = flat index within the call), which
  is cheap but changes with tiling. The global variant is tile-INVARIANT: it keys each draw on the
  element's GLOBAL index, which — because `uniform` only exposes a single scalar starting offset (the
  `(seed, offset)` key pair, fine for 1D/full-width tiles but not 2D sub-blocks) — forces
  materializing a per-element `(numel, 2)` uint64 key tensor = **4.29 GB** (16 B/element, 8× the
  0.54 GB bf16 output), written then read back by a batched Philox. That ~8.5 GB key round-trip
  triples total traffic (~12.3 GB vs ~3.76 GB), matching the 3.2× slowdown.

### Fix direction: key by global index without materializing keys

The global index only needs to reach Philox as a *counter*. Today the sole knob is the key's single
scalar `offset` — one value per call, which can only shift a 1-D contiguous stream, so it cannot
express a 2-D sub-block's global index and the recipe is forced to fold the index into a
**per-element key tensor** (`(numel, 2)` uint64, 4.29 GB). If `uniform` instead accepted a per-element
**affine counter** (a `base` plus per-dim `strides`), element `(i, j)` could take
`counter = base + i·num_col + j` computed *in-kernel from its own indices* — one shared key, zero
materialized index/key tensors. Combined with a fusible in-kernel Philox (as Triton's
`tl.rand(seed, offset)` already allows), the whole tile-invariant SR becomes one fused kernel that
never materializes `u` either — approaching the relu ceiling.

<table>
<tr><th>Current — per-element key tensor (4.29 GB)</th><th>Ideal — shared key + in-kernel affine counter</th></tr>
<tr><td>

```python
# to key on the GLOBAL index, fold it into a
# distinct key per element:
i = (global_row + arange(M)).view(-1, 1)
j = (global_col + arange(N)).view(1, -1)
gidx = (i * num_col + j).reshape(-1)   # global index
seed = key[0:1].expand(gidx.numel())
keys = stack([seed, gidx], -1).to(uint64)
#      ^ (numel, 2) uint64 = 4.29 GB  <-- materialized
u = uniform(keys, (gidx.numel(),))
#   ^ batched philox reads 4.29 GB keys back,
#     writes u (1.07 GB)  <-- also materialized
rand16 = (u * 65536).to(int32)
```

</td><td>

```python
# one shared key; per-element counter is an affine
# map of the element's coords, computed in-kernel:
u = uniform(
    key,                       # single (seed, offset) key
    (M, N),
    counter_base=global_row * num_col + global_col,
    counter_strides=(num_col, 1),
)   # counter(i,j) = base + i*num_col + j
#   no key tensor; fusible -> u never materialized
rand16 = (u * 65536).to(int32)
```

</td></tr>
</table>
