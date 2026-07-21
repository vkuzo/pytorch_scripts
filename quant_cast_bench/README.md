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
recipe                                                       gpu_time_ms    gbps    pct_peak  perf_description
------------------------------  ----------------------------------------  ------  ----------  ---------------------------------
relu (baseline)                                                   0.1784  6020.4       75.3%
fp8_tensorwise_precalc_scale                                      0.1430  5631.8       70.4%  elementwise
mxfp8_floor_swizzle                                               0.1301  6256.7       78.2%  (1,32) block, swizzle
mxfp8_floor_dim_m                                                 0.5818  1398.6       17.5%  (32,1) block, t-contig
mxfp8_32x32_floor                                                 0.3793  2123.7       26.5%  (32,32) block
fp8_deepseek_1x128                                                0.1311  6204.5       77.6%  (1,128) block
fp8_deepseek_1x128_dim_m                                          0.2466  3299.8       41.2%  (128,1) block, t-contig
fp8_deepseek_128x128                                              0.2270  3547.9       44.3%  (128,128) block
fp8_rowwise                                                       0.1224  6578.5       82.2%  (1,-1) block
fp8_colwise                                                       0.3873  2079.7       26.0%  (-1,1) block, t-contig
nvfp4_swizzle                                                     0.4149  1657.7       20.7%  (1,16) block, fp4 qdata, swizzle
bf16_rht                                                          0.4596  2336.5       29.2%  elementwise RHT
fp32_to_bf16_sr                                                   1.0286  1565.8       19.6%
fp32_to_bf16_sr_global_offsets  SKIPPED: Unsupported: Observed exception                      elementwise SR with stateless RNG
```

### `--mode triton`

```
shape: (16384, 16384)  mode: triton
recipe                          gpu_time_ms    gbps    pct_peak  perf_description
----------------------------  -------------  ------  ----------  --------------------------------
relu (baseline)                      0.1783  6020.8       75.3%
fp8_tensorwise_precalc_scale         0.1422  5663.6       70.8%  elementwise
mxfp8_floor_swizzle                  0.1248  6519.5       81.5%  (1,32) block, swizzle
mxfp8_floor_dim_m                    0.1758  4627.9       57.8%  (32,1) block, t-contig
mxfp8_32x32_floor                    0.1287  6258.0       78.2%  (32,32) block
fp8_deepseek_1x128                   0.1341  6067.4       75.8%  (1,128) block
fp8_deepseek_1x128_dim_m             0.1428  5696.7       71.2%  (128,1) block, t-contig
fp8_deepseek_128x128                 0.1306  6166.7       77.1%  (128,128) block
fp8_rowwise                          0.1291  6237.9       78.0%  (1,-1) block
fp8_colwise                          0.2845  2831.0       35.4%  (-1,1) block, t-contig
nvfp4_swizzle                        0.1373  5009.0       62.6%  (1,16) block, fp4 qdata, swizzle
```
