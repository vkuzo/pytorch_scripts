# flex_cast_quant_dense prototype

## Goal

Flexible API for quantizing a tensor, backed by performant JIT generated kernels. 90% clauded as this is a POC.

tl;dr:
* user specifies how the scaling grid works and how they want to scale and cast the data
* `torch.compile(flex_cast_quant_dense)` spits out a fast kernel, either with inductor or
  a handwritten triton template

Delta vs just using `torch.compile`:
1. today, templates are faster than compile on anything except (1, B) casts across the K dim
2. API is more constrained (some users might prefer that, some might not)

## Example

```python
# deepseek 1x128 (activaiton) and 128x128 (weight) fp8 scaling

def amax_to_scale_fn(amax: torch.Tensor) -> torch.Tensor:
    amax_fp32 = amax.clamp(min=EPS).to(torch.float32)
    return amax_fp32 / FP8_MAX

def cast_to_dtype_fn(tile: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    reciprocal = 1.0 / scale
    y = tile * reciprocal
    return y.to(torch.float8_e4m3fn)

# enable compile for performance, eager fallback is slow
flex_cast_quant_dense_c = torch.compile(flex_cast_quant_dense)

# 1x128 (activation)
# will dispatch to torchinductor and fuse with prev_op
x_q, x_scale = flex_cast_quant_dense_c(
    x,
    block_size=128,
    dim=-1,
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=amax_to_scale_fn,
    cast_to_dtype_fn=cast_to_dtype_fn,
)

# 128x128 (weight)
# will dispatch to a handwritten triton template that beats inductor
w_q, w_scale = flex_cast_quant_dense_c(
    w,
    block_size=(128, 128),
    dim=(-2, -1),
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=amax_to_scale_fn,
    cast_to_dtype_fn=cast_to_dtype_fn,
)
```

## Code structure

```
api.py - the public API, and the logic when to use inductor vs templates
example.py - one e2e example of just doing the casts
recipes.py - deepseek recipes (for testing and benchmarking)
test.py - smoke tests
benchmark.py - bench all the deepseek casts (assumes B200)
hop/* - out of tree lowering to templates (100% clauded, didn't look much)
```

## Status

* toy examples run on a B200 for deepseek 1x128 and 128x128 recipes, with inductor generating 1x128 kernels and faster HOPs for 128x1 and 128x128 kernels
* out-of-tree hop machinery works fine (although this is 100% clauded)
* heuristic based dispatch to inductor or HOP based on the scaling grid works fine
* API looks nice (I'd use it)

## Not implemented

* hierarchical scaling (such as nvfp4)
* zero_point
* mx formats 
* scale swizzling
* stochastic rounding
* any real e2e use cases

## Open questions

* API bike shedding
* validate that the design will fit the features not yet implemented (nvfp4, RS, etc)
* validate that we can lower to cuteDSL for performant versions of the casts needed for MoE
* validate API with potential users

## Performance (on a B200)

```bash
# the _hop rows demonstrate triton template beating compile
# the _triton rows are just for debugging (manual triton kernels)
# cpu overhead of _hop is slightly higher than _triton but acceptable

> python benchmark.py

shape: (16384, 16384) bfloat16
recipe                           gpu_time_ms   gpu_gbps  gpu_pct_peak  cpu_time_ms
relu (eager baseline)                 0.1770     6066.9         75.8%       0.0158
deepseek_fp8_1_128                    0.1262     6446.9         80.6%       0.0731
deepseek_fp8_1_128_dim_m              0.2855     2850.0         35.6%       0.0911
deepseek_fp8_1_128_dim_m_hop          0.1430     5690.8         71.1%       0.0788
deepseek_fp8_1_128_dim_m_triton       0.1458     5579.9         69.7%       0.0502
deepseek_fp8_128_128                  0.2295     3508.6         43.9%       0.0927
deepseek_fp8_128_128_hop              0.1297     6208.3         77.6%       0.0778
deepseek_fp8_128_128_triton           0.1290     6244.4         78.1%       0.0544
```

## Alternatives

Alternative 1: - write custom kernels for every variant
  - pros: simplicity
  - cons: maintainability

Alternative 2: - just teach torch.compile to be good at all possible quantizations
  - pros: generality
  - cons: long term good, but this is not viable to quickly chase SOTA as compiler improvements take time
