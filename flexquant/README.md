# flexquant prototype

Goal: unified flexible API for quantizing a tensor, backed by performant kernels.

User provides:
* spec for how to tile the tensor (currently with `block_size` and `dim`)
* callables for how to calculate the scale (`amax_to_scale_fn`) and cast
a datum to low precision (`cast_to_dtype_fn`)

API handles behind the scenes:
* tiling the tensor in preparation for calculating the scale
* selecting from `torch.compile` or a manual kernel, based on what we expect
  to be faster for the provided tiling spec

99% clauded

Implemented so far:
* `flex_cast_quant_dense` API with the following recipes:
  * fp8 deepseek 1x128
  * fp8 deepseek 128x128
* scaling across dim=-1 and dim=-2
* a triton template for 128x128 blockwise, torch.compile path for all others

Not implemented yet:
* hierarchical scaling (such as nvfp4)
* zero_point
* battle tested recipes (with epsilon, other edge case handling, etc)
* mx formats and nvfp4
* scale swizzle

Alternative 1: - write custom kernels for every variant
  - main pro is simplicity
  - main con is maintainability
Alternative 2: - just teach torch.compile to be good at all possible quantizations
  - main pro is generality, and native PT code looks nie
  - cons users who want to stay in eager mode, and long eng time to implement the harder cases
