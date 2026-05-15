# flexquant prototype

Goal: unified flexible API for quantizing a tensor, backed by performant kernels
Specifically:
* tiling for calculating the scale is constrained by the API
* function to calculate the scale from a tile is provided by the user (with constraints)
* function to quantize a single value is provided by the user (with constraints)
* the API either uses torch.compile to generate the kernel from scratch or lowers to a pre-selected template

100% clauded

Implemented so far:
* flex_quant_dense API with support for single stage scaling
* testing for the API for simplified deepseek 1x128, deepseek 128x128, float8 rowwise recipe stubs
* torch.compile path only
* dim=-1 and row-major output only
* no scale swizzle

Not implemented:
* hierarchical scaling (nvfp4, etc)
* zero_point
* actual recipes (with epsilon, other edge case handling, etc)
* actual templates to lower to for cases where compile is not going to generate a good kernel from scratch

Alternative 1: - just write custom kernels for every variant
  - main pro is simplicity
  - main con is maintainability
Alternative 2: - just teach torch.compile to be good at all possible quantizations
  - main pro is generality, and native PT code looks nie
  - cons users who want to stay in eager mode, and long eng time to implement the harder cases
