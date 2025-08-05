"""
Walkthrough of `torch._grouped_mm` signature, forward and backward formulas

Signature (from https://github.com/pytorch/pytorch/pull/150374/files)

  func: _grouped_mm(Tensor self, Tensor mat2, Tensor? offs=None, Tensor? bias=None, ScalarType? out_dtype=None) -> Tensor
"""

import copy

import torch
import torch.nn as nn

torch.manual_seed(0)

def run():
    M, K, N = 4, 16, 32
    # Simple routing - x[:1] to expert0, x[1:] to expert1. In the real use
    # case we would enter this flow after token shuffling.
    M_cut_idx = 1

    # create data
    x_ref = torch.randn(M, K, device="cuda").bfloat16().requires_grad_()
    expert0_ref = nn.Linear(K, N, device="cuda", bias=False).bfloat16()
    expert1_ref = nn.Linear(K, N, device="cuda", bias=False).bfloat16()
    grad_ref = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    x = copy.deepcopy(x_ref)
    expert0 = copy.deepcopy(expert0_ref)
    expert1 = copy.deepcopy(expert1_ref)
    grad = copy.deepcopy(grad_ref)

    x0_ref = x_ref[:M_cut_idx]
    x1_ref = x_ref[M_cut_idx:]

    #
    # reference - separate gemms
    #
    y0_ref = x0_ref @ expert0_ref.weight.t()
    y1_ref = x1_ref @ expert1_ref.weight.t()
    y_ref = torch.cat([y0_ref, y1_ref], dim=0) 
    y_ref.backward(grad_ref)
    # print(y_ref)

    #
    # grouped_mm
    #

    # create combined weight
    w0t = expert0.weight.t()  # [N, K] -> [K, N]
    w1t = expert1.weight.t()  # [N, K] -> [K, N]
    w01t = torch.stack([w0t, w1t], dim=0)  # [num_experts, K, N]

    # create offsets
    # offset for token 0 means slice of 0:M_cut_idx
    # offset for token 1 means slice of M_cut_idx:M
    x_offsets = torch.tensor([M_cut_idx, M], device="cuda", dtype=torch.int32)

    y = torch._grouped_mm(
        x,  # [M, K]
        w01t,  # [num_experts, K, N]
        x_offsets,  # [num_experts,]
    )  # [M, N]
    # print(y)
    y.backward(grad)

    # 
    # verify outputs match
    #

    # y
    torch.testing.assert_close(y_ref, y, atol=0, rtol=0)

    # grad_input
    torch.testing.assert_close(x_ref.grad, x.grad, atol=0, rtol=0)

    # grad_weight
    torch.testing.assert_close(expert0_ref.weight.grad, expert0.weight.grad, atol=0, rtol=0)
    torch.testing.assert_close(expert1_ref.weight.grad, expert1.weight.grad, atol=0, rtol=0)

    #
    # now, reproduce grad_input and grad_weight using calls to 
    # `torch._grouped_mm`, demonstrating proper use of offsets
    #
    # individual weight stored with shape [N, K], so weight.t() is [K, N]
    #
    # forward, with shapes: 
    #
    #   output[M, N] = input[M, K] @ weight.t()[num_experts, K, N]
    #
    # backward, with shapes:
    #
    #   grad_input[M, K] = grad_output[M, N] @ weight[num_experts, N, K]
    #
    #   grad_weight[num_experts, N, K] = grad_output.t()[N, M] @ input[M, K]
    # 

    with torch.no_grad():
        # calculate grad_input by hand
        w01 = torch.stack([expert0.weight, expert1.weight])  # num_experts, N, K
        grad_input_repro = torch._grouped_mm(grad, w01, x_offsets)
        torch.testing.assert_close(grad_input_repro, x_ref.grad, atol=0, rtol=0)

        # calculate grad_weight by hand
        grad_weight_repro = torch._grouped_mm(grad.t(), x, x_offsets)
        grad_weight_ref = torch.stack([expert0.weight.grad, expert1.weight.grad])
        torch.testing.assert_close(grad_weight_repro, grad_weight_ref, atol=0, rtol=0)
    
    print('done')

if __name__ == '__main__':
    run() 
