"""Correctness of the plain Helion deepseek 1x128 kernel vs plain-PyTorch reference.

Uses the fixed-config kernel (no autotuning) so the test runs in seconds.
"""

import pytest
import torch

from main import (
    BLOCK_N,
    deepseek_quant_1x128_fixed,
    deepseek_quant_1x128_reshape_fixed,
    deepseek_quant_1x128_reference,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA (Helion GPU kernel)"
)

# Both fixed-config Helion kernels: the nested-loop one and the host-reshape
# (reduction-axis) one. Both should be bit-exact vs the eager reference.
KERNELS = {
    "nested_loop": deepseek_quant_1x128_fixed,
    "reshape_axis": deepseek_quant_1x128_reshape_fixed,
}


@pytest.mark.parametrize("name", list(KERNELS), ids=list(KERNELS))
def test_deepseek_1x128_matches_reference(name):
    kernel = KERNELS[name]
    torch.manual_seed(0)
    M, K = 4096, 4096
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    qdata, scale = kernel(x)
    qdata_ref, scale_ref = deepseek_quant_1x128_reference(x)

    assert qdata.shape == (M, K)
    assert qdata.dtype == torch.float8_e4m3fn
    assert scale.shape == (M, K // BLOCK_N)
    assert scale.dtype == torch.float32

    # bit-exact vs reference (v1 discipline)
    assert torch.equal(qdata.to(torch.float32), qdata_ref.to(torch.float32))
    assert torch.equal(scale, scale_ref)
