"""Unit test: our dropless grouped_mm_2d_3d MoE block matches HF ``DeepseekV3MoE``.

We instantiate the real HuggingFace ``DeepseekV3MoE`` building block, copy its weights
into our ``RefMoE``, run both on the same input, and assert the outputs match. This
validates that the grouped_mm_2d_3d-based dropless implementation is numerically equivalent
to HF's per-expert ``torch.where`` loop (only the token iteration order differs).
"""

import pytest
import torch

from moe import GroupedExpertGEMM, RefMoE, grouped_mm_2d_3d

from transformers import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE


# --- small config so the test is fast but exercises group-limited routing ---
HIDDEN = 64
MOE_INTERMEDIATE = 32
N_ROUTED_EXPERTS = 8
TOP_K = 4
N_GROUP = 2
TOPK_GROUP = 2
N_SHARED = 2
ROUTED_SCALING = 2.5


def make_config():
    return DeepseekV3Config(
        hidden_size=HIDDEN,
        intermediate_size=128,  # dense MLP size (unused by the MoE block)
        moe_intermediate_size=MOE_INTERMEDIATE,
        n_routed_experts=N_ROUTED_EXPERTS,
        num_local_experts=N_ROUTED_EXPERTS,
        num_experts_per_tok=TOP_K,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
        n_shared_experts=N_SHARED,
        norm_topk_prob=True,
        routed_scaling_factor=ROUTED_SCALING,
        hidden_act="silu",
        num_hidden_layers=1,
    )


def build_pair(dtype, device, nonzero_bias):
    """Build an HF DeepseekV3MoE and a RefMoE with identical (random) weights."""
    config = make_config()

    torch.manual_seed(0)
    hf = DeepseekV3MoE(config).to(device=device, dtype=dtype)

    # Randomize all parameters (default init leaves the 3D expert params uninitialized).
    with torch.no_grad():
        for p in hf.parameters():
            p.normal_(0, 0.1)
        if nonzero_bias:
            hf.gate.e_score_correction_bias.normal_(0, 0.1)

    ref = RefMoE(
        hidden_size=HIDDEN,
        moe_intermediate_size=MOE_INTERMEDIATE,
        n_routed_experts=N_ROUTED_EXPERTS,
        num_experts_per_tok=TOP_K,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
        n_shared_experts=N_SHARED,
        norm_topk_prob=True,
        routed_scaling_factor=ROUTED_SCALING,
        hidden_act="silu",
    ).to(device=device, dtype=dtype)

    # Copy weights by name. Layouts match exactly (we transpose inside grouped_mm_2d_3d).
    with torch.no_grad():
        ref.gate.weight.copy_(hf.gate.weight)
        ref.gate.e_score_correction_bias.copy_(hf.gate.e_score_correction_bias)
        ref.experts.gate_up_proj.copy_(hf.experts.gate_up_proj)
        ref.experts.down_proj.copy_(hf.experts.down_proj)
        ref.shared_experts.gate_proj.weight.copy_(hf.shared_experts.gate_proj.weight)
        ref.shared_experts.up_proj.weight.copy_(hf.shared_experts.up_proj.weight)
        ref.shared_experts.down_proj.weight.copy_(hf.shared_experts.down_proj.weight)

    return hf, ref


def _tols(dtype):
    if dtype == torch.float32:
        return dict(rtol=1e-5, atol=1e-5)
    return dict(rtol=2e-2, atol=2e-2)  # bf16


def _devices():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 8), (1, 16), (3, 5)])
@pytest.mark.parametrize("nonzero_bias", [False, True])
def test_matches_hf_deepseek_v3(device, dtype, shape, nonzero_bias):
    hf, ref = build_pair(dtype, device, nonzero_bias)
    batch, seq = shape

    torch.manual_seed(1)
    x = torch.randn(batch, seq, HIDDEN, device=device, dtype=dtype)

    with torch.no_grad():
        out_hf = hf(x.clone())
        out_ref = ref(x.clone())

    assert out_ref.shape == out_hf.shape
    torch.testing.assert_close(out_ref, out_hf, **_tols(dtype))


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("nonzero_bias", [False, True])
def test_gradients_match_hf_deepseek_v3(device, nonzero_bias):
    """Backward equivalence: input grad and all parameter grads match HF.

    Run in float32 (gradient checks need the precision). We backprop the same scalar
    loss through both modules and compare grad w.r.t. the input and every shared
    parameter. Routed-expert weights are stored with identical layout, so their grads
    compare directly.
    """
    dtype = torch.float32
    hf, ref = build_pair(dtype, device, nonzero_bias)

    batch, seq = 2, 8
    torch.manual_seed(1)
    base = torch.randn(batch, seq, HIDDEN, device=device, dtype=dtype)

    x_hf = base.clone().requires_grad_(True)
    x_ref = base.clone().requires_grad_(True)

    # Same fixed weighting so the two scalar losses are identical functions of output.
    torch.manual_seed(2)
    grad_seed = torch.randn(batch, seq, HIDDEN, device=device, dtype=dtype)

    (hf(x_hf) * grad_seed).sum().backward()
    (ref(x_ref) * grad_seed).sum().backward()

    tols = _tols(dtype)

    # Gradient w.r.t. the input.
    torch.testing.assert_close(x_ref.grad, x_hf.grad, **tols)

    # Gradients w.r.t. parameters (names map 1:1 between the two modules).
    param_pairs = [
        (ref.gate.weight, hf.gate.weight),
        (ref.experts.gate_up_proj, hf.experts.gate_up_proj),
        (ref.experts.down_proj, hf.experts.down_proj),
        (ref.shared_experts.gate_proj.weight, hf.shared_experts.gate_proj.weight),
        (ref.shared_experts.up_proj.weight, hf.shared_experts.up_proj.weight),
        (ref.shared_experts.down_proj.weight, hf.shared_experts.down_proj.weight),
    ]
    for ref_p, hf_p in param_pairs:
        assert ref_p.grad is not None
        assert hf_p.grad is not None
        torch.testing.assert_close(ref_p.grad, hf_p.grad, **tols)


def test_dropless_property():
    """Every routed (token, expert) pair is computed: offs[-1] == T * top_k."""
    device, dtype = "cpu", torch.float32
    _, ref = build_pair(dtype, device, nonzero_bias=False)

    batch, seq = 3, 7
    T = batch * seq
    x = torch.randn(batch, seq, HIDDEN, device=device, dtype=dtype)

    idx, _ = ref.gate(x)
    counts = torch.bincount(idx.reshape(-1), minlength=N_ROUTED_EXPERTS)
    assert int(counts.sum()) == T * TOP_K  # nothing dropped


def test_grouped_expert_gemm_gradcheck():
    """The custom GroupedExpertGEMM manual backward matches autograd (float64)."""
    torch.manual_seed(0)
    E, H, I = 3, 5, 4
    # Expert-contiguous token layout with group sizes [2, 0, 3] (empty group included).
    offs = torch.tensor([2, 2, 5])
    T = int(offs[-1])

    x_sorted = torch.randn(T, H, dtype=torch.float64, requires_grad=True)
    gate_up_proj = torch.randn(E, 2 * I, H, dtype=torch.float64, requires_grad=True)
    down_proj = torch.randn(E, H, I, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        GroupedExpertGEMM.apply,
        (x_sorted, gate_up_proj, down_proj, offs),
        eps=1e-6,
        atol=1e-5,
    )


def test_grouped_mm_2d_3d_matches_dense():
    """grouped_mm_2d_3d equals per-expert dense matmul on a hand-built layout."""
    torch.manual_seed(0)
    E, K, N = 3, 4, 5
    w = torch.randn(E, K, N)
    # group sizes [2, 0, 3] -> exercises an empty group too
    x = torch.randn(5, K)
    offs = torch.tensor([2, 2, 5])

    out = grouped_mm_2d_3d(x, w, offs)
    expected = torch.cat([x[0:2] @ w[0], x[2:2] @ w[1], x[2:5] @ w[2]], dim=0)
    torch.testing.assert_close(out, expected)
