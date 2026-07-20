"""mm + flex_tile_map -> flex_gemm fusion, in BOTH forward and backward.

Forward computation is gemm -> epilogue -> gemm:

    c = a @ b                       # mm, outside the autograd function
    d = CustomUserFn.apply(c)       # forward = flex_tile_map(c, sin)
    e = d @ w

Backward of CustomUserFn is the true VJP of sin, expressed with flex_tile_map:

    grad_c = flex_tile_map(grad_d, lambda go: go * cos(c))

where grad_d = grad_e @ w.T is produced by a matmul in the backward graph. So the
backward graph ALSO contains a fusible mm -> flex_tile_map pair. The post-grad
pass runs on the forward and backward graphs separately; we assert the fusion
fires in each.
"""

import unittest

import torch
from api import (
    count_node_targets,
    flex_gemm_hop,
    flex_tile_map,
    flex_tile_map_hop,
    install_flex_tile_map_pass,
)

aten = torch.ops.aten

try:
    import cutlass  # noqa: F401

    HAS_CUTEDSL = True
except ImportError:
    HAS_CUTEDSL = False

TEST_CUDA = torch.cuda.is_available()
SM100 = TEST_CUDA and torch.cuda.get_device_capability(0) >= (10, 0)


def _the_flex_gemm_graph(graphs):
    """The single recorded post-grad graph that contains a flex_gemm node.

    ``install_flex_tile_map_pass`` records every graph the post-grad pass runs on
    (including intermediate ones with no match); pick the one where fusion landed.
    """
    matches = [g for g in graphs if count_node_targets(g, flex_gemm_hop)]
    assert len(matches) == 1, f"expected exactly one graph with a flex_gemm, got {len(matches)}"
    return matches[0]


def _sqnr(ref, actual):
    """Signal-to-quantization-noise ratio in dB (standard low-precision metric)."""
    ref = ref.double()
    actual = actual.double()
    noise = (ref - actual).pow(2).mean()
    if noise == 0:
        return float("inf")
    return (10 * torch.log10(ref.pow(2).mean() / noise)).item()


def _fwd_epilogue_fn(acc):
    return acc.sin()


class CustomUserFn(torch.autograd.Function):
    """y = sin(c), expressed via flex_tile_map. The mm stays OUTSIDE.

    Both forward and backward wrap their epilogue in flex_tile_map, so both
    graphs expose a fusible mm -> flex_tile_map pair. The backward is the true
    VJP of sin: grad_c = grad_out * cos(c), so it captures the saved input c.
    """

    @staticmethod
    def forward(ctx, c):
        ctx.save_for_backward(c)
        return flex_tile_map(c, _fwd_epilogue_fn)

    @staticmethod
    def backward(ctx, grad_out):
        (c,) = ctx.saved_tensors
        # correct VJP of sin: grad_out * cos(c). cos(c) captured into the epilogue.
        return flex_tile_map(grad_out, lambda go: go * c.cos())


class TestFlexTileMap(unittest.TestCase):
    @unittest.skipIf(not HAS_CUTEDSL, "CuteDSL required")
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100, "SM100+ required")
    def test_only_first_gemm_fuses_forward(self):
        torch._dynamo.reset()
        graphs = install_flex_tile_map_pass()

        def fn(a, b, w):
            c = torch.mm(a, b)            # first gemm         (fused with epilogue)
            d = CustomUserFn.apply(c)     # epilogue           (fusible)
            return torch.mm(d, w)         # second gemm        (not fused)

        # forward only: inputs do not require grad
        a = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(128, 32, device="cuda", dtype=torch.bfloat16)

        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        actual = compiled(a, b, w)

        # numerics vs eager bf16
        ref = torch.mm(a, b).sin() @ w
        torch.testing.assert_close(actual, ref, rtol=2e-2, atol=2e-2)

        # inspect the actual post-grad FX graph: exactly the first gemm+epilogue
        # fused into a flex_gemm, no flex_tile_map node left, and the second gemm
        # stays a plain aten.mm. (forward-only -> a single compiled graph.)
        g = _the_flex_gemm_graph(graphs)
        self.assertEqual(count_node_targets(g, flex_gemm_hop), 1)
        self.assertEqual(count_node_targets(g, flex_tile_map_hop), 0)
        self.assertEqual(count_node_targets(g, aten.mm.default), 1)

    @unittest.skipIf(not HAS_CUTEDSL, "CuteDSL required")
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100, "SM100+ required")
    def test_fusion_fires_in_forward_and_backward(self):
        torch._dynamo.reset()
        graphs = install_flex_tile_map_pass()

        def fn(a, b, w):
            c = torch.mm(a, b)
            d = CustomUserFn.apply(c)
            return torch.mm(d, w)

        def make_inputs():
            a = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            w = torch.randn(128, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            return a, b, w

        # compiled fwd + bwd
        ca, cb, cw = make_inputs()
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled(ca, cb, cw).sum().backward()

        # eager fwd + bwd on identical inputs
        ea = ca.detach().clone().requires_grad_(True)
        eb = cb.detach().clone().requires_grad_(True)
        ew = cw.detach().clone().requires_grad_(True)
        fn(ea, eb, ew).sum().backward()

        # compiled and eager both run in bf16 but with different kernel reduction
        # orders, so individual elements differ by a few ulps (occasionally more).
        # Compare with SQNR (the standard low-precision numerics metric) rather
        # than an elementwise tolerance, which would trip on benign bf16 noise.
        for cg, eg, name in [
            (ca.grad, ea.grad, "a"),
            (cb.grad, eb.grad, "b"),
            (cw.grad, ew.grad, "w"),
        ]:
            self.assertGreater(_sqnr(eg, cg), 30.0, f"grad {name} SQNR too low")

        # pin the true VJP of sin: grad_d = grad_e @ w.T (grad_e = ones from
        # .sum()), grad_c = grad_d * cos(c), grad_a = grad_c @ b.T.
        c = ea @ eb
        grad_d = torch.ones(256, 32, device="cuda", dtype=torch.bfloat16) @ ew.t()
        grad_c = grad_d * c.cos()
        grad_a_ref = grad_c @ eb.t()
        self.assertGreater(_sqnr(grad_a_ref, ca.grad), 30.0, "grad a vs pinned VJP")

        # inspect the actual post-grad FX graphs: the fusion fired in BOTH the
        # forward and the backward graph (each has one flex_gemm), and no
        # flex_tile_map node survives in any recorded graph.
        with_flex_gemm = [g for g in graphs if count_node_targets(g, flex_gemm_hop)]
        self.assertEqual(
            len(with_flex_gemm), 2, "expected a flex_gemm in the fwd AND bwd graph"
        )
        for g in with_flex_gemm:
            self.assertEqual(count_node_targets(g, flex_gemm_hop), 1)
        self.assertTrue(
            all(count_node_targets(g, flex_tile_map_hop) == 0 for g in graphs),
            "a flex_tile_map node survived fusion",
        )


if __name__ == "__main__":
    unittest.main()
