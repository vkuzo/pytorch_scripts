"""flex_tile_map_v2: a frontend that compile-lowers to flex_gemm.

The user writes two separate ops:

    c = torch.mm(a, b)
    d = flex_tile_map(c, f)          # f(acc) -> Tensor | tuple[Tensor, ...]

Keeping them separate (rather than calling ``flex_gemm(torch.mm, (a, b), f)``
directly) means the autograd boundary sits around the epilogue alone -- which is
the whole point of this frontend. Under ``torch.compile``, an Inductor post-grad
pass re-fuses the pair into a single ``flex_gemm`` call, recovering the fused
kernel. ``f`` has the identical signature to flex_gemm's epilogue.

This is deliberately minimal: only the ``mm -> flex_tile_map`` shape is handled,
and it lowers via flex_gemm's QUACK backend.
"""

import torch
from torch._higher_order_ops.base_hop import BaseHOP, FunctionWithNoFreeVars
from torch._higher_order_ops.flex_gemm import flex_gemm_hop
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch._inductor.pattern_matcher import (
    CallFunctionVarArgs,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)

__all__ = [
    "flex_tile_map",
    "flex_tile_map_hop",
    "flex_gemm_hop",
    "install_flex_tile_map_pass",
    "count_node_targets",
    "FLEX_TILE_MAP_PASS",
]

aten = torch.ops.aten


# ---------------------------------------------------------------------------
# The HOP: flex_tile_map(subgraph, input_tensor)
# ---------------------------------------------------------------------------


class FlexTileMap(BaseHOP):
    """A BaseHOP so Dynamo traces it under fullgraph=True (via BaseHOPVariable).

    BaseHOP supplies the CompositeExplicitAutograd (eager ``f(input)``), fake,
    functionalize, autograd, and ProxyTorchDispatchMode impls for free. This
    frontend is forward-only in practice, so we keep BaseHOP's defaults.
    """

    def __init__(self) -> None:
        super().__init__("flex_tile_map")


flex_tile_map_hop = FlexTileMap()


def flex_tile_map(input, f):
    """Apply the epilogue ``f`` to ``input`` as a standalone, fusible op.

    ``f(acc) -> Tensor | tuple[Tensor, ...]`` -- a single positional arg (the full
    tensor), extra tensors captured by closure. Identical to flex_gemm's epilogue.
    """
    if torch.compiler.is_dynamo_compiling():
        # Dynamo speculates ``f`` into a subgraph itself; pass it through raw.
        return flex_tile_map_hop(f, input)
    # Eager: BaseHOP.__call__ requires a wrapped callable (no free vars).
    return flex_tile_map_hop(FunctionWithNoFreeVars(f), input)


# ---------------------------------------------------------------------------
# Post-grad fusion: flex_tile_map_hop(sub, mm(a, b)) -> flex_gemm_hop(...)
# ---------------------------------------------------------------------------


FLEX_TILE_MAP_PASS = PatternMatcherPass(pass_name="flex_tile_map_fusion")


def _resolve_gm(owning_gm, arg):
    """The HOP subgraph arg is a get_attr node (or, defensively, a GraphModule)."""
    if isinstance(arg, torch.fx.Node) and arg.op == "get_attr":
        return getattr(owning_gm, arg.target)
    if isinstance(arg, torch.fx.GraphModule):
        return arg
    raise AssertionError(f"unexpected flex_tile_map subgraph arg: {arg!r}")


def _build_fused_body(epilogue_gm, tile_idx, mm_val, out_val, aux_vals):
    """Build a GraphModule ``(a, b, *aux) -> epilogue(...)`` for flex_gemm.

    This matches what flex_gemm's own body looks like: the GEMM operands ``a, b``
    are the leading placeholders, any captured epilogue tensors follow as trailing
    placeholders (flex_gemm appends captured tensors to the gemm-args tuple), the
    body contains an ``aten.mm.default`` node, and the epilogue is inlined on top.

    The epilogue's placeholders correspond positionally to the flex_tile_map
    operands. ``tile_idx`` is the operand index that is the mm tile (NOT
    necessarily 0 -- e.g. in a backward graph Dynamo may order a captured
    activation before the gradient tile). The tile placeholder maps to the mm
    result; every other epilogue placeholder maps to a fresh aux placeholder, in
    order. ``mm_val``/``out_val``/``aux_vals`` stamp ``meta['val']`` (the QUACK
    lowering reads the output's ``meta['val']``).
    """
    g = torch.fx.Graph()
    # All placeholders MUST come first and contiguously: process_subgraph_nodes
    # (the flex_gemm default-backend lowering) binds placeholder node at absolute
    # graph index i to args[i], so a placeholder appearing after the mm node would
    # read an out-of-range arg. Order: gemm operands (a, b), then captured aux.
    pa = g.placeholder("a")
    pb = g.placeholder("b")
    aux_placeholders = []
    for i, aux_val in enumerate(aux_vals):
        p = g.placeholder(f"aux{i}")
        p.meta["val"] = aux_val
        aux_placeholders.append(p)

    mm_node = g.call_function(aten.mm.default, (pa, pb))
    mm_node.meta["val"] = mm_val

    remap = {}
    aux_iter = iter(aux_placeholders)
    output_val = None
    ph_idx = 0
    for node in epilogue_gm.graph.nodes:
        if node.op == "placeholder":
            if ph_idx == tile_idx:
                remap[node] = mm_node  # the tile -> mm(a, b)
            else:
                remap[node] = next(aux_iter)  # a captured operand -> aux input
            ph_idx += 1
        elif node.op == "output":
            output_val = node
        else:
            new_node = g.node_copy(node, lambda x: remap[x])
            remap[node] = new_node

    assert output_val is not None, "epilogue graph has no output"
    out_args = torch.fx.map_arg(output_val.args, lambda x: remap[x])
    out_node = g.output(out_args[0])
    out_node.meta["val"] = out_val

    fused = torch.fx.GraphModule(epilogue_gm, g)
    fused.recompile()
    return fused


@register_graph_pattern(
    CallFunctionVarArgs(flex_tile_map_hop), pass_dict=FLEX_TILE_MAP_PASS
)
def _fuse_mm_into_flex_gemm(match: Match, *args, **kwargs):
    ftm_node = match.nodes[-1]
    # node is flex_tile_map(subgraph, *operands); operands correspond positionally
    # to the epilogue subgraph placeholders. Exactly one operand is the mm "tile";
    # the rest are captured aux tensors (which may come BEFORE the tile -- e.g. in
    # a backward graph Dynamo can order a saved activation ahead of the grad tile).
    subgraph_arg = ftm_node.args[0]
    operands = list(ftm_node.args[1:])

    def is_fusible_mm(n):
        return (
            isinstance(n, torch.fx.Node)
            and n.op == "call_function"
            and n.target is aten.mm.default
        )

    # pick the tile operand: the (first) operand that is an aten.mm. It need NOT
    # be single-use -- in a joint fwd+bwd graph the mm output commonly also feeds
    # a saved-for-backward use; we recompute the mm inside the fused body and only
    # erase the original mm if nothing else still reads it.
    tile_idx = next((i for i, n in enumerate(operands) if is_fusible_mm(n)), None)
    if tile_idx is None:
        return

    input_node = operands[tile_idx]
    aux_nodes = [n for i, n in enumerate(operands) if i != tile_idx]

    a, b = input_node.args
    graph = match.graph
    owning_gm = graph.owning_module

    epilogue_gm = _resolve_gm(owning_gm, subgraph_arg)
    mm_val = input_node.meta["val"]
    out_val = ftm_node.meta["val"]
    aux_vals = [n.meta["val"] for n in aux_nodes]
    fused_body = _build_fused_body(epilogue_gm, tile_idx, mm_val, out_val, aux_vals)

    # Register the fused body as a submodule and emit the flex_gemm node.
    body_name = _register_submodule(owning_gm, fused_body)

    with graph.inserting_before(ftm_node):
        body_attr = graph.get_attr(body_name)
        # This is exactly the node the public flex_gemm(torch.mm, (a, b),
        # epilogue_fn) builds and dispatches: the gemm op, a body graph
        # (a, b, *aux) -> epilogue_fn(..., mm(a, b), ...), the gemm args (with any
        # captured epilogue tensors appended, as flex_gemm does), empty
        # gemm_kwargs, and empty kernel_options (flex_gemm's default backend).
        # Opting into QUACK would be kernel_options={"backend": "QUACK"} -- an
        # orthogonal backend choice, currently broken in this env by a
        # vendored-QuACK/cutlass_dsl version mismatch (fails identically for
        # stock flex_gemm).
        new_node = graph.call_function(
            flex_gemm_hop,
            args=(aten.mm.default, body_attr, (a, b, *aux_nodes), {}, {}),
        )
    new_node.meta.update(ftm_node.meta)

    ftm_node.replace_all_uses_with(new_node)
    graph.erase_node(ftm_node)
    if len(input_node.users) == 0:
        graph.erase_node(input_node)


def _register_submodule(owning_gm, submod) -> str:
    """Register ``submod`` under a fresh qualname and return it."""
    i = 0
    while hasattr(owning_gm, f"flex_tile_map_fused_body_{i}"):
        i += 1
    name = f"flex_tile_map_fused_body_{i}"
    owning_gm.register_module(name, submod)
    return name


# ---------------------------------------------------------------------------
# Pass installation
# ---------------------------------------------------------------------------


class _FlexTileMapPass(CustomGraphPass):
    """Runs the fusion and records the resulting post-grad graph(s) for inspection.

    The recorded graphs are the *actual* post-grad FX graphs (one per compiled
    graph -- forward and backward are separate), so a test can iterate
    ``graph.nodes`` and count real node targets (``flex_gemm_hop``,
    ``flex_tile_map_hop``, ``aten.mm``) instead of trusting computed telemetry.
    """

    def __init__(self, recorded_graphs: list):
        self._recorded = recorded_graphs

    def __call__(self, graph: torch.fx.Graph) -> None:
        FLEX_TILE_MAP_PASS.apply(graph)
        self._recorded.append(graph)

    def uuid(self):
        # depends on external captured state, so opt out of fx graph caching
        return None


def install_flex_tile_map_pass():
    """Install the fusion pass as Inductor's post-grad custom post pass.

    Returns a list that, after compilation, holds the post-grad FX graph(s) the
    fusion ran on -- inspect their ``.nodes`` directly (see
    :func:`count_node_targets`).
    """
    recorded_graphs: list = []
    torch._inductor.config.post_grad_custom_post_pass = _FlexTileMapPass(recorded_graphs)
    return recorded_graphs


def count_node_targets(graph, target) -> int:
    """Count call_function nodes in an fx graph whose target is ``target``."""
    return sum(
        n.op == "call_function" and n.target is target for n in graph.nodes
    )
