from collections import defaultdict
from dataclasses import dataclass, field
import fire
import re
import os

from typing import *

from jinja2 import Template

from graphviz import Digraph

# aot_joint_graph only (filename is misnamed)
test_fname1 = 'test_inputs/input_aot_graphs.txt'
# aot_joint_graph and aot_graphs
test_fname2 = 'test_inputs/input_aot_joint_graph_and_aot_graphs.txt'
# also output_code
test_fname3 = 'test_inputs/input_aot_joint_graph_aot_graphs_output_code.txt'

TENSOR_NODE_COLOR = 'lavender'
FUNC_NODE_COLOR = 'white'
INPUT_EDGE_COLOR = 'lightgray'

@dataclass
class Node:
    node_name: str
    args: List[str]
    metadata: Dict[str, str]
    node_type: str

@dataclass
class ParsedGraph:
    inputs: List[Node] = field(default_factory=lambda: [])
    nodes: Dict[str, Node] = field(default_factory=lambda: {})
    outputs: List[str] = field(default_factory=lambda: [])


def parse_graph(
    list_of_lines: List[str],
    start_idx: int,
    end_idx: int,
):
    """
    Input: a list of lines from a file, and a start and end index of the region to parse into a graph
    Output: a parsed graph
    """

    g = ParsedGraph()

    # functions can be called multiple times, keep track of
    # number of calls so we can have unique nodes per function-call
    function_name_to_num_calls = defaultdict(int)

    for line in list_of_lines[start_idx:end_idx+1]:

        # specific to joint graph
        # set up primals and tangents
        # primals_1, primals_2, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
        primals_tangents_re = re.compile('[ ]+(.*), = fx_pytree.tree_flatten_spec.*')
        res = primals_tangents_re.search(line)
        if res:
            inputs_list = [s.strip() for s in res.group(1).split(',')]
            for input_str in inputs_list:
                n = Node(input_str, [], metadata={}, node_type='input')
                g.inputs.append(n)
            continue

        # specific to forward/backward graph
        # def forward(self, primals_1: "bf16[8192, 4096][4096, 1]cuda:0", primals_2: "bf16[2048, 4096][4096, 1]cuda:0"):
        forward_def_re = re.compile('[ ]+def forward\(self, (.*)\):')
        res = forward_def_re.search(line)
        if res:
            # convert
            #   primals_1: "bf16[8192, 4096][4096, 1]cuda:0", primals_2: "bf16[2048, 4096][4096, 1]cuda:0"
            # to
            #   primals_1, primals_2
            new_res = ''
            inside_quotes = False
            for char in res.group(1):
                if char == '"':
                    inside_quotes = not inside_quotes
                if not inside_quotes:
                    new_res += char
            new_res = new_res.replace('"',"")
            inputs_list = [s.replace(':', '').strip() for s in new_res.split(',')]
            for input_str in inputs_list:
                # ignore vars named `primals` and `tangents`, special cased for joint graph,
                # since we already parse out `primals_1`, etc
                if input_str in ('primals', 'tangents'):
                    continue

                n = Node(input_str, [], metadata={}, node_type='input')
                g.inputs.append(n)

        # function call
        # abs_1: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.abs.default(primals_2)
        function_call_re = re.compile('[ ]+(\w+): "(.*)" = ([\w\.]+)\((.*)\)')
        res = function_call_re.search(line)
        if res:
            var_name, var_hint, func_name, args_str = res.group(1), res.group(2), res.group(3), res.group(4)

            func_num_calls = function_name_to_num_calls[func_name]
            function_name_to_num_calls[func_name] += 1
            cur_func_node_name = f'{func_name} {func_num_calls}'

            args_list = [s.strip() for s in args_str.split(',')]

            g.nodes[cur_func_node_name] = Node(
                cur_func_node_name,
                args_list,
                metadata={},
                node_type='func',
            )
            g.nodes[var_name] = Node(
                var_name,
                [cur_func_node_name],
                metadata={'hint': var_hint},
                node_type='args',
            )

            continue

        # return statement for aot_joint_graph
        # return pytree.tree_unflatten([view_1, permute_8, view_3], self._out_spec)
        return_re = re.compile('return pytree.tree_unflatten\(\[(.*)\].*\)')
        res = return_re.search(line)
        if res:
            tokens_str = res.group(1)
            tokens_list = [s.strip() for s in tokens_str.split(',')]
            g.outputs = tokens_list
            continue

        # return statement for aot_graphs
        return_re = re.compile('return \((.*)\)')
        res = return_re.search(line)
        if res:
            tokens_str = res.group(1)
            tokens_list = [s.strip() for s in tokens_str.split(',')]
            g.outputs = tokens_list
            continue

    # verify captured information is valid
    assert len(g.inputs) > 0
    assert len(g.nodes) > 0
    assert len(g.outputs) > 0

    return g

def parse_triton_region_graph(
    list_of_lines: List[str],
    start_idx: int,
    end_idx: int,
):
    """
    Example graph to parse:

...
def triton_red_fused_abs_max_0(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
...
def triton_per_fused__to_copy_abs_clamp_max_mul_reciprocal_1(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel):
...
def call(args):
    primals_1, primals_2 = args
    ...
    with torch.cuda._DeviceGuard(0):
        ...
        buf0 = empty_strided_cuda((512, ), (1, ), torch.float32)
        triton_red_fused_abs_max_0.run(primals_1, buf0, 512, 32768, grid=grid(512), stream=stream0)
        ...
        buf1 = empty_strided_cuda((), (), torch.bfloat16)
        buf4 = empty_strided_cuda((), (), torch.float32)
        ...
        triton_per_fused__to_copy_abs_clamp_max_mul_reciprocal_1.run(buf0, buf1, buf4, 1, 512, grid=grid(1), stream=stream0)
        ...
        extern_kernels._scaled_mm(buf6, buf7, buf8, buf5, out_dtype=torch.bfloat16, use_fast_accum=True, out=buf9)
        ...
    return (buf8, buf4, reinterpret_tensor(buf9, (4096, 4096), (1, 4096), 0), reinterpret_tensor(buf12, (4096, 4096), (1, 4096), 0), buf13, )
    """
    g = ParsedGraph()

    cur_empty_buffers_set = set()
    cur_populated_buffers_set = set()

    kernel_name_to_arg_idx_to_input_output_none = {}

    # triton kernels can be called multiple times, keep track of
    # number of calls so we can have unique nodes per kernel-call
    kernel_name_to_num_calls = defaultdict(int)

    has_matched_final_return = False
    cur_kernel_type = None

    kernel_name_to_kernel_type = defaultdict(str)

    node_name_to_hint = defaultdict(str)

    for line in list_of_lines[start_idx:end_idx+1]:

        # parse the kernel type
        # @triton_heuristics.reduction(
        triton_kernel_type_re = re.compile('@triton_heuristics\.(.*)\(')
        res = triton_kernel_type_re.match(line)
        if res:
            cur_kernel_type = res.group(1)

        # parse the individual kernel definition lines, we'll use this to map from buffer to input vs output
        individual_kernel_def_re = re.compile('def (triton_.*)\((.*)\)')
        res = individual_kernel_def_re.match(line)
        if res:
            args = res.group(2).split(', ')
            arg_idx_to_category = {}
            for arg_idx, arg in enumerate(args):
                if arg.startswith('in_ptr'):
                    arg_idx_to_category[arg_idx] = 'in_ptr'
                elif arg.startswith('out_ptr'):
                    arg_idx_to_category[arg_idx] = 'out_ptr'
                else:
                    arg_idx_to_category[arg_idx] = None
            kernel_name = res.group(1)
            kernel_name_to_arg_idx_to_input_output_none[kernel_name] = arg_idx_to_category
            kernel_name_to_kernel_type[kernel_name] = cur_kernel_type

        # primals_1, primals_2 = args
        args_re = re.compile('[ ]+(.*) = args$')
        res = args_re.match(line)
        if res:
            args_str = res.group(1).split(', ')
            for arg_str in args_str:
                n = Node(arg_str, [], metadata={}, node_type='input')
                g.inputs.append(n)
            continue

        # parse size hint on incoming variables
        # assert_size_stride(permute_3, (8192, 4096), (1, 8192))
        incoming_var_size_hint_re = re.compile('[ ]+assert_size_stride\(([\w]+), (.*)\)')
        res = incoming_var_size_hint_re.match(line)
        if res:
            node_name, hint = res.group(1), res.group(2)
            node_name_to_hint[node_name] = hint

            # modify the existing input node to include the size hint
            for cur_input_node in g.inputs:
                if cur_input_node.node_name == node_name:
                    cur_input_node.metadata['hint'] = hint
                    break

            continue

        # parse size hint on created buffers
        # buf0 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buffer_create_re = re.compile('[ ]+([\w]+) = empty_strided_cuda\((.*)\)')
        res = buffer_create_re.match(line)
        if res:
            node_name, hint = res.group(1), res.group(2)
            node_name_to_hint[node_name] = hint
            continue

        # parse size hint on reused buffers
        # buf5 = reinterpret_tensor(buf2, (8192, 2048), (2048, 1), 0); del buf2  # reuse
        buffer_reuse_re = re.compile('[ ]+([\w]+) = reinterpret_tensor\(([\w]+), (.*)\);.*')
        res = buffer_reuse_re.match(line)
        if res:
            node_name, _prev_node_name, hint = res.group(1), res.group(2), res.group(3)
            node_name_to_hint[node_name] = hint
            continue

        # triton_red_fused_abs_max_0.run(primals_1, buf0, 512, 32768, grid=grid(512), stream=stream0)
        triton_kernel_call_re = re.compile('[ ]+(triton_.*)\.run\((.*)\)')
        res = triton_kernel_call_re.match(line)
        if res:
            # TODO: handle kwargs if needed
            args_str = res.group(2)
            args = args_str.split(', ')
            kernel_name = res.group(1)

            cur_node_inputs = []
            cur_node_outputs = []

            arg_idx_to_input_output_none = kernel_name_to_arg_idx_to_input_output_none[kernel_name]

            # since triton kernels are often M inputs to N outputs, we create the following nodes
            # to make the graph easy to read:
            # input_1 ... input_n
            #          |
            #        kernel
            #          |
            # output_1 ... output_n

            # look back in the parsed in_ptr / out_ptr map to categorize args into inputs/outputs
            input_names = [n.node_name for n in g.inputs]
            for arg_idx, arg in enumerate(args):
                category = arg_idx_to_input_output_none.get(arg_idx, None)
                if arg in input_names or category == 'in_ptr':
                    cur_node_inputs.append(arg)
                elif category == 'out_ptr':
                    cur_node_outputs.append(arg)

            cur_kernel_num_calls = kernel_name_to_num_calls[kernel_name]
            kernel_name_to_num_calls[kernel_name] += 1
            cur_kernel_node_name = f'{kernel_name} {cur_kernel_num_calls}'

            kernel_type = kernel_name_to_kernel_type[kernel_name]
            g.nodes[cur_kernel_node_name] = Node(
                cur_kernel_node_name,
                cur_node_inputs,
                metadata={'kernel_type': kernel_type},
                node_type='func',
            )
            for cur_node_output in cur_node_outputs:
                metadata = {'hint': node_name_to_hint[cur_node_output]}
                g.nodes[cur_node_output] = Node(cur_node_output, [cur_kernel_node_name], metadata=metadata, node_type='args')

            continue


        # extern_kernels._scaled_mm(buf6, buf7, buf8, buf5, out_dtype=torch.bfloat16, use_fast_accum=True, out=buf9)
        # TODO: other extern_kernels may have a different signature?
        extern_kernels_re = re.compile('[ ]+(extern_kernels.*)\((.*)\)')
        res = extern_kernels_re.match(line)
        if res:
            kernel_name = res.group(1)
            args = res.group(2).split(', ')
            input_names = [n.node_name for n in g.inputs]
            input_args = [a for a in args if a.startswith('buf') or a in input_names]
            output_arg = None
            for arg in args:
                if arg.startswith('out='):
                    output_arg = arg.replace('out=', '')

            cur_kernel_num_calls = kernel_name_to_num_calls[kernel_name]
            kernel_name_to_num_calls[kernel_name] += 1
            cur_kernel_node_name = f'{kernel_name} {cur_kernel_num_calls}'

            g.nodes[cur_kernel_node_name] = Node(cur_kernel_node_name, input_args, {}, node_type='func')
            metadata = {'hint': node_name_to_hint[output_arg]}
            g.nodes[output_arg] = Node(output_arg, [cur_kernel_node_name], metadata=metadata, node_type='args')

        # parse the return values
        # return (buf9, buf5, reinterpret_tensor(buf10, (8192, 4096), (1, 8192), 0), reinterpret_tensor(buf13, (2048, 4096), (1, 2048), 0), buf14, )
        #
        # To avoid matching other `return` statements, we only match the first return after `def call(args)`
        if not has_matched_final_return:
            final_return_re = re.compile('[ ]+return \((.*)\)')
            res = final_return_re.match(line)
            if res:
                has_matched_final_return = True
                output_args_tmp = res.group(1).split(', ')
                output_args = []
                for a in output_args_tmp:
                    if a.startswith('buf'):
                        output_args.append(a)
                    elif a.startswith('reinterpret_tensor('):
                        output_args.append(a.replace('reinterpret_tensor(', ''))
                g.outputs = output_args

    assert len(g.inputs) > 0
    assert len(g.nodes) > 0
    assert len(g.outputs) > 0
    return g


def create_html_summary(output_dir, graph_titles):
    """
    Creates a single html page which displays all the generated artifacts in `output_dir`
    """
    print('creating html summary')

    html_filename = os.path.join(output_dir, 'summary.html')

    template = Template("""
<html>
    <head>
        <title>{{ title }}</title>
    </head>
    <body>
        {% for row in titles_and_svgs %}
            <h1>{{ row[0] }} </h1>
            <object data="{{ row[1] }}" type="image/svg+xml"></object>
        {% endfor %}
    </body>
</html>
    """)

    titles_and_svgs = [(t, f'{t}.svg') for t in graph_titles]

    html = template.render(
        title=f"Graph summary for {output_dir}",
        titles_and_svgs=titles_and_svgs,
    )

    with open(html_filename, 'w') as f:
        f.write(html)


def shorten_func_name(s):
    """
    torch.ops.aten.abs.default -> abs
    torch.ops.prims.convert_element_type.default -> convert_element_type
    """
    s = s.replace('torch.ops.aten.', '')
    s = s.replace('torch.ops.prims.', '')
    s = s.replace('.default', '')
    return s

def create_diagram(
    g,
    output_dir,
    out_filename,
):
    print('creating diagram', output_dir, out_filename)
    dot = Digraph(comment='aot_joint_graph')
    dot.attr(label=out_filename)
    # 'b' is label at the bottom of subgraph, 't' is at top
    dot.attr(labelloc='t')
    g_idx = 0

    # Add the start nodes
    dot.node('input', 'input', shape='oval')
    for input_node in g.inputs:
        input_name_with_idx = f"{g_idx}_{input_node.node_name}"
        metadata_str = f"<font color='blue'>{input_node.metadata}</font>"
        node_comment = f"<{input_node.node_name}<br/>{metadata_str}>"

        dot.node(input_name_with_idx, node_comment, shape='rectangle', style='filled', fillcolor=TENSOR_NODE_COLOR)
        dot.edge('input', input_name_with_idx, '', color=INPUT_EDGE_COLOR)

    # Add the intermediate nodes
    for node_name, node in g.nodes.items():
        node_name_with_idx = f"{g_idx}_{node_name}"
        metadata_str = f"<font color='blue'>{node.metadata}</font>"

        # Add the node
        # use HTML syntax to make things easier to read
        if node.node_type == 'args':
            node_comment = f"<{node_name}<br/>{metadata_str}>"
        else:
            node_comment = f"<{shorten_func_name(node_name)}<br/>({node.args})<br/>{metadata_str}>"

        # using shapes to distinguish between functions and tensors leads to too-large
        # graph rendering, so use color instead
        fillcolor = TENSOR_NODE_COLOR
        if node.node_type == 'func':
            fillcolor = FUNC_NODE_COLOR

        # rectangle seems to be the most efficient shape in terms of the rendering
        # space it takes on the screen
        dot.node(node_name_with_idx, node_comment, shape='rectangle', color='black', style='filled', fillcolor=fillcolor)

        # Add edges to args
        input_names = [n.node_name for n in g.inputs]
        for arg_name in node.args:
            if arg_name in input_names or arg_name in g.nodes:
                arg_name_with_idx = f"{g_idx}_{arg_name}"
                dot.edge(arg_name_with_idx, node_name_with_idx, '')

    # Create output
    dot.node('output', 'output', shape='oval')
    for output_name in g.outputs:
        output_name_with_idx = f"{g_idx}_{output_name}"
        dot.edge(output_name_with_idx, 'output', '', color=INPUT_EDGE_COLOR)

    out_filename = os.path.join(output_dir, out_filename)
    dot.render(out_filename, format='svg', cleanup=True)


def run(
    fname: str = test_fname3,
    output_subdir: str = 'test',
):
    """
    Inputs:
    * `fname`: filename with logs
    # `output_subdir`: subdirectory of `outputs` where to store the output data

    Outputs output_subdir/summary.html with svg graphs of the text graphs found in `fname`.
    """

    # format:
    # {
    #   0: {
    #     'joint': ...,
    #     'forward': ...,
    #     'backward': ...,
    #   },
    #   ...
    # }
    aot_graph_id_to_graphs = defaultdict(dict)

    cur_captured_graph_id = None
    joint_start_idx, forward_start_idx, backward_start_idx, end_idx = None, None, None, None
    cur_aot_id, cur_fwd_bwd = None, None
    aot_triton_region_start_idx, aot_triton_region_end_idx = None, None

    # {
    #   '0': {
    #     'forward': [123, 126, None],
    #     'backward': [245, 256, None],
    #   },
    # }
    #
    # The `None` entries are for the parsed graphs, which will be filled in later
    graph_id_to_fwdbwd_to_triton_region_idxs = defaultdict(dict)

    # parse the aot_joint graph
    with open(fname, 'r') as f:
        lines = f.readlines()

        for idx, line in enumerate(lines):

            # Start of aot_graphs graph 0:
            #  ===== Joint graph 0 =====

            # TODO(later): extract the graph ID
            start_of_aot_joint_graph_re = re.compile(' ===== Joint graph ([\d]+) =====')
            res = start_of_aot_joint_graph_re.search(line)
            if res:
                joint_start_idx = idx
                cur_captured_graph_id = res.group(1)
                continue

            # Start of forward graph 0:
            # ===== Forward graph 0 =====
            start_of_aot_forward_graph_re = re.compile(' ===== Forward graph ([\d]+) =====')
            res = start_of_aot_forward_graph_re.search(line)
            if res:
                forward_start_idx = idx
                cur_captured_graph_id = res.group(1)
                continue

            # Start of backward graph 0:
            # ===== Forward graph 0 =====
            start_of_aot_backward_graph_re = re.compile(' ===== Backward graph ([\d]+) =====')
            res = start_of_aot_backward_graph_re.search(line)
            if res:
                backward_start_idx = idx
                cur_captured_graph_id = res.group(1)
                continue

            # End of aot_joint_graphs graph:
            # return pytree.tree_unflatten([view_1, permute_8, view_3], self._out_spec)
            end_of_aot_graph_re = re.compile('.*return pytree.tree_unflatten')
            if end_of_aot_graph_re.match(line):
                end_idx = idx
                aot_graph_id_to_graphs[cur_captured_graph_id]['joint'] = joint_start_idx, end_idx
                cur_captured_graph_id = None
                joint_start_idx, forward_start_idx, backward_start_idx, end_idx = None, None, None, None
                continue

            # End of aot_forward or aot_backward or triton kernel wrappers
            end_of_aot_forward_graph_re = re.compile('.*return \(.*\)')
            if end_of_aot_forward_graph_re.match(line):
                end_idx = idx
                if forward_start_idx is not None:
                    aot_graph_id_to_graphs[cur_captured_graph_id]['forward'] = forward_start_idx, end_idx
                elif backward_start_idx is not None:
                    aot_graph_id_to_graphs[cur_captured_graph_id]['backward'] = backward_start_idx, end_idx
                else:
                    # triton kernel wrappers, for now we skip these
                    pass
                cur_captured_graph_id = None
                joint_start_idx, forward_start_idx, backward_start_idx, end_idx = None, None, None, None
                continue

            # triton kernel to aot_id fwd/bwd mapping
            # # AOT ID: ['0_forward']
            triton_to_aot_id = re.compile('# AOT ID: \[(.*)\]')
            res = triton_to_aot_id.match(line)
            if res:
                contents = res.group(1).replace("'", "").split('_')
                aot_id, fwd_or_bwd = contents
                cur_aot_id = aot_id
                cur_fwd_bwd = fwd_or_bwd
                aot_triton_region_start_idx = idx

            # End of triton kernel file corresponding to AOT ID
            # torch._inductor.codecache.__output_code:Output code written to: /tmp/torchinductor_vasiliy/tmpspdynggy/at/cathtx22pafsaeak2pf2g55j64arrziidjtpayap4wm6hlgbjx5m.py
            end_of_triton_kernel_file = re.compile('.*Output code written to.*')
            if end_of_triton_kernel_file.match(line) and aot_triton_region_start_idx is not None:
                aot_triton_region_end_idx = idx
                graph_id_to_fwdbwd_to_triton_region_idxs[cur_aot_id][cur_fwd_bwd] = [aot_triton_region_start_idx, aot_triton_region_end_idx, None]
                aot_triton_region_start_idx, aot_triton_region_end_idx = None, None

    joint_graph, forward_graph, backward_graph = None, None, None

    # parse the graphs, if present
    # TODO handle multiple joint graphs per log file
    if 'joint' in aot_graph_id_to_graphs['0']:
        start_idx, end_idx = aot_graph_id_to_graphs['0']['joint']
        joint_graph = parse_graph(lines, start_idx, end_idx)
    if 'forward' in aot_graph_id_to_graphs['0']:
        start_idx, end_idx = aot_graph_id_to_graphs['0']['forward']
        forward_graph = parse_graph(lines, start_idx, end_idx)
    if 'backward' in aot_graph_id_to_graphs['0']:
        start_idx, end_idx = aot_graph_id_to_graphs['0']['backward']
        backward_graph = parse_graph(lines, start_idx, end_idx)

    # graph of a triton kernel file, with nodes being the buffers passed around
    # in the triton kernel wrapper
    triton_region_graph_forward, triton_region_graph_backward = None, None
    if 'forward' in graph_id_to_fwdbwd_to_triton_region_idxs['0']:
        start_idx, end_idx, _ = graph_id_to_fwdbwd_to_triton_region_idxs['0']['forward']
        triton_region_graph_forward = parse_triton_region_graph(lines, start_idx, end_idx)
    if 'backward' in graph_id_to_fwdbwd_to_triton_region_idxs['0']:
        start_idx, end_idx, _ = graph_id_to_fwdbwd_to_triton_region_idxs['0']['backward']
        triton_region_graph_backward = parse_triton_region_graph(lines, start_idx, end_idx)

    output_dir = os.path.join('outputs', output_subdir)

    graph_titles = []

    if joint_graph is not None:
        create_diagram(
            joint_graph,
            output_dir,
            'joint',
        )
        graph_titles.append('joint')
    if forward_graph is not None:
        create_diagram(
            forward_graph,
            output_dir,
            'forward',
        )
        graph_titles.append('forward')
    if backward_graph is not None:
        create_diagram(
            backward_graph,
            output_dir,
            'backward',
        )
        graph_titles.append('backward')

    if triton_region_graph_forward is not None:
        create_diagram(
            triton_region_graph_forward,
            output_dir,
            'triton_region_forward',
        )
        graph_titles.append('triton_region_forward')
    if triton_region_graph_backward is not None:
        create_diagram(
            triton_region_graph_backward,
            output_dir,
            'triton_region_backward',
        )
        graph_titles.append('triton_region_backward')

    create_html_summary(output_dir, graph_titles)

if __name__ == '__main__':
    fire.Fire(run)
