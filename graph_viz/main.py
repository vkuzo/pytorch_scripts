from collections import defaultdict
from dataclasses import dataclass, field
import fire
import re
import os

from typing import *

from graphviz import Digraph

# aot_joint_graph only (filename is misnamed)
test_fname1 = 'test_inputs/input_aot_graphs.txt'

# aot_joint_graph and aot_graphs
test_fname2 = 'test_inputs/input_aot_joint_graph_and_aot_graphs.txt'

# also output_code
test_fname3 = 'test_inputs/input_aot_joint_graph_aot_graphs_output_code.txt'

@dataclass
class Node:
    node_name: str
    func: str
    args: List[str]
    metadata: str

# format:
#   inputs: ['input0', ...],
#   nodes: {
#     'node_n': ['func', ['arg_0', 'arg_1']],
#     ...
#   },
#   outputs: ['output0, ...]
#   triton_kernel_str: """
# @triton_heuristics.reduction(...
#   ...
# ''', device_str='cuda')
#   """
#   triton_kernel_type: reduction,
#   triton_kernel_name: 'triton_red_fused_abs_max_0',
@dataclass
class ParsedGraph:
    inputs: List[str] = field(default_factory=lambda: [])
    nodes: Dict[str, Node] = field(default_factory=lambda: {})
    outputs: List[str] = field(default_factory=lambda: [])
    triton_kernel_str: Optional[str] = None
    # pointwise/reduction/persistent_reduction
    triton_kernel_type: Optional[str] = None
    triton_kernel_name: Optional[str] = None


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

    for line in list_of_lines[start_idx:end_idx+1]:

        # Comment
        # # File: /data/users/vasiliy/ao/torchao/float8/float8_linear.py:335 in forward, code: weight_maybe_fp8_t = self.weight.t()
        # for now, skip matching

        # Blank line
        # for now, skip matching

        # specific to joint graph
        # set up primals and tangents
        # primals_1, primals_2, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
        primals_tangents_re = re.compile('[ ]+(.*), = fx_pytree.tree_flatten_spec.*')
        res = primals_tangents_re.search(line)
        if res:
            inputs_list = [s.strip() for s in res.group(1).split(',')]
            g.inputs = inputs_list
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
            g.inputs = inputs_list

        # function call
        # abs_1: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.abs.default(primals_2)
        function_call_re = re.compile('[ ]+(\w+): ".* = ([\w\.]+)\((.*)\)')
        res = function_call_re.search(line)
        if res:
            var_name, func_name, args_str = res.group(1), res.group(2), res.group(3)
            args_list = [s.strip() for s in args_str.split(',')]
            g.nodes[var_name] = Node(var_name, func_name, args_list, None)
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
    assert g.inputs is not None
    assert len(g.nodes) > 0
    assert g.outputs is not None

    return g

def parse_triton_kernel_graph(
    list_of_lines: List[str],
    start_idx: int,
    end_idx: int,
):
    g = ParsedGraph()

    for line in list_of_lines[start_idx:end_idx+1]:
        # %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 448.0), kwargs = {})
        # TODO also parse kwargs
        triton_graph_re = re.compile('#   (\%.*)+ : .*args = \((.*)\), kwargs.*')
        res = triton_graph_re.search(line)
        if res:
            target_node = res.group(1).replace('%', '')
            args = res.group(2).split(', ')
            args = [s.replace('%', '').replace(',', '') for s in args]

            g.nodes[target_node] = Node(target_node, '', args, None)
            # TODO future parse the func as well

    # add the triton kernel text to the graph as metadata, so we can display it later
    # also add the kernel type and name
    triton_kernel_start_re = re.compile('triton_heuristics.*')
    triton_kernel_end_re = re.compile("''', device_str='cuda'.*")
    triton_kernel_name_re = re.compile('(.*) = async_compile.*')
    triton_kernel_start_idx, triton_kernel_end_idx = None, None
    cur_idx = end_idx
    while True:
        line = list_of_lines[cur_idx]
        if triton_kernel_start_re.search(line):
            triton_kernel_start_idx = cur_idx
            g.triton_kernel_type = line.replace('@triton_heuristics.', '').replace('(', '')
        elif triton_kernel_end_re.search(line):
            triton_kernel_end_idx = cur_idx
            break
        cur_idx += 1

        res = triton_kernel_name_re.search(line)
        if res:
            g.triton_kernel_name = res.group(1)
    g.triton_kernel_str = ''.join(list_of_lines[triton_kernel_start_idx:triton_kernel_end_idx])

    # populate the inputs and outputs by:
    # 1. for each node n, count the number of parent and children nodes
    # 2. for each node n with 0 parents, add it to inputs
    # 3. for each node n with 0 children, add it to outputs
    # note that for joint/fwd/bwd graph we parse inputs/outputs from the graph
    # printout, but for triton kernels we only have the graph fragment so we
    # have to resort to this hack

    # B: '', [A]
    # C: '', [A, B]
    # D: '', [C]
    # traversing B: A add one child, B add one parent
    # traversing C: A and B add one child, C add one parent
    # traversing D: C add one child, D add one parent

    # TODO(future): make this also handle kwargs
    node_name_to_num_parents_children = defaultdict(lambda: [0, 0])
    for node_name, node in g.nodes.items():
        func = node.func
        args = node.args
        metadata = node.metadata
        for arg in args:
            # increment child counter
            node_name_to_num_parents_children[arg][1] += 1
        # increment parent counter
        node_name_to_num_parents_children[node_name][0] += len(args)

    inputs = []
    outputs = []
    starts_with_az = re.compile('^[a-zA-Z]+.*')
    for node_name, (num_parents, num_children) in node_name_to_num_parents_children.items():

        # Currently args can be variables (foo_1), numbers (1e12), torch constants (torch.bfloat16), etc
        # we don't know which one because this is all done with string parsing. For now,
        # do a hacky filter of vars only by restricting the first character to be a-zA-Z, and discarding
        # the things that look like `torch.`.  This is pretty brittle.
        if not(
            starts_with_az.search(node_name)
            and not node_name.startswith('torch')
            and not node_name == 'None'
        ):
            continue
        if num_parents == 0:
            inputs.append(node_name)
        elif num_children == 0:
            outputs.append(node_name)

    g.inputs = inputs
    g.outputs = outputs

    # verify captured information is valid
    assert len(g.nodes) > 0
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
    return (buf8, buf4, reinterpret_tensor(buf9, (4096, 4096), (1, 4096), 0), reinterpret_tensor(buf12, (4096, 4096), (1, 4096), 0), buf13, )

    """

    g = ParsedGraph()

    cur_empty_buffers_set = set()
    cur_populated_buffers_set = set()

    kernel_name_to_arg_idx_to_input_output_none = {}

    # triton kernels can be called multiple times, keep track of
    # number of calls so we can have unique nodes per kernel-call
    kernel_name_to_num_calls = defaultdict(int)

    for line in list_of_lines[start_idx:end_idx+1]:

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
            kernel_name_to_arg_idx_to_input_output_none[res.group(1)] = arg_idx_to_category

        # primals_1, primals_2 = args
        args_re = re.compile('[ ]+(.*) = args$')
        res = args_re.match(line)
        if res:
            args_str = res.group(1).split(', ')
            g.inputs = args_str
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
            for arg_idx, arg in enumerate(args):
                category = arg_idx_to_input_output_none.get(arg_idx, None)
                if arg in g.inputs or category == 'in_ptr':
                    cur_node_inputs.append(arg)
                elif category == 'out_ptr':
                    cur_node_outputs.append(arg)

            cur_kernel_num_calls = kernel_name_to_num_calls[kernel_name]
            kernel_name_to_num_calls[kernel_name] += 1
            cur_kernel_node_name = f'{kernel_name}__{cur_kernel_num_calls}'

            g.nodes[cur_kernel_node_name] = Node(cur_kernel_node_name, '', cur_node_inputs, '')
            for cur_node_output in cur_node_outputs:
                g.nodes[cur_node_output] = Node(cur_node_output, '', [cur_kernel_node_name], '')

            continue

    # TODO: populate graph outputs, will do later
    return g

def fname_to_graphs(
    fname: str,
    output_subdir: str,
):
    """
    Inputs:
    * `fname`: filename with logs
    # `output_subdir`: subdirectory of `outputs` where to store the output data

    Outputs:
    * aot_joint_graph
    * aot_graphs
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

    # parse the aot_joint graph
    with open(fname, 'r') as f:

        lines = f.readlines()

        cur_captured_graph_id = None
        joint_start_idx, forward_start_idx, backward_start_idx, end_idx = None, None, None, None

        # for individual triton kernels
        triton_start_idx, triton_end_idx = None, None

        # {
        #   '0': {
        #     'forward': [[123, 126, None], [128, 134, None], ...],
        #     'backward': [[245, 256, None], ...]
        #   },
        # }
        #
        # The `None` entries are for the parsed graphs, which will be filled in later
        graph_id_to_fwdbwd_to_triton_idxs_and_graphs = defaultdict(lambda: defaultdict(list))

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

            # Start of triton kernel graph fragment
            # Graph fragment:
            start_of_triton_graph_fragment = re.compile('.*Graph fragment.*')
            if start_of_triton_graph_fragment.match(line):
                triton_start_idx = idx

            # End of triton kernel graph fragment
            # triton_red_fused_abs_max_0 = async_compile.triton('triton_red_fused_abs_max_0', '''
            end_of_triton_graph_fragment = re.compile('.* = async_compile.triton.*')
            if end_of_triton_graph_fragment.match(line):
                triton_end_idx = idx

                graph_id_to_fwdbwd_to_triton_idxs_and_graphs[cur_aot_id][cur_fwd_bwd].append([triton_start_idx, triton_end_idx, None])

            # End of triton kernel file corresponding to AOT ID
            # torch._inductor.codecache.__output_code:Output code written to: /tmp/torchinductor_vasiliy/tmpspdynggy/at/cathtx22pafsaeak2pf2g55j64arrziidjtpayap4wm6hlgbjx5m.py
            end_of_triton_kernel_file = re.compile('.*Output code written to.*')
            if end_of_triton_kernel_file.match(line) and aot_triton_region_start_idx is not None:
                pass
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

    # graph of a triton kernel file, with nodes being the aten node names from
    # graph fragments
    if 'forward' in graph_id_to_fwdbwd_to_triton_idxs_and_graphs['0']:
        for entry in graph_id_to_fwdbwd_to_triton_idxs_and_graphs['0']['forward']:
            start_idx, end_idx, _ = entry
            triton_graph = parse_triton_kernel_graph(lines, start_idx, end_idx)
            entry[2] = triton_graph
    # TODO(future): also backward

    # graph of a triton kernel file, with nodes being the buffers passed around
    # in the triton kernel wrapper
    triton_region_graph = None
    if 'forward' in graph_id_to_fwdbwd_to_triton_region_idxs['0']:
        start_idx, end_idx, _ = graph_id_to_fwdbwd_to_triton_region_idxs['0']['forward']
        print(start_idx, end_idx)
        triton_region_graph = parse_triton_region_graph(lines, start_idx, end_idx)
        print(triton_region_graph)

    output_dir = os.path.join('outputs', output_subdir)

    if joint_graph is not None:
        create_diagram(
            [joint_graph],
            output_dir,
            'joint',
        )
    if forward_graph is not None:
        create_diagram(
            [forward_graph],
            output_dir,
            'forward',
            triton_idxs_and_graphs=graph_id_to_fwdbwd_to_triton_idxs_and_graphs['0']['forward']
        )
    if backward_graph is not None:
        create_diagram(
            [backward_graph],
            output_dir,
            'backward',
        )

    triton_forward_graphs = [g for _, __, g in graph_id_to_fwdbwd_to_triton_idxs_and_graphs['0']['forward']]
    create_diagram(
        triton_forward_graphs,
        output_dir,
        'triton_forward',
    )

    if triton_region_graph is not None:
        create_diagram(
            [triton_region_graph],
            output_dir,
            'triton_region_forward',
        )

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
    graphs,
    output_dir,
    out_filename,
    triton_idxs_and_graphs=None,
):
    print('creating diagram', output_dir, out_filename)
    dot = Digraph(comment='aot_joint_graph')
    dot.attr(label=out_filename)
    dot.attr(labelloc='t')

    for g_idx, g in enumerate(graphs):

        # note: subgraph name must start with cluster_ for graphviz to render a border
        with dot.subgraph(name=f"cluster_{str(g_idx)}") as c:

            # TODO(future): render `g.triton_kernel_str` if present
            if g.triton_kernel_name is not None:
                label = f"idx: {g_idx}\nkernel_name: {g.triton_kernel_name}\ntype:{g.triton_kernel_type}"
                c.attr(label=label)
            else:
                c.attr(label=f"label: {str(g_idx)}")

            # 'b' is label at the bottom of subgraph, 't' is at top
            c.attr(labelloc='b')

            # Add the start nodes
            if len(g.inputs):
                dot.node('input', 'input', shape='oval')
            for input_name in g.inputs:
                input_name_with_idx = f"{g_idx}_{input_name}"
                c.node(input_name_with_idx, input_name, shape='oval')
                dot.edge('input', input_name_with_idx, '', color='lightgray')

            # Add the intermediate nodes
            for node_name, node in g.nodes.items():
                func_name = node.func
                args = node.args
                metadata = node.metadata
                node_name_with_idx = f"{g_idx}_{node_name}"
                if metadata is None or metadata == '':
                    metadata='a'

                metadata_str = f"<font color='red'>{metadata}</font>"
                # Add the node
                # use HTML syntax to make things easier to read
                node_comment = f"<{shorten_func_name(func_name)}({args})<br/><br/>{node_name}<br/>{metadata_str}>"
                fillcolor='white'
                c.node(node_name_with_idx, node_comment, shape='rectangle', color='black')

                # Add edges to args
                for arg_name in args:
                    if arg_name in g.inputs or arg_name in g.nodes:
                        arg_name_with_idx = f"{g_idx}_{arg_name}"
                        dot.edge(arg_name_with_idx, node_name_with_idx, '')

            # Create output
            if len(g.outputs):
                dot.node('output', 'output', shape='oval')
            for output_name in g.outputs:
                output_name_with_idx = f"{g_idx}_{output_name}"
                dot.edge(output_name_with_idx, 'output', '', color='lightgray')

    out_filename = os.path.join(output_dir, out_filename)
    dot.render(out_filename, format='svg', cleanup=True)


# from Claude
def create_debug_workflow_diagram():
    # Create a new directed graph
    dot = Digraph(comment='Workflow Diagram')
    dot.attr(rankdir='TB')  # Left to right layout

    # Add nodes
    dot.node('A', 'Start', shape='oval')
    dot.node('B', 'Process Data', shape='box')
    dot.node('C', 'Decision', shape='diamond')
    dot.node('D', 'Success', shape='box')
    dot.node('E', 'Error', shape='box')

    # Add edges
    dot.edge('A', 'B', 'begin')
    dot.edge('B', 'C', 'evaluate')
    dot.edge('C', 'D', 'yes')
    dot.edge('C', 'E', 'no')

    # Add subgraph for error handling
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Error Handling')
        c.node('E1', 'Log Error')
        c.node('E2', 'Notify Admin')
        dot.edge('E', 'E1')
        dot.edge('E1', 'E2')

    # Save and render
    dot.render('workflow', format='svg', cleanup=True)

def run(
    input_fname: str = test_fname3,
    output_subdir: str = 'test',
):
    fname_to_graphs(input_fname, output_subdir)
    print('done')


if __name__ == '__main__':
    fire.Fire(run)
    # create_debug_workflow_diagram()
