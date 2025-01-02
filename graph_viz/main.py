from collections import defaultdict
import fire
import re

from typing import *

from graphviz import Digraph

# aot_joint_graph only (filename is misnamed)
fname = 'input_aot_graphs.txt'

# aot_joint_graph and aot_graphs
fname = 'input_aot_joint_graph_and_aot_graphs.txt'

# also output_code
fname = 'input_aot_joint_graph_aot_graphs_output_code.txt'

def parse_graph(
    list_of_lines: List[str],
    start_idx: int,
    end_idx: int,
):
    """
    Input: a list of lines from a file, and a start and end index of the region to parse into a graph
    Output: a parsed graph
    """

    # format:
    # {
    #   'inputs': ['input0', ...],
    #   'nodes': {
    #     'node_n': ['func', ['arg_0', 'arg_1']],
    #     ...
    #   },
    #   'outputs': ['output0, ...]
    # }
    g = {
        'inputs': [],
        'nodes': {},
        'outputs': [],
    }

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
            g['inputs'] = inputs_list
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
            g['inputs'] = inputs_list

        # function call
        # abs_1: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.abs.default(primals_2)
        function_call_re = re.compile('[ ]+(\w+): ".* = ([\w\.]+)\((.*)\)')
        res = function_call_re.search(line)
        if res:
            var_name, func_name, args_str = res.group(1), res.group(2), res.group(3)
            args_list = [s.strip() for s in args_str.split(',')]
            g['nodes'][var_name] = [func_name, args_list]
            continue

        # return statement for aot_joint_graph
        # return pytree.tree_unflatten([view_1, permute_8, view_3], self._out_spec)
        return_re = re.compile('return pytree.tree_unflatten\(\[(.*)\].*\)')
        res = return_re.search(line)
        if res:
            tokens_str = res.group(1)
            tokens_list = [s.strip() for s in tokens_str.split(',')]
            g['outputs'] = tokens_list
            continue

        # return statement for aot_graphs
        return_re = re.compile('return \((.*)\)')
        res = return_re.search(line)
        if res:
            tokens_str = res.group(1)
            tokens_list = [s.strip() for s in tokens_str.split(',')]
            g['outputs'] = tokens_list
            continue

    # verify captured information is valid
    assert g['inputs'] is not None
    assert len(g['nodes']) > 0
    assert g['outputs'] is not None

    return g

def parse_triton_graph(
    list_of_lines: List[str],
    start_idx: int,
    end_idx: int,
):
    # format:
    # {
    #   'inputs': ['input0', ...],
    #   'nodes': {
    #     'node_n': ['func', ['arg_0', 'arg_1']],
    #     ...
    #   },
    #   'outputs': ['output0, ...]
    # }
    g = {
        'inputs': [],
        'nodes': {},
        'outputs': [],
    }

    for line in list_of_lines[start_idx:end_idx+1]:
        # %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 448.0), kwargs = {})
        # TODO also parse kwargs
        triton_graph_re = re.compile('#   (\%.*)+ : .*args = \((.*)\), kwargs.*')
        res = triton_graph_re.search(line)
        if res:
            target_node = res.group(1).replace('%', '')
            args = res.group(2).split(', ')
            args = [s.replace('%', '').replace(',', '') for s in args]

            g['nodes'][target_node] = ['', args]
            # TODO future parse the func as well
            # TODO(future): inputs and outputs

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
    for node_name, (func, args) in g['nodes'].items():
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

    g['inputs'] = inputs
    g['outputs'] = outputs

    # verify captured information is valid
    assert len(g['nodes']) > 0
    return g

def fname_to_graphs(
    fname: str,
):
    """
    Inputs:
    * `fname`: filename with logs

    Outputs:
    * aot_joint_graph
    * aot_graphs
    """

    # format:
    # {
    #   'inputs': ['input0', ...],
    #   'nodes': {
    #     'node_n': ['func', ['arg_0', 'arg_1']],
    #     ...
    #   },
    #   'outputs': ['output0, ...]
    # }
    aot_graph = {
        'inputs': None,
        'nodes': {},
        'outputs': None,
    }

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

            # TODO(next): save the aot_id and fwd/bwd to triton graph fragments and maybe kernels,
            # and display correspondence

            # triton kernel to aot_id fwd/bwd mapping
            # # AOT ID: ['0_forward']
            triton_to_aot_id = re.compile('# AOT ID: \[(.*)\]')
            res = triton_to_aot_id.match(line)
            if res:
                contents = res.group(1).replace("'", "").split('_')
                aot_id, fwd_or_bwd = contents
                cur_aot_id = aot_id
                cur_fwd_bwd = fwd_or_bwd

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

    # triton kernels
    if 'forward' in graph_id_to_fwdbwd_to_triton_idxs_and_graphs['0']:
        for entry in graph_id_to_fwdbwd_to_triton_idxs_and_graphs['0']['forward']:
            start_idx, end_idx, _ = entry
            triton_graph = parse_triton_graph(lines, start_idx, end_idx)
            entry[2] = triton_graph

    if joint_graph is not None:
        create_diagram(
            [joint_graph],
            'joint',
        )
    if forward_graph is not None:
        create_diagram(
            [forward_graph],
            'forward',
            triton_idxs_and_graphs=graph_id_to_fwdbwd_to_triton_idxs_and_graphs['0']['forward']
        )
    if backward_graph is not None:
        create_diagram(
            [backward_graph],
            'backward',
        )

    triton_forward_graphs = [g for _, __, g in graph_id_to_fwdbwd_to_triton_idxs_and_graphs['0']['forward']]
    create_diagram(
        triton_forward_graphs,
        'triton_forward',
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
    out_filename,
    triton_idxs_and_graphs=None,
):
    print('create_diagram', out_filename)
    print(triton_idxs_and_graphs)

    dot = Digraph(comment='aot_joint_graph')
    # dot.attr(fontsize='9')
    dot.attr(label=out_filename)
    dot.attr(labelloc='t')

    for g_idx, g in enumerate(graphs):

        # note: subgraph name must start with cluster_ for graphviz to render a border
        with dot.subgraph(name=f"cluster_{str(g_idx)}") as c:
            c.attr(label=f"label: {str(g_idx)}\nasdfasdf\nasdfasd")

            # 'b' is label at the bottom of subgraph, 't' is at top
            c.attr(labelloc='b')

            # Add the start nodes
            if len(g['inputs']):
                dot.node('input', 'input', shape='oval')
            for input_name in g['inputs']:
                input_name_with_idx = f"{g_idx}_{input_name}"
                c.node(input_name_with_idx, input_name, shape='oval')
                dot.edge('input', input_name_with_idx, '', color='lightgray')

            # Add the intermediate nodes
            for node_name, (func_name, args) in g['nodes'].items():
                node_name_with_idx = f"{g_idx}_{node_name}"

                category_str = f"<font color='red'>test</font>"
                # Add the node
                # use HTML syntax to make things easier to read
                node_comment = f"<{shorten_func_name(func_name)}({args})<br/><br/>{node_name}<br/>{category_str}>"
                fillcolor='white'
                c.node(node_name_with_idx, node_comment, shape='rectangle', color='black')

                # Add edges to args
                for arg_name in args:
                    if arg_name in g['inputs'] or arg_name in g['nodes']:
                        arg_name_with_idx = f"{g_idx}_{arg_name}"
                        dot.edge(arg_name_with_idx, node_name_with_idx, '')

            # Create output
            if len(g['outputs']):
                dot.node('output', 'output', shape='oval')
            for output_name in g['outputs']:
                output_name_with_idx = f"{g_idx}_{output_name}"
                dot.edge(output_name_with_idx, 'output', '', color='lightgray')

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

def run():
    fname_to_graphs(fname)

    print('done')


if __name__ == '__main__':
    fire.Fire(run)
    # create_debug_workflow_diagram()
