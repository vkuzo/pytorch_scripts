from collections import defaultdict
import fire
import re

from typing import *

from graphviz import Digraph

# aot_joint_graph only (filename is misnamed)
fname = 'input_aot_graphs.txt'

# aot_joint_graph and aot_graphs
fname = 'input_aot_joint_graph_and_aot_graphs.txt'

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
        'inputs': None,
        'nodes': {},
        'outputs': None,
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

            # End of aot_forward or aot_backward
            end_of_aot_forward_graph_re = re.compile('.*return \(.*\)')
            if end_of_aot_forward_graph_re.match(line):
                end_idx = idx
                if forward_start_idx is not None:
                    aot_graph_id_to_graphs[cur_captured_graph_id]['forward'] = forward_start_idx, end_idx
                else:
                    assert backward_start_idx is not None
                    aot_graph_id_to_graphs[cur_captured_graph_id]['backward'] = backward_start_idx, end_idx
                cur_captured_graph_id = None
                joint_start_idx, forward_start_idx, backward_start_idx, end_idx = None, None, None, None
                continue

    print(aot_graph_id_to_graphs)

    # aot_graph = parse_graph(lines, start_idx, end_idx)

    for name in ('joint', 'forward', 'backward'):
        print(name)
        if name not in aot_graph_id_to_graphs['0']:
            continue
        start_idx, end_idx = aot_graph_id_to_graphs['0'][name]
        graph = parse_graph(lines, start_idx, end_idx)
        create_aot_joint_graph_diagram(graph, name)

def shorten_func_name(s):
    """
    torch.ops.aten.abs.default -> abs
    torch.ops.prims.convert_element_type.default -> convert_element_type
    """
    s = s.replace('torch.ops.aten.', '')
    s = s.replace('torch.ops.prims.', '')
    s = s.replace('.default', '')
    return s

def create_aot_joint_graph_diagram(g, out_filename):
    print('here')

    dot = Digraph(comment='aot_joint_graph')
    dot.attr(fontsize='9')

    # Add the start nodes
    dot.node('input', 'input', shape='oval')
    for input_name in g['inputs']:
        dot.node(input_name, input_name, shape='oval')
        dot.edge('input', input_name, '')

    # Add the intermediate nodes
    for node_name, (func_name, args) in g['nodes'].items():
        # Add the node
        # use HTML syntax to make things easier to read
        node_comment = f"<{shorten_func_name(func_name)}({args})<br/><br/>{node_name}>"
        dot.node(node_name, node_comment, shape='rectangle')

        # Add edges to args
        for arg_name in args:
            if arg_name in g['inputs'] or arg_name in g['nodes']:
                dot.edge(arg_name, node_name, '')

    # Create output
    dot.node('output', 'output', shape='oval')
    for output_name in g['outputs']:
        dot.edge(output_name, 'output', '')

    dot.render(out_filename, format='svg', cleanup=True)


# from Claude
def create_debug_workflow_diagram():
    # Create a new directed graph
    dot = Digraph(comment='Workflow Diagram')
    dot.attr(rankdir='LR')  # Left to right layout

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
