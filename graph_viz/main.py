import fire
import re

from graphviz import Digraph

fname = 'input_aot_graphs.txt'

def fname_to_graphs(
    fname: str,
):
    """
    Inputs:
    * `fname`: filename with logs

    Outputs:
    * aot_graph
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

    # parse the aot_joint graph
    with open(fname, 'r') as f:

        for line in f:

            # Start of aot_graphs graph 0:
            #  ===== Joint graph 0 =====

            # TODO(later): extract the graph ID
            start_of_aot_graph_re = re.compile(' ===== Joint graph ([\d]+) =====')
            res = start_of_aot_graph_re.search(line)
            if res:
                print('START')
                continue

            # Comment
            # # File: /data/users/vasiliy/ao/torchao/float8/float8_linear.py:335 in forward, code: weight_maybe_fp8_t = self.weight.t()
            # for now, skip matching

            # Blank line
            # for now, skip matching

            # set up primals and tangents
            # primals_1, primals_2, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
            primals_tangents_re = re.compile('[ ]+(.*), = fx_pytree.tree_flatten_spec.*')
            res = primals_tangents_re.search(line)
            if res:
                inputs_list = [s.strip() for s in res.group(1).split(',')]
                aot_graph['inputs'] = inputs_list
                continue

            # function call
            # abs_1: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.abs.default(primals_2)
            function_call_re = re.compile('[ ]+(\w+): ".* = ([\w\.]+)\((.*)\)')
            res = function_call_re.search(line)
            if res:
                var_name, func_name, args_str = res.group(1), res.group(2), res.group(3)
                args_list = [s.strip() for s in args_str.split(',')]
                aot_graph['nodes'][var_name] = [func_name, args_list]
                continue

            # return statement
            # return pytree.tree_unflatten([view_1, permute_8, view_3], self._out_spec)
            return_re = re.compile('return pytree.tree_unflatten\(\[(.*)\].*\)')
            res = return_re.search(line)
            if res:
                tokens_str = res.group(1)
                tokens_list = [s.strip() for s in tokens_str.split(',')]
                aot_graph['outputs'] = tokens_list
                continue

            # End of aot_graphs graph:
            # return pytree.tree_unflatten([view_1, permute_8, view_3], self._out_spec)
            end_of_aot_graph_re = re.compile('.*return pytree.tree_unflatten')
            if end_of_aot_graph_re.match(line):
                print('END')
                continue

    # verify captured information is valid
    assert aot_graph['inputs'] is not None
    assert len(aot_graph['nodes']) > 0
    assert aot_graph['outputs'] is not None

    print(aot_graph['inputs'])
    for k, v in aot_graph['nodes'].items():
        print(k, v)

        output_node_name = k
        func_name, args = v

        # look for matches in the dict
        for arg in args:
            if arg in aot_graph['nodes']:
                # match
                pass

    print(aot_graph['outputs'])

    create_aot_joint_graph_diagram(aot_graph, 'aot_joint_graph')

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
        dot.edge(input_name, 'input', '')

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
