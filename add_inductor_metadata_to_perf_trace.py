import json
import re

import fire


def run(
    perf_trace_file: str,
    torch_logs_file: str,
    modified_perf_trace_file: str,
):
    """
    Input 1: a perf trace generated by using `torch.profiler.profile` inside of 
      some_program.py, and containing torch.compile + inductor kernels
    Input 2: a text file with the output of 
      TORCH_LOGS="output_code" python some_program.py
    Input 3: filename for the modified perf trace

    This script does the following for each triton kernel in input 1:
    - navigate to the kernel information in the logs from input 2
    - copy over the kernel metadata (aten graph, triton code, etc) to the JSON 
      in input 1

    The end result is that Input 1 is modified so that the kernel metadata is 
    directly visible in tools like chrome://tracing and perfetto.
    """

    external_id_to_cpu_ops = dict()
    external_id_to_kernels = dict()

    # open the torch logs file
    torch_logs_str = None
    with open(torch_logs_file, 'r') as f:
        torch_logs_str = f.readlines()

    # strip away the torch_logs prefix
    torch_logs_only = []
    for line in torch_logs_str:
        line = line.replace('\n', '')
        match = re.match('.* \[__output_code\] (.*)', line)
        if match:
            torch_logs_only.append(match.group(1))

    # Find the locations of the kernel metadata in the logs.
    # metadata format, haven't been extensively tested so may be brittle:
    #
    #   ...[__output_code]: # kernel_path: /tmp/torchinductor_...
    #   ...[__output_code]: ...
    #   ...[__output_code]: triton_red_fused_LayerNorm_3 = async_compile.triton('triton_', '''
    #   ...[__output_code]: ...
    #   ...[__output_code]: ''', device_str='cuda')
    #
    # We look for the first and last line and save everything in between
    name_to_start_end = {}
    cur_start, cur_end, cur_name = None, None, None
    for line_num, line in enumerate(torch_logs_only):
        match_start = re.match('\# kernel path: .*', line)
        if match_start:
            cur_start = line_num

        # triton_red_fused_LayerNorm_3 = async_compile.triton('triton_', '''
        match_name = re.match("([\w_]+) = async_compile.*", line)
        if match_name:
            cur_name = match_name.group(1)

        match_end = re.match("''', device_str='cuda'\)", line)
        if match_end:
            cur_end = line_num

            # populate the mapping and reset
            name_to_start_end[cur_name] = (cur_start, cur_end)
            cur_start, cur_end, cur_name = None, None, None

    # ensure matching didn't have loose ends
    assert cur_start is None and cur_end is None and cur_name is None

    # Now, go through the JSON file and populate the extra metadata
    # Format of the relevant parts of the perf trace JSON:
    # {
    #   ...
    #   // CPU ops, with names matchable to triton kernels from inductor output code
    #   {
    #     # "cat": "cpu_op", 
    #     # "name": "triton_red_fused_LayerNorm_abs_max_0",
    #     # "args": {"External id": 1030, ...},
    #     # ...
    #   },
    #   // Inductor kernels, with wall time
    #   {
    #     # "cat": "kernel", 
    #     # "name": "triton_",  // we don't depend on this name, including for context
    #     # "args": {"External id": 1030, ...},
    #     # "ts": 4275686082015.124, // start time
    #     # "dur": 208.640,  // duration
    #     # ...
    #   },
    # }
    #
    # We can't assume any ordering, so we do two passes:
    # 1. Find mapping of cpu_op to external_id
    # 2. Using 1, add the metadata to triton kernels

    # open the perf trace json
    with open(perf_trace_file, 'r') as f:
        perf_trace = json.load(f)

    # find mapping of cpu_op to external_id
    external_id_to_cpu_op = dict()
    for record in perf_trace['traceEvents']:
        # print(record)
        is_cpu_op = record.get('cat') == 'cpu_op'
        if is_cpu_op:
            external_id_to_cpu_op[record['args']['External id']] = record['name']

    # add the metadata to triton kernels
    for record in perf_trace['traceEvents']:
        is_triton_kernel = record.get('cat') == 'kernel' and 'triton' in record.get('name', '')
        if not is_triton_kernel:
            continue
        op_name = external_id_to_cpu_op.get(record['args']['External id'])
        if op_name is None:
            continue
        start, end = name_to_start_end[op_name]
        triton_code = torch_logs_only[start:end+1]
        s = ''
        for line in triton_code:
            s += f'{line}\n'
        record['args']['triton_code'] = s

    # write the modified file
    # out_file = perf_trace_file.replace('.json', '') + '_with_metadata.json'
    with open(modified_perf_trace_file, 'w') as f:
        json.dump(perf_trace, f)

    print('done')

if __name__ == '__main__':
    fire.Fire(run)