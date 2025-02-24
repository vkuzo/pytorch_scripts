# graph_viz

This is a script which converts the output of `TORCH_LOGS_FORMAT=short TORCH_LOGS=aot_graphs,output_code your_torch_compile_script.py` to an html visualization of the joint AOT graph, partitioned AOT graphs, and triton kernels.  This is useful when reasoning about the system.

At some point it would be nice to fold this into tlparse.

## usage

```bash
# first, create a text file with your logs
TORCH_LOGS_FORMAT=short TORCH_LOGS=aot_graphs,output_code your_torch_compile_script.py > your_logs.txt 2>&1

# then, create the html summary
python main.py your_logs.txt ~/local/tmp/your_output_dir
```
