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
## example output

Note that below is a screenshot of partial output.  In the actual output you can zoom, scroll and text search.

### AOT joint

<img width="2224" alt="Image" src="https://github.com/user-attachments/assets/701efd44-2ce9-4991-8696-fad7bb531928" />

### AOT forward and backward

<img width="1535" alt="Image" src="https://github.com/user-attachments/assets/44c71edc-0e97-46c1-9d63-1eba852cb111" />

### triton forward and backward

<img width="1631" alt="Image" src="https://github.com/user-attachments/assets/c5e628f6-1ab6-4389-96df-63be21bdd132" />
