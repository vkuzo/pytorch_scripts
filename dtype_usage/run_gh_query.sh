#!/bin/bash

# query the github CLI for usage of PyTorch lowp dtypes

# e8m0
# get raw results
with-proxy gh search code "torch.float8_e8m0fnu" --json path,repository,sha,textMatches,url --limit 1000 > results_e8m0.json
# parse them
python parse_dtype_results.py --in_file results_e8m0.json

# get raw results
with-proxy gh search code "torch.float4_e2m1fn_x2" --json path,repository,sha,textMatches,url --limit 1000 > results_fp4.json
# parse them
python parse_dtype_results.py --in_file results_fp4.json
