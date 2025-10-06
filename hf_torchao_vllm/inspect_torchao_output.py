# inspects the output of model created with torchao
# via the `torchao_hf_script.py` script

import json
import os
import pathlib
import torch
import torchao  # this is needed to run torch.serialization.add_safe_globals([torchao.quantization.Float8Tensor])
import fire

from utils import inspect_model_state_dict

# not sure why I still need this
torch.serialization.add_safe_globals([getattr])

def run(dir_name: str = 'data/torchao/fp8-opt-125m'):
    json_config_name = f'{dir_name}/config.json'

    # inspect the config
    with open(json_config_name, 'r') as f:
        data = json.load(f)
        print(json.dumps(data, indent=2))

    # inspect the data
    # 
    # if there is a single chunk, the state dict is named `pytorch_model.bin`
    #
    # if there are multiple chunks, the state dict is spread across multiple files:
    #
    #   pytorch_model-00001-of-00004.bin
    #   ...
    #   pytorch_model-00004-of-00004.bin
    #   pytorch_model.bin.index.json
    #
    model_name, model_extension = 'pytorch_model', 'bin'
    inspect_model_state_dict(dir_name, model_name, model_extension)

if __name__ == '__main__':
    fire.Fire(run)
