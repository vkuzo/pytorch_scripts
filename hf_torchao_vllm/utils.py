import json
import os
import pathlib

import safetensors

import torch

torch.serialization.add_safe_globals([getattr])

def _inspect_state_dict_file(model_name):
    if str(model_name).endswith('safetensors'):
        # safetensors format
        with safetensors.safe_open(model_name, framework='pt', device='cpu') as f:
            print(f.metadata())
            for k in f.keys():
                t = f.get_tensor(k)
                print(k, type(t), t.shape, t.dtype)
    else:
        # pytorch format
        state_dict = torch.load(model_name, weights_only=True)
        for k, v in state_dict.items():
            print(k, type(v), v.shape, v.dtype)

def inspect_model_state_dict(dir_name, model_name, model_extension) -> None:
    """
    Inspect the model state_dict from HuggingFace and print data to stdout.
    For example, if model_name == `pytorch_model` and extension == `bin`,
    1. if there is a single chunk, the state dict is named `pytorch_model.bin`
    2. if there are multiple chunks, the state dict is spread across multiple
       files:

      pytorch_model-00001-of-00004.bin
      ...
      pytorch_model-00004-of-00004.bin
      pytorch_model.bin.index.json
    """
    is_single_chunk = os.path.isfile(f'{dir_name}/{model_name}.{model_extension}')
    if is_single_chunk:
        print('single state dict file')
        model_name = f'{dir_name}/{model_name}.{model_extension}'
        _inspect_state_dict_file(model_name)
    else:
        print('multiple state dict files')

        index_name = f'{dir_name}/{model_name}.{model_extension}.index.json'
        print(index_name)
        with open(index_name, 'r') as f:
            data = json.load(f)
            print(json.dumps(data, indent=2))

        # iterate through each file
        for file_path in pathlib.Path(dir_name).iterdir():
            if not file_path.is_file():
                continue
            if not (model_name in str(file_path) and str(file_path).endswith(model_extension)):
                continue
            print(file_path)
            _inspect_state_dict_file(file_path)
