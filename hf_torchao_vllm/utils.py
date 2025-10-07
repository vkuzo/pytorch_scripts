import copy
import json
import os
from typing import List, Dict, Any
import pathlib

import safetensors
from safetensors.torch import save_file

import torch
import torchao
from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor

from torchao.core.config import AOBaseConfig, config_from_dict
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow
from torchao.prototype.mx_formats.inference_workflow import NVFP4InferenceConfig
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
from torchao.prototype.mx_formats.utils import from_blocked
from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor


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

def convert_pt_statedict_to_safetensors(
    pt_statedict_filename,
    safetensors_statedict_filename,
) -> None:
    old_state_dict = torch.load(pt_statedict_filename, weights_only=True)
    new_state_dict = {}

    for k, v in old_state_dict.items():
        print(k, v.shape, type(v))
        if type(v) == torch.Tensor:
            
            if "lm_head" in k:
                # work around issues detailed in 
                # https://huggingface.co/docs/safetensors/torch_shared_tensors
                v = copy.deepcopy(v)

            new_state_dict[k] = v

        elif type(v) == Float8Tensor:
            new_state_dict[k] = v.qdata
            # for now, manually cast scale to bfloat16 to match current 
            # llm-compressor script
            # TODO(future): prob needs to be user controllable 
            new_state_dict[k + '_scale'] = v.scale.bfloat16()

        elif type(v) == NVFP4Tensor:
            # example checkpoint format: https://www.internalfb.com/phabricator/paste/view/P1981272933
            
            # torchao does not support nvfp4 activation calibration yet, 
            # set activation global scale to 1.0
            new_state_dict[k.replace('weight', 'input_global_scale')] = torch.tensor([1.0])
            # torchao does not support fusion-aware nvfp4 weight global scale yet,
            # set weight scale to 1.0
            new_state_dict[k.replace('weight', 'weight_global_scale')] = torch.tensor([1.0])
            new_state_dict[k + '_packed'] = v.qdata.view(torch.uint8)
            # compressed-tensors stores the nvfp4 scale in row-major format,
            # convert from swizzled to row-major
            swizzled_scale = v._scale_e4m3
            original_rows = v.qdata.shape[0]
            # multiply by 2 to undo the packing, then divide by nvfp4 block size of 16
            original_cols = v.qdata.shape[1] * 2 // 16
            # TODO(future) also do the padding calculation here and remove the
            # assertions
            assert original_rows % 128 == 0, "unsupported"
            assert original_cols % 4 == 0, "unsupported"
            # import pdb; pdb.set_trace()
            row_major_scale = from_blocked(
                swizzled_scale,
                original_rows,
                original_cols,
            )
            new_state_dict[k + '_scale'] = row_major_scale

        else:
            raise AssertionError(f'unsupported type {type(v)}')
    save_file(new_state_dict, safetensors_statedict_filename)

def convert_pt_multifile_index_to_safetensors(
    source_filename: str,
    target_filename: str,
    model_part_filenames: List[str],
) -> None:
    """
    Source format

    {
        "metadata": {...},
        "weight_map": {
            "foo": "pytorch_model-00001-of-00004.bin",
            "bar": "pytorch_model-00002-of-00004.bin",
            ...
        }
    }

    Target format

    {
        "metadata": {...},
        "weight_map": {
            # weight already in high precision
            "foo": "pytorch_model-00001-of-00004.bin",
            # weight original stored as tensor subclass, but now decomposed
            # into qdata and scale
            "bar": "model-00002-of-00004.safetensors",
            "bar_scale": "model-00002-of-00004.safetensors",
            ...
        }
    }

    For now, metadata is not updated.
    """

    # generate the new fqn to weight location map from the new safetensors files
    new_weight_map = {}
    for model_part_filename in model_part_filenames:
        # print(model_part_filename)

        # get the file_name from dir_name/file_name
        basename = os.path.basename(model_part_filename)
        # print(basename)

        with safetensors.safe_open(model_part_filename, framework='pt', device='cpu') as f:
            for k in f.keys():
                new_weight_map[k] = basename

    # save the updated mapping
    with open(source_filename, 'r') as f:
        source_mapping = json.load(f)
    source_mapping['weight_map'] = new_weight_map
    # print(json.dumps(source_mapping, indent=2))
    with open(target_filename, 'w') as f:
        json.dump(source_mapping, f, indent=2) 


def ao_config_to_compressed_tensors_config(aobaseconfig: AOBaseConfig) -> Dict[str, Any]:
    # for now, allowlist of recipes we know how to convert and hand convert
    # them here
    # for a production version, we'll need a more scalable way to do this

    if isinstance(aobaseconfig, Float8DynamicActivationFloat8WeightConfig):
        assert aobaseconfig.granularity == [PerRow(), PerRow()], "unsupported"

        ct_config = {
            "format": "float-quantized",
            "input_activations": {
                "dynamic": True,
                "num_bits": 8,
                "strategy": "token",
                "symmetric": True,
                "type": "float",
            },
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "dynamic": False,
                "num_bits": 8,
                "observer": "minmax",
                "strategy": "channel",
                "symmetric": True,
                "type": "float",
            },
        }

    elif isinstance(aobaseconfig, NVFP4InferenceConfig):

        ct_config = {
            "format": "nvfp4-pack-quantized",
            "input_activations": {
              "dynamic": "local",
              "group_size": 16,
              "num_bits": 4,
              "observer": "minmax",
              "observer_kwargs": {},
              "strategy": "tensor_group",
              "symmetric": True,
              "type": "float",
            },
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
              "dynamic": False,
              "group_size": 16,
              "num_bits": 4,
              "observer": "minmax",
              "observer_kwargs": {},
              "strategy": "tensor_group",
              "symmetric": True,
              "type": "float"
            },
        }

    else:
        raise AssertionError(f"unsupported type {type(aobaseconfig)}")
    return ct_config 
