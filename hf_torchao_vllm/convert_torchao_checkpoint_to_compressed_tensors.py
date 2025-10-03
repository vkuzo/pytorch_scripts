import copy
import filecmp
import json
import pathlib
import shutil
import subprocess
from typing import Dict, Any

import fire

import torch
from torchao.core.config import AOBaseConfig, config_from_dict
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow
from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor

from safetensors import safe_open
from safetensors.torch import save_file

def ao_config_to_compressed_tensors_config(aobaseconfig: AOBaseConfig) -> Dict[str, Any]:
    # for now, allowlist of recipes we know how to convert and hand convert
    # them here
    # for a production version, we'll need a more scalable way to do this

    assert isinstance(aobaseconfig, Float8DynamicActivationFloat8WeightConfig), "unsupported"
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
    return ct_config 

def run(
    # original torchao checkpoint
    dir_source: str = 'data/torchao/fp8-opt-125m',
    # new compressed-tensors checkpoint
    dir_target: str = 'data/torchao_compressed_tensors/fp8-opt-125m',
    # existing compressed-tensors checkpoint to validate against
    dir_validation: str = 'data/llmcompressor/fp8-opt-125m',
    skip_conversion: bool = False,
):
    config_name_source = f"{dir_source}/config.json"
    config_name_target = f"{dir_target}/config.json"
    config_name_validation = f"{dir_validation}/config.json"
    weights_name_source = f"{dir_source}/pytorch_model.bin" 
    weights_name_target = f"{dir_target}/model.safetensors"
    weights_name_validation = f"{dir_validation}/model.safetensors"

    if not skip_conversion:
        #
        # convert config.json
        #

        with open(config_name_source, 'r') as f:
            config_source = json.load(f)

        # get torchao config format
        # example: https://www.internalfb.com/phabricator/paste/view/P1975688376
        # we need to translate it to compressed-tensors format
        # example: https://www.internalfb.com/phabricator/paste/view/P1975642629
        old_hf_quantization_config = config_source["quantization_config"]
        fqn_to_serialized_aobaseconfig = old_hf_quantization_config["quant_type"]
        assert len(fqn_to_serialized_aobaseconfig) == 1, "unsupported"

        new_hf_quantization_config = {
            "config_groups": {},
            "format": "float-quantized",
            "ignore": ["lm_head"],
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "sparsity_config": {},
            "transform_config": {},
            "version": "torchao_hack",
        }

        for fqn, serialized_aobaseconfig in fqn_to_serialized_aobaseconfig.items():
            print(fqn, serialized_aobaseconfig)
            aobaseconfig = config_from_dict(serialized_aobaseconfig)
            print(aobaseconfig)
            ct_config = ao_config_to_compressed_tensors_config(aobaseconfig)
            print(json.dumps(ct_config, indent=2))

            assert fqn == "default", "unsupported"
            new_hf_quantization_config["config_groups"]["group_0"] = ct_config

        # for now, modify config_source inplace
        config_source["quantization_config"] = new_hf_quantization_config

        # save to new location
        with open(config_name_target, 'w') as f:
            json.dump(config_source, f, indent=2)

        #
        # convert the checkpoint
        #

        # not sure why I still need this
        torch.serialization.add_safe_globals([getattr])

        old_state_dict = torch.load(weights_name_source, weights_only=True)
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
                # for now, manually cast scale to bfloat16 to match currnt 
                # llm-compressor script
                # TODO(future): prob needs to be user controllable 
                new_state_dict[k + '_scale'] = v.scale.bfloat16()
            else:
                raise AssertionError(f'unsupported type {type(v)}')
        save_file(new_state_dict, weights_name_target)

        # move all the other files over
        for dir_and_file_path in pathlib.Path(dir_source).iterdir():
            if not dir_and_file_path.is_file():
                continue
            file_path = dir_and_file_path.parts[-1]
            if file_path in ('config.json', 'pytorch_model.bin'):
                # these are converted in custom logic elsewhere in this script
                continue
            # if we got here, we just need to copy the file over without any changes
            target_file_path = f"{dir_target}/{str(file_path)}"
            shutil.copyfile(dir_and_file_path, target_file_path)

    # validate target_dir vs validation_dir
    for dir_and_file_path in pathlib.Path(dir_target).iterdir():
        if not dir_and_file_path.is_file():
            continue
        file_path_target = dir_and_file_path.parts[-1]
        print("\nvalidating", file_path_target)
        dir_and_file_path_validation = f"{dir_validation}/{str(file_path_target)}"

        if file_path_target == 'config.json':
            # for now just diff and print the output to stdout
            command = f'diff {dir_and_file_path} {dir_and_file_path_validation}'
            try:
                result = subprocess.run(command, capture_output=False, text=True, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                # this will always fail, for now, as we are not perfectly matching
                print(e.stderr) 

        elif file_path_target == 'model.safetensors':
            # TODO implement me
            pass

            with safe_open(dir_and_file_path, framework='pt') as f_target:
                with safe_open(dir_and_file_path_validation, framework='pt') as f_validation:
                    k_target_seen = set()
                    for k_target in f_target.keys():
                        v_target = f_target.get_tensor(k_target)
                        v_validation = f_validation.get_tensor(k_target)

                        # ensure metadata matches
                        if v_target.shape != v_validation.shape:
                            print(f"shape mismatch: {k_target=}, {v_target.shape=}, {v_validation.shape=}")

                        if v_target.dtype != v_validation.dtype: 
                            print(f"dtype mismatch: {k_target=}, {v_target.dtype=}, {v_validation.dtype=}")

                        # for now, no numerical checks

                        k_target_seen.add(k_target)

                    for k_validation in f_validation.keys():
                        if k_validation not in k_target_seen:
                            print(f"key {k_validation} not present in target")

        else:
            # approx check, currently fails because modification timestamp is not the
            # same. Since we copy these files ourselves, low-pri to make this better.
            is_equal = filecmp.cmp(dir_and_file_path, dir_and_file_path_validation, shallow=False)
            print('filecmp equal', is_equal)

if __name__ == '__main__':
    fire.Fire(run)
