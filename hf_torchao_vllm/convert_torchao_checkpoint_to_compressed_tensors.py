import filecmp
import json
import os
import pathlib
import shutil
import subprocess

import fire
from safetensors import safe_open

import torch
from torchao.core.config import config_from_dict
from utils import (
    ao_config_to_compressed_tensors_config,
    convert_pt_multifile_index_to_safetensors,
    convert_pt_statedict_to_safetensors,
)


def run(
    # original torchao checkpoint
    dir_source: str = "data/torchao/fp8-opt-125m",
    # new compressed-tensors checkpoint
    dir_target: str = "data/torchao_compressed_tensors/fp8-opt-125m",
    # existing compressed-tensors checkpoint to validate against
    dir_validation: str = "data/llmcompressor/fp8-opt-125m",
    skip_conversion: bool = False,
):
    dir_source = dir_source.rstrip("/")
    dir_target = dir_target.rstrip("/")
    dir_validation = dir_validation.rstrip("/")

    config_name_source = f"{dir_source}/config.json"
    config_name_target = f"{dir_target}/config.json"
    config_name_validation = f"{dir_validation}/config.json"
    weights_name_source = f"{dir_source}/pytorch_model.bin"
    weights_name_target = f"{dir_target}/model.safetensors"
    weights_name_validation = f"{dir_validation}/model.safetensors"

    # create new dir if not yet exists
    os.makedirs(dir_target, exist_ok=True)

    if not skip_conversion:
        source_converted_filenames = set()

        #
        # convert config.json
        #

        with open(config_name_source) as f:
            config_source = json.load(f)
            print(json.dumps(config_source, indent=2))

        # get torchao config format
        # example: https://www.internalfb.com/phabricator/paste/view/P1975688376
        # we need to translate it to compressed-tensors format
        # example: https://www.internalfb.com/phabricator/paste/view/P1975642629
        old_hf_quantization_config = config_source["quantization_config"]
        fqn_to_serialized_aobaseconfig = old_hf_quantization_config[
            "quant_type"
        ]
        assert len(fqn_to_serialized_aobaseconfig) == 1, "unsupported"

        if (
            fqn_to_serialized_aobaseconfig["default"]["_type"]
            == "ModuleFqnToConfig"
        ):
            fqn_to_serialized_aobaseconfig = fqn_to_serialized_aobaseconfig[
                "default"
            ]["_data"]["module_fqn_to_config"]

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

        for (
            fqn,
            serialized_aobaseconfig,
        ) in fqn_to_serialized_aobaseconfig.items():
            if serialized_aobaseconfig is None:
                new_hf_quantization_config["ignore"].append(fqn)
                continue

            aobaseconfig = config_from_dict(serialized_aobaseconfig)
            ct_config = ao_config_to_compressed_tensors_config(aobaseconfig)
            # print(aobaseconfig)
            # print(ct_config)
            # return

            # assert fqn in ("default", "_default"), "unsupported"
            new_hf_quantization_config["config_groups"]["group_0"] = ct_config

        # for now, modify config_source inplace
        config_source["quantization_config"] = new_hf_quantization_config

        # HACK: manually assign `ignore` based on the specific recipe
        # used to quantize `Llama-4-Scout-17B-16E-Instruct`:
        # 1. only quantize FFN
        for layer_idx in range(48):
            new_ignore_list = [
                f"model.layers.{layer_idx}.self_attn.q_proj",
                f"model.layers.{layer_idx}.self_attn.k_proj",
                f"model.layers.{layer_idx}.self_attn.v_proj",
                f"model.layers.{layer_idx}.self_attn.o_proj",
            ]
            new_hf_quantization_config["ignore"].extend(new_ignore_list)

        # save to new location
        with open(config_name_target, "w") as f:
            json.dump(config_source, f, indent=2)

        source_converted_filenames.add(config_name_source)

        #
        # convert the checkpoint
        #

        # not sure why I still need this
        torch.serialization.add_safe_globals([getattr])

        is_single_chunk = os.path.isfile(f"{dir_source}/pytorch_model.bin")
        if is_single_chunk:
            convert_pt_statedict_to_safetensors(
                weights_name_source, weights_name_target
            )
            source_converted_filenames.add(weights_name_source)
        else:
            # convert each model state_dict file
            model_part_filenames = []
            for file_path in pathlib.Path(dir_source).iterdir():
                if not file_path.is_file():
                    continue
                if not (
                    ("pytorch_model") in str(file_path)
                    and str(file_path).endswith("bin")
                ):
                    continue
                pt_sd_filename = str(file_path)
                # dir_source/pytorch_model-00001-of-00004.bin -> dir_target/model-00001-of-00004.safetensors
                safetensors_sd_filename = pt_sd_filename.replace(
                    dir_source, dir_target
                )
                safetensors_sd_filename = safetensors_sd_filename.replace(
                    "pytorch_model", "model"
                )
                safetensors_sd_filename = safetensors_sd_filename.replace(
                    ".bin", ".safetensors"
                )
                model_part_filenames.append(safetensors_sd_filename)
                print(pt_sd_filename, safetensors_sd_filename)
                convert_pt_statedict_to_safetensors(
                    pt_sd_filename, safetensors_sd_filename
                )
                source_converted_filenames.add(pt_sd_filename)

            # convert pytorch_model.bin.index.json
            convert_pt_multifile_index_to_safetensors(
                f"{dir_source}/pytorch_model.bin.index.json",
                f"{dir_target}/model.safetensors.index.json",
                model_part_filenames,
            )
            source_converted_filenames.add(
                f"{dir_source}/pytorch_model.bin.index.json"
            )

        print(source_converted_filenames)

        # move all the other files over
        for dir_and_file_path in pathlib.Path(dir_source).iterdir():
            if not dir_and_file_path.is_file():
                continue
            if str(dir_and_file_path) in source_converted_filenames:
                # these are converted in custom logic elsewhere in this script
                continue
            # if we got here, we just need to copy the file over without any changes
            file_path = dir_and_file_path.parts[-1]
            target_file_path = f"{dir_target}/{str(file_path)}"
            print(f"copying {dir_and_file_path} to {target_file_path}")
            shutil.copyfile(dir_and_file_path, target_file_path)

    # validate target_dir vs validation_dir
    for dir_and_file_path in pathlib.Path(dir_target).iterdir():
        if not dir_and_file_path.is_file():
            continue
        file_path_target = dir_and_file_path.parts[-1]
        print("\nvalidating", file_path_target)
        dir_and_file_path_validation = (
            f"{dir_validation}/{str(file_path_target)}"
        )

        if file_path_target == "config.json":
            # for now just diff and print the output to stdout
            command = f"diff {dir_and_file_path} {dir_and_file_path_validation}"
            try:
                result = subprocess.run(
                    command,
                    capture_output=False,
                    text=True,
                    shell=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                # this will always fail, for now, as we are not perfectly matching
                print(e.stderr)

        # TODO(future, as needed): also validate the other files, they are unlikely to match
        # exactly for any model with >1 chunk of state dict files since we are not
        # trying to enfore that the same tensors live in the same chunks.

        elif file_path_target == "model.safetensors":
            with safe_open(dir_and_file_path, framework="pt") as f_target:
                with safe_open(
                    dir_and_file_path_validation, framework="pt"
                ) as f_validation:
                    k_target_seen = set()
                    for k_target in f_target.keys():
                        v_target = f_target.get_tensor(k_target)
                        v_validation = f_validation.get_tensor(k_target)

                        # ensure metadata matches
                        if v_target.shape != v_validation.shape:
                            print(
                                f"shape mismatch: {k_target=}, {v_target.shape=}, {v_validation.shape=}"
                            )

                        if v_target.dtype != v_validation.dtype:
                            print(
                                f"dtype mismatch: {k_target=}, {v_target.dtype=}, {v_validation.dtype=}"
                            )

                        # for now, no numerical checks

                        k_target_seen.add(k_target)

                    for k_validation in f_validation.keys():
                        if k_validation not in k_target_seen:
                            print(f"key {k_validation} not present in target")

        else:
            # approx check, currently fails because modification timestamp is not the
            # same. Since we copy these files ourselves, low-pri to make this better.
            is_equal = filecmp.cmp(
                dir_and_file_path, dir_and_file_path_validation, shallow=False
            )
            print("filecmp equal", is_equal)


if __name__ == "__main__":
    fire.Fire(run)
