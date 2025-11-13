"""
Converts a LLaMa 4 Scout checkpoint to have the expert weights in MNK (from MKN) layout.

This is important because vLLM needs MKN layout, so we convert it here AOT instead
of converting during model loading to vLLM. For LLaMa 4 Scout on H100, this saves
~2 minutes in model loading time.

Modifies the checkpoint inplace.
"""

import pathlib

import fire

import torch

# import torchao to ensure we can load torchao subclass weights
# note: noqa: F401 is required for ruff to not remove torchao from the list of
# imports
import torchao  # noqa: F401


def _modify_state_dict_file(model_name):
    state_dict = torch.load(model_name, weights_only=True)
    for k, v in state_dict.items():
        if "Float8" in str(type(v)) and len(v.shape) == 3:
            print(1, v.shape, v.qdata.shape, v.qdata.stride())
            v.qdata = v.qdata.transpose(-2, -1).contiguous().transpose(-2, -1)
            print(2, v.shape, v.qdata.shape, v.qdata.stride())
        print(k, type(v), v.shape, v.dtype)
    torch.save(state_dict, model_name)


def run(
    dir_name: str = "data/torchao/fp8-experts-only-mnk-Llama-4-Scout-17B-16E-Instruct/",
):
    model_name, model_extension = "pytorch_model", "bin"

    # iterate through each file
    for file_path in pathlib.Path(dir_name).iterdir():
        if not file_path.is_file():
            continue
        if not (
            model_name in str(file_path)
            and str(file_path).endswith(model_extension)
        ):
            continue
        print(file_path)
        _modify_state_dict_file(file_path)

    print("done")


if __name__ == "__main__":
    fire.Fire(run)
