"""
Test that SGLANG is not broken by 
https://github.com/pytorch/ao/issues/1690

Note: I don't have a working sglang install, so the test below is a hack to verify
that just the torchao API still works.
"""

import fire
import torch
import torchao
import torch.nn as nn


def run():
    import sglang
    import sglang.srt.layers.torchao_utils as torchao_utils

    print(f"torch version: {torch.__version__}")
    print(f"torchao version: {torchao.__version__}")
    print(f"sglang version: {sglang.__version__}")

    m = nn.Sequential(nn.Linear(256, 256, bias=False, device="cuda"))
    torchao_config = "int8wo"
    filter_fn = lambda mod, fqn: isinstance(mod, torch.nn.Linear)
    m = torchao_utils.apply_torchao_config_to_model(m, torchao_config, filter_fn)
    print(m)

if __name__ == '__main__':
    fire.Fire(run)
