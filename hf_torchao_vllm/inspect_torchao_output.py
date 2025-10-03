# inspects the output of model created with torchao
# via the `torchao_hf_script.py` script

import json
import torch
import torchao  # this is needed to run torch.serialization.add_safe_globals([torchao.quantization.Float8Tensor])
import fire

# not sure why I still need this
torch.serialization.add_safe_globals([getattr])

def run(dir_name: str = 'data/torchao/fp8-opt-125m'):
    json_config_name = f'{dir_name}/config.json'

    # inspect the config
    with open(json_config_name, 'r') as f:
        data = json.load(f)
        # TODO: pretty print
        print(json.dumps(data, indent=2))

    # inspect the data
    model_name = f'{dir_name}/pytorch_model.bin'
    state_dict = torch.load(model_name, weights_only=True)
    for k, v in state_dict.items():
        print(k, v.shape, type(v))

if __name__ == '__main__':
    fire.Fire(run)
