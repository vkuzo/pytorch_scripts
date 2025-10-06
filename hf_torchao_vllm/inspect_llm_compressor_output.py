# inspects the output of model created with llm-compressor
# via the `run_llm_compressor.py` script

import safetensors
import json
import fire

from utils import inspect_model_state_dict

def run(
    dir_name: str = 'data/llmcompressor/fp8-opt-125m',
):
    json_config_name = f'{dir_name}/config.json'
    with open(json_config_name, 'r') as f:
        data = json.load(f)
        # TODO: pretty print
        print(json.dumps(data, indent=2))

    model_name, model_extension = 'model', 'safetensors'
    inspect_model_state_dict(dir_name, model_name, model_extension)

if __name__ == '__main__':
    fire.Fire(run)
