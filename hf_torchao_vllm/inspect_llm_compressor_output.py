# inspects the output of model created with llm-compressor
# via the `run_llm_compressor.py` script

import safetensors
import json
import fire

def run(
    dir_name: str = 'data/llmcompressor/fp8-opt-125m',
):
    json_config_name = f'{dir_name}/config.json'
    with open(json_config_name, 'r') as f:
        data = json.load(f)
        # TODO: pretty print
        print(json.dumps(data, indent=2))

    # inpect the model, saved in safetensors format
    model_name = f'{dir_name}/model.safetensors'
    with safetensors.safe_open(model_name, framework='pt', device='cpu') as f:
        print(f.metadata())
        for k in f.keys():
            t = f.get_tensor(k)
            print(k, t.shape, t.dtype)

if __name__ == '__main__':
    fire.Fire(run)
