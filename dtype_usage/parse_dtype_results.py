from  collections import defaultdict
import json

import fire

def run(in_file: str):
    print('in_file', in_file)
    results = json.load(open(in_file, 'r')) 

    assert len(results) < 1000, "github CLI has a limit of 1000 returned results, and we are at this limit"

    # example format:
    # {
    #   'path': 'modules/ComfyUI/comfy/model_management.py', 
    #   'repository': {'id': 'R_kgDONuTC3g', 'isFork': False, 'isPrivate': False, 'nameWithOwner': 'Chunn241529/ollama_api', 'url': 'https://github.com/Chunn241529/ollama_api'}, 
    #   'sha': '28381561180c492d66aee4be502ac5b9f922c4c1', 
    #   'textMatches': [{'fragment': '    try:\n        float8_types.append(torch.float8_e8m0fnu)\n    except:', 'matches': [{'indices': [37, 57], 'text': 'torch.float8_e8m0fnu'}], 'property': 'content', 'type': 'FileContent'}, {'fragment': '    if args.fp8_e8m0fnu_unet:\n        return torch.float8_e8m0fnu\n', 'matches': [{'indices': [45, 65], 'text': 'torch.float8_e8m0fnu'}], 'property': 'content', 'type': 'FileContent'}], 
    #   'url': 'https://github.com/Chunn241529/ollama_api/blob/5032feab6030b57534cf8a25927869c06e350183/modules/ComfyUI/comfy/model_management.py'
    # } 

    repo_to_count = defaultdict(int)

    for row in results:
        # print(row)
        repo_to_count[row['repository']['nameWithOwner']] += 1


    repo_to_count = sorted(repo_to_count.items(), key=lambda item: item[1], reverse=True)

    # filter out pytorch stuff
    repo_to_count = [x for x in repo_to_count if not x[0].startswith('pytorch')]

    print('distinct non-pytorch repositories with callsites', len(repo_to_count))
    print('cumulative non-pytorch number of callsites', sum(x[1] for x in repo_to_count))

    for item in repo_to_count:
        print(item)
    print()

if __name__ == '__main__':
    fire.Fire(run)
