#!/bin/bash

# wrapper around https://github.com/vllm-project/vllm/blob/main/use_existing_torch.py
# 1. stash local changes, if applicable
# 2. run the script
# 3. unstash local changes

# ensure we are in the vllm working directory
if [[ $(basename "$PWD") != "vllm" ]]; then
    echo "Error: Current directory must end with 'vllm'" >&2
    exit 1
fi
echo "passed"

# add a dummy file, this will ensure `git stash` in the next command
# will always capture something and remove the need to handle the 
# "no local changes" case
touch dummy_file.txt

# stash existing changes
git stash push

# build vLLM with local pytorch
# see https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#use-an-existing-pytorch-installation
python use_existing_torch.py
with-proxy pip install -r requirements/build.txt
with-proxy pip install --no-build-isolation -e .

# undo the changes made by `use_existing_torch.py`
git restore .

# restore local changes
git stash pop

# remove the dummy file
rm dummy_file.txt
