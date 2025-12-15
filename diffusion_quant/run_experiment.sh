#!/bin/bash

with-proxy python diffusion_quant/main.py --mode full_quant --model flux-2.dev --prompt_set calibration 2>&1 | tee diffusion_quant/outputs/flux-2.dev/full_quant_fp8_calibration_log.txt
with-proxy python diffusion_quant/main.py --mode use_sweep_results --model flux-2.dev --prompt_set calibration 2>&1 | tee diffusion_quant/outputs/flux-2.dev/use_sweep_results_fp8_calibration_log.txt
with-proxy python diffusion_quant/main.py --mode full_quant --model flux-2.dev --prompt_set test 2>&1 | tee diffusion_quant/outputs/flux-2.dev/full_quant_fp8_test_log.txt
with-proxy python diffusion_quant/main.py --mode use_sweep_results --model flux-2.dev --prompt_set test 2>&1 | tee diffusion_quant/outputs/flux-2.dev/use_sweep_results_fp8_test_log.txt
