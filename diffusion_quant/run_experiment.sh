#!/bin/bash

with-proxy python diffusion_quant/main.py --mode full_quant --model flux --prompt_set calibration 2>&1 | tee diffusion_quant/outputs/flux/full_quant_fp8_calibration_log.txt
with-proxy python diffusion_quant/main.py --mode use_sweep_results --model flux --prompt_set calibration 2>&1 | tee diffusion_quant/outputs/flux/use_sweep_results_fp8_calibration_log.txt
with-proxy python diffusion_quant/main.py --mode full_quant --model flux --prompt_set test 2>&1 | tee diffusion_quant/outputs/flux/full_quant_fp8_test_log.txt
with-proxy python diffusion_quant/main.py --mode use_sweep_results --model flux --prompt_set test 2>&1 | tee diffusion_quant/outputs/flux/use_sweep_results_fp8_test_log.txt
