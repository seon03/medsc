#!/bin/bash
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm-decompose
cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 scripts/decompose_radfact_predictions_vllm.py \
    --predictions_path /home/seona/prj-brl/baseline_rrg/R2GenGPT/results/r2gengpt_delta_mimic_test/predictions.json \
    --output_file /home/seona/prj-brl/baseline_rrg/R2GenGPT/results/r2gengpt_delta_mimic_test/radfact_candidate_findings.json \
    --tensor_parallel_size 4
