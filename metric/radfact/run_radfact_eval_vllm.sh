#!/bin/bash
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm-decompose
cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 scripts/run_eval_radfact_vllm.py \
    --gt_findings_path gt_cache/mimic-cxr_test_gt_radfact_findings.json \
    --candidate_findings_path /home/seona/prj-brl/baseline_rrg/R2GenGPT/results/r2gengpt_delta_mimic_test/radfact_candidate_findings.json \
    --output_file /home/seona/prj-brl/baseline_rrg/R2GenGPT/results/r2gengpt_delta_mimic_test/radfact_scores.json \
    --tensor_parallel_size 4
