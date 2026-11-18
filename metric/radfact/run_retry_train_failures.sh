#!/bin/bash
# Re-decomposes only the 3,980 parse_error records left over from the full train run
# (JSON-malformed LLM output), in place in the same gt_cache file. ~137,375 already-ok
# records are kept untouched and just carried through.
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm-decompose
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=1,2,3,8 python3 scripts/decompose_radfact_vllm.py \
    --output_file gt_cache/mimic-cxr_train_gt_radfact_findings.json \
    --tensor_parallel_size 4 \
    --retry_failed
