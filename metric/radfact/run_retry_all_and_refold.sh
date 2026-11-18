#!/bin/bash
# Retries every known decompose parse_error (train GT, test GT, R2GenGPT candidate) with a
# small retry temperature (0.3) instead of pure greedy -- greedy retries mostly reproduce
# the same malformed JSON deterministically (confirmed: only 207/3980 recovered on the
# train set's first greedy-temperature retry). Then re-runs the NLI eval fresh (status
# filtering now picks up whatever got newly recovered) and re-folds scores.json.
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm-decompose
cd "$(cd "$(dirname "$0")" && pwd)"

echo "=== 1/4: retry train GT failures (3,773 remaining) ==="
CUDA_VISIBLE_DEVICES=1,2,3,4 python3 scripts/decompose_radfact_vllm.py \
    --output_file gt_cache/mimic-cxr_train_gt_radfact_findings.json \
    --tensor_parallel_size 4 --retry_failed --retry_temperature 0.3

echo "=== 2/4: retry test GT failures (53 remaining) ==="
CUDA_VISIBLE_DEVICES=1,2,3,4 python3 scripts/decompose_radfact_vllm.py \
    --output_file gt_cache/mimic-cxr_test_gt_radfact_findings.json \
    --tensor_parallel_size 4 --retry_failed --retry_temperature 0.3

echo "=== 3/4: retry R2GenGPT candidate failures (496 remaining) ==="
CUDA_VISIBLE_DEVICES=1,2,3,4 python3 scripts/decompose_radfact_predictions_vllm.py \
    --output_file /home/seona/prj-brl/baseline_rrg/R2GenGPT/results/r2gengpt_delta_mimic_test/radfact_candidate_findings.json \
    --tensor_parallel_size 4 --retry_failed --retry_temperature 0.3

echo "=== 4/4: re-run NLI eval fresh (picks up newly-recovered pairs) + re-fold ==="
rm -f /home/seona/prj-brl/baseline_rrg/R2GenGPT/results/r2gengpt_delta_mimic_test/radfact_scores.json
CUDA_VISIBLE_DEVICES=1,2,3,4 python3 scripts/run_eval_radfact_vllm.py \
    --gt_findings_path gt_cache/mimic-cxr_test_gt_radfact_findings.json \
    --candidate_findings_path /home/seona/prj-brl/baseline_rrg/R2GenGPT/results/r2gengpt_delta_mimic_test/radfact_candidate_findings.json \
    --output_file /home/seona/prj-brl/baseline_rrg/R2GenGPT/results/r2gengpt_delta_mimic_test/radfact_scores.json \
    --tensor_parallel_size 4

python3 fold_radfact_scores.py \
    /home/seona/prj-brl/baseline_rrg/R2GenGPT/results/r2gengpt_delta_mimic_test/radfact_scores.json \
    /home/seona/prj-brl/baseline_rrg/R2GenGPT/results/r2gengpt_delta_mimic_test/scores.json \
    --candidate_findings_path /home/seona/prj-brl/baseline_rrg/R2GenGPT/results/r2gengpt_delta_mimic_test/radfact_candidate_findings.json \
    --gt_findings_path gt_cache/mimic-cxr_test_gt_radfact_findings.json

cd "$(dirname "$0")/.."
python3 collect_results.py
echo "=== done ==="
