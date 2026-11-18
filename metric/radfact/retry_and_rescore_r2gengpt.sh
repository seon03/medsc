#!/bin/bash
set -e
RUN_DIR="/home/seona/prj-brl/baseline_rrg/R2GenGPT/results/r2gengpt_delta_mimic_test"
GT_CACHE="/home/seona/prj-brl/metric/radfact/gt_cache/mimic-cxr_test_gt_radfact_findings.json"

cd "$(dirname "$0")"

echo "===== Step 1: retry the 223 failed decompose samples (temperature=0.3) ====="
python3 scripts/decompose_radfact_predictions_vllm.py \
    --output_file "$RUN_DIR/radfact_candidate_findings.json" \
    --retry_failed \
    --retry_temperature 0.3 \
    --tensor_parallel_size 4

echo
echo "===== Step 2: re-run NLI comparison on the full (now-fixed) candidate set ====="
rm -f "$RUN_DIR/radfact_scores.json"
python3 scripts/run_eval_radfact_vllm.py \
    --gt_findings_path "$GT_CACHE" \
    --candidate_findings_path "$RUN_DIR/radfact_candidate_findings.json" \
    --output_file "$RUN_DIR/radfact_scores.json" \
    --tensor_parallel_size 4

echo
echo "===== Step 3: fold into scores.json ====="
python3 fold_radfact_scores.py "$RUN_DIR/radfact_scores.json" "$RUN_DIR/scores.json" \
    --candidate_findings_path "$RUN_DIR/radfact_candidate_findings.json" \
    --gt_findings_path "$GT_CACHE"

echo
echo "===== Done. Remaining failed ids (if any): ====="
python3 -c "
import json
with open('$RUN_DIR/radfact_candidate_findings.json') as f:
    cand = json.load(f)
failed = [c for c in cand if c['status'] != 'ok']
print(f'{len(failed)} / {len(cand)} still failed')
for c in failed[:5]:
    print(' -', c['id'], c['status'])
"
