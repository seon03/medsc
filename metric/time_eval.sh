#!/bin/bash
# -----------------------------------------------------------------------------------------
# Same steps as run_eval.sh, but wraps each one with timing so you can see per-metric and
# total evaluation time for a given run_dir -- useful for estimating how long evaluating
# each future baseline will take. Does not modify run_eval.sh itself.
#
# Usage: same args as run_eval.sh
#   ./time_eval.sh <run_dir> --dataset mimic-cxr --split test [--metrics ...] [--gpus 0,2,3]
# -----------------------------------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALL_METRICS="radgraph,f1chexbert,radcliq,green,radfact"

RUN_DIR="$1"; shift
DATASET=""
SPLIT=""
METRICS="$ALL_METRICS"
GPUS=""
RADFACT_TP=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --metrics) METRICS="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --radfact_tp) RADFACT_TP="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi

TIMING_LOG="$RUN_DIR/eval_timing.log"
: > "$TIMING_LOG"
TOTAL_START=$(date +%s)

run_timed() {
  local label="$1"; shift
  local start=$(date +%s)
  echo "=== $label ==="
  "$@"
  local end=$(date +%s)
  local elapsed=$((end - start))
  echo "[$label] ${elapsed}s" | tee -a "$TIMING_LOG"
}

IFS=',' read -ra METRIC_ARR <<< "$METRICS"
RUN_RADFACT=0
for m in "${METRIC_ARR[@]}"; do
  if [[ "$m" == "radfact" ]]; then
    RUN_RADFACT=1
  else
    run_timed "$m" python3 "$SCRIPT_DIR/radeval_metrics/$m.py" "$RUN_DIR"
  fi
done

if [[ "$RUN_RADFACT" -eq 1 ]]; then
  GT_CACHE="$SCRIPT_DIR/radfact/gt_cache/${DATASET}_${SPLIT}_gt_radfact_findings.json"
  CAND_FINDINGS="$RUN_DIR/radfact_candidate_findings.json"
  RADFACT_SCORES="$RUN_DIR/radfact_scores.json"

  if [[ ! -f "$CAND_FINDINGS" ]]; then
    run_timed "radfact_decompose" conda run -n vllm-decompose --no-capture-output python3 \
      "$SCRIPT_DIR/radfact/scripts/decompose_radfact_predictions_vllm.py" \
      --predictions_path "$RUN_DIR/predictions.json" \
      --output_file "$CAND_FINDINGS" \
      --tensor_parallel_size "$RADFACT_TP"
  else
    echo "-- candidate findings cache hit, skipping decompose timing --"
  fi

  if [[ ! -f "$RADFACT_SCORES" ]]; then
    run_timed "radfact_nli" conda run -n vllm-decompose --no-capture-output python3 \
      "$SCRIPT_DIR/radfact/scripts/run_eval_radfact_vllm.py" \
      --gt_findings_path "$GT_CACHE" \
      --candidate_findings_path "$CAND_FINDINGS" \
      --output_file "$RADFACT_SCORES" \
      --tensor_parallel_size "$RADFACT_TP"
  else
    echo "-- radfact scores cache hit, skipping NLI timing --"
  fi

  python3 "$SCRIPT_DIR/radfact/fold_radfact_scores.py" "$RADFACT_SCORES" "$RUN_DIR/scores.json" \
    --candidate_findings_path "$CAND_FINDINGS" --gt_findings_path "$GT_CACHE"
fi

TOTAL_END=$(date +%s)
echo "[TOTAL] $((TOTAL_END - TOTAL_START))s" | tee -a "$TIMING_LOG"

echo
echo "===== Timing summary ($TIMING_LOG) ====="
cat "$TIMING_LOG"
