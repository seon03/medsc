#!/bin/bash
# -----------------------------------------------------------------------------------------
# Single entrypoint to evaluate ANY baseline's predictions.json against every metric in
# this project: RadGraph-F1, CheXbert-F1, RadCliQ, RaTEScore, GREEN (all via radeval_metrics/,
# the official `radeval` package) and RadFact-F1 (via radfact/, our own CXR pipeline).
#
# INPUT CONTRACT (see README.md): <run_dir>/predictions.json must be a JSON list of
#   {"id": ..., "study_id": ..., "gt_report": "...", "generated_report": "..."}
# This is the one format every baseline in this project writes to (directly, like
# base_model/MedGemma/generate_reports.py, or via a converter, like
# baseline_rrg/R2GenGPT/convert_results.py). Match this and `run_eval.sh` works for you.
#
# RadFact-F1 needs to know which dataset/split this run's GT came from, to look up the
# pre-decomposed GT findings cache (radfact/gt_cache/<dataset>_<split>_gt_radfact_findings.json)
# shared across every baseline evaluated on the same test set. Pass --dataset/--split
# explicitly rather than relying on run_dir/config.json (not every baseline's converted
# output has one, e.g. R2GenGPT's).
#
# GT decomposition is NOT done automatically here -- it's a one-time, dataset-level step,
# done once per dataset/split and reused by every baseline after that. If the cache is
# missing, this script tells you the exact command to populate it
# (radfact/scripts/decompose_radfact_vllm.py) and exits; it does not fall back to the slow
# sequential decomposer.
#
# The candidate (this baseline's generated reports) decompose + NLI-eval steps DO run
# automatically here, via the vLLM-batched scripts (radfact/scripts/*_vllm.py) -- these need
# a separate `vllm-decompose` conda env (vllm pins its own torch/CUDA build), invoked here via
# `conda run -n vllm-decompose` so you don't have to activate it yourself or leave the env
# you're running the radeval metrics in.
#
# Requires TWO conda envs: `radeval` (for radgraph/f1chexbert/radcliq/ratescore/green, and to
# run this script itself) and `vllm-decompose` (for the radfact candidate-decompose/NLI steps,
# invoked automatically via `conda run`).
#
# Usage:
#   conda activate radeval
#   ./run_eval.sh <run_dir> --dataset mimic-cxr --split test [--metrics radgraph,f1chexbert,radcliq,ratescore,green,radfact] [--gpus 0,2,3] [--radfact_tp 4]
#
#   # run everything (default) on an R2GenGPT result:
#   ./run_eval.sh <run_dir> \
#       --dataset mimic-cxr --split test
#
#   # radeval metrics only, skip the expensive RadFact-F1 LLM pass:
#   ./run_eval.sh <run_dir> --dataset mimic-cxr --split test --metrics radgraph,f1chexbert,radcliq,ratescore,green
# -----------------------------------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALL_METRICS="radgraph,f1chexbert,radcliq,ratescore,green,radfact"

RUN_DIR=""
DATASET=""
SPLIT=""
METRICS="$ALL_METRICS"
GPUS=""
RADFACT_TP=4

RUN_DIR="$1"; shift
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

if [[ -z "$RUN_DIR" || ! -f "$RUN_DIR/predictions.json" ]]; then
  echo "Usage: $0 <run_dir> --dataset mimic-cxr|iu-xray --split train|val|test [--metrics ...] [--gpus 0,2,3]"
  echo "  <run_dir>/predictions.json not found."
  exit 1
fi

case "$DATASET" in
  mimic-cxr) ANNOTATION_PATH="/mnt/.cache/huggingface/datasets/mimic-cxr/annotation.json" ;;
  iu-xray) ANNOTATION_PATH="/mnt/.cache/huggingface/datasets/iu_xray/annotation.json" ;;
  "") ANNOTATION_PATH="" ;;
  *) echo "Unknown --dataset '$DATASET' (expected mimic-cxr or iu-xray)"; exit 1 ;;
esac

# Split requested metrics into "radeval_metrics/" ones vs radfact
IFS=',' read -ra METRIC_ARR <<< "$METRICS"
RADEVAL_SUBSET=()
RUN_RADFACT=0
for m in "${METRIC_ARR[@]}"; do
  if [[ "$m" == "radfact" ]]; then
    RUN_RADFACT=1
  else
    RADEVAL_SUBSET+=("$m")
  fi
done

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi

for m in "${RADEVAL_SUBSET[@]}"; do
  echo "=== $m ==="
  if [[ "$m" == "ratescore" ]]; then
    # ratescore's medspacy->spacy->thinc chain needs numpy<2.0, which conflicts with
    # green's pyarrow/datasets dependency (needs numpy>=2.0) in the same env. Run it in
    # its own env (radeval-rate, cloned from radeval with numpy<2.0) instead of toggling
    # numpy back and forth in radeval every time.
    conda run -n radeval-rate --no-capture-output python3 "$SCRIPT_DIR/radeval_metrics/$m.py" "$RUN_DIR"
  else
    python3 "$SCRIPT_DIR/radeval_metrics/$m.py" "$RUN_DIR"
  fi
done

if [[ "$RUN_RADFACT" -eq 1 ]]; then
  if [[ -z "$DATASET" || -z "$SPLIT" ]]; then
    echo "RadFact-F1 requires --dataset and --split (to find/cache GT decomposition)."
    exit 1
  fi

  GT_CACHE="$SCRIPT_DIR/radfact/gt_cache/${DATASET}_${SPLIT}_gt_radfact_findings.json"
  CAND_FINDINGS="$RUN_DIR/radfact_candidate_findings.json"
  RADFACT_SCORES="$RUN_DIR/radfact_scores.json"

  echo "=== RadFact-F1 ==="
  if [[ ! -f "$GT_CACHE" ]]; then
    echo "GT findings cache missing: $GT_CACHE"
    echo "This is a one-time, dataset/split-level step shared by every baseline -- populate it with:"
    echo "  conda activate vllm-decompose"
    echo "  cd $SCRIPT_DIR/radfact"
    echo "  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/decompose_radfact_vllm.py \\"
    echo "      --annotation_path $ANNOTATION_PATH --split $SPLIT \\"
    echo "      --output_file gt_cache/${DATASET}_${SPLIT}_gt_radfact_findings.json --tensor_parallel_size 4"
    echo "Then re-run this command."
    exit 1
  fi
  echo "-- GT findings cache hit: $GT_CACHE --"

  if [[ -f "$CAND_FINDINGS" ]]; then
    echo "-- candidate findings cache hit: $CAND_FINDINGS (delete it to force re-decompose) --"
  else
    echo "-- decomposing this baseline's generated reports (vLLM, env: vllm-decompose) --"
    conda run -n vllm-decompose --no-capture-output python3 \
      "$SCRIPT_DIR/radfact/scripts/decompose_radfact_predictions_vllm.py" \
      --predictions_path "$RUN_DIR/predictions.json" \
      --output_file "$CAND_FINDINGS" \
      --tensor_parallel_size "$RADFACT_TP"
  fi

  if [[ -f "$RADFACT_SCORES" ]]; then
    echo "-- radfact scores cache hit: $RADFACT_SCORES (delete it to force re-eval) --"
  else
    echo "-- running NLI comparison (vLLM, env: vllm-decompose) --"
    conda run -n vllm-decompose --no-capture-output python3 \
      "$SCRIPT_DIR/radfact/scripts/run_eval_radfact_vllm.py" \
      --gt_findings_path "$GT_CACHE" \
      --candidate_findings_path "$CAND_FINDINGS" \
      --output_file "$RADFACT_SCORES" \
      --tensor_parallel_size "$RADFACT_TP"
  fi

  # Fold P/R/F1 into run_dir/scores.json so one file has every metric. Uses RadFact's own
  # aggregation (mean(P), mean(R), THEN harmonic mean -- not mean of per-sample F1) and
  # cross-references candidate/GT decompose status to drop any upstream-failure contamination.
  python3 "$SCRIPT_DIR/radfact/fold_radfact_scores.py" "$RADFACT_SCORES" "$RUN_DIR/scores.json" \
    --candidate_findings_path "$CAND_FINDINGS" --gt_findings_path "$GT_CACHE"
fi

echo
echo "=== Done. Combined scores: $RUN_DIR/scores.json ==="
cat "$RUN_DIR/scores.json" 2>/dev/null | python3 -m json.tool 2>/dev/null || cat "$RUN_DIR/scores.json"
