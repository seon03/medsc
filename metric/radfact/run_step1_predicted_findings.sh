#!/bin/bash
set -e
cd "$(dirname "$0")"
python3 scripts/run_decompose_radfact_predictions.py \
    --predictions_path /home/seona/prj-brl/baseline_sc/SC-SFT/pilot/medgemma_predictions/predictions.json \
    --output_file /home/seona/prj-brl/baseline_sc/SC-SFT/pilot/predicted_findings.json
