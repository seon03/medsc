#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------
# RadCliQ-v1: composite metric (BERTScore + RadGraph + semantic embeddings + BLEU-2 via a
# learned linear model). Output key is "radcliq_v1" (not "radcliq") -- higher is better
# (returns 1/mean of the underlying raw distance scores). See ../README.md for the input
# contract (<run_dir>/predictions.json).
#
# Usage: python3 radcliq.py <run_dir> [run_dir2 ...]
# -----------------------------------------------------------------------------------------

from __future__ import annotations

import json
import sys
from pathlib import Path

from radeval import RadEval

from common import load_predictions, save_evidence, save_results

METRIC = "radcliq"
PER_SAMPLE_KEYS = ["radcliq_v1"]


def main() -> None:
    run_dirs = [Path(p) for p in sys.argv[1:]]
    if not run_dirs:
        print(f"Usage: python3 {Path(__file__).name} <run_dir> [run_dir2 ...]")
        sys.exit(1)

    evaluator = RadEval(metrics=[METRIC], per_sample=True, detailed=True)

    for run_dir in run_dirs:
        refs, hyps = load_predictions(run_dir)
        print(f"\n=== {run_dir} (N={len(refs)}) metric={METRIC} ===")
        results = evaluator(refs=refs, hyps=hyps)
        print(json.dumps(results, indent=2))
        save_results(run_dir, results, PER_SAMPLE_KEYS, METRIC)
        save_evidence(run_dir, METRIC, evaluator._active_metrics[0].last_evidence)


if __name__ == "__main__":
    main()
