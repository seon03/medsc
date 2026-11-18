#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------
# RadGraph-F1: extracts an entity+relation graph from GT and generated report text (via the
# radgraph-xl model) and computes set-F1 at three granularities (entity-only, entity+status,
# entity+relations). See ../README.md for the input contract (<run_dir>/predictions.json).
#
# Usage: python3 radgraph.py <run_dir> [run_dir2 ...]
# -----------------------------------------------------------------------------------------

from __future__ import annotations

import json
import sys
from pathlib import Path

from radeval import RadEval

from common import load_predictions, save_evidence, save_results

METRIC = "radgraph"
PER_SAMPLE_KEYS = ["radgraph_simple", "radgraph_partial", "radgraph_complete"]


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
