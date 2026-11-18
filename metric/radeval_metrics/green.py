#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------
# GREEN: LLM-based evaluation using StanfordAIMI/GREEN-radllama2-7b (~7B model). NOT cached
# locally as of 2026-11-10 -- running this for the first time triggers an automatic
# multi-GB download. See ../README.md's model-download table. Also requires the
# `radeval` env's `datasets`/`pyarrow`/`numpy` stack to actually be import-compatible
# (pyarrow needs numpy>=2.0; if this script fails with
# "pyarrow requires NumPy 2.0 or newer", run `pip install -U "numpy>=2.0"` in radeval).
#
# Usage: python3 green.py <run_dir> [run_dir2 ...]
# -----------------------------------------------------------------------------------------

from __future__ import annotations

import json
import sys
from pathlib import Path

from radeval import RadEval

from common import load_predictions, save_evidence, save_results

METRIC = "green"
PER_SAMPLE_KEYS = ["green"]


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
