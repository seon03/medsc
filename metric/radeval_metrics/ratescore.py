#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------
# RaTEScore: entity-aware metric using NER (Angelakeke/RaTE-NER-Deberta) + synonym-aware
# embeddings (FremyCompany/BioLORD-2023-C) to score entity-level precision/recall. Both
# backing models are already cached locally -- no download needed. See ../README.md for
# the input contract (<run_dir>/predictions.json).
#
# Usage: python3 ratescore.py <run_dir> [run_dir2 ...]
# -----------------------------------------------------------------------------------------

from __future__ import annotations

import json
import sys
from pathlib import Path

from radeval import RadEval

from common import load_predictions, save_results

METRIC = "ratescore"
PER_SAMPLE_KEYS = ["ratescore"]


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


if __name__ == "__main__":
    main()
