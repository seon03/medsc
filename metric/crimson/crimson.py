#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------
# CRIMSON: clinically-grounded LLM-based report metric (Baharoon et al. 2026,
# arXiv:2603.06183), NOT part of the `radeval` package -- lives under metric/crimson/
# (sibling to metric/radfact/, not metric/radeval_metrics/) for that reason, same as
# RadFact. Uses the CRIMSON repo cloned at metric/CRIMSON directly
# (pip install -e'd into its own env, see setup_crimson_env.sh -- torch>=2.10/python>=3.12
# conflicts with radeval's pins, so this needs a separate env, same reason RaTEScore does).
#
# Scores (-1, 1] per sample via the default HF model (rajpurkarlab/medgemma-4b-it-crimson).
# No patient_context (age/indication) is passed -- we don't have that data for MIMIC-CXR/IU
# X-Ray (same reason our EditGRPO-style prompts always use INDICATION: omitted/None, see
# baseline_rrg/Qwen3-VL-SFT/convert_to_llamafactory_multiview.py).
#
# Reuses radeval_metrics/common.py's load_predictions/save_results (predictions.json /
# scores.json contract) directly -- CRIMSON is a single evaluate() call per sample, same
# shape as green.py/radgraph.py, so it doesn't need RadFact's separate decompose+eval+merge
# pipeline.
#
# Usage:
#   conda activate crimson
#   python3 crimson.py <run_dir> [run_dir2 ...]
# -----------------------------------------------------------------------------------------

from __future__ import annotations

import sys
import time
from pathlib import Path

METRIC_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(METRIC_DIR / "CRIMSON"))
from CRIMSON import CRIMSONScore  # noqa: E402

sys.path.insert(0, str(METRIC_DIR / "radeval_metrics"))
from common import load_predictions, save_evidence, save_results  # noqa: E402

METRIC = "crimson"
PER_SAMPLE_KEYS = ["crimson"]


def main() -> None:
    run_dirs = [Path(p) for p in sys.argv[1:]]
    if not run_dirs:
        print(f"Usage: python3 {Path(__file__).name} <run_dir> [run_dir2 ...]")
        sys.exit(1)

    print("Loading rajpurkarlab/medgemma-4b-it-crimson ...")
    scorer = CRIMSONScore()  # api="hf" default

    for run_dir in run_dirs:
        refs, hyps = load_predictions(run_dir)
        print(f"\n=== {run_dir} (N={len(refs)}) metric={METRIC} ===")

        scores: list[float] = []
        evidence: list[dict] = []
        for i, (ref, hyp) in enumerate(zip(refs, hyps)):
            start = time.monotonic()
            result = scorer.evaluate(reference_findings=ref, predicted_findings=hyp)
            elapsed = time.monotonic() - start
            scores.append(result["crimson_score"])
            evidence.append(result)
            print(f"  [{i + 1}/{len(refs)}] ({elapsed:.1f}s) crimson_score={result['crimson_score']:.3f}")

        results = {"crimson": scores}
        save_results(run_dir, results, PER_SAMPLE_KEYS, METRIC)
        save_evidence(run_dir, METRIC, {"per_sample": evidence})


if __name__ == "__main__":
    main()
