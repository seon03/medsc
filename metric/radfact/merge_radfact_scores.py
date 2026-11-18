#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------
# run_eval_radfact.py writes its own standalone <output_file> (list of per-sample
# {"id", "logical_precision", "logical_recall", "logical_f1"}), separate from the
# <run_dir>/scores.json contract the radeval_metrics/ scripts use (see common.py's
# save_results). collect_results.py only reads scores.json, so RadFact-F1 never shows up
# in RRG_RESULTS.md until this merge step runs.
#
# Usage:
#   python3 merge_radfact_scores.py <radfact_scores_output.json> <run_dir>
# -----------------------------------------------------------------------------------------

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: python3 {Path(__file__).name} <radfact_scores_output.json> <run_dir>")
        sys.exit(1)

    radfact_path, run_dir = Path(sys.argv[1]), Path(sys.argv[2])
    with open(radfact_path) as f:
        results = json.load(f)

    ok_results = [r for r in results if r["status"] == "ok"]
    scores_path = run_dir / "scores.json"
    existing = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    existing["radfact_f1"] = [r["logical_f1"] for r in ok_results]
    existing["radfact_precision"] = [r["logical_precision"] for r in ok_results]
    existing["radfact_recall"] = [r["logical_recall"] for r in ok_results]
    scores_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))

    print(f"{len(ok_results)} / {len(results)} RadFact scores merged into {scores_path}")


if __name__ == "__main__":
    main()
