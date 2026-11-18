#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------
# Re-aggregate an already-scored run_dir's per-sample metrics (merged into
# predictions.json's "radeval_scores" field by common.py's save_results/
# merge_per_sample_scores) restricted to frontal-view (PA/AP) images only -- no
# re-running the actual scoring models needed, since radeval_scores are already
# per-sample. Frontal/lateral is read from MIMIC-CXR's official
# mimic-cxr-2.0.0-metadata.csv.gz (ViewPosition column), matched by dicom_id == our
# predictions.json "id" field.
#
# Writes <run_dir>/scores_frontal.json (same shape as scores.json, but averaged over
# the frontal-only subset) and prints how many of the run_dir's samples were frontal.
#
# Usage:
#   python3 filter_frontal_and_reaggregate.py <run_dir> [run_dir2 ...]
# -----------------------------------------------------------------------------------------

from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

METADATA_PATH = "/mnt/.cache/huggingface/datasets/mimic-cxr/mimic-cxr-2.0.0-metadata.csv.gz"
FRONTAL_VIEWS = {"PA", "AP"}


def load_frontal_dicom_ids() -> set[str]:
    frontal_ids: set[str] = set()
    with gzip.open(METADATA_PATH, "rt") as f:
        for row in csv.DictReader(f):
            if row["ViewPosition"] in FRONTAL_VIEWS:
                frontal_ids.add(row["dicom_id"])
    return frontal_ids


def reaggregate(run_dir: Path, frontal_ids: set[str]) -> None:
    predictions_path = run_dir / "predictions.json"
    if not predictions_path.exists():
        print(f"SKIP {run_dir}: no predictions.json")
        return
    with open(predictions_path) as f:
        predictions = json.load(f)

    frontal_preds = [p for p in predictions if p["id"] in frontal_ids]
    print(f"{run_dir}: {len(frontal_preds)} / {len(predictions)} samples are frontal (PA/AP)")
    if not frontal_preds:
        print(f"SKIP {run_dir}: no frontal samples found (id format mismatch with dicom_id?)")
        return

    # Collect every radeval_scores key that's present and numeric across the frontal subset.
    all_keys: set[str] = set()
    for p in frontal_preds:
        all_keys.update(p.get("radeval_scores", {}).keys())

    frontal_scores: dict[str, float] = {}
    for key in sorted(all_keys):
        values = [
            p["radeval_scores"][key]
            for p in frontal_preds
            if key in p.get("radeval_scores", {}) and isinstance(p["radeval_scores"][key], (int, float))
        ]
        if values:
            frontal_scores[key] = sum(values) / len(values)

    frontal_scores["_num_samples"] = len(frontal_preds)
    frontal_scores["_num_samples_total"] = len(predictions)

    out_path = run_dir / "scores_frontal.json"
    with open(out_path, "w") as f:
        json.dump(frontal_scores, f, indent=2, ensure_ascii=False)
    print(f"  -> {out_path}")
    print(json.dumps(frontal_scores, indent=2))


def main() -> None:
    run_dirs = [Path(p) for p in sys.argv[1:]]
    if not run_dirs:
        print(f"Usage: python3 {Path(__file__).name} <run_dir> [run_dir2 ...]")
        sys.exit(1)

    print("Loading MIMIC-CXR view metadata...")
    frontal_ids = load_frontal_dicom_ids()
    print(f"{len(frontal_ids)} frontal (PA/AP) dicom_ids total in MIMIC-CXR.\n")

    for run_dir in run_dirs:
        reaggregate(run_dir, frontal_ids)


if __name__ == "__main__":
    main()
