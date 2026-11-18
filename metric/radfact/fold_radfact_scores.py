#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------
# Folds a run_eval_radfact(_vllm).py output's P/R/F1 into <run_dir>/scores.json.
#
# Two things this fixes relative to the original version of this script:
#
# 1. Aggregation formula. RadFact's own metric/schema.py::RadFactScore.from_aggregate()
#    computes dataset-level precision/recall as nanmean(per-sample P), nanmean(per-sample R),
#    THEN takes logical_f1 as a derived @property = 2*P*R/(P+R) of those MEANS -- it does
#    NOT average the per-sample F1 values. The two are not the same number in general. This
#    script now matches the original: mean(P) and mean(R) first, F1 computed from those.
#
# 2. Upstream decompose-failure contamination. If run_eval_radfact(_vllm).py was run BEFORE
#    its own status-filtering fix (it now skips any pair where candidate or GT decompose
#    status != "ok"), its output can contain real-valued (non-NaN) precision/recall for
#    records whose upstream decompose actually failed -- e.g. a candidate with a JSON
#    parse_error has findings=[], which gets scored as "the model found literally nothing"
#    (a real, low recall) rather than excluded as "we don't know, decomposition broke."
#    Pass --candidate_findings_path/--gt_findings_path (the same files run_eval_radfact_vllm
#    used) to cross-reference and drop those contaminated ids before aggregating, without
#    re-running the expensive NLI step. New runs (after the status-filtering fix) won't have
#    any contaminated ids to begin with, so passing these is a no-op safety net for them.
#
# Usage:
#   python3 fold_radfact_scores.py <radfact_scores.json> <run_dir/scores.json> \
#       [--candidate_findings_path <run_dir/radfact_candidate_findings.json>] \
#       [--gt_findings_path <gt_cache/..._gt_radfact_findings.json>]
import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("radfact_scores_path")
    parser.add_argument("scores_path")
    parser.add_argument("--candidate_findings_path", default=None)
    parser.add_argument("--gt_findings_path", default=None)
    return parser.parse_args()


def load_bad_ids(candidate_findings_path: str | None, gt_findings_path: str | None) -> set[str]:
    """ids whose upstream candidate or GT decompose failed (status != 'ok')."""
    bad_ids: set[str] = set()
    if candidate_findings_path:
        with open(candidate_findings_path) as f:
            candidates = json.load(f)
        bad_ids |= {c["id"] for c in candidates if c["status"] != "ok"}
    if gt_findings_path:
        with open(gt_findings_path) as f:
            gt_groups = json.load(f)
        for group in gt_groups:
            if group["status"] != "ok":
                bad_ids |= set(group["ids"])
    return bad_ids


def main() -> None:
    args = parse_args()
    with open(args.radfact_scores_path) as f:
        records = json.load(f)

    bad_ids = load_bad_ids(args.candidate_findings_path, args.gt_findings_path)
    num_contaminated = sum(1 for r in records if r["id"] in bad_ids)

    ok = [r for r in records if r["status"] == "ok" and r["id"] not in bad_ids]

    def mean(key):
        vals = [r[key] for r in ok if r[key] is not None and r[key] == r[key]]  # == self: not NaN
        return sum(vals) / len(vals) if vals else None

    p = mean("logical_precision")
    r = mean("logical_recall")
    if p is not None and r is not None and (p + r) > 0:
        f1 = 2 * p * r / (p + r)
    elif p is not None and r is not None:
        f1 = 0.0
    else:
        f1 = None

    scores = {}
    try:
        with open(args.scores_path) as f:
            scores = json.load(f)
    except FileNotFoundError:
        pass
    scores["radfact_precision"] = p
    scores["radfact_recall"] = r
    scores["radfact_f1"] = f1
    with open(args.scores_path, "w") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    print(
        f"Folded RadFact-F1 (n={len(ok)}/{len(records)} used; "
        f"{num_contaminated} excluded for upstream decompose failure) into {args.scores_path}"
    )
    print(f"  precision={p}, recall={r}, f1={f1}")


if __name__ == "__main__":
    main()
