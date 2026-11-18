#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------
# Decompose any dataset's reports (MIMIC-CXR or IU X-Ray, anything with the same
# annotation.json {train,val,test} shape) into RadGraph entity/relation graphs, then
# reconstruct a human-readable "atomic finding" list via
# decomposition/report_to_findings_radgraph.py::reconstruct_findings (observation +
# status + location + modifiers, from RadGraph's located_at/modify relations).
#
# Dataset-agnostic via extract_findings_text(): strips a "Findings:" header when one
# exists (MIMIC), falls back to using the report as-is when there's none (IU X-Ray).
#
# No LLM generation here -- RadGraph is a single BERT-family forward pass per report, so
# (unlike the radfact driver) this runs the whole split in ONE batched call instead of a
# slow per-sample loop. Must run in the `radeval` conda env (radgraph package + the
# transformers==4.49.0 pin that fixed its vendored allennlp code -- see PLAN.md history).
#
# Usage:
#   conda activate radeval
#   cd metric/radfact
#   python3 scripts/run_decompose_radgraph.py \
#       --annotation_path /mnt/.cache/huggingface/datasets/iu_xray/annotation.json \
#       --split test \
#       --output_file iuxray_test_radgraph_findings.json
# -----------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radgraph import RadGraph

from decomposition.mimic_text_utils import extract_findings_text
from decomposition.report_to_findings_radgraph import reconstruct_findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--num_samples", type=int, default=10**9)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--model_type", default="radgraph-xl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.annotation_path) as f:
        ann = json.load(f)
    records = ann[args.split][args.skip : args.skip + args.num_samples]
    texts = [extract_findings_text(rec["report"]) for rec in records]

    print(f"Loading RadGraph ({args.model_type}) ...")
    rad = RadGraph(model_type=args.model_type)

    print(f"Running RadGraph on {len(texts)} reports ...")
    start = time.monotonic()
    outputs = rad(texts)  # {"0": {"entities": {...}}, "1": {...}, ...}
    elapsed = time.monotonic() - start
    print(f"Done in {elapsed:.1f}s ({elapsed / len(texts):.3f}s/report)\n")

    def format_finding(f: dict) -> str:
        loc = f"loc:{','.join(f['location'])}" if f["location"] else ""
        mod = f"mod:{','.join(f['modifiers'])}" if f["modifiers"] else ""
        extra = ",".join(x for x in (loc, mod) if x)
        return f"{f['observation']}({f['status']}{',' + extra if extra else ''})"

    results = []
    num_empty = 0
    status_counts: Counter[str] = Counter()
    observation_counts: Counter[str] = Counter()

    for i, rec in enumerate(records):
        entities = outputs[str(i)]["entities"]
        findings = reconstruct_findings(entities)
        if not findings:
            num_empty += 1
        for f in findings:
            status_counts[f["status"]] += 1
            observation_counts[f["observation"].lower()] += 1

        results.append(
            {
                "id": rec["id"],
                "report": rec["report"],
                "radgraph_entities": entities,
                "findings": findings,
            }
        )
        # Per-report log: count + content preview, not just a bare number -- this is
        # exactly what would have surfaced the label-string mismatch bug immediately
        # instead of silently printing "0 findings" for every report.
        preview = " | ".join(format_finding(f) for f in findings) or "(없음)"
        print(f"[{rec['id']}] {len(findings)} findings: {preview}")

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("요약")
    print(f"  처리한 report 수: {len(results)}")
    print(f"  finding 0개로 나온 report 수: {num_empty} / {len(results)}")
    print(f"  status 분포: {dict(status_counts)}")
    print(f"  가장 빈번한 observation 상위 15개:")
    for obs, cnt in observation_counts.most_common(15):
        print(f"    {obs:<20} {cnt}")
    print(f"  저장 위치: {args.output_file}")


if __name__ == "__main__":
    main()
