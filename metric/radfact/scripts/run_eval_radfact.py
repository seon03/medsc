#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------
# Final RadFact-F1 aggregation step: given GT findings (from run_decompose_radfact.py,
# grouped by "ids" since GT reports are deduped by text) and candidate findings (from
# run_decompose_radfact_predictions.py, one "id" per generated report), run the
# bidirectional NLI comparison (decomposition/nli_local.py) for each matched pair and
# report per-sample + aggregate RadFactScore (logical precision/recall/F1).
#
# Matches candidate records to their GT group by looking up which GT group's "ids" list
# contains the candidate's "id" (GT dedup groups share one report across multiple image
# records, so several candidates can map to the same GT group).
#
# Writes after EVERY sample so an interruption partway through a long run doesn't lose
# everything already scored.
#
# Usage:
#   cd metric/radfact
#   CUDA_VISIBLE_DEVICES=0,2,3 python3 scripts/run_eval_radfact.py \
#       --gt_findings_path mimic_test_gt_radfact_findings.json \
#       --candidate_findings_path mimic_test_r2gengpt_radfact_findings.json \
#       --output_file r2gengpt_mimic_test_radfact_scores.json
# -----------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from decomposition.nli_local import build_messages, compute_radfact_score, parse_model_output

MODEL_PATH = (
    "/mnt/.cache/huggingface/models/models--meta-llama--Meta-Llama-3-70B-Instruct/"
    "snapshots/50fd307e57011801c7833c87efa1984ddf2db42f"
)  # matches RadFact's own README: "we used LLama-3-70B-Instruct" for entailment verification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_findings_path", required=True, help="Output of run_decompose_radfact.py")
    parser.add_argument(
        "--candidate_findings_path", required=True, help="Output of run_decompose_radfact_predictions.py"
    )
    parser.add_argument("--output_file", required=True)
    return parser.parse_args()


def build_gt_lookup(gt_findings_path: str) -> dict[str, tuple[list[str], str]]:
    """Map every original record id -> (its GT group's findings list, that group's decompose status)."""
    with open(gt_findings_path) as f:
        gt_groups = json.load(f)
    lookup: dict[str, tuple[list[str], str]] = {}
    for group in gt_groups:
        for record_id in group["ids"]:
            lookup[record_id] = (group["findings"], group["status"])
    return lookup


def main() -> None:
    args = parse_args()
    gt_lookup = build_gt_lookup(args.gt_findings_path)
    with open(args.candidate_findings_path) as f:
        candidates = json.load(f)

    print(f"Loading {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    results = []
    num_parse_failures = 0
    precisions, recalls, f1s = [], [], []

    for cand in candidates:
        cand_id = cand["id"]
        gt_entry = gt_lookup.get(cand_id)
        if gt_entry is None:
            print(f"[{cand_id}] skip - no matching GT findings")
            continue
        gt_findings, gt_status = gt_entry
        if cand["status"] != "ok":
            print(f"[{cand_id}] skip - candidate decompose failed ({cand['status']})")
            continue
        if gt_status != "ok":
            print(f"[{cand_id}] skip - GT decompose failed")
            continue

        cand_findings = cand["findings"]
        messages = build_messages(cand_findings, gt_findings)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
        raw = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        try:
            evidence = parse_model_output(raw, cand_findings, gt_findings)
            score = compute_radfact_score(evidence)
            status = "ok"
            if score.logical_precision == score.logical_precision:  # not NaN
                precisions.append(score.logical_precision)
            if score.logical_recall == score.logical_recall:  # not NaN
                recalls.append(score.logical_recall)
            if score.logical_f1 == score.logical_f1:  # not NaN
                f1s.append(score.logical_f1)
            print(f"[{cand_id}] P={score.logical_precision:.2f} R={score.logical_recall:.2f} F1={score.logical_f1:.2f}")
        except Exception as exc:  # noqa: BLE001 - want to see everything, keep going
            score = None
            status = f"parse_error: {exc}"
            num_parse_failures += 1
            print(f"[{cand_id}] {status}")

        results.append(
            {
                "id": cand_id,
                "status": status,
                "logical_precision": score.logical_precision if score else None,
                "logical_recall": score.logical_recall if score else None,
                "logical_f1": score.logical_f1 if score else None,
            }
        )
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("요약")
    print(f"  처리한 샘플 수: {len(results)}")
    print(f"  파싱 실패: {num_parse_failures}")
    if precisions:
        print(f"  mean logical_precision: {sum(precisions)/len(precisions):.4f} (n={len(precisions)})")
    if recalls:
        print(f"  mean logical_recall: {sum(recalls)/len(recalls):.4f} (n={len(recalls)})")
    if f1s:
        print(f"  mean logical_f1: {sum(f1s)/len(f1s):.4f} (n={len(f1s)})")
    print(f"  저장 위치: {args.output_file}")


if __name__ == "__main__":
    main()
