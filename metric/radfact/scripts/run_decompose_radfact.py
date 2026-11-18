#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------
# Decompose any dataset's reports (MIMIC-CXR or IU X-Ray, anything with the same
# annotation.json {train,val,test} shape) into RadFact-style one-finding-per-sentence
# phrases, using decomposition/report_to_findings_radfact.py (RadFact's own CXR prompt +
# 9 few-shot examples, called against a local Qwen2.5-32B-Instruct -- no `radfact` package
# install, see that file's header for why).
#
# Dataset-agnostic via extract_findings_text(): strips a "Findings:" header when one
# exists (MIMIC), falls back to using the report as-is when there's none (IU X-Ray has no
# header at all -- the report string IS the findings text directly).
#
# Dedupes by report TEXT before decomposing: MIMIC-CXR's train split has 270,790
# image-level records but only ~141,355 distinct report strings (multiple images --
# e.g. PA + lateral views of the same study -- share one report), so decomposing every
# record independently would waste ~2x the LLM calls on MIMIC train specifically. IU
# X-Ray is already 1:1 (one record per unique report), so this is a no-op there -- safe
# to re-run on existing IU X-Ray outputs without changing them.
# Each output record keeps "ids" (all record ids that shared this report text) instead of
# a single "id", so downstream code can still map findings back to every image record.
#
# Writes after EVERY sample (not just at the end) so an interruption partway through a long
# run doesn't lose everything already decomposed.
#
# Usage:
#   cd metric/radfact
#   CUDA_VISIBLE_DEVICES=0,2,3 python3 scripts/run_decompose_radfact.py \
#       --annotation_path /mnt/.cache/huggingface/datasets/iu_xray/annotation.json \
#       --split test \
#       --output_file iuxray_test_radfact_findings.json
#
#   CUDA_VISIBLE_DEVICES=0,2,3 python3 scripts/run_decompose_radfact.py \
#       --annotation_path /mnt/.cache/huggingface/datasets/mimic-cxr/annotation.json \
#       --split test \
#       --output_file mimic_test_radfact_findings.json
# -----------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from decomposition.mimic_text_utils import extract_findings_text
from decomposition.report_to_findings_radfact import build_messages, parse_model_output

MODEL_PATH = (
    "/mnt/.cache/huggingface/models/models--meta-llama--Meta-Llama-3-70B-Instruct/"
    "snapshots/50fd307e57011801c7833c87efa1984ddf2db42f"
)  # matches RadFact's own README: "we used LLama-3-70B-Instruct" for entailment verification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--num_samples", type=int, default=10**9)
    parser.add_argument("--skip", type=int, default=0)
    return parser.parse_args()


def dedupe_by_report_text(records: list[dict]) -> list[dict]:
    """Group records sharing the exact same report string (common on MIMIC-CXR, where
    multiple image views of one study repeat the same report text) into one entry with
    an "ids" list, so each distinct report is only decomposed once. First-seen order is
    preserved. No-op for datasets where every record already has a unique report (e.g.
    IU X-Ray)."""
    groups: dict[str, dict] = {}
    for rec in records:
        report = rec["report"]
        if report not in groups:
            groups[report] = {"ids": [], "report": report}
        groups[report]["ids"].append(rec["id"])
    return list(groups.values())


def main() -> None:
    args = parse_args()
    with open(args.annotation_path) as f:
        ann = json.load(f)
    records = ann[args.split][args.skip : args.skip + args.num_samples]
    groups = dedupe_by_report_text(records)

    print(f"Loading {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print(f"{len(records)} records -> {len(groups)} distinct reports to decompose.\n")

    results = []
    num_parse_failures = 0
    num_sentences = 0
    num_empty_new = 0
    durations: list[float] = []

    for group in groups:
        findings_text = extract_findings_text(group["report"])
        group_id = group["ids"][0]  # used only as the LLM-facing id for this call

        messages = build_messages(findings_text)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.monotonic()
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        elapsed = time.monotonic() - start
        durations.append(elapsed)

        raw = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        try:
            parsed = parse_model_output(group_id, raw)
            findings = []
            for sentence in parsed.sentence_list:
                num_sentences += 1
                if not sentence.new:
                    num_empty_new += 1
                findings.extend(sentence.new)
            status = "ok"
        except Exception as exc:  # noqa: BLE001 - want to see everything, keep going
            findings = []
            status = f"parse_error: {exc}"
            num_parse_failures += 1

        print(f"[{group_id}] (+{len(group['ids'])-1} dup) ({elapsed:.1f}s) {status} - {len(findings)} findings")
        results.append(
            {"ids": group["ids"], "report": group["report"], "findings": findings, "status": status}
        )

        # Write after EVERY sample, not just at the end -- a long run (hundreds to
        # thousands of reports) getting killed partway through should not lose everything.
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("요약")
    print(f"  처리한 report 수: {len(results)}")
    print(f"  JSON 파싱 실패: {num_parse_failures} / {len(results)}")
    print(f"  빈 new 리스트 발생: {num_empty_new} / {num_sentences} 문장")
    if durations:
        avg = sum(durations) / len(durations)
        print(f"  report당 평균 소요 시간: {avg:.1f}s (min {min(durations):.1f}s, max {max(durations):.1f}s)")
    print(f"  저장 위치: {args.output_file}")


if __name__ == "__main__":
    main()
