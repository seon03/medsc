#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------
# Same RadFact decomposition as run_decompose_radfact.py, but for RRG-MODEL-GENERATED
# reports instead of GT reports: reads a generate_reports.py run_dir's predictions.json
# (each record has "generated_report", plus "id"/"study_id"/"gt_report") and decomposes
# the generated_report field.
#
# No report-text dedup here (unlike run_decompose_radfact.py for GT): each prediction is
# generated independently per image/study, so predictions.json records don't have the
# "multiple records share identical GT text" redundancy that MIMIC's annotation.json has.
#
# Writes after EVERY sample so an interruption partway through a long run doesn't lose
# everything already decomposed.
#
# Usage:
#   cd metric/radfact
#   CUDA_VISIBLE_DEVICES=0,2,3 python3 scripts/run_decompose_radfact_predictions.py \
#       --predictions_path /home/seona/prj-brl/MedGemma/results/medgemma4b_mimic_train/predictions.json \
#       --output_file mimic_train_medgemma_radfact_findings.json
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
    parser.add_argument("--predictions_path", required=True, help="A generate_reports.py run_dir's predictions.json")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--num_samples", type=int, default=10**9)
    parser.add_argument("--skip", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.predictions_path) as f:
        records = json.load(f)
    records = records[args.skip : args.skip + args.num_samples]

    print(f"Loading {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print(f"{len(records)} generated reports to decompose.\n")

    results = []
    num_parse_failures = 0
    num_sentences = 0
    num_empty_new = 0
    durations: list[float] = []

    for rec in records:
        findings_text = extract_findings_text(rec["generated_report"])

        messages = build_messages(findings_text)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.monotonic()
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=3072, do_sample=False)
        elapsed = time.monotonic() - start
        durations.append(elapsed)

        raw = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        try:
            parsed = parse_model_output(rec["id"], raw)
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

        print(f"[{rec['id']}] ({elapsed:.1f}s) {status} - {len(findings)} findings")
        results.append(
            {
                "id": rec["id"],
                "study_id": rec.get("study_id", rec["id"]),
                "generated_report": rec["generated_report"],
                "findings": findings,
                "status": status,
            }
        )

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
