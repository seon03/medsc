#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------
# Sanity test for the RadFact-style NLI entailment step (decomposition/nli_local.py).
#
# We don't have real RRG model outputs yet (t1/r1 generation is a later step), so this
# validates the pipeline end-to-end on 2 real-ish chest X-ray reports vs a hand-corrupted
# version of each (one finding deliberately flipped) -- confirming the pipeline actually
# detects the injected error (precision/recall should drop, and the flipped phrase should
# show up as non-entailed), not just that the JSON parses.
#
# Usage:
#   cd metric/radfact
#   CUDA_VISIBLE_DEVICES=0,2,3 python3 scripts/test_nli_score.py
# -----------------------------------------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from decomposition.nli_local import build_messages as build_nli_messages
from decomposition.nli_local import compute_radfact_score
from decomposition.nli_local import parse_model_output as parse_nli_output
from decomposition.report_to_findings_radfact import build_messages as build_phrase_messages
from decomposition.report_to_findings_radfact import parse_model_output as parse_phrase_output

MODEL_PATH = (
    "/mnt/.cache/huggingface/models/models--meta-llama--Meta-Llama-3-70B-Instruct/"
    "snapshots/50fd307e57011801c7833c87efa1984ddf2db42f"
)  # matches RadFact's own README: "we used LLama-3-70B-Instruct" for entailment verification

TEST_CASES = [
    {
        "report_id": "sanity_1",
        "reference": (
            "There is no focal consolidation, pleural effusion or pneumothorax. "
            "The cardiomediastinal silhouette is normal."
        ),
        "candidate_correct": (
            "There is no focal consolidation, pleural effusion or pneumothorax. "
            "The cardiomediastinal silhouette is normal."
        ),
        "candidate_corrupted": (
            # flipped: pleural effusion falsely asserted present instead of absent
            "There is no focal consolidation or pneumothorax. A pleural effusion is present. "
            "The cardiomediastinal silhouette is normal."
        ),
    },
    {
        "report_id": "sanity_2",
        "reference": "Mild cardiomegaly with bibasilar atelectasis. No pleural effusion or pneumothorax.",
        "candidate_correct": (
            "Mild cardiomegaly with bibasilar atelectasis. No pleural effusion or pneumothorax."
        ),
        "candidate_corrupted": (
            # flipped: severity changed mild->severe, and atelectasis negated instead of present
            "Severe cardiomegaly. No atelectasis. No pleural effusion or pneumothorax."
        ),
    },
]


def generate(tokenizer, model, messages: list[dict[str, str]]) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def decompose(tokenizer, model, report_id: str, text: str) -> list[str]:
    messages = build_phrase_messages(text)
    raw = generate(tokenizer, model, messages)
    parsed = parse_phrase_output(report_id, raw)
    return [phrase for sentence in parsed.sentence_list for phrase in sentence.new]


def main() -> None:
    print(f"Loading Llama-3-70B-Instruct from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print("Model loaded.\n")

    for case in TEST_CASES:
        print("=" * 80)
        print(f"[{case['report_id']}]")
        ref_phrases = decompose(tokenizer, model, case["report_id"] + "_ref", case["reference"])
        correct_phrases = decompose(tokenizer, model, case["report_id"] + "_correct", case["candidate_correct"])
        corrupted_phrases = decompose(
            tokenizer, model, case["report_id"] + "_corrupted", case["candidate_corrupted"]
        )

        print(f"reference phrases:            {ref_phrases}")
        print(f"candidate(correct) phrases:   {correct_phrases}")
        print(f"candidate(corrupted) phrases: {corrupted_phrases}\n")

        for label, cand_phrases in [("correct", correct_phrases), ("corrupted", corrupted_phrases)]:
            nli_messages = build_nli_messages(cand_phrases, ref_phrases)
            raw = generate(tokenizer, model, nli_messages)
            try:
                evidence = parse_nli_output(raw)
                score = compute_radfact_score(evidence)
                print(
                    f"  [{label}] precision={score.logical_precision:.2f} "
                    f"recall={score.logical_recall:.2f} f1={score.logical_f1:.2f}"
                )
                for p in evidence.phrases_A_evidenced:
                    if p.status != "entailment":
                        print(f"    ⚠️  candidate phrase NOT entailed ({p.status}): {p.phrase}")
                for p in evidence.phrases_B_evidenced:
                    if p.status != "entailment":
                        print(f"    ⚠️  reference phrase NOT entailed ({p.status}): {p.phrase}")
            except Exception as exc:  # noqa: BLE001 - prototype script, want to see everything
                print(f"  [{label}] 파싱 실패: {exc}\n  raw: {raw}")
        print()


if __name__ == "__main__":
    main()
