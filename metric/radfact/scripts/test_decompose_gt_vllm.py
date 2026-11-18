#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------
# Same decomposition test as test_decompose_gt.py, but using vLLM's offline batch API
# instead of raw `transformers.generate()`.
#
# Two reasons this should be much faster than the transformers version (measured at
# 15.7s/report average on 20 real MIMIC-CXR GT reports):
#   1. Continuous batching: all reports are submitted to `llm.chat()` as one batch
#      instead of one sequential `generate()` call per report.
#   2. Automatic prefix caching: our system prompt + 9 few-shot examples are BYTE-FOR-BYTE
#      IDENTICAL across every single request (only the final query differs), so vLLM can
#      reuse that shared KV cache across the whole batch instead of re-prefilling it for
#      every report.
#
# Model choice: still Qwen2.5-32B-Instruct, already validated (20/20 parsed, few content
# issues -- see prior test_decompose_gt.py run). Deliberately NOT switching to Qwen3-32B
# (would need a new ~65GB download + handling its thinking-mode output format + a fresh
# quality re-validation) -- per discussion, sticking with the already-validated model.
#
# Requires a SEPARATE conda env ("vllm-decompose"): vllm pins its own torch/CUDA build,
# which can conflict with the `medgemma`/`base` envs used elsewhere in this project.
#   conda create -n vllm-decompose python=3.11 -y
#   conda activate vllm-decompose
#   pip install vllm
#
# Usage:
#   conda activate vllm-decompose
#   cd metric/radfact
#   CUDA_VISIBLE_DEVICES=0,2 python3 scripts/test_decompose_gt_vllm.py
#
# NOTE: TENSOR_PARALLEL_SIZE below must match the number of GPU ids you pass via
# CUDA_VISIBLE_DEVICES (2 GPUs here, since Qwen2.5-32B-Instruct in bf16 is ~62GB and
# doesn't comfortably fit on one GPU alongside other users' jobs on this machine).
# -----------------------------------------------------------------------------------------

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm import LLM, SamplingParams

from decomposition.mimic_text_utils import extract_findings_text
from decomposition.report_to_findings_radfact import build_messages, parse_model_output

MODEL_PATH = (
    "/mnt/.cache/huggingface/models/models--meta-llama--Meta-Llama-3-70B-Instruct/"
    "snapshots/50fd307e57011801c7833c87efa1984ddf2db42f"
)  # matches RadFact's own README: "we used LLama-3-70B-Instruct" for entailment verification
ANNOTATION_PATH = "/mnt/.cache/huggingface/datasets/mimic-cxr/annotation.json"
NUM_SAMPLES = 20
TENSOR_PARALLEL_SIZE = 2


def load_sample_reports(n: int) -> list[dict]:
    """Same dedup-by-report-text sampling as test_decompose_gt.py, so results are
    directly comparable to the transformers-engine run."""
    with open(ANNOTATION_PATH) as f:
        ann = json.load(f)
    seen_reports: set[str] = set()
    sampled: list[dict] = []
    for rec in ann["train"]:
        if rec["report"] in seen_reports:
            continue
        seen_reports.add(rec["report"])
        sampled.append(rec)
        if len(sampled) >= n:
            break
    return sampled


def main() -> None:
    print(f"Loading Llama-3-70B-Instruct via vLLM from {MODEL_PATH} ...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)

    records = load_sample_reports(NUM_SAMPLES)
    findings_texts = [extract_findings_text(rec["report"]) for rec in records]
    conversations = [build_messages(text) for text in findings_texts]

    print(f"Running batched generation for {len(conversations)} reports ...")
    start = time.monotonic()
    outputs = llm.chat(conversations, sampling_params)
    elapsed = time.monotonic() - start
    print(
        f"Total batch time: {elapsed:.1f}s "
        f"({elapsed / len(conversations):.2f}s/report amortized)\n"
    )

    num_parse_failures = 0
    num_sentences = 0
    num_empty_new = 0

    for rec, findings_text, output in zip(records, findings_texts, outputs):
        report_id = rec["id"]
        raw_output = output.outputs[0].text

        print("=" * 80)
        print(f"[{report_id}] findings_text:\n{findings_text}\n")
        try:
            parsed = parse_model_output(report_id, raw_output)
            print("파싱 성공:")
            for sentence in parsed.sentence_list:
                num_sentences += 1
                print(f"  orig: {sentence.orig}")
                if not sentence.new:
                    num_empty_new += 1
                    print("    ⚠️  new가 빈 리스트입니다")
                for phrase in sentence.new:
                    print(f"    -> {phrase}")
        except Exception as exc:  # noqa: BLE001 - prototype script, want to see everything
            num_parse_failures += 1
            print(f"파싱 실패: {exc}")
            print(f"raw output:\n{raw_output}")
        print()

    print("=" * 80)
    print("요약")
    print(f"  처리한 report 수: {len(records)}")
    print(f"  JSON 파싱 실패: {num_parse_failures} / {len(records)}")
    print(f"  빈 new 리스트 발생: {num_empty_new} / {num_sentences} 문장")
    print(f"  총 배치 소요 시간: {elapsed:.1f}s ({elapsed / len(records):.2f}s/report amortized)")


if __name__ == "__main__":
    main()
