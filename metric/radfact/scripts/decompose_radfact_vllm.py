#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------
# Production vLLM-batched version of run_decompose_radfact.py, for splits too large for
# the plain transformers.generate() sequential loop to finish in reasonable time (MIMIC
# train: 141,355 unique reports after dedup -- at ~15-30s/report sequential, that's 3-5+
# weeks; vLLM's continuous batching + automatic prefix caching -- our system prompt + 9
# few-shot examples are byte-for-byte identical across every request -- should cut this
# drastically. See test_decompose_gt_vllm.py's original 20-sample validation for the
# technique; this generalizes it to any split/size and writes the same gt_cache/ output
# format run_eval.sh expects (list of {"ids": [...], "report": ..., "findings": [...],
# "status": ...}), INCLUDING the report-text dedup run_decompose_radfact.py does.
#
# Writes incrementally (after each completed vLLM batch) so a long train-scale run surviving
# a restart doesn't lose everything, same principle as the other decompose scripts in this
# project -- but note vLLM's own batch itself is all-or-nothing per call, so BATCH_SIZE below
# controls how much work is at risk if interrupted, not per-report granularity.
#
# Requires a SEPARATE conda env ("vllm-decompose"): vllm pins its own torch/CUDA build.
#   conda create -n vllm-decompose python=3.11 -y && conda activate vllm-decompose && pip install vllm
#
# Usage:
#   conda activate vllm-decompose
#   cd metric/radfact
#   CUDA_VISIBLE_DEVICES=4,5,6 python3 scripts/decompose_radfact_vllm.py \
#       --annotation_path /mnt/.cache/huggingface/datasets/mimic-cxr/annotation.json \
#       --split train \
#       --output_file gt_cache/mimic-cxr_train_gt_radfact_findings.json \
#       --tensor_parallel_size 3
# -----------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
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
)  # matches RadFact's own README: "we used LLama-3-70B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", required=False, help="Required unless --retry_failed.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=3)
    parser.add_argument(
        "--batch_size", type=int, default=2000,
        help="How many reports to submit to vLLM per llm.chat() call. Larger batches "
             "amortize scheduling overhead better; smaller batches checkpoint more often.",
    )
    parser.add_argument("--skip", type=int, default=0, help="Skip this many distinct reports (resume).")
    parser.add_argument(
        "--retry_failed", action="store_true",
        help="Instead of decomposing --split fresh, load --output_file (must already exist), "
             "re-run only the records whose status != 'ok' (e.g. JSON parse_error from a "
             "prior run), and overwrite --output_file with those replaced in place. "
             "--annotation_path/--split are ignored in this mode.",
    )
    parser.add_argument(
        "--retry_temperature", type=float, default=0.3,
        help="Sampling temperature for --retry_failed only (main run always uses greedy "
             "temperature=0.0). A JSON parse_error at temperature=0 tends to reproduce "
             "deterministically on retry -- a small positive temperature gives the retry an "
             "actual chance at a differently-shaped (hopefully valid) output.",
    )
    return parser.parse_args()


def dedupe_by_report_text(records: list[dict]) -> list[dict]:
    """Same as run_decompose_radfact.py::dedupe_by_report_text."""
    groups: dict[str, dict] = {}
    for rec in records:
        report = rec["report"]
        if report not in groups:
            groups[report] = {"ids": [], "report": report}
        groups[report]["ids"].append(rec["id"])
    return list(groups.values())


def main() -> None:
    args = parse_args()

    done_records: list[dict] = []
    if args.retry_failed:
        with open(args.output_file) as f:
            existing = json.load(f)
        done_records = [r for r in existing if r["status"] == "ok"]
        groups = [{"ids": r["ids"], "report": r["report"]} for r in existing if r["status"] != "ok"]
        print(
            f"--retry_failed: {len(done_records)} already-ok records kept as-is, "
            f"{len(groups)} failed records will be re-decomposed.\n"
        )
    else:
        with open(args.annotation_path) as f:
            ann = json.load(f)
        groups = dedupe_by_report_text(ann[args.split])
        groups = groups[args.skip :]
        print(f"{len(groups)} distinct reports to decompose (after --skip={args.skip}).\n")

    print(f"Loading Llama-3-70B-Instruct via vLLM from {MODEL_PATH} ...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )
    temperature = args.retry_temperature if args.retry_failed else 0.0
    sampling_params = SamplingParams(temperature=temperature, max_tokens=1024)

    results = list(done_records)
    num_parse_failures = 0
    start_all = time.monotonic()

    for batch_start in range(0, len(groups), args.batch_size):
        batch = groups[batch_start : batch_start + args.batch_size]
        findings_texts = [extract_findings_text(g["report"]) for g in batch]
        conversations = [build_messages(t) for t in findings_texts]
        group_ids = [g["ids"][0] for g in batch]  # LLM-facing id only, doesn't need to be unique across batches

        start = time.monotonic()
        outputs = llm.chat(conversations, sampling_params)
        elapsed = time.monotonic() - start
        done = batch_start + len(batch)
        print(
            f"[batch {batch_start}-{done}/{len(groups)}] {elapsed:.1f}s "
            f"({elapsed / len(batch):.2f}s/report amortized)"
        )

        for group, group_id, output in zip(batch, group_ids, outputs):
            raw = output.outputs[0].text
            try:
                parsed = parse_model_output(group_id, raw)
                findings = [p for s in parsed.sentence_list for p in s.new]
                status = "ok"
            except Exception as exc:  # noqa: BLE001 - want to see everything, keep going
                findings = []
                status = f"parse_error: {exc}"
                num_parse_failures += 1

            results.append(
                {"ids": group["ids"], "report": group["report"], "findings": findings, "status": status}
            )

        # Checkpoint after every batch, not just at the end.
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    total_elapsed = time.monotonic() - start_all
    print("\n" + "=" * 80)
    print("요약")
    print(f"  처리한 report 수: {len(results)}")
    print(f"  JSON 파싱 실패: {num_parse_failures} / {len(results)}")
    print(f"  총 소요 시간: {total_elapsed:.1f}s ({total_elapsed / len(results):.2f}s/report amortized)")
    print(f"  저장 위치: {args.output_file}")


if __name__ == "__main__":
    main()
