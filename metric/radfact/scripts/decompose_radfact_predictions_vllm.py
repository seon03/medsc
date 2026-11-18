#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------
# vLLM-batched version of run_decompose_radfact_predictions.py. The plain-transformers
# version loads Llama-3-70B-Instruct with device_map="auto" across however many GPUs are
# visible and generates ONE report at a time (batch_size=1) -- for R2GenGPT's 3858-sample
# test set that measured ~30s/report (confirmed: 43/3858 after 25 minutes), i.e. ~37h total.
# Same fix as decompose_radfact_vllm.py applied to the GT side: vLLM's continuous batching +
# PagedAttention + prefix caching (the few-shot system prompt is identical across every
# request) cuts this by roughly two orders of magnitude.
#
# No report-text dedup (same as the original): each prediction is generated independently
# per image/study, so there's no guaranteed "many records share identical text" redundancy
# the way MIMIC's annotation.json GT has.
#
# Requires the SEPARATE "vllm-decompose" conda env (vllm pins its own torch/CUDA build):
#   conda create -n vllm-decompose python=3.11 -y && conda activate vllm-decompose && pip install vllm
#
# Usage:
#   conda activate vllm-decompose
#   cd metric/radfact
#   CUDA_VISIBLE_DEVICES=1,2,3,4 python3 scripts/decompose_radfact_predictions_vllm.py \
#       --predictions_path /home/seona/prj-brl/baseline_rrg/R2GenGPT/results/r2gengpt_delta_mimic_test/predictions.json \
#       --output_file /home/seona/prj-brl/baseline_rrg/R2GenGPT/results/r2gengpt_delta_mimic_test/radfact_candidate_findings.json \
#       --tensor_parallel_size 4
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
    parser.add_argument("--predictions_path", required=False, help="A generate_reports.py run_dir's predictions.json. Required unless --retry_failed.")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument(
        "--batch_size", type=int, default=1000,
        help="How many reports to submit to vLLM per llm.chat() call.",
    )
    parser.add_argument("--num_samples", type=int, default=10**9)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument(
        "--retry_failed", action="store_true",
        help="Load --output_file (must already exist), re-run only records whose "
             "status != 'ok', and overwrite --output_file with those replaced in place. "
             "--predictions_path/--skip/--num_samples are ignored in this mode.",
    )
    parser.add_argument(
        "--retry_temperature", type=float, default=0.3,
        help="Sampling temperature for --retry_failed only (main run always uses greedy "
             "temperature=0.0). A JSON parse_error at temperature=0 tends to reproduce "
             "deterministically on retry -- a small positive temperature gives the retry an "
             "actual chance at a differently-shaped (hopefully valid) output.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024,
        help="vLLM max_tokens for the decompose call. NOT a truncation fix: diagnosed "
             "directly (diagnose_parse_errors.py, 2026-11-16) on 8 still-failing samples -- "
             "all had finish_reason='stop' at 187-285 tokens, nowhere near this limit. The "
             "actual cause is the model occasionally emitting ']'/'}' in the wrong order "
             "(e.g. '...\"}], {\"orig\":' instead of '...\"]}, {\"orig\":'), a genuine JSON "
             "formatting mistake that --retry_temperature (resampling) addresses, not this.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    done_records: list[dict] = []
    if args.retry_failed:
        with open(args.output_file) as f:
            existing = json.load(f)
        done_records = [r for r in existing if r["status"] == "ok"]
        records = [r for r in existing if r["status"] != "ok"]
        print(
            f"--retry_failed: {len(done_records)} already-ok records kept as-is, "
            f"{len(records)} failed records will be re-decomposed at temperature={args.retry_temperature}.\n"
        )
    else:
        with open(args.predictions_path) as f:
            records = json.load(f)
        records = records[args.skip : args.skip + args.num_samples]
        print(f"{len(records)} generated reports to decompose.\n")

    print(f"Loading Llama-3-70B-Instruct via vLLM from {MODEL_PATH} ...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )
    temperature = args.retry_temperature if args.retry_failed else 0.0
    sampling_params = SamplingParams(temperature=temperature, max_tokens=args.max_tokens)

    results = list(done_records)
    num_parse_failures = 0
    start_all = time.monotonic()

    for batch_start in range(0, len(records), args.batch_size):
        batch = records[batch_start : batch_start + args.batch_size]
        findings_texts = [extract_findings_text(rec["generated_report"]) for rec in batch]
        conversations = [build_messages(t) for t in findings_texts]

        start = time.monotonic()
        outputs = llm.chat(conversations, sampling_params)
        elapsed = time.monotonic() - start
        done = batch_start + len(batch)
        print(
            f"[batch {batch_start}-{done}/{len(records)}] {elapsed:.1f}s "
            f"({elapsed / len(batch):.2f}s/report amortized)"
        )

        for rec, output in zip(batch, outputs):
            raw = output.outputs[0].text
            try:
                parsed = parse_model_output(rec["id"], raw)
                findings = [p for s in parsed.sentence_list for p in s.new]
                status = "ok"
            except Exception as exc:  # noqa: BLE001 - want to see everything, keep going
                findings = []
                status = f"parse_error: {exc}"
                num_parse_failures += 1

            results.append(
                {
                    "id": rec["id"],
                    "study_id": rec.get("study_id", rec["id"]),
                    "generated_report": rec["generated_report"],
                    "findings": findings,
                    "status": status,
                }
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
