#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------
# vLLM-batched version of run_eval_radfact.py (same device_map="auto" + batch_size=1
# slowness as the other two scripts this mirrors -- decompose_radfact_vllm.py and
# decompose_radfact_predictions_vllm.py). nli_local.build_messages() already returns
# standard {"role", "content"} dicts, so no custom chat-template handling needed here
# (unlike GREEN's "from"/"value" format) -- vLLM's llm.chat() takes them directly.
#
# Requires the SEPARATE "vllm-decompose" conda env:
#   conda create -n vllm-decompose python=3.11 -y && conda activate vllm-decompose && pip install vllm
#
# Usage:
#   conda activate vllm-decompose
#   cd metric/radfact
#   CUDA_VISIBLE_DEVICES=4,5,6,7 python3 scripts/run_eval_radfact_vllm.py \
#       --gt_findings_path gt_cache/mimic-cxr_test_gt_radfact_findings.json \
#       --candidate_findings_path /path/to/run_dir/radfact_candidate_findings.json \
#       --output_file /path/to/run_dir/radfact_scores.json \
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

from decomposition.nli_local import build_messages, compute_radfact_score, parse_model_output

MODEL_PATH = (
    "/mnt/.cache/huggingface/models/models--meta-llama--Meta-Llama-3-70B-Instruct/"
    "snapshots/50fd307e57011801c7833c87efa1984ddf2db42f"
)  # matches RadFact's own README: "we used LLama-3-70B-Instruct" for entailment verification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_findings_path", required=True, help="Output of run_decompose_radfact.py / decompose_radfact_vllm.py")
    parser.add_argument(
        "--candidate_findings_path", required=True,
        help="Output of run_decompose_radfact_predictions.py / decompose_radfact_predictions_vllm.py",
    )
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1000)
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

    # Skip pairs where there's no matching GT group, OR where either side's upstream
    # decompose failed (status != "ok"). A decompose parse_error leaves findings=[], which
    # would otherwise be silently scored as "report genuinely has zero findings" -- for the
    # GT side this produces a spurious NaN recall (correctly dropped already) but a spurious
    # LOW real-valued recall for candidates with empty findings (the model is asked to match
    # nothing against a non-empty reference, scoring ~0) -- i.e. decompose failures were
    # silently counting as "model found nothing," pulling recall down for reasons that have
    # nothing to do with the model's report quality. Same risk symmetrically for GT-side
    # failures contaminating precision.
    pairs = []
    num_no_gt = 0
    num_bad_candidate = 0
    num_bad_gt = 0
    for cand in candidates:
        gt_entry = gt_lookup.get(cand["id"])
        if gt_entry is None:
            num_no_gt += 1
            continue
        gt_findings, gt_status = gt_entry
        if cand["status"] != "ok":
            num_bad_candidate += 1
            continue
        if gt_status != "ok":
            num_bad_gt += 1
            continue
        pairs.append((cand["id"], cand["findings"], gt_findings))
    print(
        f"{len(pairs)} candidate/GT pairs to score "
        f"({num_no_gt} no matching GT, {num_bad_candidate} candidate decompose failed, "
        f"{num_bad_gt} GT decompose failed -- all excluded from scoring).\n"
    )

    print(f"Loading Llama-3-70B-Instruct via vLLM from {MODEL_PATH} ...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

    results = []
    num_parse_failures = 0
    precisions, recalls, f1s = [], [], []
    start_all = time.monotonic()

    for batch_start in range(0, len(pairs), args.batch_size):
        batch = pairs[batch_start : batch_start + args.batch_size]
        conversations = [build_messages(cand_findings, gt_findings) for _, cand_findings, gt_findings in batch]

        start = time.monotonic()
        outputs = llm.chat(conversations, sampling_params)
        elapsed = time.monotonic() - start
        done = batch_start + len(batch)
        print(
            f"[batch {batch_start}-{done}/{len(pairs)}] {elapsed:.1f}s "
            f"({elapsed / len(batch):.2f}s/report amortized)"
        )

        for (cand_id, cand_findings, gt_findings), output in zip(batch, outputs):
            raw = output.outputs[0].text
            try:
                evidence = parse_model_output(raw, cand_findings, gt_findings)
                score = compute_radfact_score(evidence)
                status = "ok"
                if score.logical_precision == score.logical_precision:  # not NaN
                    precisions.append(score.logical_precision)
                if score.logical_recall == score.logical_recall:
                    recalls.append(score.logical_recall)
                if score.logical_f1 == score.logical_f1:
                    f1s.append(score.logical_f1)
            except Exception as exc:  # noqa: BLE001 - want to see everything, keep going
                score = None
                status = f"parse_error: {exc}"
                num_parse_failures += 1

            results.append(
                {
                    "id": cand_id,
                    "status": status,
                    "logical_precision": score.logical_precision if score else None,
                    "logical_recall": score.logical_recall if score else None,
                    "logical_f1": score.logical_f1 if score else None,
                }
            )

        # Checkpoint after every batch, not just at the end.
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    total_elapsed = time.monotonic() - start_all
    print("\n" + "=" * 80)
    print("요약 (참고용 -- 최종 집계는 fold_radfact_scores.py가 원본 RadFact 공식으로 다시 계산함)")
    print(f"  처리한 샘플 수: {len(results)}")
    print(f"  NLI 출력 파싱 실패: {num_parse_failures}")
    # RadFact's own aggregation (metric/schema.py::RadFactScore.from_aggregate): mean(P) and
    # mean(R) first, THEN the harmonic mean -- NOT mean(per-sample F1). See fold_radfact_scores.py.
    p_mean = sum(precisions) / len(precisions) if precisions else None
    r_mean = sum(recalls) / len(recalls) if recalls else None
    if p_mean is not None:
        print(f"  mean logical_precision: {p_mean:.4f} (n={len(precisions)})")
    if r_mean is not None:
        print(f"  mean logical_recall: {r_mean:.4f} (n={len(recalls)})")
    if p_mean is not None and r_mean is not None and (p_mean + r_mean) > 0:
        print(f"  logical_f1 (of mean P/R, RadFact's own formula): {2 * p_mean * r_mean / (p_mean + r_mean):.4f}")
    print(f"  총 소요 시간: {total_elapsed:.1f}s ({total_elapsed / len(results):.2f}s/report amortized)")
    print(f"  저장 위치: {args.output_file}")


if __name__ == "__main__":
    main()
