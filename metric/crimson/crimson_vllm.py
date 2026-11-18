#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------
# vLLM-batched version of crimson.py. The plain `pipeline()` version (api="hf", default)
# generates ONE sample at a time -- measured ~17.4s/sample on qwen3vl8b_base_mimic_test3858
# (sample 1252/3858 after 6h), i.e. ~19h for that run_dir alone. CRIMSONScore already ships
# an api="vllm" mode (CRIMSON/generate_score.py) using vLLM's continuous batching, same fix
# as metric/radfact/scripts/decompose_radfact_predictions_vllm.py applied to RadFact.
#
# Unlike the plain crimson.py, this checkpoints after every CHUNK (not just at the end of a
# run_dir) to <run_dir>/evidence/crimson_partial.json, and resumes from there if interrupted
# -- a run_dir with thousands of samples at ~a few seconds/sample amortized is still minutes
# long, and losing that to a kill partway through would be wasteful.
#
# Skips any run_dir that already has a "crimson" key in scores.json (idempotent -- safe to
# pass the same run_dirs again after a partial run).
#
# Requires the CRIMSON repo importable (sys.path, no pip install needed) + vllm installed --
# use the "vllm-decompose" env (already has vllm; CRIMSON's api="vllm" path only needs vllm
# itself, not its HF-pipeline deps like peft/accelerate/sentence-transformers).
#
# Usage:
#   conda activate vllm-decompose
#   cd metric/crimson
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python3 crimson_vllm.py <run_dir> [run_dir2 ...]
# -----------------------------------------------------------------------------------------

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

METRIC_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(METRIC_DIR / "CRIMSON"))
from CRIMSON import CRIMSONScore  # noqa: E402

sys.path.insert(0, str(METRIC_DIR / "radeval_metrics"))
from common import load_predictions, save_evidence, save_results  # noqa: E402

METRIC = "crimson"
PER_SAMPLE_KEYS = ["crimson"]
CHUNK_SIZE = 500

# Local snapshot path for rajpurkarlab/medgemma-4b-it-crimson -- moved here from the
# default HF cache (~/.cache/huggingface/hub/) into the team's shared cache convention
# (2026-11-18). Passing this directly (not the bare HF repo id) skips CRIMSONScore's
# resolve_model_for_vllm() hub lookup entirely, so it loads straight from disk instead of
# re-downloading/re-symlinking into /tmp on every run.
MODEL_PATH = (
    "/mnt/.cache/huggingface/models/models--rajpurkarlab--medgemma-4b-it-crimson/"
    "snapshots/02e8a99bb55fe427bab965222c0c60ee0e76b125"
)


def already_done(run_dir: Path) -> bool:
    scores_path = run_dir / "scores.json"
    if not scores_path.exists():
        return False
    return "crimson" in json.loads(scores_path.read_text())


def load_partial(run_dir: Path) -> tuple[list[float], list[dict]]:
    partial_path = run_dir / "evidence" / "crimson_partial.json"
    if not partial_path.exists():
        return [], []
    data = json.loads(partial_path.read_text())
    return data["scores"], data["evidence"]


def save_partial(run_dir: Path, scores: list[float], evidence: list[dict]) -> None:
    partial_path = run_dir / "evidence" / "crimson_partial.json"
    partial_path.parent.mkdir(exist_ok=True)
    partial_path.write_text(json.dumps({"scores": scores, "evidence": evidence}, indent=2, ensure_ascii=False))


def main() -> None:
    run_dirs = [Path(p) for p in sys.argv[1:]]
    if not run_dirs:
        print(f"Usage: python3 {Path(__file__).name} <run_dir> [run_dir2 ...]")
        sys.exit(1)

    pending = [d for d in run_dirs if not already_done(d)]
    skipped = [d for d in run_dirs if d not in pending]
    for d in skipped:
        print(f"Skipping {d} -- crimson already in scores.json")
    if not pending:
        print("Nothing to do.")
        return

    print(f"Loading rajpurkarlab/medgemma-4b-it-crimson via vLLM from {MODEL_PATH} ...")
    scorer = CRIMSONScore(api="vllm", model_name=MODEL_PATH)

    for run_dir in pending:
        refs, hyps = load_predictions(run_dir)
        scores, evidence = load_partial(run_dir)
        start_idx = len(scores)
        if start_idx:
            print(f"\n=== {run_dir} (N={len(refs)}) resuming from {start_idx} (checkpoint found) ===")
        else:
            print(f"\n=== {run_dir} (N={len(refs)}) metric={METRIC} ===")

        for chunk_start in range(start_idx, len(refs), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(refs))
            ref_chunk = refs[chunk_start:chunk_end]
            hyp_chunk = hyps[chunk_start:chunk_end]

            start = time.monotonic()
            # include_guidelines=False explicitly: CRIMSONScore normally infers this itself
            # by checking `model_name == DEFAULT_HF_MODEL` (the bare HF repo id string), but
            # we pass a local snapshot path instead (see MODEL_PATH above) so that check
            # would silently fail and flip guidelines back on, changing the prompt (and
            # therefore scores) vs. every run_dir already scored today. Must match.
            results = scorer.evaluate_batch(ref_chunk, hyp_chunk, include_guidelines=False)
            elapsed = time.monotonic() - start

            for r in results:
                if r is None:
                    scores.append(float("nan"))
                    evidence.append({})
                else:
                    scores.append(r["crimson_score"])
                    evidence.append(r)

            save_partial(run_dir, scores, evidence)
            n = chunk_end - chunk_start
            print(
                f"  [{chunk_end}/{len(refs)}] {elapsed:.1f}s "
                f"({elapsed / n:.2f}s/sample amortized) -- checkpointed"
            )

        results_out = {"crimson": scores}
        save_results(run_dir, results_out, PER_SAMPLE_KEYS, METRIC)
        save_evidence(run_dir, METRIC, {"per_sample": evidence})
        # Final save_results/save_evidence above is now the source of truth; drop the
        # partial checkpoint so a later re-run doesn't think there's unfinished work.
        partial_path = run_dir / "evidence" / "crimson_partial.json"
        partial_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
