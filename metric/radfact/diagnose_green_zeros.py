#!/usr/bin/env python3
# One-off diagnostic: for a sample of R2GenGPT predictions whose GREEN score came out
# exactly 0, recompute GREEN directly (not through the RadEval wrapper, which discards the
# raw LLM analysis text) and inspect whether "[Matched Findings]:" actually parsed from the
# model's own output, or whether compute_green() silently defaulted to 0 because the format
# didn't match. Distinguishes "genuinely bad report" from "parsing swallowed a failure."
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "RadEval"))
from radeval.metrics.green_score.green import GREEN

with open("/home/seona/prj-brl/baseline_rrg/R2GenGPT/results/r2gengpt_delta_mimic_test/predictions.json") as f:
    preds = json.load(f)

zero_samples = [p for p in preds if p.get("radeval_scores", {}).get("green") == 0][:20]
print(f"Recomputing GREEN directly for {len(zero_samples)} zero-score samples...\n")

scorer = GREEN("StanfordAIMI/GREEN-radllama2-7b", output_dir=".")
refs = [p["gt_report"] for p in zero_samples]
hyps = [p["generated_report"] for p in zero_samples]
mean, std, scores, df = scorer(refs, hyps)

num_format_missing = 0
num_genuine_zero = 0
for i, (score, analysis) in enumerate(zip(scores, df["green_analysis"])):
    has_matched_findings_section = "[Matched Findings]" in analysis
    verdict = "FORMAT_MISSING (regex couldn't find the section at all)" if not has_matched_findings_section else "genuine (section present, model said 0)"
    if not has_matched_findings_section:
        num_format_missing += 1
    else:
        num_genuine_zero += 1
    print(f"--- sample {i} --- score={score} --- {verdict}")
    print(analysis[:400].replace("\n", " "))
    print()

print("=" * 80)
print(f"format-missing (parse failure, silently -> 0): {num_format_missing}/{len(zero_samples)}")
print(f"genuine zero (model explicitly said 0 matched findings): {num_genuine_zero}/{len(zero_samples)}")
