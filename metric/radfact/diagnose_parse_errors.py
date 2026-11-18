#!/usr/bin/env python3
# One-off: decompose a few of R2GenGPT's remaining candidate parse_error reports and print
# the RAW (unparsed) LLM output + its length in tokens, to check whether failures are
# truncation (hitting max_tokens mid-JSON) or a formatting mistake elsewhere in the text.
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vllm import LLM, SamplingParams
from decomposition.mimic_text_utils import extract_findings_text
from decomposition.report_to_findings_radfact import build_messages

MODEL_PATH = (
    "/mnt/.cache/huggingface/models/models--meta-llama--Meta-Llama-3-70B-Instruct/"
    "snapshots/50fd307e57011801c7833c87efa1984ddf2db42f"
)

with open("/home/seona/prj-brl/baseline_rrg/R2GenGPT/results/r2gengpt_delta_mimic_test/radfact_candidate_findings.json") as f:
    d = json.load(f)
failures = [r for r in d if r["status"] != "ok"][:8]
print(f"Decomposing {len(failures)} still-failing candidates at temperature=0.0, max_tokens=1024 (same as main run) to inspect raw output...\n")

llm = LLM(model=MODEL_PATH, tensor_parallel_size=4, dtype="bfloat16", enable_prefix_caching=True)
sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)

conversations = [build_messages(extract_findings_text(r["generated_report"])) for r in failures]
outputs = llm.chat(conversations, sampling_params)

for r, output in zip(failures, outputs):
    raw = output.outputs[0].text
    num_tokens = len(output.outputs[0].token_ids)
    finish_reason = output.outputs[0].finish_reason
    print(f"--- id={r['id']} --- tokens_generated={num_tokens} finish_reason={finish_reason}")
    print(f"error was: {r['status'][:150]}")
    print(f"raw output length: {len(raw)} chars")
    print(f"raw output tail (last 300 chars): ...{raw[-300:]!r}")
    print()
