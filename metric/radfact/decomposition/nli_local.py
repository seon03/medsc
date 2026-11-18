# -----------------------------------------------------------------------------------------
# Dependency-light re-implementation of RadFact's NLI (entailment verification) prompt,
# analogous to report_to_phrases_local.py: reuse RadFact's actual prompt assets, call a
# local HF `transformers`/vLLM chat model instead of Azure/OpenAI via langchain.
#
# Reused as-is from RadFact (MIT licensed, microsoft/RadFact):
#   - the CXR entailment-verification system prompt text
#   - the 10 CXR few-shot examples (used here in their native bidirectional
#     {phrases_A, phrases_B} -> {phrases_A_evidenced, phrases_B_evidenced} shape, NOT
#     exploded into per-phrase examples the way RadFact's own single-phrase processor does)
#
# See nli_schema.py for why we use one bidirectional call per report pair instead of
# RadFact's per-phrase calls.
# -----------------------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path

from decomposition.nli_schema import BidirectionalEvidence, EvidencedPhrase, RadFactScore

RADFACT_NLI_PROMPTS_DIR = (
    Path(__file__).resolve().parents[1] / "RadFact/src/radfact/llm_utils/nli/prompts/cxr"
)

JSON_FORMAT_INSTRUCTIONS = """
You will be given two lists of phrases from two chest X-ray reports: "phrases_A" (the
candidate report) and "phrases_B" (the reference report).

For EVERY phrase in phrases_A, decide whether it is entailed by (true according to) the
phrases in phrases_B taken together, and vice versa for every phrase in phrases_B against
phrases_A. Use status "entailment" if true given the other report, "contradiction" if the
other report says the opposite, "neutral" if the other report says nothing relevant.

Return your answer as a single JSON object with exactly this structure, and nothing else:
{
  "phrases_A_evidenced": [
    {"phrase": "<one of phrases_A, verbatim>", "evidence": ["<phrase(s) from phrases_B supporting the status, verbatim>"], "status": "entailment" | "contradiction" | "neutral"}
  ],
  "phrases_B_evidenced": [
    {"phrase": "<one of phrases_B, verbatim>", "evidence": ["<phrase(s) from phrases_A supporting the status, verbatim>"], "status": "entailment" | "contradiction" | "neutral"}
  ]
}
Every phrase in phrases_A must appear exactly once in phrases_A_evidenced (and likewise for
phrases_B). Entailed or contradicted phrases must have at least one evidence item from the
OTHER list. Neutral phrases must have an empty evidence list.
""".strip()


def load_system_prompt() -> str:
    base_prompt = (RADFACT_NLI_PROMPTS_DIR / "system_message_ev_singlephrase.txt").read_text()
    return f"{base_prompt}\n\n{JSON_FORMAT_INSTRUCTIONS}"


def load_few_shot_messages() -> list[dict[str, str]]:
    examples = json.loads((RADFACT_NLI_PROMPTS_DIR / "few_shot_examples.json").read_text())
    messages: list[dict[str, str]] = []
    for example in examples:
        human = json.dumps(example["input"], ensure_ascii=False)
        ai = json.dumps(example["output"], ensure_ascii=False)
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": ai})
    return messages


def build_messages(candidate_phrases: list[str], reference_phrases: list[str]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": load_system_prompt()}]
    messages.extend(load_few_shot_messages())
    query = json.dumps(
        {"phrases_A": candidate_phrases, "phrases_B": reference_phrases}, ensure_ascii=False
    )
    messages.append({"role": "user", "content": query})
    return messages


def parse_model_output(
    raw_text: str, candidate_phrases: list[str], reference_phrases: list[str]
) -> BidirectionalEvidence:
    """Parse + validate the model's bidirectional-NLI JSON output.

    RadFact's own per-phrase processor only ever asks about one phrase at a time, so it
    can't silently drop one -- our one-call-per-report-pair approach can, if the model
    forgets a phrase or merges two. We don't reconstruct per original RadFact (no retry
    budget here), but we DO require phrases_A_evidenced/phrases_B_evidenced to cover every
    input phrase exactly once -- any other count is treated as a parse failure (status=
    parse_error upstream) rather than silently scored on a partial phrase list, which would
    otherwise inflate precision/recall by shrinking the denominator.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[len("json") :]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output: {raw_text!r}")
    payload = json.loads(text[start : end + 1])
    evidence = BidirectionalEvidence.model_validate(payload)

    if len(evidence.phrases_A_evidenced) != len(candidate_phrases):
        raise ValueError(
            f"phrases_A_evidenced has {len(evidence.phrases_A_evidenced)} entries, "
            f"expected {len(candidate_phrases)} (one per candidate phrase)."
        )
    if len(evidence.phrases_B_evidenced) != len(reference_phrases):
        raise ValueError(
            f"phrases_B_evidenced has {len(evidence.phrases_B_evidenced)} entries, "
            f"expected {len(reference_phrases)} (one per reference phrase)."
        )
    return evidence


def compute_radfact_score(evidence: BidirectionalEvidence) -> RadFactScore:
    def entailed_fraction(phrases: list[EvidencedPhrase]) -> float:
        if not phrases:
            return float("nan")
        return sum(1 for p in phrases if p.status == "entailment") / len(phrases)

    return RadFactScore(
        logical_precision=entailed_fraction(evidence.phrases_A_evidenced),
        logical_recall=entailed_fraction(evidence.phrases_B_evidenced),
        num_candidate_phrases=len(evidence.phrases_A_evidenced),
        num_reference_phrases=len(evidence.phrases_B_evidenced),
    )
