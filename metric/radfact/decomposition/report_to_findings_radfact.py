# -----------------------------------------------------------------------------------------
# Dependency-light re-implementation of RadFact's report-to-phrases prompt construction
# (radfact/llm_utils/processor/structured_processor.py::QueryTemplate.format_messages),
# adapted to call a local HF `transformers` chat model instead of Azure/OpenAI via langchain.
#
# We deliberately do NOT `pip install radfact`: its pyproject.toml pulls in azureml-sdk,
# azure-keyvault, hi-ml-azure, langchain-openai, and pins pydantic==1.10.17, which conflicts
# with the pydantic 2.x already installed in this environment. Instead we read RadFact's
# actual prompt assets straight from the cloned repo at metric/radfact/RadEval and
# reproduce just the message-construction logic:
#
#   [system prompt (+ our own JSON format instructions)]
#   + for each of the 9 CXR few-shot examples: [human: findings_text] [assistant: JSON]
#   + [human: the real query]
#
# Reused as-is from RadFact (MIT licensed, microsoft/RadFact):
#   - the CXR system prompt text (one-finding-per-sentence splitting rules)
#   - the 9 CXR few-shot examples
#   - the {orig, new[]} sentence_list output schema
# -----------------------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path

from decomposition.schema import ParsedReport

RADFACT_CXR_PROMPTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "RadFact/src/radfact/llm_utils/report_to_phrases/prompts/cxr"
)

# RadFact's own pipeline gets this "for free" from langchain's PydanticOutputParser
# (`parser.get_format_instructions()`, appended to the system prompt in
# StructuredProcessor.__init__). We spell it out manually here since we're not using
# langchain.
JSON_FORMAT_INSTRUCTIONS = """
Return your answer as a single JSON object with exactly this structure, and nothing else:
{
  "id": "<will be overwritten by the caller, put anything>",
  "sentence_list": [
    {"orig": "<verbatim original sentence from the report>", "new": ["<one or more one-finding-per-sentence phrases derived from it>"]}
  ]
}
""".strip()


def load_system_prompt() -> str:
    base_prompt = (RADFACT_CXR_PROMPTS_DIR / "system_message.txt").read_text()
    return f"{base_prompt}\n\n{JSON_FORMAT_INSTRUCTIONS}"


def load_few_shot_messages() -> list[dict[str, str]]:
    """Mirrors QueryTemplate.prepare_few_shot_examples: each example becomes a
    (human: findings_text) / (assistant: JSON of the parsed_report) message pair.
    """
    examples = json.loads((RADFACT_CXR_PROMPTS_DIR / "few_shot_examples.json").read_text())
    messages: list[dict[str, str]] = []
    for example in examples:
        parsed_report = example["parsed_report"]
        assistant_json = json.dumps(
            {"id": parsed_report["id"], "sentence_list": parsed_report["sentence_list"]},
            ensure_ascii=False,
        )
        messages.append({"role": "user", "content": example["findings_text"]})
        messages.append({"role": "assistant", "content": assistant_json})
    return messages


def build_messages(findings_text: str) -> list[dict[str, str]]:
    """Build the full chat message list for one report, matching
    QueryTemplate.format_messages's ordering: system, few-shot pairs, then the query.
    """
    messages = [{"role": "system", "content": load_system_prompt()}]
    messages.extend(load_few_shot_messages())
    messages.append({"role": "user", "content": findings_text})
    return messages


def parse_model_output(report_id: str, raw_text: str) -> ParsedReport:
    """Parse the model's raw text output into a ParsedReport.

    Tolerates markdown code fences and any leading/trailing commentary around the
    JSON object (smaller/less-instruction-tuned models often add these even when
    told not to).
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
    payload["id"] = report_id  # always use our own id, don't trust the model's copy of it
    return ParsedReport.model_validate(payload)
