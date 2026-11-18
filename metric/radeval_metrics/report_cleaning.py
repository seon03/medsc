# -----------------------------------------------------------------------------------------
# Shared text-cleaning for GT reports before they're used as RadEval/CheXbert/RadGraph
# references. IU X-Ray reports keep NLM's original de-identification placeholders
# ("XXXX") in the raw text -- measured on the 590-study test split: 35.9% of GT reports
# contain at least one "XXXX". Left in, these get fed to RadGraph/CheXbert as if they were
# real words, which can perturb entity extraction and label classification on over a third
# of the dataset. Stripping them is a straightforward text cleanup, not a methodology
# choice that needs much justification -- "XXXX" was never meant to be evaluated against.
# -----------------------------------------------------------------------------------------

from __future__ import annotations

import re

_XXXX_RE = re.compile(r"\bX{2,}\b")
_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,])")


def clean_report(text: str) -> str:
    """Remove NLM de-identification placeholders ("XXXX", "XX", ...) and collapse the
    resulting whitespace/punctuation artifacts."""
    cleaned = _XXXX_RE.sub("", text)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    return cleaned.strip()
