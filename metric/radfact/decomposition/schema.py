# -----------------------------------------------------------------------------------------
# Minimal pydantic (v2) mirror of RadFact's report_to_phrases output schema
# (radfact/llm_utils/report_to_phrases/schema.py: SentenceWithRephrases, ParsedReport).
# RadFact itself pins pydantic==1.10.17, which conflicts with the pydantic 2.x already
# installed here, so we re-declare just the two fields we need instead of importing
# the radfact package.
# -----------------------------------------------------------------------------------------

from __future__ import annotations

from pydantic import BaseModel


class SentenceWithRephrases(BaseModel):
    """One original report sentence and its one-finding-per-sentence rephrasings."""

    orig: str
    new: list[str]


class ParsedReport(BaseModel):
    """A report decomposed into (orig sentence -> rephrased finding sentences) pairs."""

    id: str
    sentence_list: list[SentenceWithRephrases]
