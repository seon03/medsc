# -----------------------------------------------------------------------------------------
# Minimal pydantic (v2) mirror of RadFact's bidirectional NLI output schema
# (radfact/llm_utils/nli/schema.py: EvidencedPhrase, BidirectionalEvidence), plus a
# RadFactScore that reproduces radfact/metric/schema.py::RadFactScore's precision/recall
# definition (entailed_fraction per direction).
#
# We use the BIDIRECTIONAL format directly (one LLM call per candidate/reference pair)
# rather than RadFact's own per-phrase "exploded" single-phrase calls
# (radfact/llm_utils/nli/processor.py::get_ev_processor_singlephrase, which issues one call
# PER PHRASE for extra robustness). That's ~N_candidate + N_reference calls per report --
# too expensive for our 10-sample prototype. We're trading a bit of per-phrase reliability
# for far fewer LLM calls; worth revisiting if evidence/status quality turns out too noisy.
# -----------------------------------------------------------------------------------------

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

NLIStatus = Literal["entailment", "contradiction", "neutral"]


class EvidencedPhrase(BaseModel):
    phrase: str
    evidence: list[str]
    status: NLIStatus


class BidirectionalEvidence(BaseModel):
    phrases_A_evidenced: list[EvidencedPhrase]  # A = candidate
    phrases_B_evidenced: list[EvidencedPhrase]  # B = reference


class RadFactScore(BaseModel):
    """Mirrors radfact/metric/schema.py::RadFactScore's logical precision/recall, i.e. the
    fraction of candidate phrases entailed by the reference (precision) and the fraction of
    reference phrases entailed by the candidate (recall)."""

    logical_precision: float
    logical_recall: float
    num_candidate_phrases: int
    num_reference_phrases: int

    @property
    def logical_f1(self) -> float:
        p, r = self.logical_precision, self.logical_recall
        if p != p or r != r:  # NaN check (no phrases on one side)
            return float("nan")
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
