# -----------------------------------------------------------------------------------------
# Small helper to strip section-header labels out of our MIMIC-CXR annotation.json report
# strings for RadFact decomposition, keeping BOTH impression and findings content (our
# MedGemma generation prompt explicitly asks for "findings and impression ... as a single
# continuous report", matching the standard R2Gen-style convention of treating the whole
# report as one generation/evaluation target -- so decomposition should cover all of it,
# not just the findings section).
#
# This dataset's report strings always use the pattern "impression: ... Findings: ..."
# (impression BEFORE findings) when both are present. An earlier version of this function
# only kept text AFTER "findings:", which silently dropped the (often clinically important,
# e.g. progression/diagnosis calls) impression sentence whenever it preceded "Findings:" --
# verified on the 20-sample pilot: 17/20 reports have this exact impression-then-findings
# shape, 3/20 have no header at all (already plain findings-only text, used as-is).
# -----------------------------------------------------------------------------------------

from __future__ import annotations

import re

_SECTION_HEADER_RE = re.compile(r"(impression|findings)\s*:", re.IGNORECASE)
_FINDINGS_HEADER_RE = re.compile(r"findings\s*:", re.IGNORECASE)


def extract_findings_text(report: str) -> str:
    return _SECTION_HEADER_RE.sub("", report).strip()


def extract_findings_only(report: str) -> str:
    """Keep only the text AFTER the "Findings:" header, dropping any preceding
    "Impression:" section entirely. For the Qwen3-VL-8B multi-view SFT, whose target is
    explicitly findings-only (not findings+impression) -- unlike extract_findings_text(),
    which keeps both. Falls back to the whole (header-stripped) report when there's no
    "Findings:" header at all (same ~3/20 case noted in extract_findings_text's docstring:
    already plain findings-only text with no header)."""
    parts = _FINDINGS_HEADER_RE.split(report, maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()
    return extract_findings_text(report)
