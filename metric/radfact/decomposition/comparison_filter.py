# -----------------------------------------------------------------------------------------
# Filters out GT findings that depend on a prior-study comparison the model was never
# shown (single-view/frontal-only evaluation has no prior image -- unlike the multi-view
# setup in convert_to_llamafactory_multiview.py, which was built specifically because
# 62.6% of findings-only targets contain this kind of ungrounded comparison language).
#
# Keyword-based, not LLM-based: fast, free, auditable. Deliberately conservative --
# "increased"/"decreased" alone are NOT included, since they're ambiguous (can describe
# current severity without implying a prior comparison, e.g. "increased density in the
# right lower lobe"); only terms that strongly and specifically imply a temporal/prior
# reference are listed.
# -----------------------------------------------------------------------------------------

from __future__ import annotations

import re

COMPARISON_KEYWORDS = [
    "prior", "previous", "preceding", "earlier",
    "compared", "comparison",
    "unchanged", "interval change", "interval",
    "redemonstrat",  # redemonstrated / redemonstrates / redemonstration
    "again seen", "again noted", "again demonstrated", "again identified",
    "persist",  # persistent / persists / persisting
    "resolved", "resolution of",
    "new since", "progression", "progressed", "progressive",
    "improved", "improving", "improvement",
    "worsened", "worsening",
    "stable",
]
_PATTERN = re.compile("|".join(re.escape(k) for k in COMPARISON_KEYWORDS), re.IGNORECASE)


def is_comparison_dependent(finding: str) -> bool:
    return bool(_PATTERN.search(finding))


def filter_comparison_findings(findings: list[str]) -> tuple[list[str], list[str]]:
    """Returns (kept, removed)."""
    kept, removed = [], []
    for f in findings:
        (removed if is_comparison_dependent(f) else kept).append(f)
    return kept, removed
