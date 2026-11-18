# -----------------------------------------------------------------------------------------
# RadGraph-based atomic finding reconstruction: run the `radgraph` package on report text,
# then regroup its entity/relation graph into a human-readable finding list by attaching
# each Observation entity's located_at anatomy and modify-relation modifiers -- the
# "RadGraph Option B" reconstruction discussed in this project's initial research phase.
#
# Verified against a real radgraph-xl call (2026-10-10) -- its actual schema differs from
# what an earlier version of this file assumed:
#   - entity["label"] is a verbose string like "Observation::definitely present" /
#     "Observation::definitely absent" / "Observation::uncertain" / "Anatomy::definitely
#     present" -- NOT the short "OBS-DP"/"ANAT-DP" codes from the original RadGraph paper.
#   - entity["tokens"] is a plain string (e.g. "cardiomegaly"), not a list of tokens.
#   - the "modify" relation points FROM the modifier word TO the thing it modifies (e.g.
#     "Mild" --modify--> "cardiomegaly"), not the other way around. The modifier word is
#     itself tagged as its own Observation entity, so it must be excluded from the finding
#     list (it's not an independent finding) and instead attached as a modifier to whatever
#     it points at.
# -----------------------------------------------------------------------------------------

from __future__ import annotations

STATUS_BY_LABEL = {
    "Observation::definitely present": "present",
    "Observation::definitely absent": "absent",
    "Observation::uncertain": "uncertain",
}


def _lookup_entity(entities: dict, idx) -> dict | None:
    """Entity/relation target indices come back as whichever of str/int RadGraph used;
    try both since this has been inconsistent across radgraph versions."""
    return entities.get(idx) or entities.get(str(idx))


def reconstruct_findings(entities: dict) -> list[dict]:
    """Group each observation entity with its located_at anatomy and modify modifiers,
    from one report's RadGraph entity dict (the "entities" value of one RadGraph(...)
    output item)."""
    # Entities that are themselves the SOURCE of a "modify" relation are modifier words
    # (e.g. "Mild"), not independent findings -- exclude them from the finding list.
    modifier_source_ids = {
        idx for idx, ent in entities.items() for rel_type, _ in ent.get("relations", []) if rel_type == "modify"
    }

    findings = []
    for idx, ent in entities.items():
        if idx in modifier_source_ids:
            continue
        if ent["label"] not in STATUS_BY_LABEL:
            continue  # e.g. Anatomy-only entities with no observation status

        location = []
        for rel_type, tgt_idx in ent.get("relations", []):
            tgt = _lookup_entity(entities, tgt_idx)
            if tgt is not None and rel_type == "located_at" and tgt["label"].startswith("Anatomy"):
                location.append(tgt["tokens"])

        # modify relations point INTO this entity (modifier -> target), so look for
        # other entities whose relation list names this entity's id as the target.
        modifiers = [
            other_ent["tokens"]
            for other_ent in entities.values()
            for rel_type, tgt_idx in other_ent.get("relations", [])
            if rel_type == "modify" and (tgt_idx == idx or str(tgt_idx) == str(idx))
        ]

        findings.append(
            {
                "observation": ent["tokens"],
                "status": STATUS_BY_LABEL[ent["label"]],
                "location": location,
                "modifiers": modifiers,
            }
        )
    return findings
