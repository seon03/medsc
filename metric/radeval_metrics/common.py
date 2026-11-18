# -----------------------------------------------------------------------------------------
# Shared I/O helpers for the per-metric scripts in this folder (radgraph.py, f1chexbert.py,
# radcliq.py, ratescore.py, green.py). Each metric is its own standalone script (not one
# combined runner) so each can be read/run/tested in isolation -- the only thing they share
# is "how to read predictions.json" and "how to write results back", which lives here.
# -----------------------------------------------------------------------------------------

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from report_cleaning import clean_report


def load_predictions(run_dir: Path) -> tuple[list[str], list[str]]:
    """Read <run_dir>/predictions.json (the project-wide input contract: a list of
    {"id", "study_id", "gt_report", "generated_report"} records) and return
    (refs, hyps) as plain string lists, cleaned of NLM de-identification placeholders."""
    with open(run_dir / "predictions.json") as f:
        data = json.load(f)
    refs = [clean_report(rec["gt_report"]) for rec in data]
    hyps = [clean_report(rec["generated_report"]) for rec in data]
    return refs, hyps


def merge_per_sample_scores(run_dir: Path, results: dict, per_sample_keys: list[str]) -> None:
    """Write each report's own per-sample scores back into predictions.json, alongside
    the gt_report/generated_report it already has."""
    predictions_path = run_dir / "predictions.json"
    with open(predictions_path) as f:
        predictions = json.load(f)

    for key in per_sample_keys:
        values = results.get(key)
        if not isinstance(values, list):
            continue  # this key isn't actually per-sample here (unexpected, but don't crash)
        if len(values) != len(predictions):
            print(f"WARNING: {key} has {len(values)} values but {len(predictions)} predictions -- skipping merge for this key.")
            continue
        for pred, value in zip(predictions, values):
            pred.setdefault("radeval_scores", {})[key] = value

    with open(predictions_path, "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


def save_results(run_dir: Path, results: dict, per_sample_keys: list[str], metric_name: str) -> None:
    """Merge `results` into run_dir/scores.json (without wiping scores other metric
    scripts already wrote there), merge per-sample values into predictions.json, and
    record which metric/when into run_dir/config.json."""
    scores_path = run_dir / "scores.json"
    existing = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    existing.update(results)
    with open(scores_path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    merge_per_sample_scores(run_dir, results, per_sample_keys)

    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    config.setdefault("scoring", {}).setdefault("metrics", [])
    if metric_name not in config["scoring"]["metrics"]:
        config["scoring"]["metrics"].append(metric_name)
    config["scoring"]["tool"] = "radeval (official jbdel/RadEval v2.2.2)"
    config["scoring"][f"{metric_name}_scored_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))

    print(
        f"Saved {metric_name} -> {scores_path}, per-sample merged into "
        f"{run_dir / 'predictions.json'}, recorded in {config_path}"
    )


def save_evidence(run_dir: Path, metric_name: str, evidence: dict) -> None:
    """Persist a metric's raw intermediate evidence (e.g. RadGraph's extracted
    entity/relation graphs, GREEN's LLM error-analysis text, CheXbert's label
    vectors) to <run_dir>/evidence/<metric_name>.json -- kept separate from the
    lean scores.json summary since this data is bulky and per-sample."""
    def _json_default(obj):
        # numpy scalars (e.g. from pandas-derived error-count columns) aren't
        # natively JSON-serializable -- coerce via .item() if present.
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    evidence_path = evidence_dir / f"{metric_name}.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False, default=_json_default)
    print(f"Saved {metric_name} evidence -> {evidence_path}")
