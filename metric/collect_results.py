#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------
# Regenerates RRG_RESULTS.md from each registered baseline's <run_dir>/scores.json. scores.json
# stores per-sample lists (the project's per_sample=True contract -- see radeval_metrics/
# common.py), so the mean is computed here rather than trusted from hand-copied numbers.
#
# Four tables: Qwen3-VL-8B-Instruct and MedGemma-4B, each on MIMIC-CXR test and IU X-Ray
# test. Columns: RadFact-F1, CRIMSON, 1/RadCliQ-v1 (inverted so higher-is-better, matching
# the other columns), RadGraph-F1, RaTEScore, GREEN.
#
# A row can have a run_dir (we computed it), a literature-reference value (from ReXrank or
# another paper, not something we ran), or both. When both exist for the same cell, both
# are shown as "computed (구현) / literature*" so they're never silently conflated --
# CRIMSON and RaTEScore/GREEN for MedGemma-4B/base on IU X-Ray are examples where we only
# have some metrics computed ourselves and fill the rest from ReXrank's published numbers.
#
# To add a baseline we evaluated ourselves: add one entry to the relevant ROW_TO_RUN_DIR.
# To add a literature number: add/extend an entry in LITERATURE_REFERENCE_ROWS. Rows with
# neither render blank, not removed, so the table keeps its place for baselines still
# planned.
#
# Usage: python3 collect_results.py
# -----------------------------------------------------------------------------------------

from __future__ import annotations

import json
import statistics
from pathlib import Path

METRIC_DIR = Path(__file__).resolve().parent
RESULTS_PATH = METRIC_DIR.parent / "results/RRG_RESULTS.md"
BASE_MODEL_DIR = METRIC_DIR.parent / "base_model"

COLUMNS = ["radfact_f1", "crimson", "inv_radcliq", "radgraph", "ratescore", "green"]

# ---- Qwen3-VL-8B-Instruct ----
QWEN_MIMIC_ROW_TO_RUN_DIR: dict[str, Path] = {
    "base": BASE_MODEL_DIR / "Qwen/results/qwen3vl8b_base_mimic_test3858",
    "SFT (mimic-cxr)": BASE_MODEL_DIR / "Qwen/results/qwen3vl8b_sft_mimic_test3858",
    "Prompt_SC": METRIC_DIR.parent / "baseline_sc/Prompt-SC/results/qwen3vl_sft_rci_full_test",
}
QWEN_IUXRAY_ROW_TO_RUN_DIR: dict[str, Path] = {
    "base": BASE_MODEL_DIR / "Qwen/results/qwen3vl8b_base_iuxray_test590",
    "SFT (mimic-cxr)": BASE_MODEL_DIR / "Qwen/results/qwen3vl8b_sft_iuxray_test590",
}
QWEN_MIMIC_ALL_ROWS = [
    "base", "SFT (mimic-cxr)", "UniRG-CXR", "EditGRPO", "Prompt_SC", "SFT_SC",
    "SC-Captioner (RL_SC)", "SFT_SC + RL_SC", "+ 다른 sc method", "Ours",
]
QWEN_IUXRAY_ALL_ROWS = ["base", "SFT (mimic-cxr)", "UniRG-CXR", "Ours"]

# ---- MedGemma-4B ----
MEDGEMMA_MIMIC_ROW_TO_RUN_DIR: dict[str, Path] = {}  # no run_dir matched the literature numbers below
MEDGEMMA_IUXRAY_ROW_TO_RUN_DIR: dict[str, Path] = {
    "base": BASE_MODEL_DIR / "MedGemma/results/medgemma4b_iuxray_test590",
}
MEDGEMMA_MIMIC_ALL_ROWS = ["base", "Ours"]
MEDGEMMA_IUXRAY_ALL_ROWS = ["base", "Ours"]

# label (as rendered in the table, including dataset context) -> {column: value}.
# Rendered with a "*" marker. When the SAME row also has a run_dir result for a given
# column, both are shown ("computed (구현) / literature*") instead of one silently
# overriding the other.
LITERATURE_REFERENCE_ROWS: dict[str, dict[str, float]] = {
    "UniRG-CXR/mimic": {
        "crimson": 0.181, "inv_radcliq": 1.180, "radgraph": 0.268,
        "ratescore": 0.602, "green": 0.360,
    },
    "UniRG-CXR/iuxray": {
        "crimson": 0.632, "inv_radcliq": 3.907, "radgraph": 0.396,
        "ratescore": 0.729, "green": 0.664,
    },
    "MedGemma-4B/base/mimic": {
        "crimson": 0.082, "inv_radcliq": 0.744, "radgraph": 0.159,
        "ratescore": 0.549, "green": 0.293,
    },
    "MedGemma-4B/base/iuxray": {
        "crimson": 0.561, "inv_radcliq": 1.339, "radgraph": 0.263,
        "ratescore": 0.675, "green": 0.706,
    },
}


def mean_of(scores: dict, key: str) -> float | None:
    v = scores.get(key)
    if isinstance(v, list) and v:
        # Some per-sample scorers (e.g. crimson) record NaN for samples whose LLM
        # response failed to parse -- exclude those rather than letting a single NaN
        # poison statistics.mean() for the whole run_dir.
        valid = [x for x in v if isinstance(x, (int, float)) and x == x]
        return statistics.mean(valid) if valid else None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def computed_values(run_dir: Path | None) -> dict[str, float | None]:
    if run_dir is None or not (run_dir / "scores.json").exists():
        return {}
    with open(run_dir / "scores.json") as f:
        s = json.load(f)
    radcliq = mean_of(s, "radcliq_v1")
    return {
        "radfact_f1": mean_of(s, "radfact_f1"),
        "crimson": mean_of(s, "crimson"),
        "inv_radcliq": (1 / radcliq) if radcliq else None,
        "radgraph": mean_of(s, "radgraph_partial"),
        "ratescore": mean_of(s, "ratescore"),
        "green": mean_of(s, "green"),
    }


def format_cell(computed: float | None, literature: float | None) -> str:
    if computed is not None and literature is not None:
        return f"{computed:.4f} (구현) / {literature:.4f}*"
    if computed is not None:
        return f"{computed:.4f}"
    if literature is not None:
        return f"{literature:.4f}*"
    return ""


def format_row(label: str, run_dir: Path | None, lit_key: str | None) -> str:
    computed = computed_values(run_dir)
    literature = LITERATURE_REFERENCE_ROWS.get(lit_key, {}) if lit_key else {}
    cells = [format_cell(computed.get(col), literature.get(col)) for col in COLUMNS]
    return f"| {label} | " + " | ".join(cells) + " |"


def render_table(
    title: str, rows: list[str], row_to_run_dir: dict[str, Path], lit_dataset_suffix: str, lit_prefix: str = ""
) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Model | RadFact-F1 | CRIMSON | 1/RadCliQ-v1 | RadGraph | RaTEScore | GREEN |",
        "|---|---|---|---|---|---|---|",
    ]
    for label in rows:
        candidate_key = f"{lit_prefix}{label}/{lit_dataset_suffix}"
        lit_key = candidate_key if candidate_key in LITERATURE_REFERENCE_ROWS else None
        lines.append(format_row(label, row_to_run_dir.get(label), lit_key))
    lines.append("")
    return lines


def main() -> None:
    lines = [
        "# 베이스라인별 평가 결과 요약",
        "",
        "각 셀의 숫자는 해당 baseline의 `<run_dir>/scores.json`에 저장된 per-sample 값의 mean입니다",
        "(원본은 그 파일에 있고, 여기는 집계된 요약본 -- 직접 손으로 고치지 말고",
        "`python3 collect_results.py`로 재생성할 것). `*` 표시된 값은 저희가 직접 돌린 게 아니라",
        "외부 출처(ReXrank, 논문 등)에서 그대로 가져온 참고값입니다. 둘 다 있는 셀은",
        "\"계산값 (구현) / 참고값*\" 형태로 같이 보여줍니다.",
        "",
        "## Report Generation (MIMIC-CXR test)",
        "",
    ]
    lines += render_table("Qwen3-VL-8B-Instruct", QWEN_MIMIC_ALL_ROWS, QWEN_MIMIC_ROW_TO_RUN_DIR, "mimic")
    lines += render_table(
        "MedGemma-4B", MEDGEMMA_MIMIC_ALL_ROWS, MEDGEMMA_MIMIC_ROW_TO_RUN_DIR, "mimic", lit_prefix="MedGemma-4B/"
    )

    lines += [
        "## Report Generation (IU X-Ray test)",
        "",
    ]
    lines += render_table("Qwen3-VL-8B-Instruct", QWEN_IUXRAY_ALL_ROWS, QWEN_IUXRAY_ROW_TO_RUN_DIR, "iuxray")
    lines += render_table(
        "MedGemma-4B", MEDGEMMA_IUXRAY_ALL_ROWS, MEDGEMMA_IUXRAY_ROW_TO_RUN_DIR, "iuxray", lit_prefix="MedGemma-4B/"
    )

    lines += [
        "## 업데이트 방법",
        "",
        "```bash",
        "cd metric",
        "python3 collect_results.py",
        "```",
        "",
        "새 baseline을 저희가 평가했으면 해당 `*_ROW_TO_RUN_DIR` 딕셔너리에 행 이름:run_dir 추가.",
        "외부 참고값을 추가/수정하려면 `LITERATURE_REFERENCE_ROWS`에 \"행 이름/mimic\" 또는",
        "\"행 이름/iuxray\" 키로 추가하면 됩니다.",
    ]

    RESULTS_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
