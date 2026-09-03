"""Batch Excel reporting. One summary workbook per batch run, listing every
subject: decision, whether the model corrected it, iterations, dim-fix, timing,
errors. Mirrors MotionDetectionProgram's summary and adds correction fields."""
from __future__ import annotations

import os
from typing import Dict, List

# columns that are large arrays / in-memory volumes -> never written to Excel
_DROP = {"p_slice", "slice_logits", "loop_history", "_corrected_volume"}


def _row(r: Dict) -> Dict:
    corrected = bool(r.get("corrected", False))
    return {
        "subject": r.get("subject", ""),
        "status": r.get("status", ""),
        "mode": r.get("mode", ""),
        "decision": r.get("decision", ""),
        "corrected": "YES" if corrected else "NO",
        "iterations": r.get("iterations", 0),
        "p_subject": _fmt(r.get("p_subject", r.get("p_subject_final"))),
        "p_subject_initial": _fmt(r.get("p_subject_initial")),
        "threshold": _fmt(r.get("threshold")),
        "topk": r.get("topk", ""),
        "n_bag": r.get("n_bag", ""),
        "orig_shape": _shape(r.get("orig_shape")),
        "dim_fixed": "YES" if r.get("dim_fixed") else "NO",
        "dim_op": r.get("dim_op", ""),
        "seconds": r.get("seconds", ""),
        "output_file": r.get("output_file", ""),
        "input_file": r.get("input_file", ""),
        "error": r.get("error", ""),
    }


def _fmt(v):
    return "" if v is None else (f"{float(v):.4f}" if isinstance(v, (int, float)) else v)


def _shape(s):
    return "x".join(str(x) for x in s) if s else ""


def write_summary(output_path: str, results: List[Dict]) -> None:
    import pandas as pd

    rows = [_row(r) for r in results]
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(output_path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Batch Summary", index=False)

        corr = [r for r in results if r.get("corrected")]
        skip = [r for r in results if not r.get("corrected") and r.get("status") == "ok"]
        errs = [r for r in results if r.get("status") == "error"]
        overview = pd.DataFrame({
            "metric": ["Total subjects", "Corrected", "Not corrected", "Errors"],
            "value": [len(results), len(corr), len(skip), len(errs)],
        })
        overview.to_excel(w, sheet_name="Overview", index=False)


def clean_for_json(result: Dict) -> Dict:
    """Strip large/in-memory fields so a result can be JSON-logged."""
    return {k: v for k, v in result.items() if k not in _DROP}
