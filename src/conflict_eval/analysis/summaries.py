"""Descriptive summaries over real pilot results.

Every function here raises on empty input rather than returning an empty
or fabricated-looking table, per docs/decisions.md and
docs/phase2_research_design.md ("Figures"): nothing here should be able to
produce output that looks like a real summary but was not built from
recorded generations.
"""

from __future__ import annotations

import pandas as pd


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    if not records:
        raise ValueError("Cannot build a summary from zero records — run the real pilot first.")
    return pd.DataFrame(records)


def condition_summary(records: list[dict]) -> pd.DataFrame:
    """C0-C4 rates by model and knowledge group (Plot 3's underlying
    table). conflict_status is retained per row so agreement and conflict
    conditions stay visibly distinguished.
    """
    df = records_to_frame(records)
    grouped = (
        df.groupby(["model_id", "knowledge_group", "condition", "conflict_status"])
        .agg(
            n=("context_adopted", "size"),
            context_adoption_rate=("context_adopted", "mean"),
            final_accuracy=("final_correct", "mean"),
            abstention_rate=("decision", lambda s: (s == "uncertain").mean()),
        )
        .reset_index()
    )
    return grouped


def override_summary(records: list[dict]) -> pd.DataFrame:
    """Corrective vs. harmful override, by source role (Plot 2's
    underlying table).
    """
    from conflict_eval.evaluation.metrics import (
        corrective_override_rate,
        harmful_override_rate,
    )

    rows = []
    for model_id in sorted({r["model_id"] for r in records}):
        model_records = [r for r in records if r["model_id"] == model_id]
        for source_role in ("preferred", "dispreferred"):
            subset = [r for r in model_records if r.get("source_role") == source_role]
            hor = harmful_override_rate(subset)
            cor = corrective_override_rate(subset)
            rows.append(
                {
                    "model_id": model_id,
                    "source_role": source_role,
                    "harmful_override_rate": hor.rate,
                    "harmful_override_n": hor.n,
                    "corrective_override_rate": cor.rate,
                    "corrective_override_n": cor.n,
                }
            )
    if not rows:
        raise ValueError("Cannot build override_summary from zero records.")
    return pd.DataFrame(rows)


def abstention_summary(records: list[dict]) -> pd.DataFrame:
    """Abstention rate under conflict, by model and evidence truth
    (Plot 4's underlying table). Exploratory.
    """
    df = records_to_frame(records)
    conflict_df = df[df["conflict_status"] == "conflict"]
    if conflict_df.empty:
        raise ValueError("No conflict trials found for abstention_summary.")
    grouped = (
        conflict_df.groupby(["model_id", "knowledge_group", "evidence_truth"])
        .agg(
            n=("decision", "size"),
            abstention_rate=("decision", lambda s: (s == "uncertain").mean()),
        )
        .reset_index()
    )
    return grouped
