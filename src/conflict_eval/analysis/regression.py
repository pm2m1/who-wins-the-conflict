"""Exploratory logistic regression on conflict trials.

Implements the conceptual model in docs/phase2_research_design.md,
"Primary statistical analysis":

    logit P(Y=1) = b0 + b1*B + b2*S + b3*T + b4*(B*S) + b5*(S*T) + b6*(B*T)

Y = context_adopted, B = parametric_margin, S = source_role == preferred,
T = evidence_truth == true (corrective vs. harmful context). The primary
interaction of interest is the B*S coefficient. This is ordinary logistic
regression on pilot-scale data — exploratory only, not a substitute for a
pre-registered, adequately powered, mixed-effects analysis at scale.
"""

from __future__ import annotations

import pandas as pd


def prepare_conflict_frame(records: list[dict]) -> pd.DataFrame:
    conflict_records = [r for r in records if r.get("conflict_status") == "conflict"]
    if not conflict_records:
        raise ValueError("No conflict trials found; cannot run the primary regression.")
    df = pd.DataFrame(conflict_records)
    df["Y"] = df["context_adopted"].astype(int)
    df["B"] = df["parametric_margin"].astype(float)
    df["S"] = (df["source_role"] == "preferred").astype(int)
    df["T"] = (df["evidence_truth"] == "true").astype(int)
    return df


def fit_primary_model(records: list[dict]):
    """Fit the conceptual model above via statsmodels. Returns the fitted
    result object (has .params, .conf_int(), .summary()). Raises if there
    are too few conflict trials or if the fit fails to converge —
    callers should surface that as a pilot limitation, not retry with
    fabricated data.
    """
    import statsmodels.formula.api as smf

    df = prepare_conflict_frame(records)
    if df["Y"].nunique() < 2:
        raise ValueError(
            "context_adopted has no variation in the conflict trials — "
            "logistic regression is not meaningful on this sample."
        )
    model = smf.logit("Y ~ B * S + S * T + B * T", data=df)
    return model.fit(disp=0)
