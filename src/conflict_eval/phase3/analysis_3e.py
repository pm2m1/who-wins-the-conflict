"""Phase 3E: the preregistered analysis, and only that.

Phase 3C sealed which analyses exist and what status each carries; Phase 3D
produced the observations. This module joins the two and computes exactly
the analyses the frozen registry names — no more, and nothing chosen after
seeing a number.

Every statistical procedure is imported from `paired_stats`, which
implements the single pre-specified method for each role (§26.2): the
paired risk difference, the 95% Tango matched-pair score interval, and the
exact two-sided McNemar / exact binomial test on discordant pairs. There is
no alternative interval or test anywhere in this module, and no argument
that would select one.

Four frozen rules are enforced structurally rather than by care:

- **Saturation is checked before interpretation.** `classify_replication`
  evaluates the §30 ceiling/floor flag and the discordance floor *first*
  and returns INCONCLUSIVE if either fires, so a saturated cell can never
  be read as a null (§37, evaluation order fixed).
- **An aliased observation is counted once.** Outcomes are looked up
  through the sealed deduplication alias map, so a generation that serves
  two conditions contributes its single outcome to whichever contrast
  references it and never enters an estimate twice (§22).
- **Holm sees only the frozen secondary family.** Membership comes from
  the registry's `declared_secondary_family` — which already excludes
  Cohort A and the `counted_once_with` Qwen common-arm row — minus rows
  removed by the §32 eligibility gate. The primary test is never pooled
  with it.
- **Eligibility gates are read, not decided.** Cohort B confirmatory
  eligibility comes from the sealed manifest, where Phase 3C recorded it
  pre-outcome.

Exploratory and diagnostic analyses are computed and reported, and are
never multiplicity-corrected or promoted (§28).
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

from conflict_eval.phase3.constants import (
    COMMON_SOURCE_A,
    COMMON_SOURCE_B,
    MARGIN_STRATA,
    MIN_INFORMATIVE_DISCORDANT_PAIRS,
    PHASE2_QWEN_CORRECTIVE_DELTA,
    PHASE3_RELATIONS,
)
from conflict_eval.phase3.paired_stats import holm_adjusted, paired_source_result

#: §37 replication categories.
FULL_REPLICATION = "FULL CONFIRMATORY REPLICATION"
ATTENUATED_REPLICATION = "DIRECTIONAL / ATTENUATED REPLICATION"
NON_REPLICATION = "NON-REPLICATION"
INCONCLUSIVE = "INCONCLUSIVE DUE TO SATURATION OR INSUFFICIENT INFORMATION"

#: Confirmatory alpha for the single-test primary family (§28).
ALPHA = 0.05

#: Which condition pair expresses a *conflict* in the common arm, by
#: knowledge group (§22). KW items conflict under correct evidence (K1/K2
#: assert the gold answer against the model's wrong memory); KC items
#: conflict under false evidence (K3/K4 assert the foil against correct
#: memory). The complementary pair is the agreement control.
COMMON_CONFLICT_PAIR = {"KW": ("K1", "K2"), "KC": ("K3", "K4")}
COMMON_AGREEMENT_PAIR = {"KW": ("K3", "K4"), "KC": ("K1", "K2")}

#: The model-specific arm is always a conflict contrast (§22).
MODEL_SPECIFIC_PAIR = ("M1", "M2")


class Phase3EError(RuntimeError):
    """Raised when the preregistered analysis cannot be run as frozen."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_observations(root: str | Path) -> list[dict[str, Any]]:
    """Every returned Phase 3D observation, in block order per model."""
    root = Path(root)
    records: list[dict[str, Any]] = []
    for model_dir in sorted(p for p in root.iterdir() if (p / "blocks").is_dir()):
        for path in sorted((model_dir / "blocks").glob("block_*.jsonl")):
            records.extend(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    if not records:
        raise Phase3EError(f"no Phase 3D observations found under {root}")
    return records


def build_outcome_lookup(
    manifest: dict[str, Any], records: list[dict[str, Any]]
) -> dict[tuple[str, str, str], bool]:
    """`(model, item, condition) -> context_adopted`, through the alias map.

    An observation that serves several conditions is registered under each
    of them, which is exactly the §22 semantics: one generation, referenced
    by several planned contrasts, counted once in any single estimate
    because no contrast asks for the same condition twice.
    """
    by_observation = {r["observation_id"]: r for r in records}
    lookup: dict[tuple[str, str, str], bool] = {}
    for key, observation_id in (manifest.get("deduplication_alias_map") or {}).items():
        model_key, item_id, condition = key.split("|")
        record = by_observation.get(observation_id)
        if record is None:
            raise Phase3EError(
                f"planned trial {key!r} resolves to observation "
                f"{observation_id!r}, which was not returned"
            )
        outcome = record.get("context_adopted")
        if not isinstance(outcome, bool):
            raise Phase3EError(
                f"observation {observation_id!r} has non-boolean context_adopted; "
                "a missing outcome is not a zero (§24)"
            )
        lookup[(model_key, item_id, condition)] = outcome
    return lookup


def cohort_item_ids(manifest: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    """Frozen cohort membership as `cohort -> model -> sorted item ids`.

    Read from the sealed manifest's membership map, never recomputed.
    """
    membership: dict[str, dict[str, set[str]]] = {}
    for key, labels in (manifest.get("cohort_membership_map") or {}).items():
        model_key, item_id = key.split("|")
        for label in labels:
            membership.setdefault(label, {}).setdefault(model_key, set()).add(item_id)
    return {
        label: {model: sorted(items) for model, items in sorted(by_model.items())}
        for label, by_model in sorted(membership.items())
    }


def item_attributes(manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """`(model, item) -> {knowledge_group, relation, margin, stratum}`.

    Sourced from the per-model screening provenance the freeze recorded, so
    the analysis groups items by exactly the labels the cohorts were built
    from.
    """
    attributes: dict[tuple[str, str], dict[str, Any]] = {}
    for model_key, entry in (manifest.get("models") or {}).items():
        for item_id, margin in (entry.get("margins") or {}).items():
            attributes[(model_key, str(item_id))] = {
                "knowledge_group": margin.get("knowledge_group"),
                "relation": margin.get("relation"),
                "parametric_margin": margin.get("parametric_margin"),
                "margin_stratum": margin.get("margin_stratum"),
            }
    return attributes


# ---------------------------------------------------------------------------
# Contrasts
# ---------------------------------------------------------------------------


def paired_outcomes(
    lookup: dict[tuple[str, str, str], bool],
    model_key: str,
    item_ids: list[str],
    condition_a: str,
    condition_b: str,
) -> list[tuple[bool, bool]]:
    """`(outcome under A, outcome under B)` for every complete pair.

    An item missing either condition is skipped: an incomplete pair cannot
    contribute to a paired comparison (Phase 2 precedent).
    """
    pairs: list[tuple[bool, bool]] = []
    for item_id in item_ids:
        key_a = (model_key, str(item_id), condition_a)
        key_b = (model_key, str(item_id), condition_b)
        if key_a in lookup and key_b in lookup:
            pairs.append((lookup[key_a], lookup[key_b]))
    return pairs


def result_to_dict(result) -> dict[str, Any]:
    """Serialize a `PairedSourceResult` with every §26.2 mandatory field."""
    diagnostics = result.diagnostics
    return {
        "n": result.n,
        "both": result.both,
        "a_only": result.a_only,
        "b_only": result.b_only,
        "neither": result.neither,
        "discordant_pairs": result.discordant,
        "rate_a": result.rate_a,
        "rate_b": result.rate_b,
        "risk_difference": result.risk_difference,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "ci_method": "Tango matched-pair score interval, 95%",
        "exact_p": result.exact_p,
        "test": "exact two-sided McNemar (exact binomial on discordant pairs)",
        "diagnostics": {
            "discordant_pairs": diagnostics.discordant_pairs,
            "discordant_rate": diagnostics.discordant_rate,
            "rate_a": diagnostics.rate_a,
            "rate_b": diagnostics.rate_b,
            "near_boundary": diagnostics.near_boundary,
            "ci_width": diagnostics.ci_width,
            "both_fraction": diagnostics.both_fraction,
            "neither_fraction": diagnostics.neither_fraction,
            "low_information": diagnostics.low_information,
            "saturated_uninformative": diagnostics.saturated_uninformative,
        },
    }


def run_contrast(
    lookup: dict[tuple[str, str, str], bool],
    *,
    model_key: str,
    item_ids: list[str],
    condition_a: str,
    condition_b: str,
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    """One source contrast under the single pre-specified procedure."""
    outcomes = paired_outcomes(lookup, model_key, item_ids, condition_a, condition_b)
    if not outcomes:
        return {
            "n": 0,
            "estimable": False,
            "reason": "no complete pairs",
            "condition_a": condition_a,
            "condition_b": condition_b,
            "source_a": label_a,
            "source_b": label_b,
        }
    payload = result_to_dict(paired_source_result(outcomes))
    payload.update(
        {
            "estimable": True,
            "model_key": model_key,
            "condition_a": condition_a,
            "condition_b": condition_b,
            "source_a": label_a,
            "source_b": label_b,
        }
    )
    return payload


# ---------------------------------------------------------------------------
# §37 replication classification
# ---------------------------------------------------------------------------


def classify_replication(
    contrast: dict[str, Any],
    *,
    phase2_point: float = PHASE2_QWEN_CORRECTIVE_DELTA,
    cohort_eligibility_limited: bool = False,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Assign one of the four frozen §37 categories, mechanically.

    The evaluation order is the frozen one and is not negotiable: the
    ceiling/floor flag and the discordance floor are checked **first**, so a
    saturated contrast is INCONCLUSIVE and the remaining categories are not
    considered. That ordering is the study's main guard against reading a
    ceiling as a scientific null.
    """
    if not contrast.get("estimable"):
        return {
            "category": INCONCLUSIVE,
            "reasons": ["contrast has no complete pairs"],
            "inputs": {},
        }
    diagnostics = contrast["diagnostics"]
    delta = contrast["risk_difference"]
    lower, upper = contrast["ci_lower"], contrast["ci_upper"]
    p_value = contrast["exact_p"]
    excludes_zero = lower is not None and upper is not None and not (lower <= 0 <= upper)
    contains_phase2 = (
        lower is not None and upper is not None and lower <= phase2_point <= upper
    )
    contains_zero = not excludes_zero
    inputs = {
        "direction_positive": delta is not None and delta > 0,
        "delta": delta,
        "ci": [lower, upper],
        "ci_excludes_zero": excludes_zero,
        "ci_contains_phase2_point": contains_phase2,
        "exact_p": p_value,
        "p_below_alpha": p_value is not None and p_value < alpha,
        "saturated_uninformative": diagnostics["saturated_uninformative"],
        "discordant_pairs": diagnostics["discordant_pairs"],
        "phase2_point": phase2_point,
    }

    # --- Fixed evaluation order: (f) and the discordance floor first ------
    reasons: list[str] = []
    if diagnostics["saturated_uninformative"]:
        reasons.append(
            
                "§30 flags the contrast SATURATED / UNINFORMATIVE (an arm is beyond "
                "the 0.95/0.05 boundary and discordant pairs are fewer than 5)"
            
        )
    if diagnostics["discordant_pairs"] < MIN_INFORMATIVE_DISCORDANT_PAIRS:
        reasons.append(
            f"discordant pairs {diagnostics['discordant_pairs']} < "
            f"{MIN_INFORMATIVE_DISCORDANT_PAIRS}; the test was not informative"
        )
    if cohort_eligibility_limited:
        reasons.append(
            "Cohort A is eligibility-limited under the narrow §15.1 condition"
        )
    if contains_zero and contains_phase2:
        reasons.append(
            
                "the 95% interval contains both 0 and the Phase 2 point estimate, so "
                "it is compatible with no effect and with the full Phase 2 effect"
            
        )
    if reasons:
        return {"category": INCONCLUSIVE, "reasons": reasons, "inputs": inputs}

    positive = delta is not None and delta > 0
    significant = p_value is not None and p_value < alpha
    if positive and excludes_zero and significant:
        if contains_phase2:
            return {
                "category": FULL_REPLICATION,
                "reasons": [
                    (
                        "Delta > 0; the 95% Tango interval excludes 0; exact p < 0.05; "
                        "the interval contains the Phase 2 point estimate; not "
                        "saturated and discordant pairs >= 5"
                    )
                ],
                "inputs": inputs,
            }
        return {
            "category": ATTENUATED_REPLICATION,
            "reasons": [
                (
                    "Delta > 0; the 95% Tango interval excludes 0; exact p < 0.05; "
                    "not saturated and discordant pairs >= 5 -- but the interval "
                    "does not contain the Phase 2 point estimate, so the Phase 3 "
                    "effect is materially smaller. Criterion (e) is SECONDARY and "
                    "never overrides the Phase 3 estimate (§37)"
                )
            ],
            "inputs": inputs,
        }
    detail = []
    if not positive:
        detail.append("Delta <= 0")
    if contains_zero:
        detail.append("the 95% interval includes 0")
    if not significant:
        detail.append("exact p >= 0.05")
    return {
        "category": NON_REPLICATION,
        "reasons": [
            (
                "the test was genuinely informative (not saturated, discordant "
                "pairs >= 5) and " + "; ".join(detail)
            )
        ],
        "inputs": inputs,
    }


#: Applied to a secondary contrast that is positive, interval-excluding-zero
#: and Holm-surviving. It is deliberately NOT called FULL CONFIRMATORY
#: REPLICATION: the FULL/ATTENUATED split turns on §37 criterion (e), the
#: comparison against a frozen Phase 2 point estimate, and the only contrast
#: with such a comparator is the Qwen corrective frozen-pair test. Inventing
#: a Phase 2 comparator for any other cell after results exist is exactly
#: what the freeze forbids.
DIRECTIONAL_EFFECT_CONFIRMED = "DIRECTIONAL EFFECT CONFIRMED"

#: Positive, interval excludes 0, informative -- but the Holm-adjusted
#: p-value does not clear alpha. §37 makes Holm survival a requirement for
#: calling a secondary contrast a FULL CONFIRMATORY REPLICATION; it does NOT
#: say that failing Holm makes it a NON-REPLICATION, whose definition is
#: `Delta <= 0` or an interval containing 0. Collapsing the two would
#: manufacture a negative finding the frozen design never defined.
DIRECTIONAL_NOT_MULTIPLICITY_SURVIVING = (
    "DIRECTIONAL EFFECT, NOT MULTIPLICITY-SURVIVING"
)

#: A row that rests on the same observations as another family member and is
#: therefore counted once (§22, §28). It is not an independent test and has
#: no pass/fail of its own.
COUNTED_ONCE = "COUNTED ONCE (shared observations)"


def classify_secondary(
    contrast: dict[str, Any],
    *,
    holm_adjusted_p: float | None,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Apply the §37 categories to a secondary contrast.

    Same fixed evaluation order as the primary -- saturation and the
    discordance floor first -- with one frozen difference: a secondary test
    must survive Holm correction within its family to count as confirmed
    (§37, final paragraph). Criterion (e) is not evaluated, because no
    frozen Phase 2 comparator exists for these cells; that is stated in the
    result rather than worked around.
    """
    if not contrast.get("estimable") or "risk_difference" not in contrast:
        return {
            "category": INCONCLUSIVE,
            "reasons": [contrast.get("reason") or "contrast not estimable"],
        }
    if contrast.get("counted_once_with"):
        return {
            "category": COUNTED_ONCE,
            "reasons": [
                (
                    "this contrast rests on the same observations as "
                    f"{contrast['counted_once_with']!r} (§19, §22), so it is "
                    "reported once and is not an independent test. It has no "
                    "pass/fail of its own and must never be presented as "
                    "corroboration of the result it shares observations with "
                    "(§26.1, §28)"
                )
            ],
        }
    diagnostics = contrast["diagnostics"]
    reasons: list[str] = []
    if diagnostics["saturated_uninformative"]:
        reasons.append(
            "§30 flags this contrast SATURATED / UNINFORMATIVE; its null may not be "
            "counted as evidence against H2b, nor aggregated with non-saturated nulls"
        )
    if diagnostics["discordant_pairs"] < MIN_INFORMATIVE_DISCORDANT_PAIRS:
        reasons.append(
            f"discordant pairs {diagnostics['discordant_pairs']} < "
            f"{MIN_INFORMATIVE_DISCORDANT_PAIRS}; the manipulation had almost no "
            "opportunity to express an effect"
        )
    if reasons:
        return {"category": INCONCLUSIVE, "reasons": reasons}

    delta = contrast["risk_difference"]
    lower, upper = contrast["ci_lower"], contrast["ci_upper"]
    excludes_zero = lower is not None and upper is not None and not (lower <= 0 <= upper)
    survives = holm_adjusted_p is not None and holm_adjusted_p < alpha
    if delta is not None and delta > 0 and excludes_zero and survives:
        return {
            "category": DIRECTIONAL_EFFECT_CONFIRMED,
            "reasons": [
                (
                    "Delta > 0, the 95% Tango interval excludes 0, the test "
                    "survives Holm correction within the secondary family, and the "
                    "contrast is neither saturated nor below the discordance "
                    "floor. FULL vs ATTENUATED is not assigned: §37 criterion (e) "
                    "needs a frozen Phase 2 comparator, which exists only for the "
                    "Qwen corrective frozen-pair primary test"
                )
            ],
            "holm_adjusted_p": holm_adjusted_p,
        }
    if delta is not None and delta > 0 and excludes_zero:
        # Positive and interval-excluding-zero, but not Holm-surviving. §37
        # withholds FULL status; it does not convert this into a negative
        # finding, and this module will not invent one.
        return {
            "category": DIRECTIONAL_NOT_MULTIPLICITY_SURVIVING,
            "reasons": [
                (
                    "Delta > 0 and the 95% Tango interval excludes 0, but the "
                    "Holm-adjusted p-value does not clear alpha within the "
                    "secondary family. §37 requires Holm survival before a "
                    "secondary contrast may be called a confirmed replication; it "
                    "does not make a Holm-failing positive result a NON-REPLICATION"
                )
            ],
            "holm_adjusted_p": holm_adjusted_p,
        }
    detail = []
    if delta is None or delta <= 0:
        detail.append("Delta <= 0")
    if not excludes_zero:
        detail.append("the 95% interval includes 0")
    return {
        "category": NON_REPLICATION,
        "reasons": [
            "the contrast was genuinely informative (not saturated, discordant "
            "pairs >= 5) and " + "; ".join(detail)
        ],
        "holm_adjusted_p": holm_adjusted_p,
    }


# ---------------------------------------------------------------------------
# H1: continuous margin (§26.2)
# ---------------------------------------------------------------------------


def _cluster_robust_logit(y, x, clusters):
    """Ordinary logistic regression with item-clustered robust SEs.

    Returns `None` when the fit is not estimable -- non-convergence,
    quasi/complete separation, or a non-finite standard error -- which is
    the exact trigger for the frozen Firth fallback rather than something
    to retry with a different specification.
    """
    import numpy as np
    import statsmodels.api as sm

    design = sm.add_constant(np.asarray(x, dtype=float).reshape(-1, 1), has_constant="add")
    y = np.asarray(y, dtype=float)
    if len(set(y.tolist())) < 2:
        return None
    try:
        fit = sm.Logit(y, design).fit(
            disp=0, maxiter=200, cov_type="cluster",
            cov_kwds={"groups": np.asarray(clusters)},
        )
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError, RuntimeError):
        # statsmodels raises these on a singular design or a failed fit;
        # both are the frozen trigger for the Firth fallback (§26.2).
        return None
    if not getattr(fit, "mle_retvals", {}).get("converged", True):
        return None
    coefficient = float(fit.params[1])
    std_error = float(fit.bse[1])
    if not (math.isfinite(coefficient) and math.isfinite(std_error)) or std_error == 0:
        return None
    # A coefficient this large with this SE is the numerical signature of
    # separation, which statsmodels does not always flag.
    if abs(coefficient) > 25 or std_error > 25:
        return None
    lower, upper = fit.conf_int()[1]
    return {
        "method": "ordinary logistic regression, item-clustered robust SE",
        "coefficient": coefficient,
        "std_error": std_error,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value": float(fit.pvalues[1]),
        "n_trials": len(y),
        "n_items": len(set(map(str, clusters))),
        "estimable": True,
    }


def _firth_logit(y, x, clusters, max_iter: int = 200, tol: float = 1e-8):
    """Firth penalized (bias-reduced) logistic regression.

    The single pre-specified alternative when ordinary logit is not
    estimable (§26.2). The estimand is the same log-odds coefficient on the
    margin; the Jeffreys prior penalty keeps it finite under separation,
    which is precisely the Phase 2 Llama failure this ladder exists for.
    """
    import numpy as np

    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(y)), np.asarray(x, dtype=float)])
    beta = np.zeros(design.shape[1])
    for _ in range(max_iter):
        eta = design @ beta
        prob = 1.0 / (1.0 + np.exp(-eta))
        w = prob * (1 - prob)
        fisher = design.T @ (design * w[:, None])
        try:
            fisher_inv = np.linalg.inv(fisher)
        except np.linalg.LinAlgError:
            return None
        hat = np.sum((design @ fisher_inv) * design, axis=1) * w
        score = design.T @ (y - prob + hat * (0.5 - prob))
        step = fisher_inv @ score
        beta = beta + step
        if not np.all(np.isfinite(beta)):
            return None
        if np.max(np.abs(step)) < tol:
            break
    else:
        return None
    eta = design @ beta
    prob = 1.0 / (1.0 + np.exp(-eta))
    w = prob * (1 - prob)
    try:
        covariance = np.linalg.inv(design.T @ (design * w[:, None]))
    except np.linalg.LinAlgError:
        return None
    std_error = float(np.sqrt(covariance[1, 1]))
    coefficient = float(beta[1])
    if not (math.isfinite(coefficient) and math.isfinite(std_error)):
        return None
    return {
        "method": "Firth penalized (bias-reduced) logistic regression",
        "coefficient": coefficient,
        "std_error": std_error,
        "ci_lower": coefficient - 1.959963985 * std_error,
        "ci_upper": coefficient + 1.959963985 * std_error,
        "p_value": None,
        "interval_note": (
            "Wald interval on the penalized estimate; reported as "
            "Firth-penalized (§26.2)"
        ),
        "n_trials": len(y),
        "n_items": len(set(map(str, clusters))),
        "estimable": True,
    }


def h1_margin_effect(
    lookup: dict[tuple[str, str, str], bool],
    attributes: dict[tuple[str, str], dict[str, Any]],
    *,
    model_key: str,
    item_ids: list[str],
    conflict_conditions: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    """H1: adoption on the continuous parametric margin, conflict trials only.

    Follows the frozen two-step ladder exactly -- ordinary clustered logit,
    then Firth if that is not estimable, then NOT ESTIMABLE. There is no
    third fallback and no rank-based escape hatch (§26.2).
    """
    y: list[int] = []
    x: list[float] = []
    clusters: list[str] = []
    for item_id in item_ids:
        attribute = attributes.get((model_key, str(item_id)))
        if attribute is None or attribute.get("parametric_margin") is None:
            continue
        group = attribute["knowledge_group"]
        pair = conflict_conditions.get(group)
        if pair is None:
            continue
        for condition in pair:
            key = (model_key, str(item_id), condition)
            if key in lookup:
                y.append(int(lookup[key]))
                x.append(float(attribute["parametric_margin"]))
                clusters.append(str(item_id))
    if len(y) < 10 or len(set(y)) < 2:
        return {
            "estimable": False,
            "status": "NOT ESTIMABLE",
            "reason": (
                "fewer than 10 conflict trials, or no variation in "
                "context_adopted; a not-estimable result is never converted "
                "into a null (§26.2)"
            ),
            "n_trials": len(y),
        }
    primary = _cluster_robust_logit(y, x, clusters)
    if primary is not None:
        primary["status"] = "ESTIMATED (primary specification)"
        return primary
    firth = _firth_logit(y, x, clusters)
    if firth is not None:
        firth["status"] = "ESTIMATED (Firth fallback; ordinary logit not estimable)"
        return firth
    return {
        "estimable": False,
        "status": "NOT ESTIMABLE",
        "reason": (
            "ordinary clustered logistic regression was not estimable and Firth "
            "penalized regression did not produce a finite estimate; no further "
            "model variants are tried (§26.2)"
        ),
        "n_trials": len(y),
    }


# ---------------------------------------------------------------------------
# Diagnostics and exploratory
# ---------------------------------------------------------------------------


def descriptive_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The §25/§44 DIAGNOSTIC descriptives. Never confirmatory."""
    by_model: dict[str, dict[str, Any]] = {}
    for record in records:
        bucket = by_model.setdefault(
            record["model_key"],
            {
                "observations": 0,
                "abstentions": 0,
                "manual_review": 0,
                "final_correct": 0,
                "answer_class": {},
                "tentative_context_content": 0,
            },
        )
        bucket["observations"] += 1
        bucket["abstentions"] += int(record.get("decision") == "uncertain")
        bucket["manual_review"] += int(bool(record.get("manual_review")))
        bucket["final_correct"] += int(bool(record.get("final_correct")))
        cls = record.get("answer_class")
        bucket["answer_class"][cls] = bucket["answer_class"].get(cls, 0) + 1
        # §25 SECONDARY mechanistic: contextual answer text present while the
        # model declined to commit. Never merged into context_adopted.
        asserted = record.get("asserted_answer")
        if (
            record.get("decision") == "uncertain"
            and asserted
            and record.get("parsed_answer")
            and str(record["parsed_answer"]).strip().lower()
            == str(asserted).strip().lower()
        ):
            bucket["tentative_context_content"] += 1
    for bucket in by_model.values():
        n = bucket["observations"]
        bucket["abstention_rate"] = bucket["abstentions"] / n if n else None
        bucket["parsed_answer_accuracy"] = bucket["final_correct"] / n if n else None
        bucket["parsing_failure_rate"] = bucket["manual_review"] / n if n else None
        bucket["tentative_context_content_rate"] = (
            bucket["tentative_context_content"] / n if n else None
        )
    return by_model


def margin_bin_display(
    lookup: dict[tuple[str, str, str], bool],
    attributes: dict[tuple[str, str], dict[str, Any]],
    *,
    model_key: str,
    item_ids: list[str],
    conflict_conditions: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    """DIAGNOSTIC adoption rate by frozen margin stratum (§44).

    Descriptive only. §26.2 forbids inferring nonlinearity from three bin
    means, which is why H3 is modeled on the continuous margin instead.
    """
    bins: dict[str, dict[str, int]] = {
        stratum: {"adopted": 0, "trials": 0} for stratum in MARGIN_STRATA
    }
    for item_id in item_ids:
        attribute = attributes.get((model_key, str(item_id)))
        if attribute is None:
            continue
        stratum = attribute.get("margin_stratum")
        pair = conflict_conditions.get(attribute.get("knowledge_group"))
        if stratum not in bins or pair is None:
            continue
        for condition in pair:
            key = (model_key, str(item_id), condition)
            if key in lookup:
                bins[stratum]["trials"] += 1
                bins[stratum]["adopted"] += int(lookup[key])
    for bucket in bins.values():
        bucket["rate"] = (
            bucket["adopted"] / bucket["trials"] if bucket["trials"] else None
        )
    return bins


def h3_margin_by_source(
    lookup: dict[tuple[str, str, str], bool],
    attributes: dict[tuple[str, str], dict[str, Any]],
    *,
    model_key: str,
    item_ids: list[str],
    conflict_conditions: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    """H3 EXPLORATORY: quadratic margin x source interaction (§26.2).

    Modeled on the continuous standardized margin, never inferred from bin
    means. EXPLORATORY regardless of outcome, and never Holm-corrected.
    """
    import numpy as np
    import statsmodels.api as sm

    rows: list[tuple[int, float, int, str]] = []
    for item_id in item_ids:
        attribute = attributes.get((model_key, str(item_id)))
        if attribute is None or attribute.get("parametric_margin") is None:
            continue
        pair = conflict_conditions.get(attribute.get("knowledge_group"))
        if pair is None:
            continue
        for index, condition in enumerate(pair):
            key = (model_key, str(item_id), condition)
            if key in lookup:
                rows.append(
                    (int(lookup[key]), float(attribute["parametric_margin"]),
                     1 - index, str(item_id))
                )
    if len(rows) < 20 or len({r[0] for r in rows}) < 2:
        return {"estimable": False, "status": "NOT ESTIMABLE", "n_trials": len(rows)}
    y = np.array([r[0] for r in rows], dtype=float)
    margin = np.array([r[1] for r in rows], dtype=float)
    standardized = (margin - margin.mean()) / (margin.std() or 1.0)
    source = np.array([r[2] for r in rows], dtype=float)
    design = sm.add_constant(
        np.column_stack(
            [standardized, standardized**2, source, standardized * source,
             (standardized**2) * source]
        ),
        has_constant="add",
    )
    try:
        fit = sm.Logit(y, design).fit(
            disp=0, maxiter=200, cov_type="cluster",
            cov_kwds={"groups": np.array([r[3] for r in rows])},
        )
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError, RuntimeError):
        return {"estimable": False, "status": "NOT ESTIMABLE", "n_trials": len(rows)}
    names = ["const", "margin", "margin_sq", "source", "margin_x_source",
             "margin_sq_x_source"]
    return {
        "estimable": True,
        "status": "EXPLORATORY",
        "model": "logit(adopt) ~ z + z^2 + S + z:S + z^2:S, item-clustered SE",
        "n_trials": len(rows),
        "coefficients": {
            name: {
                "estimate": float(fit.params[i]),
                "std_error": float(fit.bse[i]),
                "p_value": float(fit.pvalues[i]),
            }
            for i, name in enumerate(names)
        },
        "note": (
            "EXPLORATORY regardless of outcome; never multiplicity-corrected and "
            "never reported as confirmatory (§28)."
        ),
    }


# ---------------------------------------------------------------------------
# Secondary family assembly
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FamilyMember:
    name: str
    p_value: float | None
    included: bool
    reason: str


def secondary_family(
    declared: list[str],
    results: dict[str, dict[str, Any]],
    cohort_b_eligible: dict[str, bool],
) -> tuple[list[FamilyMember], dict[str, float]]:
    """Assemble the Holm family from the frozen declaration.

    `declared` is the registry's `declared_secondary_family`, which already
    excludes Cohort A and the `counted_once_with` Qwen common-arm row. Two
    further removals are applied, both decided by structure and never by an
    observed p-value:

    - a row whose Cohort B model x group is not confirmatory-eligible under
      §32 rule 4 is removed;
    - a row that yields no single pre-specified test statistic (an interval
      robustness check, or a set of recomputations) cannot be a Holm member
      and is recorded as such rather than silently dropped.

    Every declared row is returned either way, with its reason, so the
    realized family size is auditable rather than asserted.
    """
    members: list[FamilyMember] = []
    for name in declared:
        result = results.get(name)
        if result is None:
            members.append(
                FamilyMember(name, None, False, "not computed: no result produced")
            )
            continue
        gate = result.get("eligibility_gate")
        if gate and not gate.get("eligible", True):
            members.append(
                FamilyMember(name, None, False, gate.get("reason", "not eligible"))
            )
            continue
        p_value = result.get("holm_p_value")
        if p_value is None:
            members.append(
                FamilyMember(
                    name, None, False,
                    result.get("holm_exclusion_reason")
                    or "no single pre-specified test statistic",
                )
            )
            continue
        members.append(FamilyMember(name, float(p_value), True, "included"))
    adjusted = holm_adjusted(
        {m.name: m.p_value for m in members if m.included and m.p_value is not None}
    )
    return members, adjusted


# ---------------------------------------------------------------------------
# Orchestration: registry row -> concrete contrast
# ---------------------------------------------------------------------------


def pooled_conflict_pairs(
    lookup: dict[tuple[str, str, str], bool],
    attributes: dict[tuple[str, str], dict[str, Any]],
    *,
    model_key: str,
    item_ids: list[str],
    pair_by_group: dict[str, tuple[str, str]],
) -> list[tuple[bool, bool]]:
    """Paired outcomes where each item's condition pair depends on its group.

    The common arm expresses conflict through different conditions for KC
    and KW items (§22), so a per-model common-arm contrast must select the
    pair item by item. Each item still contributes exactly one pair.
    """
    pairs: list[tuple[bool, bool]] = []
    for item_id in item_ids:
        attribute = attributes.get((model_key, str(item_id)))
        if attribute is None:
            continue
        pair = pair_by_group.get(attribute.get("knowledge_group"))
        if pair is None:
            continue
        key_a = (model_key, str(item_id), pair[0])
        key_b = (model_key, str(item_id), pair[1])
        if key_a in lookup and key_b in lookup:
            pairs.append((lookup[key_a], lookup[key_b]))
    return pairs


def _wrap(outcomes: list[tuple[bool, bool]], **meta) -> dict[str, Any]:
    if not outcomes:
        return {"estimable": False, "n": 0, "reason": "no complete pairs", **meta}
    payload = result_to_dict(paired_source_result(outcomes))
    payload.update({"estimable": True, **meta})
    payload["holm_p_value"] = payload["exact_p"]
    return payload


def build_phase3e_tables(
    *,
    manifest: dict[str, Any],
    registry: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute every preregistered Phase 3E analysis.

    The set of analyses comes from the frozen registry; this function only
    supplies each one's concrete operationalization, all of which were
    fixed by the freeze (which cohort, which conditions, which sources).
    """
    lookup = build_outcome_lookup(manifest, records)
    attributes = item_attributes(manifest)
    cohorts = cohort_item_ids(manifest)
    models = manifest["models"]
    realization = registry["realization"]
    cohort_b_eligible = realization["cohort_b_confirmatory_eligible"]
    disabled = set(realization["model_specific_arm_disabled"])

    def cohort_b_items(model_key: str, group: str | None = None) -> list[str]:
        if group:
            return cohorts.get(f"B:{group}", {}).get(model_key, [])
        return sorted(
            set(cohorts.get("B:KC", {}).get(model_key, []))
            | set(cohorts.get("B:KW", {}).get(model_key, []))
        )

    results: dict[str, dict[str, Any]] = {}

    # --- PRIMARY (§26.1) --------------------------------------------------
    cohort_a = cohorts["A"]["qwen"]
    primary = _wrap(
        paired_outcomes(lookup, "qwen", cohort_a, *MODEL_SPECIFIC_PAIR),
        model_key="qwen",
        cohort="A",
        knowledge_group="KW",
        condition_a=MODEL_SPECIFIC_PAIR[0],
        condition_b=MODEL_SPECIFIC_PAIR[1],
        source_a=models["qwen"]["preferred_source"],
        source_b=models["qwen"]["dispreferred_source"],
        trials="KW corrective conflict",
    )
    primary["holm_p_value"] = None
    primary["holm_exclusion_reason"] = (
        "single-test primary family; no multiplicity correction is required or "
        "permitted, and it is never pooled with the secondary family (§28)"
    )
    cohort_a_provenance = manifest["cohorts"]["A"]
    primary["cohort_provenance"] = {
        "status": cohort_a_provenance["status"],
        "realized_total": cohort_a_provenance["realized_total"],
        "per_stratum_selected": cohort_a_provenance["per_stratum_selected"],
        "realized_relation_distribution": cohort_a_provenance[
            "realized_relation_distribution"
        ],
        "relation_dominance_share": cohort_a_provenance["relation_dominance_share"],
        "relation_dominance_flag": cohort_a_provenance["relation_dominance_flag"],
        "excluded_phase2_count": cohort_a_provenance["excluded_phase2_count"],
    }
    primary["shared_observation_note"] = (
        "Qwen's frozen pair is the common pair (§19), so M1/M2 are "
        "prompt-identical to K1/K2 and this contrast rests on the same "
        "observations as Qwen's common-arm contrast. That contrast is therefore "
        "counted once and never presented as independent corroboration (§22, §26.1)."
    )
    results["cohort_a_qwen_corrective_frozen_pair"] = primary
    replication = classify_replication(
        primary,
        cohort_eligibility_limited=(cohort_a_provenance["status"] != "COMPLETE"),
    )

    # --- SECONDARY: Cohort B model-specific contrasts ---------------------
    for model_key in sorted(models):
        for group, suffix in (("KW", "corrective"), ("KC", "harmful")):
            name = f"cohort_b_{model_key}_{suffix}"
            if model_key in disabled:
                results[name] = {
                    "estimable": False,
                    "status": "NOT APPLICABLE",
                    "reason": (
                        f"{model_key}'s model-specific arm was disabled by the "
                        "frozen §34 calibration rule, so M1/M2 were never "
                        "generated; this contrast was never measured and is NOT a "
                        "null result"
                    ),
                    "holm_p_value": None,
                }
                continue
            items = cohort_b_items(model_key, group)
            entry = _wrap(
                paired_outcomes(lookup, model_key, items, *MODEL_SPECIFIC_PAIR),
                model_key=model_key,
                cohort=f"B:{group}",
                knowledge_group=group,
                condition_a=MODEL_SPECIFIC_PAIR[0],
                condition_b=MODEL_SPECIFIC_PAIR[1],
                source_a=models[model_key]["preferred_source"],
                source_b=models[model_key]["dispreferred_source"],
            )
            eligible = cohort_b_eligible.get(f"{model_key}|{group}", False)
            entry["eligibility_gate"] = {
                "eligible": eligible,
                "reason": (
                    "included"
                    if eligible
                    else (
                        f"Cohort B {model_key}|{group} qualifies on fewer than three "
                        "of the four PRIMARY relations, so the whole model x group "
                        "is ELIGIBILITY-LIMITED / EXPLORATORY and is removed from "
                        "the confirmatory families (§32 rule 4)"
                    )
                ),
            }
            entry["cohort_provenance"] = {
                k: manifest["cohorts"]["B"][f"{model_key}|{group}"][k]
                for k in ("status", "confirmatory_eligible", "realized_cell_count",
                          "qualifying_relations", "excluded_short_relations")
            }
            results[name] = entry

    # --- SECONDARY: common fixed-source contrasts (H2a) -------------------
    for model_key in sorted(models):
        name = f"common_fixed_source_{model_key}"
        items = cohort_b_items(model_key)
        entry = _wrap(
            pooled_conflict_pairs(
                lookup, attributes, model_key=model_key, item_ids=items,
                pair_by_group=COMMON_CONFLICT_PAIR,
            ),
            model_key=model_key,
            cohort="B",
            knowledge_group="KC+KW conflict trials",
            condition_a="K1 (KW) / K3 (KC)",
            condition_b="K2 (KW) / K4 (KC)",
            source_a=COMMON_SOURCE_A,
            source_b=COMMON_SOURCE_B,
        )
        entry["per_group"] = {
            group: _wrap(
                paired_outcomes(
                    lookup, model_key, cohort_b_items(model_key, group),
                    *COMMON_CONFLICT_PAIR[group],
                ),
                knowledge_group=group,
            )
            for group in ("KW", "KC")
        }
        eligible_groups = [
            g for g in ("KW", "KC") if cohort_b_eligible.get(f"{model_key}|{g}", False)
        ]
        entry["eligibility_gate"] = {
            "eligible": bool(eligible_groups),
            "reason": (
                "included"
                if eligible_groups
                else (
                    f"neither Cohort B group for {model_key} is confirmatory-eligible "
                    "under §32 rule 4"
                )
            ),
        }
        if model_key == "qwen":
            entry["counted_once_with"] = "cohort_a_qwen_corrective_frozen_pair"
            entry["holm_exclusion_reason"] = (
                "Qwen's common-arm contrast rests on the same observations as its "
                "frozen-pair contrast (§19, §22) and is counted once; it does not "
                "re-enter the secondary family as an additional independent test "
                "(§28)"
            )
        results[name] = entry

    # --- DIAGNOSTIC: common-arm agreement control -------------------------
    results["common_arm_agreement_control"] = {
        "status": "DIAGNOSTIC",
        "note": (
            "Agreement cells are controls and are never interpreted as identifying "
            "a source-caused adoption effect (§24)."
        ),
        "per_model": {
            model_key: _wrap(
                pooled_conflict_pairs(
                    lookup, attributes, model_key=model_key,
                    item_ids=cohort_b_items(model_key),
                    pair_by_group=COMMON_AGREEMENT_PAIR,
                ),
                model_key=model_key,
            )
            for model_key in sorted(models)
        },
        "holm_p_value": None,
    }

    # --- SECONDARY: H1 continuous margin ----------------------------------
    for model_key in sorted(models):
        name = f"parametric_strength_h1_{model_key}"
        entry = h1_margin_effect(
            lookup, attributes,
            model_key=model_key,
            item_ids=cohort_b_items(model_key),
            conflict_conditions=COMMON_CONFLICT_PAIR,
        )
        entry["holm_p_value"] = entry.get("p_value")
        if entry.get("p_value") is None:
            entry["holm_exclusion_reason"] = (
                "no p-value: " + entry.get("status", "not estimable")
            )
        eligible_groups = [
            g for g in ("KW", "KC") if cohort_b_eligible.get(f"{model_key}|{g}", False)
        ]
        entry["eligibility_gate"] = {
            "eligible": bool(eligible_groups),
            "reason": (
                "included"
                if eligible_groups
                else f"no confirmatory-eligible Cohort B group for {model_key} (§32)"
            ),
        }
        results[name] = entry

    # --- SECONDARY: Cohort C ----------------------------------------------
    for model_key in sorted(models):
        name = f"cohort_c_model_specific_{model_key}"
        if model_key in disabled:
            results[name] = {
                "estimable": False,
                "status": "NOT APPLICABLE",
                "reason": (
                    f"{model_key} has no model-specific arm (§34); never measured, "
                    "not a null"
                ),
                "holm_p_value": None,
            }
            continue
        items = cohorts.get("C", {}).get(model_key, [])
        results[name] = _wrap(
            paired_outcomes(lookup, model_key, items, *MODEL_SPECIFIC_PAIR),
            model_key=model_key,
            cohort="C",
            condition_a=MODEL_SPECIFIC_PAIR[0],
            condition_b=MODEL_SPECIFIC_PAIR[1],
            source_a=models[model_key]["preferred_source"],
            source_b=models[model_key]["dispreferred_source"],
        )

    cross_model_pairs: list[tuple[bool, bool]] = []
    for model_key in sorted(models):
        cross_model_pairs.extend(
            pooled_conflict_pairs(
                lookup, attributes, model_key=model_key,
                item_ids=cohorts.get("C", {}).get(model_key, []),
                pair_by_group=COMMON_CONFLICT_PAIR,
            )
        )
    results["cohort_c_common_cross_model"] = _wrap(
        cross_model_pairs,
        cohort="C",
        knowledge_group="all models, conflict trials",
        source_a=COMMON_SOURCE_A,
        source_b=COMMON_SOURCE_B,
    )

    # --- SECONDARY: cross-model model x source interaction ----------------
    results["cross_model_model_by_source_interaction"] = _model_by_source_interaction(
        lookup, attributes, cohorts, sorted(models)
    )

    # --- SECONDARY: §29 sensitivity analyses ------------------------------
    results.update(
        _sensitivity_analyses(lookup, attributes, cohorts, manifest, cohort_a)
    )

    # --- SECONDARY (mechanistic) and EXPLORATORY / DIAGNOSTIC -------------
    diagnostics = descriptive_diagnostics(records)
    results["tentative_answer_content_vs_commitment"] = {
        "status": "SECONDARY (mechanistic)",
        "multiplicity_family": "none",
        "note": (
            "Contextual answer text under Decision: uncertain. It never replaces or "
            "merges into context_adopted (§25), and its family is 'none', so it is "
            "not Holm-corrected."
        ),
        "per_model": {
            k: {
                "tentative_context_content": v["tentative_context_content"],
                "rate": v["tentative_context_content_rate"],
            }
            for k, v in sorted(diagnostics.items())
        },
        "holm_p_value": None,
    }
    results["source_by_parametric_strength_h3"] = {
        "status": "EXPLORATORY",
        "per_model": {
            model_key: h3_margin_by_source(
                lookup, attributes, model_key=model_key,
                item_ids=cohort_b_items(model_key),
                conflict_conditions=COMMON_CONFLICT_PAIR,
            )
            for model_key in sorted(models)
        },
        "holm_p_value": None,
    }
    results["margin_bin_displays"] = {
        "status": "DIAGNOSTIC",
        "per_model": {
            model_key: margin_bin_display(
                lookup, attributes, model_key=model_key,
                item_ids=cohort_b_items(model_key),
                conflict_conditions=COMMON_CONFLICT_PAIR,
            )
            for model_key in sorted(models)
        },
        "holm_p_value": None,
    }

    members, adjusted = secondary_family(
        realization["declared_secondary_family"], results, cohort_b_eligible
    )

    # §37 categories for every contrast, not only the primary. Saturation is
    # evaluated first in every case, so a ceiling can never be read as a null.
    secondary_classification = {
        name: classify_secondary(entry, holm_adjusted_p=adjusted.get(name))
        for name, entry in sorted(results.items())
        if entry.get("estimable") and "risk_difference" in entry
        and name != "cohort_a_qwen_corrective_frozen_pair"
    }
    saturated = sorted(
        name
        for name, entry in results.items()
        if entry.get("estimable")
        and entry.get("diagnostics", {}).get("saturated_uninformative")
    )
    low_information = sorted(
        name
        for name, entry in results.items()
        if entry.get("estimable")
        and entry.get("diagnostics", {}).get("low_information")
    )
    return {
        "secondary_classification": secondary_classification,
        "saturation_summary": {
            "saturated_uninformative": saturated,
            "below_discordance_floor": low_information,
            "rule": (
                "A contrast is SATURATED / UNINFORMATIVE when either arm exceeds "
                "0.95 or falls below 0.05 adoption AND discordant pairs are fewer "
                "than 5 (§30). Such a null is reported as 'this design could not "
                "detect a source effect in this regime', is never counted as "
                "evidence against H2b, and is never aggregated with non-saturated "
                "nulls."
            ),
        },
        "primary": primary,
        "replication_classification": replication,
        "results": results,
        "diagnostics": diagnostics,
        "secondary_family": {
            "declared": realization["declared_secondary_family"],
            "members": [dataclasses.asdict(m) for m in members],
            "included": sorted(m.name for m in members if m.included),
            "family_size": sum(1 for m in members if m.included),
            "holm_adjusted_p": adjusted,
            "significant_after_holm": sorted(
                name for name, p in adjusted.items() if p < ALPHA
            ),
        },
        "cohort_status": {
            "A": manifest["cohorts"]["A"]["status"],
            "B": {
                key: {
                    "status": value["status"],
                    "confirmatory_eligible": value["confirmatory_eligible"],
                    "realized_cell_count": value["realized_cell_count"],
                    "qualifying_relations": value["qualifying_relations"],
                    "excluded_short_relations": value["excluded_short_relations"],
                }
                for key, value in sorted(manifest["cohorts"]["B"].items())
            },
            "C": {
                "status": manifest["cohorts"]["C"]["status"],
                "items": len(manifest["cohorts"]["C"]["items"]),
                "relation_distribution": manifest["cohorts"]["C"][
                    "relation_distribution"
                ],
            },
        },
    }


def _model_by_source_interaction(lookup, attributes, cohorts, models) -> dict[str, Any]:
    """RQ-B: does the source effect differ across models? (§27)"""
    import numpy as np
    import statsmodels.api as sm

    rows: list[tuple[int, int, str, str]] = []
    for model_key in models:
        items = sorted(
            set(cohorts.get("B:KC", {}).get(model_key, []))
            | set(cohorts.get("B:KW", {}).get(model_key, []))
        )
        for item_id in items:
            attribute = attributes.get((model_key, str(item_id)))
            if attribute is None:
                continue
            pair = COMMON_CONFLICT_PAIR.get(attribute.get("knowledge_group"))
            if pair is None:
                continue
            for index, condition in enumerate(pair):
                key = (model_key, str(item_id), condition)
                if key in lookup:
                    rows.append(
                        (int(lookup[key]), 1 - index, model_key,
                         f"{model_key}|{item_id}")
                    )
    if len(rows) < 40 or len({r[0] for r in rows}) < 2:
        return {"estimable": False, "status": "NOT ESTIMABLE", "holm_p_value": None}
    y = np.array([r[0] for r in rows], dtype=float)
    source = np.array([r[1] for r in rows], dtype=float)
    present = [m for m in models if any(r[2] == m for r in rows)]
    reference = present[0]
    columns = [source]
    names = ["source"]
    for model_key in present[1:]:
        indicator = np.array([1.0 if r[2] == model_key else 0.0 for r in rows])
        columns.extend([indicator, indicator * source])
        names.extend([f"model[{model_key}]", f"source_x_model[{model_key}]"])
    design = sm.add_constant(np.column_stack(columns), has_constant="add")
    try:
        fit = sm.Logit(y, design).fit(
            disp=0, maxiter=200, cov_type="cluster",
            cov_kwds={"groups": np.array([r[3] for r in rows])},
        )
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError, RuntimeError):
        return {"estimable": False, "status": "NOT ESTIMABLE", "holm_p_value": None}
    interaction_idx = [i + 1 for i, n in enumerate(names) if n.startswith("source_x_")]
    try:
        constraint = np.zeros((len(interaction_idx), design.shape[1]))
        for row, idx in enumerate(interaction_idx):
            constraint[row, idx] = 1.0
        wald = fit.wald_test(constraint, scalar=True)
        p_value = float(wald.pvalue)
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError, RuntimeError):
        p_value = None
    return {
        "estimable": True,
        "status": "SECONDARY CONFIRMATORY",
        "model": "logit(adopt) ~ source * model, item-clustered SE, conflict trials",
        "reference_model": reference,
        "n_trials": len(rows),
        "joint_interaction_p": p_value,
        "holm_p_value": p_value,
        "coefficients": {
            name: {
                "estimate": float(fit.params[i + 1]),
                "std_error": float(fit.bse[i + 1]),
                "p_value": float(fit.pvalues[i + 1]),
            }
            for i, name in enumerate(names)
        },
    }


def _sensitivity_analyses(lookup, attributes, cohorts, manifest, cohort_a):
    """The §29 sensitivity analyses that carry secondary status.

    Each is computed. Whether it can be a Holm member is decided by
    structure -- does the frozen specification define a single test
    statistic for it? -- and never by the value it produced. A set of
    recomputations and an interval robustness check are reported in full
    and recorded as non-members with that reason.
    """
    out: dict[str, dict[str, Any]] = {}
    qwen_pair = MODEL_SPECIFIC_PAIR

    per_relation = {}
    for dropped in PHASE3_RELATIONS:
        kept = [
            i for i in cohort_a
            if (attributes.get(("qwen", str(i))) or {}).get("relation") != dropped
        ]
        per_relation[f"without_{dropped}"] = _wrap(
            paired_outcomes(lookup, "qwen", kept, *qwen_pair), dropped_relation=dropped
        )
    out["leave_one_relation_out"] = {
        "status": "SECONDARY CONFIRMATORY",
        "basis": "primary Cohort A contrast recomputed dropping each relation",
        "per_relation": per_relation,
        "holm_p_value": None,
        "holm_exclusion_reason": (
            "a set of recomputations, not a single pre-specified test statistic; "
            "reported in full and excluded from the Holm family rather than "
            "summarized by an invented statistic (§29)"
        ),
    }

    country_only = [
        i for i in cohort_a
        if (attributes.get(("qwen", str(i))) or {}).get("relation") == "country"
    ]
    out["country_only_sensitivity"] = _wrap(
        paired_outcomes(lookup, "qwen", country_only, *qwen_pair),
        basis="primary contrast restricted to the country relation",
    )

    shared = [i for i in cohort_a if i in set(cohorts.get("C", {}).get("qwen", []))]
    out["shared_cohort_restriction"] = _wrap(
        paired_outcomes(lookup, "qwen", shared, *qwen_pair),
        basis="primary contrast restricted to Cohort A items also in Cohort C",
    )

    b_kw = set(cohorts.get("B:KW", {}).get("qwen", []))
    restricted = [i for i in cohort_a if i in b_kw]
    out["model_specific_cohort_restriction"] = _wrap(
        paired_outcomes(lookup, "qwen", restricted, *qwen_pair),
        basis="primary contrast restricted to Cohort A items also in Cohort B KW",
    )

    strata = {}
    for stratum in MARGIN_STRATA:
        subset = [
            i for i in cohort_a
            if (attributes.get(("qwen", str(i))) or {}).get("margin_stratum") == stratum
        ]
        strata[stratum] = _wrap(paired_outcomes(lookup, "qwen", subset, *qwen_pair))
    out["margin_standardization_robustness"] = {
        "status": "SECONDARY CONFIRMATORY",
        "basis": "primary contrast within each frozen margin stratum",
        "per_stratum": strata,
        "holm_p_value": None,
        "holm_exclusion_reason": (
            "a set of stratum-wise recomputations, not a single pre-specified test "
            "statistic (§29)"
        ),
    }

    out["bootstrap_interval_robustness"] = {
        "status": "SECONDARY CONFIRMATORY",
        "basis": "percentile bootstrap over Cohort A items, 10000 resamples",
        "interval": _bootstrap_interval(lookup, cohort_a, qwen_pair),
        "holm_p_value": None,
        "holm_exclusion_reason": (
            "an interval robustness check with no test statistic; §26.2 states it "
            "is sensitivity-only and never substitutes for the Tango interval"
        ),
    }
    return out


def _bootstrap_interval(lookup, item_ids, pair, resamples: int = 10000, seed: int = 42):
    """Percentile bootstrap over items. SENSITIVITY ONLY (§26.2, §29.7)."""
    import numpy as np

    outcomes = paired_outcomes(lookup, "qwen", item_ids, *pair)
    if not outcomes:
        return {"estimable": False}
    differences = np.array([int(a) - int(b) for a, b in outcomes], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(resamples, len(differences)), replace=True)
    deltas = draws.mean(axis=1)
    return {
        "estimable": True,
        "point_estimate": float(differences.mean()),
        "ci_lower": float(np.percentile(deltas, 2.5)),
        "ci_upper": float(np.percentile(deltas, 97.5)),
        "resamples": resamples,
        "note": (
            "Never the confirmatory interval; the Tango score interval is the "
            "single pre-specified method (§26.2)."
        ),
    }


__all__ = [
    "ALPHA",
    "ATTENUATED_REPLICATION",
    "COMMON_AGREEMENT_PAIR",
    "COMMON_CONFLICT_PAIR",
    "COMMON_SOURCE_A",
    "COMMON_SOURCE_B",
    "COUNTED_ONCE",
    "DIRECTIONAL_EFFECT_CONFIRMED",
    "DIRECTIONAL_NOT_MULTIPLICITY_SURVIVING",
    "FULL_REPLICATION",
    "INCONCLUSIVE",
    "MODEL_SPECIFIC_PAIR",
    "NON_REPLICATION",
    "PHASE3_RELATIONS",
    "FamilyMember",
    "Phase3EError",
    "build_outcome_lookup",
    "build_phase3e_tables",
    "classify_replication",
    "classify_secondary",
    "cohort_item_ids",
    "descriptive_diagnostics",
    "h1_margin_effect",
    "h3_margin_by_source",
    "item_attributes",
    "load_observations",
    "margin_bin_display",
    "paired_outcomes",
    "pooled_conflict_pairs",
    "result_to_dict",
    "run_contrast",
    "secondary_family",
]
