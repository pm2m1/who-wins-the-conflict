"""The four documented pilot figures (docs/phase2_research_design.md,
"Figures"). Matplotlib only, restrained defaults, no decorative themes.
Every function raises on empty or too-small input rather than plotting
placeholder numbers — see docs/decisions.md.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from conflict_eval.analysis.summaries import abstention_summary, condition_summary, override_summary


def _require(records: list[dict], label: str) -> None:
    if not records:
        raise ValueError(f"Cannot generate {label} from zero records — run the real pilot first.")


def plot_signature_interaction(
    records: list[dict], out_path: str | Path, n_bins: int = 4
) -> Path:
    """Plot 1: context adoption vs. parametric margin, split by source
    role and evidence truth (corrective vs. harmful). Uses empirical
    quantile bins, not a manufactured smooth curve.
    """
    _require(records, "Plot 1 (signature interaction)")
    conflict = [r for r in records if r.get("conflict_status") == "conflict"]
    _require(conflict, "Plot 1 (signature interaction)")

    fig, ax = plt.subplots(figsize=(7, 5))
    series = [
        ("preferred", "true", "tab:green", "-o", "preferred source, corrective"),
        ("preferred", "false", "tab:red", "-o", "preferred source, harmful"),
        ("dispreferred", "true", "tab:green", "--s", "dispreferred source, corrective"),
        ("dispreferred", "false", "tab:red", "--s", "dispreferred source, harmful"),
    ]
    plotted_any = False
    for source_role, evidence_truth, color, style, label in series:
        subset = [
            r
            for r in conflict
            if r["source_role"] == source_role and r["evidence_truth"] == evidence_truth
        ]
        if len(subset) < n_bins:
            continue
        margins = np.array([r["parametric_margin"] for r in subset], dtype=float)
        adopted = np.array([int(r["context_adopted"]) for r in subset], dtype=float)
        edges = np.quantile(margins, np.linspace(0, 1, n_bins + 1))
        xs, ys = [], []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (margins >= lo) & (margins <= hi if i == n_bins - 1 else margins < hi)
            if mask.sum() == 0:
                continue
            xs.append(float(margins[mask].mean()))
            ys.append(float(adopted[mask].mean()))
        if xs:
            ax.plot(xs, ys, style, color=color, label=label)
            plotted_any = True

    if not plotted_any:
        raise ValueError(
            "Not enough conflict trials in any (source_role, evidence_truth) group "
            f"to form {n_bins} empirical bins for Plot 1."
        )

    ax.set_xlabel("Parametric preference margin (memory answer - conflicting context answer)")
    ax.set_ylabel("Empirical context adoption rate")
    ax.set_title("Context adoption vs. parametric preference strength")
    ax.legend(fontsize=8, loc="best")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_corrective_vs_harmful(records: list[dict], out_path: str | Path) -> Path:
    """Plot 2: 2x2 summary — corrective/harmful override rate by
    preferred/dispreferred source.
    """
    _require(records, "Plot 2 (corrective vs. harmful override)")
    df = override_summary(records)

    fig, ax = plt.subplots(figsize=(6, 4))
    roles = ["dispreferred", "preferred"]
    x = np.arange(len(roles))
    width = 0.35

    harmful = [df[df["source_role"] == r]["harmful_override_rate"].mean() for r in roles]
    corrective = [df[df["source_role"] == r]["corrective_override_rate"].mean() for r in roles]

    ax.bar(x - width / 2, harmful, width, label="harmful override rate (lower is better)", color="tab:red")
    ax.bar(x + width / 2, corrective, width, label="corrective override rate (higher is better)", color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels(roles)
    ax.set_ylabel("Rate")
    ax.set_title("Corrective vs. harmful override, by source role")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_condition_summary(records: list[dict], out_path: str | Path) -> Path:
    """Plot 3: C0-C4 context-adoption rates (only — not
    `parsed_answer_accuracy`) by model and knowledge group, distinguishing
    agreement vs. conflict conditions. `condition_summary`
    (analysis/summaries.py) also computes `parsed_answer_accuracy` and
    `abstention_rate` per condition, but only `context_adoption_rate` is
    plotted here; consult the underlying table directly for the other two.
    """
    _require(records, "Plot 3 (condition summary)")
    df = condition_summary(records)

    conditions = ["C0", "C1", "C2", "C3", "C4"]
    groups = sorted(df[["model_id", "knowledge_group"]].drop_duplicates().itertuples(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / max(len(groups), 1)
    x_base = np.arange(len(conditions))

    for i, group in enumerate(groups):
        model_id, knowledge_group = group
        rates = []
        for cond in conditions:
            row = df[
                (df["model_id"] == model_id)
                & (df["knowledge_group"] == knowledge_group)
                & (df["condition"] == cond)
            ]
            rates.append(row["context_adoption_rate"].iloc[0] if not row.empty else np.nan)
        ax.bar(x_base + i * width, rates, width, label=f"{model_id} / {knowledge_group}")

    ax.set_xticks(x_base + width * (len(groups) - 1) / 2)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Context adoption rate")
    ax.set_title("C0-C4 condition summary by model and knowledge group")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_abstention(records: list[dict], out_path: str | Path) -> Path:
    """Plot 4: abstention rate under conflict, by model and evidence
    truth. Exploratory.
    """
    _require(records, "Plot 4 (abstention under conflict)")
    df = abstention_summary(records)

    groups = sorted(df["model_id"].unique())
    truths = ["true", "false"]
    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, truth in enumerate(truths):
        rates = []
        for model_id in groups:
            row = df[(df["model_id"] == model_id) & (df["evidence_truth"] == truth)]
            rates.append(row["abstention_rate"].mean() if not row.empty else np.nan)
        offset = (i - 0.5) * width
        label = "corrective (true) context" if truth == "true" else "harmful (false) context"
        ax.bar(x + offset, rates, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Abstention rate (Decision == uncertain)")
    ax.set_title("Abstention under conflict (exploratory)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
