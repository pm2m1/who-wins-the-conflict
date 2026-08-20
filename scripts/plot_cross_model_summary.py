"""Generate curated, reader-facing summary figures from already-frozen
cross-model aggregate numbers.

This script does NOT re-run any analysis and does NOT read raw per-item
data (raw/interim/results/figures runtime artifacts are intentionally not
committed — see data/README.md, results/README.md, figures/README.md).
Every number below is copied verbatim from the frozen result document,
`docs/cross_model_pilot_results.md`, with an inline comment citing the
exact section it came from. If that document is ever amended, these
constants must be updated by hand and re-checked against it — this script
has no way to detect drift on its own.

Output: two restrained, publication-style PNGs under docs/assets/:

    docs/assets/cross_model_harmful_vs_corrective.png
    docs/assets/cross_model_margin_strength.png

Run with:

    python scripts/plot_cross_model_summary.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets"

# --- Section 4, "Primary descriptive results" -----------------------------
# Committed context adoption (preferred vs. dispreferred source), per model,
# for harmful override (KC, false conflicting evidence) and corrective
# override (KW, correct conflicting evidence).
HARMFUL_CORRECTIVE = {
    "Qwen2.5-7B-Instruct": {
        "harmful": {"preferred": 20.0, "dispreferred": 10.0},
        "corrective": {"preferred": 83.3, "dispreferred": 56.7},
    },
    "Llama-3.1-8B-Instruct": {
        "harmful": {"preferred": 33.3, "dispreferred": 36.7},
        "corrective": {"preferred": 96.7, "dispreferred": 100.0},
    },
}

# --- Section 7, "Parametric preference strength" --------------------------
# Adoption rate by margin bin (low/medium/high). Qwen's table reports
# preferred-source-only adoption (n=10 per bin, from the 30 KC / 30 KW
# items split three ways); Llama's table reports adoption pooled across
# both source conditions (n=20 per bin). These two series use different
# pooling definitions and are plotted as separate panels, not overlaid, so
# the figure does not imply they are directly comparable.
MARGIN_STRENGTH = {
    "Qwen2.5-7B-Instruct": {
        "note": "preferred-source only, n=10/bin",
        "harmful": [30.0, 20.0, 10.0],  # KC: 3/10, 2/10, 1/10
        "corrective": [100.0, 80.0, 70.0],  # KW: 10/10, 8/10, 7/10
    },
    "Llama-3.1-8B-Instruct": {
        "note": "pooled across both sources, n=20/bin",
        "harmful": [20.0, 65.0, 20.0],  # KC: 4/20, 13/20, 4/20
        "corrective": [100.0, 100.0, 95.0],  # KW: 20/20, 20/20, 19/20
    },
}

MARGIN_BIN_LABELS = ["low", "medium", "high"]


def plot_harmful_vs_corrective(output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    conditions = [("harmful", "Harmful override (KC)"), ("corrective", "Corrective override (KW)")]
    models = list(HARMFUL_CORRECTIVE.keys())
    x = range(len(models))
    width = 0.35

    for ax, (key, title) in zip(axes, conditions):
        preferred = [HARMFUL_CORRECTIVE[m][key]["preferred"] for m in models]
        dispreferred = [HARMFUL_CORRECTIVE[m][key]["dispreferred"] for m in models]
        ax.bar([i - width / 2 for i in x], preferred, width, label="preferred source", color="#4c72b0")
        ax.bar([i + width / 2 for i in x], dispreferred, width, label="dispreferred source", color="#c44e52")
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylim(0, 105)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Committed context adoption (%)")
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle("Preferred- vs. dispreferred-source committed adoption, by model")
    fig.text(
        0.5, -0.02,
        "Source: docs/cross_model_pilot_results.md, Section 4 (frozen aggregate results). Pilot-scale, two models.",
        ha="center", fontsize=7, color="#555555",
    )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cross_model_harmful_vs_corrective.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_margin_strength(output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

    for ax, (model, series) in zip(axes, MARGIN_STRENGTH.items()):
        ax.plot(MARGIN_BIN_LABELS, series["harmful"], marker="o", label="harmful (KC)", color="#c44e52")
        ax.plot(MARGIN_BIN_LABELS, series["corrective"], marker="o", label="corrective (KW)", color="#4c72b0")
        ax.set_title(f"{model}\n({series['note']})", fontsize=9)
        ax.set_xlabel("Parametric margin bin")
        ax.set_ylim(0, 105)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Context adoption (%)")
    axes[0].legend(loc="lower left", fontsize=9, frameon=False)
    fig.suptitle("Context adoption by parametric margin, per model")
    fig.text(
        0.5, -0.05,
        "Source: docs/cross_model_pilot_results.md, Section 7 (frozen aggregate results).\n"
        "Qwen and Llama panels use different pooling definitions (see script docstring) — not directly comparable.",
        ha="center", fontsize=7, color="#555555",
    )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cross_model_margin_strength.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    paths = [plot_harmful_vs_corrective(OUTPUT_DIR), plot_margin_strength(OUTPUT_DIR)]
    for path in paths:
        size = path.stat().st_size
        print(f"wrote {path} ({size} bytes)")


if __name__ == "__main__":
    main()
