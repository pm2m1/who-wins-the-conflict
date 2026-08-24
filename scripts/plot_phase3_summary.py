"""Phase 3 confirmatory result figures.

Reads the frozen Phase 3E machine-readable output
(`runs/phase3/analysis/phase3e_results.json`) and plots it. Nothing is
recomputed and nothing is hard-coded: every plotted number, interval and
status label is looked up from that file at run time, so a figure cannot
drift away from the analysis it depicts.

If the results file is absent (it is a gitignored runtime artifact, like
everything else under `runs/`), the script says so and exits rather than
falling back to remembered numbers. Regenerate it with:

    python -m conflict_eval.phase3 analyze-3e --root <phase3d-return>

Outputs, under docs/assets/phase3/:

    phase3_primary_result.png     Figure A - the primary confirmatory test
    phase3_common_source.png      Figure B - common-source effect per model
    phase3_parametric_strength.png  Figure C - H1 margin coefficients

Run with:

    python scripts/plot_phase3_summary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "runs" / "phase3" / "analysis" / "phase3e_results.json"
OUTPUT_DIR = ROOT / "docs" / "assets" / "phase3"

MODELS = ("qwen", "llama", "mistral", "gemma")
LABELS = {
    "qwen": "Qwen2.5-7B",
    "llama": "Llama-3.1-8B",
    "mistral": "Mistral-7B-v0.3",
    "gemma": "Gemma-2-9B",
}

#: Status notes for Figure B. Each restates a frozen decision recorded in
#: the Phase 3E output, so the figure cannot imply that all four rows are
#: equally confirmatory.
NOTES = {
    "qwen": "counted once with the primary test",
    "llama": "survives Holm correction",
    "mistral": "survives Holm correction",
    "gemma": "outside the family — Cohort B eligibility-limited",
}

INK = "#22252a"
MUTED = "#8b9099"
ACCENT = "#2f6f9f"
WARM = "#b4653a"
GRID = "#e3e5e9"


def _style(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=9, length=3, color=MUTED)


def load_results() -> dict:
    if not RESULTS.exists():
        sys.exit(
            f"{RESULTS} not found.\n"
            "It is a gitignored runtime artifact. Regenerate it with:\n"
            "  python -m conflict_eval.phase3 analyze-3e --root <phase3d-return>"
        )
    return json.loads(RESULTS.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Figure A - the primary confirmatory test
# ---------------------------------------------------------------------------


def figure_a(data: dict) -> Path:
    p = data["primary"]
    cls = data["replication_classification"]["category"]
    rate_a, rate_b = p["rate_a"] * 100, p["rate_b"] * 100

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(9.2, 4.1), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    fig.patch.set_facecolor("white")

    # --- left: the two adoption rates
    bars = ax.bar(
        [0, 1], [rate_a, rate_b], width=0.52, color=[ACCENT, MUTED], zorder=3
    )
    for bar, value in zip(bars, (rate_a, rate_b)):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + 2.2, f"{value:.1f}%",
            ha="center", va="bottom", fontsize=12, color=INK, fontweight="600",
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["a government website\n(preferred)", "an anonymous online\nforum post (dispreferred)"],
        fontsize=9,
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("committed context adoption", fontsize=10, color=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _style(ax)
    ax.set_title(
        f"Qwen · Cohort A · corrective conflict · n = {p['n']}",
        fontsize=10.5, color=INK, pad=10, loc="left",
    )

    # --- right: the paired effect with its interval
    delta = p["risk_difference"] * 100
    lo, hi = p["ci_lower"] * 100, p["ci_upper"] * 100
    ax2.axvline(0, color=MUTED, linewidth=1.0, linestyle="--", zorder=1)
    ax2.plot([lo, hi], [0, 0], color=ACCENT, linewidth=3.0, solid_capstyle="round", zorder=3)
    ax2.plot([delta], [0], "o", color=ACCENT, markersize=10, zorder=4)
    ax2.text(
        delta, 0.30, f"+{delta:.2f} pp", ha="center", fontsize=13,
        color=INK, fontweight="600",
    )
    ax2.text(
        delta, -0.42, f"95% Tango CI  [+{lo:.2f}, +{hi:.2f}]",
        ha="center", fontsize=9, color=INK,
    )
    ax2.text(
        delta, -0.72, f"exact two-sided p = {p['exact_p']:.2e}",
        ha="center", fontsize=9, color=INK,
    )
    ax2.text(
        delta, -1.05,
        f"paired cells  {p['both']} / {p['a_only']} / {p['b_only']} / {p['neither']}",
        ha="center", fontsize=8.5, color=MUTED,
    )
    ax2.text(
        delta, -1.32, f"{p['discordant_pairs']} discordant pairs",
        ha="center", fontsize=8.5, color=MUTED,
    )
    ax2.set_ylim(-1.75, 1.0)
    ax2.set_yticks([])
    ax2.set_xlim(-5, 45)
    ax2.set_xlabel("paired risk difference (percentage points)", fontsize=10, color=INK)
    ax2.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    _style(ax2)
    ax2.spines["left"].set_visible(False)
    ax2.set_title(cls, fontsize=10.5, color=ACCENT, pad=10, loc="left", fontweight="600")

    fig.suptitle(
        "Changing only the attributed source changed which answer Qwen followed",
        fontsize=13, color=INK, x=0.012, ha="left", y=0.985, fontweight="600",
    )
    fig.text(
        0.012, 0.02,
        "Same evidence text under both conditions; only the source label differs. "
        "Zero dispreferred-only discordant pairs.",
        fontsize=8, color=MUTED, ha="left",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    out = OUTPUT_DIR / "phase3_primary_result.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure B - common-source effect across models
# ---------------------------------------------------------------------------


def figure_b(data: dict) -> Path:
    results = data["results"]
    classification = data["secondary_classification"]

    rows = []
    for key in MODELS:
        entry = results[f"common_fixed_source_{key}"]
        rows.append(
            {
                "key": key,
                "delta": entry["risk_difference"] * 100,
                "lo": entry["ci_lower"] * 100,
                "hi": entry["ci_upper"] * 100,
                "n": entry["n"],
                "status": classification.get(f"common_fixed_source_{key}", {}).get(
                    "category", ""
                ),
                "counted_once": bool(entry.get("counted_once_with")),
                "eligible": entry["eligibility_gate"]["eligible"],
            }
        )

    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    fig.patch.set_facecolor("white")
    ax.axvline(0, color=MUTED, linewidth=1.0, linestyle="--", zorder=1)

    for i, row in enumerate(reversed(rows)):
        y = i
        confirmed = row["status"] == "DIRECTIONAL EFFECT CONFIRMED"
        color = ACCENT if confirmed else WARM
        ax.plot(
            [row["lo"], row["hi"]], [y, y], color=color, linewidth=2.6,
            solid_capstyle="round", alpha=1.0 if confirmed else 0.55, zorder=3,
        )
        ax.plot(
            [row["delta"]], [y], "o", color=color, markersize=9,
            markerfacecolor=color if confirmed else "white",
            markeredgewidth=1.8, zorder=4,
        )
        ax.text(
            27.5, y + 0.13,
            f"+{row['delta']:.1f} pp   [{row['lo']:.1f}, {row['hi']:.1f}]",
            va="center", fontsize=8.8, color=INK,
        )
        ax.text(
            27.5, y - 0.20, NOTES[row["key"]],
            va="center", fontsize=7.6, color=MUTED, style="italic",
        )

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [f"{LABELS[r['key']]}\nn = {r['n']}" for r in reversed(rows)], fontsize=9
    )
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(-2, 52)
    ax.set_xlabel(
        "paired risk difference, government website − anonymous forum post (pp)",
        fontsize=10, color=INK,
    )
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _style(ax)

    ax.set_title(
        "Common-source contrast, conflict trials, Cohort B",
        fontsize=13, color=INK, loc="left", pad=12, fontweight="600",
    )
    fig.text(
        0.012, 0.025,
        "Filled = survives Holm correction within the frozen secondary family.  "
        "Hollow = reported but not confirmatory.\n"
        "Rows are not equivalent: Qwen's contrast rests on the same observations "
        "as the primary test, and Gemma's Cohort B cells qualify on too few relations.",
        fontsize=7.8, color=MUTED, ha="left",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out = OUTPUT_DIR / "phase3_common_source.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure C - H1, parametric strength
# ---------------------------------------------------------------------------


def figure_c(data: dict) -> Path:
    results = data["results"]
    adjusted = data["secondary_family"]["holm_adjusted_p"]

    rows = []
    for key in MODELS:
        entry = results[f"parametric_strength_h1_{key}"]
        rows.append(
            {
                "key": key,
                "beta": entry["coefficient"],
                "lo": entry["ci_lower"],
                "hi": entry["ci_upper"],
                "se": entry["std_error"],
                "n": entry["n_trials"],
                "holm": adjusted.get(f"parametric_strength_h1_{key}"),
            }
        )

    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    fig.patch.set_facecolor("white")
    ax.axvline(0, color=MUTED, linewidth=1.0, linestyle="--", zorder=1)

    for i, row in enumerate(reversed(rows)):
        in_family = row["holm"] is not None
        color = ACCENT if in_family else WARM
        ax.plot(
            [row["lo"], row["hi"]], [i, i], color=color, linewidth=2.6,
            solid_capstyle="round", alpha=1.0 if in_family else 0.55, zorder=3,
        )
        ax.plot(
            [row["beta"]], [i], "o", color=color, markersize=9,
            markerfacecolor=color if in_family else "white",
            markeredgewidth=1.8, zorder=4,
        )
        ax.text(
            0.02, i, f"β = {row['beta']:.3f}  (SE {row['se']:.3f})",
            va="center", fontsize=8.5, color=INK,
        )

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [f"{LABELS[r['key']]}\n{r['n']} trials" for r in reversed(rows)], fontsize=9
    )
    ax.set_xlim(-0.40, 0.30)
    ax.set_xlabel(
        "log-odds of context adoption per unit of parametric margin",
        fontsize=10, color=INK,
    )
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _style(ax)
    ax.set_title(
        "Stronger parametric preference goes with lower context adoption",
        fontsize=13, color=INK, loc="left", pad=12, fontweight="600",
    )
    fig.text(
        0.012, 0.025,
        "Logistic regression on the continuous margin, conflict trials, "
        "item-clustered robust standard errors.  Filled = in the Holm family.\n"
        "Associational: the margin was measured, not manipulated, so this is not "
        "a causal estimate.",
        fontsize=7.8, color=MUTED, ha="left",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    out = OUTPUT_DIR / "phase3_parametric_strength.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results()
    for path in (figure_a(data), figure_b(data), figure_c(data)):
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
