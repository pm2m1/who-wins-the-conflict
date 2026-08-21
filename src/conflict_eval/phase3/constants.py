"""Frozen Phase 3 design constants.

Every value here is transcribed from `docs/phase3_scaled_study_design.md`
with the originating section cited. Changing a value in this module
changes the scientific design, which Phase 3B is forbidden to do — if an
implementation need appears to require a different value, STOP and report
it rather than editing here (frozen design, §34 "real-bug" discipline).
"""

from __future__ import annotations

from typing import Final

# --- Screening (§11) -------------------------------------------------------
SCREENING_BLOCK_SIZE: Final[int] = 250
SCREENING_CEILING_PER_MODEL: Final[int] = 2000
# "The reserve is 2 items per cell/stratum throughout" (§11).
SUPPLY_RESERVE: Final[int] = 2

# --- Cohort A (§15.1) ------------------------------------------------------
COHORT_A_MODEL: Final[str] = "qwen"
COHORT_A_KNOWLEDGE_GROUP: Final[str] = "KW"
COHORT_A_PER_STRATUM_TARGET: Final[int] = 32
COHORT_A_TOTAL_TARGET: Final[int] = 96  # 32 low + 32 medium + 32 high
# Early-stop supply criterion: 32 target + 2 reserve = 34 per stratum (§11).
COHORT_A_PER_STRATUM_SUPPLY: Final[int] = (
    COHORT_A_PER_STRATUM_TARGET + SUPPLY_RESERVE
)
# Relation-dominance DIAGNOSTIC threshold; non-gating by construction (§15.1).
COHORT_A_RELATION_DOMINANCE_FLAG: Final[float] = 0.60

# --- Cohort B (§15.2, §32) -------------------------------------------------
COHORT_B_CELL_TARGET: Final[int] = 8
COHORT_B_CELL_MINIMUM: Final[int] = 6
# Cohort B supply criterion: target + reserve = 10 per cell (§11).
COHORT_B_CELL_SUPPLY: Final[int] = COHORT_B_CELL_TARGET + SUPPLY_RESERVE
COHORT_B_PER_GROUP_TARGET: Final[int] = 96  # 4 relations x 3 strata x 8

# --- Strata (§14) ----------------------------------------------------------
MARGIN_STRATA: Final[tuple[str, str, str]] = ("low", "medium", "high")

# --- Relations (§9; unchanged from the committed Phase 2 policy) -----------
# Deliberately mirrors conflict_eligibility.PRIMARY_RELATIONS; Phase 3 does
# not expand the relation set (§9).
PHASE3_RELATIONS: Final[tuple[str, ...]] = (
    "country",
    "sport",
    "place of birth",
    "mother",
)

# --- Conditions (§22) ------------------------------------------------------
CONDITION_C0: Final[str] = "C0"
COMMON_ARM_CONDITIONS: Final[tuple[str, ...]] = ("K1", "K2", "K3", "K4")
MODEL_SPECIFIC_ARM_CONDITIONS: Final[tuple[str, ...]] = ("M1", "M2")
NOMINAL_CONDITIONS: Final[tuple[str, ...]] = (
    (CONDITION_C0,) + COMMON_ARM_CONDITIONS + MODEL_SPECIFIC_ARM_CONDITIONS
)

ARM_BASELINE: Final[str] = "baseline"
ARM_COMMON: Final[str] = "common"
ARM_MODEL_SPECIFIC: Final[str] = "model_specific"

# --- Source identities (§19) ----------------------------------------------
# FROZEN common pair. These are fixed source IDENTITIES, not
# preferred/dispreferred roles for any model (§19).
COMMON_SOURCE_A: Final[str] = "a government website"
COMMON_SOURCE_B: Final[str] = "an anonymous online forum post"

# --- Frozen Phase 2 model-specific pairs (§20.1) ---------------------------
# Reused verbatim for the confirmatory contrast. Fresh Phase 3 calibration
# for these two models is DIAGNOSTIC only and may never redefine them.
FROZEN_MODEL_SOURCE_PAIRS: Final[dict[str, dict[str, str]]] = {
    "qwen": {
        "preferred_source": "a government website",
        "dispreferred_source": "an anonymous online forum post",
    },
    "llama": {
        "preferred_source": "a government website",
        "dispreferred_source": "a social media post",
    },
}

# --- Frozen Phase 2 model revisions (§7) -----------------------------------
FROZEN_MODEL_REVISIONS: Final[dict[str, dict[str, str]]] = {
    "qwen": {
        "hf_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
    },
    "llama": {
        "hf_model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
    },
}

# --- Approved-but-unresolved new model families (§7, §42.1) ----------------
# Approved at FAMILY level only. Exact repository IDs, releases, and commit
# SHAs are deliberately unresolved until Phase 3C and must never be
# invented here.
UNRESOLVED_MODEL_FAMILIES: Final[tuple[str, ...]] = (
    "Mistral-7B-Instruct",
    "Gemma-2-9B-it",
)

# --- Dataset (§8) ----------------------------------------------------------
PHASE3_DATASET_REVISION: Final[str] = "098765c79ea10a2cb19c828324e33281b8336ec0"

# --- Analysis status (§44) -------------------------------------------------
STATUS_PRIMARY: Final[str] = "PRIMARY CONFIRMATORY"
STATUS_SECONDARY: Final[str] = "SECONDARY CONFIRMATORY"
STATUS_EXPLORATORY: Final[str] = "EXPLORATORY"
STATUS_DIAGNOSTIC: Final[str] = "DIAGNOSTIC"
ANALYSIS_STATUSES: Final[tuple[str, ...]] = (
    STATUS_PRIMARY,
    STATUS_SECONDARY,
    STATUS_EXPLORATORY,
    STATUS_DIAGNOSTIC,
)

# --- Diagnostics (§30, §37) ------------------------------------------------
# "either arm exceeds 0.95 or falls below 0.05 adoption AND discordant
# pairs are fewer than 5" (§30).
SATURATION_UPPER_BOUND: Final[float] = 0.95
SATURATION_LOWER_BOUND: Final[float] = 0.05
MIN_INFORMATIVE_DISCORDANT_PAIRS: Final[int] = 5

# --- Frozen Phase 2 replication target (§15.1, §26.1) ----------------------
# Recorded for reporting/classification only; never recomputed.
PHASE2_QWEN_CORRECTIVE_DELTA: Final[float] = 0.2667

# --- Compute bounds (§23) --------------------------------------------------
NOMINAL_CONDITION_SLOTS_FULL_TARGET: Final[int] = 5376  # 4 x 192 x 7
BASELINE_SCREEN_CEILING_TOTAL: Final[int] = 8000  # 4 x 2000

# --- Synthetic-output marking (§17 of the Phase 3B brief; Phase 2 precedent)
SYNTHETIC_PREFIX: Final[str] = "synthetic"
DRYRUN_PREFIX: Final[str] = "dryrun"

# --- Canonical item identity ----------------------------------------------
# The repository's real baseline records (src/conflict_eval/cli.py,
# cmd_screen) key each item as `item_id`. Phase 3 uses that same canonical
# field everywhere rather than introducing a competing `id` schema; an
# earlier Phase 3B draft read `id`, which only worked because the synthetic
# fixtures happened to match it.
ITEM_ID_FIELD: Final[str] = "item_id"

# Number of Qwen KW items in the frozen Phase 2 pilot. Used ONLY to
# sanity-check the size of the exclusion list a Phase 3C operator supplies
# (docs/phase3_scaled_study_design.md, §15.1: "The 30 Qwen KW items used in
# the frozen Phase 2 pilot are excluded"). The ids themselves are never
# hardcoded or invented here.
PHASE2_QWEN_KW_ITEM_COUNT: Final[int] = 30
