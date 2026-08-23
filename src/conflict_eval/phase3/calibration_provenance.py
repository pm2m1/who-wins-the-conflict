"""Structural validation of a new model's calibration provenance (§20.2, §36).

`docs/phase3_scaled_study_design.md` §36 requires the pre-run freeze manifest
to record, **per new model**:

> its calibration output SHA256, preference matrix, and the researcher's
> stated reason for the selected pair (§20.2)

and §20.2 fixes how that calibration is produced:

> all 15 unordered pairs from the six labels in `configs/sources.yaml`, both
> AB and BA presentation orders, strict `^Choice:\\s*([12])\\s*$` parsing,
> per-model preference matrix, `calibration_prompt_version: v2`.

This module validates that record *structurally*. It never recomputes,
adjusts, or infers a calibration outcome -- the numbers and hashes are
transcribed from artifacts produced on the GPU host, and the only thing done
here is to check that what was transcribed is internally consistent and
well-formed.

Two levels are deliberately separated:

- **Shape** (`validate_calibration_provenance`): every field that is present
  must be well-formed and mutually consistent. Applied at config load, so a
  malformed or placeholder value is rejected immediately, but an
  *incomplete* record is still loadable -- Phase 3C assembles provenance
  incrementally and the config is explicitly not frozen yet.
- **Completeness** (`missing_required_fields`): every §36-required field is
  present. Applied at the freeze manifest and the real-run gate -- the
  points at which §36 actually binds. An incomplete record therefore
  BLOCKS a real run rather than passing silently.

That split is what lets a missing artifact hash surface as a named,
auditable pre-freeze blocker instead of tempting anyone to invent one.
"""

from __future__ import annotations

import re
from typing import Any

from conflict_eval.phase3.constants import (
    CALIBRATION_DECISIONS,
    CALIBRATION_PAIR_COUNT,
    CALIBRATION_PRESENTATIONS_PER_PAIR,
    CALIBRATION_SOURCE_LABELS,
)

#: A SHA256 digest as this project records it: lowercase, exactly 64 hex
#: characters. Deliberately strict -- uppercase, truncated, over-long, and
#: placeholder values are all rejected, because a provenance field that
#: accepts prose is not provenance.
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

#: SHA256 fields §36 requires for every new model's calibration.
CALIBRATION_SHA_FIELDS: tuple[str, ...] = (
    "calibration_output_sha256",
    "calibration_summary_sha256",
    "calibration_archive_sha256",
)

#: Every field §36 requires for a `role == "new"` model. `preference_matrix`
#: is handled separately: it is required only when the calibration produced
#: at least one parser-valid trial, and must be explicitly null otherwise.
CALIBRATION_REQUIRED_FIELDS: tuple[str, ...] = CALIBRATION_SHA_FIELDS + (
    "calibration_prompt_version",
    "parser_total_trials",
    "parser_valid_trials",
    "malformed_trials",
    "parser_relaxed",
    "rerun_performed",
    "decision",
    "decision_reason",
)

_MATRIX_KEYS = ("source_a", "source_b", "a_wins", "b_wins")


class CalibrationProvenanceError(ValueError):
    """Raised when a calibration provenance record is malformed."""


def _is_int(value: Any) -> bool:
    # bool is a subclass of int; a boolean trial count is a mistake.
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_sha_fields(where: str, record: dict[str, Any]) -> None:
    for field in CALIBRATION_SHA_FIELDS:
        if field not in record:
            continue
        value = record[field]
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            raise CalibrationProvenanceError(
                f"{where}: {field} must be a lowercase 64-character hex SHA256 "
                f"digest, got {value!r}"
                + (
                    f" (length {len(value)})"
                    if isinstance(value, str)
                    else ""
                )
                + ". Placeholder or prose values are not provenance (§36)."
            )


def _validate_counts(where: str, record: dict[str, Any]) -> None:
    total = record.get("parser_total_trials")
    valid = record.get("parser_valid_trials")
    malformed = record.get("malformed_trials")
    for field, value in (
        ("parser_total_trials", total),
        ("parser_valid_trials", valid),
        ("malformed_trials", malformed),
    ):
        if value is not None and not (_is_int(value) and value >= 0):
            raise CalibrationProvenanceError(
                f"{where}: {field} must be a non-negative integer, got {value!r}"
            )
    if (
        total is not None
        and valid is not None
        and malformed is not None
        and valid + malformed != total
    ):
        raise CalibrationProvenanceError(
            f"{where}: parser_valid_trials ({valid}) + malformed_trials "
            f"({malformed}) must equal parser_total_trials ({total}); every "
            "presentation is either parsed or malformed (§20.2, §34)."
        )

    for field in ("stable_pairs", "order_reversal_pairs"):
        value = record.get(field)
        if value is not None and not (_is_int(value) and value >= 0):
            raise CalibrationProvenanceError(
                f"{where}: {field} must be a non-negative integer, got {value!r}"
            )

    for field in ("parser_relaxed", "rerun_performed"):
        value = record.get(field)
        if value is not None and not isinstance(value, bool):
            raise CalibrationProvenanceError(
                f"{where}: {field} must be a boolean, got {value!r}"
            )


def _validate_preference_matrix(where: str, record: dict[str, Any]) -> None:
    """Validate the pairwise preference record, if one is present.

    Stored as a deterministic list of pair records rather than a dense
    matrix: it is self-describing, diffs legibly, and cannot silently
    transpose. Every one of the 15 unordered pairs from the six frozen
    labels must appear exactly once, in either orientation but never both.
    """
    matrix = record.get("preference_matrix")
    valid = record.get("parser_valid_trials")

    if matrix is None:
        # Explicit null is the correct representation when the strict parser
        # yielded nothing: there is no matrix to record, and inventing an
        # empty one would read as "measured, all ties".
        if valid is not None and valid > 0:
            raise CalibrationProvenanceError(
                f"{where}: parser_valid_trials is {valid} but preference_matrix "
                "is null. A calibration with parser-valid trials produced a "
                "preference matrix and §36 requires it to be recorded."
            )
        return

    if valid == 0:
        raise CalibrationProvenanceError(
            f"{where}: parser_valid_trials is 0 but a preference_matrix is "
            "recorded. With no parser-valid output there is nothing to build a "
            "matrix from; it must be explicitly null (§20.2, §34)."
        )

    if not isinstance(matrix, list) or not matrix:
        raise CalibrationProvenanceError(
            f"{where}: preference_matrix must be a non-empty list of pair "
            f"records or explicitly null, got {type(matrix).__name__}"
        )

    labels = set(CALIBRATION_SOURCE_LABELS)
    seen: set[frozenset[str]] = set()
    total_wins = 0
    stable = 0
    reversals = 0

    for index, pair in enumerate(matrix):
        location = f"{where}: preference_matrix[{index}]"
        if not isinstance(pair, dict):
            raise CalibrationProvenanceError(
                f"{location} must be a mapping with keys {list(_MATRIX_KEYS)}"
            )
        missing = [k for k in _MATRIX_KEYS if k not in pair]
        if missing:
            raise CalibrationProvenanceError(f"{location} is missing {missing}")

        source_a = pair["source_a"]
        source_b = pair["source_b"]
        for field, label in (("source_a", source_a), ("source_b", source_b)):
            if label not in labels:
                raise CalibrationProvenanceError(
                    f"{location}: {field}={label!r} is not one of the six frozen "
                    f"calibration source labels {list(CALIBRATION_SOURCE_LABELS)} "
                    "(configs/sources.yaml, §20.2)"
                )
        if source_a == source_b:
            raise CalibrationProvenanceError(
                f"{location}: a pair cannot compare {source_a!r} with itself"
            )

        key = frozenset((source_a, source_b))
        if key in seen:
            raise CalibrationProvenanceError(
                f"{location}: the unordered pair {sorted(key)} appears more than "
                "once; each of the 15 pairs is recorded exactly once (§20.2)"
            )
        seen.add(key)

        a_wins = pair["a_wins"]
        b_wins = pair["b_wins"]
        for field, value in (("a_wins", a_wins), ("b_wins", b_wins)):
            if not (_is_int(value) and value >= 0):
                raise CalibrationProvenanceError(
                    f"{location}: {field} must be a non-negative integer, got "
                    f"{value!r}"
                )
        if a_wins + b_wins != CALIBRATION_PRESENTATIONS_PER_PAIR:
            raise CalibrationProvenanceError(
                f"{location}: a_wins ({a_wins}) + b_wins ({b_wins}) must equal "
                f"{CALIBRATION_PRESENTATIONS_PER_PAIR}; §20.2 presents every "
                "pair in both AB and BA order"
            )
        total_wins += a_wins + b_wins
        if a_wins == b_wins:
            reversals += 1
        else:
            stable += 1

    if len(matrix) != CALIBRATION_PAIR_COUNT:
        raise CalibrationProvenanceError(
            f"{where}: preference_matrix has {len(matrix)} pairs; §20.2 "
            f"enumerates all {CALIBRATION_PAIR_COUNT} unordered pairs from the "
            "six frozen source labels"
        )

    if valid is not None and total_wins != valid:
        raise CalibrationProvenanceError(
            f"{where}: preference_matrix accounts for {total_wins} presentations "
            f"but parser_valid_trials is {valid}; the matrix must be derived "
            "from exactly the parser-valid presentations"
        )

    # These two summaries are derivable from the matrix, so they are checked
    # rather than trusted: a mismatch means the summary and the raw record
    # disagree, which is exactly the kind of transcription error that would
    # otherwise reach the freeze manifest unnoticed.
    recorded_stable = record.get("stable_pairs")
    if recorded_stable is not None and recorded_stable != stable:
        raise CalibrationProvenanceError(
            f"{where}: stable_pairs is recorded as {recorded_stable} but the "
            f"preference_matrix implies {stable} (pairs won 2-0 in one "
            "direction)"
        )
    recorded_reversals = record.get("order_reversal_pairs")
    if recorded_reversals is not None and recorded_reversals != reversals:
        raise CalibrationProvenanceError(
            f"{where}: order_reversal_pairs is recorded as {recorded_reversals} "
            f"but the preference_matrix implies {reversals} (pairs split 1-1 "
            "across AB/BA order)"
        )


def validate_calibration_provenance(where: str, record: Any) -> None:
    """Validate the SHAPE of a calibration provenance record.

    Every field that is present must be well-formed and consistent with the
    others. Absent fields are tolerated here and caught by
    `missing_required_fields` at the freeze, so an in-progress Phase 3C
    record stays loadable while an incorrect one never does.
    """
    if record is None:
        return
    if not isinstance(record, dict):
        raise CalibrationProvenanceError(
            f"{where}: calibration_provenance must be a mapping, got "
            f"{type(record).__name__}"
        )

    _validate_sha_fields(where, record)
    _validate_counts(where, record)
    _validate_preference_matrix(where, record)

    decision = record.get("decision")
    if decision is not None and decision not in CALIBRATION_DECISIONS:
        raise CalibrationProvenanceError(
            f"{where}: decision must be one of {list(CALIBRATION_DECISIONS)}, got "
            f"{decision!r}"
        )
    reason = record.get("decision_reason")
    if reason is not None and not (isinstance(reason, str) and reason.strip()):
        raise CalibrationProvenanceError(
            f"{where}: decision_reason must be a non-empty string when present; "
            "§20.2 requires the researcher's stated reason, and §36 records it"
        )
    version = record.get("calibration_prompt_version")
    if version is not None and not (isinstance(version, str) and version.strip()):
        raise CalibrationProvenanceError(
            f"{where}: calibration_prompt_version must be a non-empty string, got "
            f"{version!r}"
        )


def missing_required_fields(record: Any) -> list[str]:
    """Return the §36-required calibration fields that are absent.

    Presence only -- shape is `validate_calibration_provenance`'s job. Used
    by the freeze manifest and the real-run gate, which are the points at
    which §36 binds. An empty list means the record is §36-complete.
    """
    if not isinstance(record, dict):
        return ["calibration_provenance"]

    missing = [f for f in CALIBRATION_REQUIRED_FIELDS if record.get(f) is None]

    # `preference_matrix` is required when the calibration produced
    # parser-valid trials, and must be *explicitly* null otherwise -- so a
    # merely absent key is a gap in both cases.
    valid = record.get("parser_valid_trials")
    absent = "preference_matrix" not in record
    null_but_required = record.get("preference_matrix") is None and (
        valid is None or valid > 0
    )
    if absent or null_but_required:
        missing.append("preference_matrix")

    return sorted(missing)
