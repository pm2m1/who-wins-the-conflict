"""Deterministic answer normalization and matching.

Used both for baseline correctness (KC/KW classification) and for
classifying experimental responses (evaluation/answer_match.py reuses
these functions). Normalization is comparison-only: the raw/original
answer text is always preserved separately on result records.
"""

from __future__ import annotations

import re
import string

_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, drop leading
    articles. Standard short-answer QA normalization (SQuAD-style), with
    article dropping restricted to a single leading article so that
    articles appearing mid-phrase (which can be semantically load-bearing
    for some answers) are left alone.
    """
    if text is None:
        return ""
    text = text.strip().lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ")
    if words and words[0] in _ARTICLES:
        words = words[1:]
    return " ".join(words)


def normalized_alias_set(gold: str, aliases: list[str] | None) -> set[str]:
    """Build the full set of normalized strings that should count as
    correct for a given gold answer, including the gold answer itself.
    """
    values = [gold] + list(aliases or [])
    return {normalize_answer(v) for v in values if v is not None and v.strip() != ""}


def is_match(candidate: str, gold: str, aliases: list[str] | None = None) -> bool:
    """Exact/alias match after normalization. This is the primary
    classifier used throughout the pipeline; no LLM judge is used.
    """
    candidate_norm = normalize_answer(candidate)
    if candidate_norm == "":
        return False
    return candidate_norm in normalized_alias_set(gold, aliases)


def token_f1(candidate: str, gold: str) -> float:
    """Token-level F1, retained only as a diagnostic (docs/methodology.md
    and docs/phase2_research_design.md: token F1 does not replace
    exact/alias classification).
    """
    cand_tokens = normalize_answer(candidate).split()
    gold_tokens = normalize_answer(gold).split()
    if not cand_tokens or not gold_tokens:
        return float(cand_tokens == gold_tokens)

    common: dict[str, int] = {}
    for tok in cand_tokens:
        if tok in gold_tokens:
            common[tok] = common.get(tok, 0) + 1
    num_same = sum(min(common.get(tok, 0), gold_tokens.count(tok)) for tok in set(cand_tokens))
    if num_same == 0:
        return 0.0
    precision = num_same / len(cand_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
