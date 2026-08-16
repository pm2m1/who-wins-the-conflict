# Reference Implementations

Network access was available during scaffolding. The five resources below
were inspected read-only (via fetched repository/dataset pages and the
GitHub license API); nothing was cloned into this project, nothing was
vendored, and no source files were copied. Where a repository's own README
could not be fully retrieved in one pass, that is noted rather than
guessed at.

General policy (see `docs/decisions.md`): repositories without a clearly
verified permissive license are treated as reference-only. Methodological
ideas are independently reimplemented. Even for the one repository with a
confirmed permissive license relevant to our data source
(`AlexTMallen/adaptive-retrieval`, MIT), we still reimplement independently
rather than reuse code, because the pilot's needs are small and specific.

---

## 1. HALoGEN

- **Paper/project:** HALoGEN: Fantastic LLM Hallucinations and Where to Find Them
- **Repository:** https://github.com/AbhilashaRavichander/HALoGEN
- **Relevance:** motivates the Type A/B/C hallucination taxonomy discussed
  in `docs/phase1_literature_synthesis.md`, and the general pattern of
  separating prompts / model generations / verification / scoring into
  distinct pipeline stages.
- **Methodological idea adapted:** modular separation of generation,
  evaluation, and scoring stages; explicit abstention handling; organizing
  experiment output by stage. Reflected in this project's
  `experiment/` vs. `evaluation/` vs. `analysis/` package split.
- **What was NOT adapted:** the verifier/decomposer stack (atomic-fact
  decomposition, external API-based verification against Semantic
  Scholar/web sources). Our pilot uses deterministic short-answer
  normalization and exact/alias matching instead, which is a much simpler
  evaluation problem than open-ended long-form fact verification.
- **Source code copied:** no.
- **License status:** GitHub's license API returned no detected license
  file for this repository as of this writing (checked via
  `GET /repos/AbhilashaRavichander/HALoGEN/license`, which returned "Not
  Found"). Treated as reference-only; no code reuse.
- **Citation:** citation verification pending (BibTeX not independently
  verified against the published paper record).

---

## 2. The Curious Case of Factuality Finetuning ("epistemic-training")

- **Repository:** https://github.com/bnewm0609/epistemic-training
- **Relevance:** motivates the idea that model-internal factuality signals
  carry information about correctness, and that this information does not
  necessarily equal what is expressed in the final generation (see
  `docs/phase1_literature_synthesis.md`, item 3).
- **Methodological idea adapted:** the conceptual distinction between an
  internal factuality signal and a behavioral/output-level signal. For
  this pilot we use a **behavioral** proxy — length-normalized sequence
  log-probability margin between candidate answers — rather than a
  hidden-state probe.
- **What was NOT adapted:** the hidden-state probe training infrastructure
  and the finetuning pipeline. The published codebase's probe workflow
  requires substantially more compute (the repository documents
  ~150GB RAM for probe training and a custom vLLM fork) than this pilot
  needs or can justify. Hidden-state probes are recorded as future work in
  `docs/phase2_research_design.md` and `docs/decisions.md`, not built now.
- **Source code copied:** no.
- **License status:** GitHub's license API returned no detected license
  file for this repository as of this writing. Treated as reference-only.
- **Citation:** citation verification pending.

---

## 3. LLM-Latent-Source-Preferences

- **Repository:** https://github.com/aflah02/LLM-Latent-Source-Preferences
  (branch `ICLR-2026`, confirmed to exist)
- **Relevance:** the most directly relevant reference for this project's
  source-preference calibration methodology (see
  `docs/phase1_literature_synthesis.md`, item 4).
- **Methodological idea adapted:** the general design of pairwise source
  comparison with A/B and B/A counterbalancing to separate position
  preference from source preference, and the split between **direct**
  (explicitly-elicited) and **indirect/latent** (behaviorally-inferred)
  preference measurement. The repository's own structure separates
  `Direct_Experiments/` and `Indirect_Experiments/`, which corroborates
  this project's decision to implement only the direct-calibration branch
  for the pilot and to reserve indirect/latent measurement for later
  comparison.
- **What was NOT adapted:** the indirect/latent preference experiments
  (news selection, academic ranking, e-commerce recommendation case
  studies) and any of the repository's task-specific harnesses. Our direct
  calibration prompt, pairwise statistics, and preference matrix are
  independently implemented in `src/conflict_eval/source_preference/`.
- **Terminology note:** per project convention, our own direct
  measurement is called "direct source preference" or "calibrated source
  preference," not "latent preference" — that term is reserved for
  indirect/behavioral measurement, which this project does not implement.
- **Source code copied:** no.
- **License status:** GitHub's license API returned no detected license
  file for this repository as of this writing. Treated as reference-only.
- **Citation:** citation verification pending.

---

## 4. WildHallucinations dataset

- **Dataset:** https://huggingface.co/datasets/wentingzhao/WildHallucinations
- **Relevance:** motivates the long-tail/entity-rarity framing in
  `docs/phase1_literature_synthesis.md`, item 1; recorded as a **future**
  ecological-validation experiment, not the pilot dataset.
- **Confirmed fields (from the dataset card):** `entity`, `perplexity`
  (measured with Llama-3-8B), `info` (scraped web text, top 10 pages per
  entity), `category` (20 categories), `wiki` (boolean Wikipedia-sourced
  flag). 7,917 entities total, sourced from filtered WildChat
  conversations.
  documented as MIT-licensed on the dataset card.
- **Methodological idea adapted for later work:** using entity popularity
  (`s_pop`/`o_pop` in PopQA, or `perplexity`/`wiki` here) as a continuous
  proxy for parametric-knowledge weakness, to test whether the source
  effect grows as parametric knowledge weakens or entity rarity increases.
  Not implemented in the pilot.
- **Source code copied:** no (this is a dataset, not a codebase).
- **License status:** MIT, as stated on the Hugging Face dataset card.
- **Citation:** citation verification pending.

---

## 5. WildBench

- **Repository/Space:** https://github.com/allenai/WildBench,
  https://huggingface.co/spaces/allenai/WildBench
- **Relevance:** recorded as a **future-only** reference for open-ended
  generalization experiments; not used in the pilot.
- **Methodological idea (not used now):** LLM-judge evaluation
  (checklist-based scoring with GPT-4-turbo/Claude-3-Opus judges,
  producing WB-Score/WB-Reward-Mix metrics). The pilot deliberately uses
  short-answer, deterministically evaluated responses instead, per
  `docs/decisions.md` (why no LLM-as-judge).
- **Source code copied:** no.
- **License status:** Apache-2.0, confirmed via the GitHub license API
  (`GET /repos/allenai/WildBench/license` returned `spdx_id: Apache-2.0`).
- **Citation:** citation verification pending.

---

## Dataset source note: PopQA

Not one of the four numbered reference implementations above, but recorded
here for completeness since it was inspected the same way. The pilot uses
the Hugging Face dataset `akariasai/PopQA` (test split, 14,300 rows),
maintained by Akari Asai, a co-author of the PopQA paper ("When Not to
Trust Language Models: Investigating Effectiveness of Parametric and
Non-Parametric Memories", Mallen et al.). Confirmed fields: `id`, `subj`,
`prop`, `obj`, `subj_id`, `prop_id`, `obj_id`, `s_aliases`, `o_aliases`,
`s_uri`, `o_uri`, `s_wiki_title`, `o_wiki_title`, `s_pop`, `o_pop`,
`question`, `possible_answers`. No license is stated on the Hugging Face
dataset card itself. The original code repository associated with the
PopQA paper, `AlexTMallen/adaptive-retrieval`, is MIT-licensed (confirmed
via the GitHub license API), which is independent of, and does not by
itself establish, a license for the dataset artifact. See
`docs/decisions.md` for why PopQA was selected and how this is handled
(raw data is not committed; recreation instructions are documented
instead). See `data/README.md` for exact recreation steps.

---

## Summary table

| Resource | Type | Relevance | Code copied | License status |
|---|---|---|---|---|
| HALoGEN | repo | pipeline structure, Type A/B/C taxonomy | no | none detected; reference-only |
| epistemic-training | repo | internal factuality signal concept | no | none detected; reference-only |
| LLM-Latent-Source-Preferences | repo | direct/indirect source preference methodology | no | none detected; reference-only |
| WildHallucinations | dataset | future ecological validation | n/a | MIT (dataset card) |
| WildBench | repo/space | future LLM-judge generalization | no | Apache-2.0 (confirmed) |
| PopQA (`akariasai/PopQA`) | dataset | pilot dataset | n/a | none stated on card; source repo `adaptive-retrieval` is MIT |
