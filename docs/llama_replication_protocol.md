# Llama-3.1-8B-Instruct Replication Protocol

## Status

- Preparation completed.
- Real Llama model **not yet run**.
- No Llama baseline results.
- No Llama calibration results.
- No Llama source roles yet (`configs/replication/llama_pilot.yaml` has
  both set to `null`).
- No Llama pilot results.
- **The Qwen pilot remains frozen** — see `docs/qwen_pilot_results.md`
  and `docs/decisions.md`, "Freeze the first Qwen pilot after validated
  analysis." Nothing in this document changes it.

No Llama scientific result exists yet. Do not read anything below as a
result — it is a precommitted specification for a run that has not
happened.

## Scientific purpose

Replicate the same controlled method (`docs/phase2_research_design.md`,
`docs/pilot_protocol.md`) on a second model, `meta-llama/Llama-3.1-8B-
Instruct`, to test whether the pattern observed in the Qwen2.5-7B-
Instruct pilot generalizes to a different model. Replication is not
expected to succeed or fail in any particular direction; this document
does not predict an outcome. Effect direction, magnitude, uncertainty,
and a failure to replicate are all scientifically informative results.

## Frozen methodological invariants

The following are carried over from the Qwen pilot's method unchanged,
so the two models are evaluated under the same precommitted procedure:

- `seed: 42`
- Exact PopQA snapshot: `akariasai/PopQA`, split `test`, revision
  `098765c79ea10a2cb19c828324e33281b8336ec0`
- `screening_candidates: 500`
- `candidate_pool: primary_conflict_relations`
- The already-committed PRIMARY relation and subject-multiplicity policy
  (unchanged)
- Target: 30 KC + 30 KW items
- Margin bins: `low`, `medium`, `high`
- C0-C4 condition definitions (unchanged)
- Prompt templates and prompt versions, including
  `calibration_prompt_version: v2` (unchanged)
- Evaluation rules (`context_adopted`, `final_correct`,
  `parsed_answer_accuracy`, abstention semantics — unchanged)
- The six source-calibration candidate labels (unchanged): Wikipedia, a
  personal blog, a government website, a news article, a social media
  post, an anonymous online forum post
- The direct pairwise source-calibration method (unchanged)
- The planned exploratory `B*S + S*T + B*T` regression form (unchanged)
- The country-only sensitivity concept (unchanged; see "Country-only
  sensitivity" below)

These invariants live in the shared, unmodified files
`configs/prompts.yaml`, `configs/sources.yaml`, `prompts/baseline.txt`,
`prompts/evidence.txt`, `prompts/source_calibration.txt`, and the
already-committed relation/eligibility/scoring/evaluation code under
`src/conflict_eval/`. This protocol references them; it does not
duplicate or re-specify them.

## Model-specific quantities that MUST be re-estimated

The following are **not** copied from the Qwen pilot and must be
independently measured for Llama:

- Baseline (no-evidence) answers
- KC/KW classification
- Parametric margins
- Margin bins (the bin *edges*, computed from Llama's own margin
  distribution)
- Manual-review decisions
- Source calibration results
- Preferred source / dispreferred source
- The final selected 60-item pilot sample

Natural overlap between Qwen's and Llama's selected items is allowed —
both draw from the same 500-candidate frame. What is forbidden is
intentionally copying Qwen's final 60-item selection, or Qwen's
preferred/dispreferred source pair, into Llama's config or pilot
construction.

## Environment/access gate

`meta-llama/Llama-3.1-8B-Instruct` is a gated Hugging Face model
(`configs/replication/models_llama.yaml`, `requires_gated_access: true`).
A future researcher must obtain legitimate access (accept the model's
license on Hugging Face) and authenticate locally (e.g. `huggingface-cli
login` or an `HF_TOKEN` environment variable) before any real run. No
credential or token is committed anywhere in this repository — see
`.gitignore`'s `.env`/`.env.*` rules and `docs/decisions.md`, "Local
secret-file protection."

## Exact revision lock

`configs/replication/models_llama.yaml` deliberately commits
`revision: null`. The exact immutable Hugging Face commit SHA for
`meta-llama/Llama-3.1-8B-Instruct` is not known at the time of writing
this protocol, because the model has not been accessed for this run, and
this repository does not commit a guessed SHA.

Before any substantive baseline screen, a future researcher must resolve
that SHA via a metadata-only lookup (no weight download), using the
existing generic helper:

```python
from conflict_eval.models.hf_causal import resolve_model_revision

sha = resolve_model_revision("meta-llama/Llama-3.1-8B-Instruct", None)
print(sha)
```

This call requires live Hub access and gated-model authentication and
was **not** run as part of this preparation task. The future researcher
must then:

1. Obtain legitimate gated-model access.
2. Authenticate privately (no committed token/credential).
3. Resolve the exact model SHA as above.
4. Copy `configs/replication/models_llama.yaml` to an ungitted scratch
   runtime config.
5. Replace `revision: null` with the resolved exact SHA in that scratch
   copy only.
6. Verify the value.
7. Only then proceed to real-model validation/screening.

The committed template stays generic and pre-run; the resolved SHA lives
only in the scratch copy and, once a real run occurs, in that run's
recorded manifest and per-record `resolved_revision` fields.

## Hardware/precision gate

The committed template (`configs/replication/models_llama.yaml`) fixes:

- `dtype: float16` — an infrastructure/comparability choice matching the
  frozen Qwen pilot's real-run precision (`docs/qwen_pilot_results.md`,
  "Frozen provenance"), not a changed scientific hypothesis.
- No quantization (no `load_in_4bit`, `load_in_8bit`, bitsandbytes, AWQ,
  GPTQ, or GGUF).
- No disk offload.
- `device_map: auto`
- `max_memory: null`

A future researcher running on the same RTX 3090 host used for the Qwen
pilot may choose and record a scratch memory cap (as the Qwen run did,
documented in `docs/qwen_pilot_results.md`), but it must not become a
universal repository default — it belongs only in an ungitted scratch
configuration for that specific runtime environment.

Generation settings are unchanged from the project default:
`do_sample: false`, `max_new_tokens: 32`, `num_beams: 1`. Prompt decoding
semantics are not modified.

## Phase A — environment validation

Future execution only; no scientific data yet.

- Confirm exact repository commit.
- Set up the virtualenv and install the pinned dependencies.
- Run `pytest -q` and `ruff check .`.
- Check `nvidia-smi` and the torch/CUDA installation.
- Confirm gated Llama access is authenticated.
- Resolve the exact Llama model revision (see "Exact revision lock").

## Phase B — small model feasibility validation

Before the 500-item research screen, using the exact pinned Llama
revision:

- Load the model and tokenizer at that revision.
- Verify deterministic generation (`do_sample: false`, `num_beams: 1`).
- Verify scoring/generation Answer-prefix alignment.
- Inspect multi-token candidate handling.
- Verify finite normalized parametric-margin scores.
- Confirm no quantization is in effect.
- Record GPU placement (`device_map` resolution).

This is infrastructure validation, not a scientific result. Experiment
design is not changed based on these diagnostics unless a genuine
implementation failure is found, and any such failure must be documented
in `docs/decisions.md` before the scientific run proceeds (see also
"Real-bug discovery rule" below).

## Phase C — reconstruct the exact PopQA sampling frame

Using `configs/replication/llama_pilot.yaml`: exact PopQA revision
`098765c79ea10a2cb19c828324e33281b8336ec0`, `candidate_pool:
primary_conflict_relations`, `seed: 42`, `screening_candidates: 500`.

Record: raw row count, interim count, PRIMARY-eligible count,
deduplicated count, candidate count, candidate-file SHA256, and the
candidate IDs.

## Phase D — Llama baseline screen

Run `screen --model llama` only. Record: total records, KC count, KW
count, excluded count, manual-review count, malformed count, requested
model revision, resolved model revision.

## Phase E — eligibility/manual audit

Inspect every `manual_review` and malformed exclusion before any item
enters the KC/KW pools (docs/pilot_protocol.md, step 4). Do not silently
convert ambiguous cases into KC/KW.

## Phase F — independent Llama source calibration

Run `calibrate-sources --model llama` using the same six labels and
`calibration_prompt_version: v2`. Record: all 15 unordered pairs, both
AB and BA presentations, valid/malformed counts, AB/BA consistency, and
the pairwise summary, plus the model revision used.

Calibration remains **direct stated source preference** — not "latent
source preference," not ground-truth credibility, not a universal
hierarchy. The researcher selects a defensible preferred/dispreferred
pair only after inspecting this output; there is no automatic
source-pair selection.

**Stop condition:** if calibration produces heavy malformed output,
unstable AB/BA reversals, or only ambiguous ties that cannot support the
intended preferred/dispreferred manipulation, stop before `build-pilot`.
Do not force Qwen's pair (government website preferred / anonymous forum
post dispreferred) onto Llama, and do not invent a new selection
threshold that is not already present in the pre-result methodology.

## Phase G — source-role lock

Once the researcher selects Llama's preferred/dispreferred pair, record
it in a scratch locked config (or directly in
`configs/replication/llama_pilot.yaml` if the researcher chooses to
commit the measured roles at that point) along with the reasoning for
why the pair is defensible. Do not describe either source as universally
best or worst — only as Llama's own measured direct preference, the same
discipline already applied to the Qwen result.

## Phase H — build the Llama pilot

`build-pilot --model llama`: 30 KC + 30 KW items, balanced across
`low`/`medium`/`high` margin bins. Record: the 60 unique item IDs, their
relations, margin bins, the pilot-trials file SHA256, and the 300 C0-C4
trial specifications. Audit a sample of rendered prompts before spending
generation compute (docs/pilot_protocol.md, step 8).

### Insufficient-sample rule (precommitted now)

If the 500 targeted candidates do not yield a valid 30 KC + 30 KW pilot
under the existing eligibility rules, **stop**. Do not automatically
loosen eligibility, add excluded relations, permit multi-object
ambiguity, redefine KC/KW, change baseline uncertainty rules, change
parser rules, copy Qwen's items, alter margin definitions or bins, or
silently increase the candidate count. Instead report: valid KC count,
valid KW count, counts by margin bin, counts by relation, exclusions and
manual-review counts, and the exact reason the pilot cannot be
constructed. The researcher then makes and records a new, pre-outcome
expansion decision in `docs/decisions.md` before any C0-C4 Llama
generation occurs. No adaptation based on Llama C0-C4 outcomes is
permitted.

## Phase I — pre-run freeze

Before any C0-C4 model generation, create and SHA256 an archived manifest
containing at minimum: repository commit SHA; model id; requested and
resolved model revision; dataset id, split, and revision; candidate file
SHA256 and IDs; baseline and baseline-exclusion file SHA256; selected
item IDs; KC/KW membership; margins and margin bins; relation counts;
source prompt version; source-config SHA256; selected Llama source
roles; pilot-trials file SHA256; precision; quantization status (none);
`device_map`; the machine-specific `max_memory` value actually used (if
any); Python, torch, CUDA, transformers, datasets, and accelerate
versions; GPU name and VRAM. Archive this state before generation begins
— the same discipline the frozen Qwen pilot already followed
(`docs/qwen_pilot_results.md`, "Frozen provenance").

## Phase J — run C0-C4

Run exactly 300 generations (60 items x 5 conditions). No mid-run design
changes.

## Phase K — integrity check before analysis

Verify: 300 records; 300 unique item-condition keys; 60 records for each
of C0-C4; 150 KC trial records; 150 KW trial records; a single consistent
resolved model revision across all records; no unexpected manual review;
no missing records.

## Phase L — C0 reproducibility

Compare every C0 result against its corresponding baseline record:
raw generation, parsed answer, decision, and confidence must match
exactly (the same check the frozen Qwen pilot passed 60/60,
`docs/qwen_pilot_results.md`, "Reproducibility checks"). If C0 does not
reproduce exactly, stop before any scientific interpretation and report
the mismatch — do not hide it or proceed past it.

## Phase M — analysis

Use the existing, unmodified analysis definitions:

- **Primary:** HOR, COR, `Delta_harm`, `Delta_correct`.
- **Secondary:** paired preferred-vs-dispreferred checks
  (`analysis/paired_comparison.py`).
- **Planned exploratory:** the continuous `B*S + S*T + B*T` logistic
  regression.
- **Planned sensitivity:** the country-only comparison (see below).
- **Pre-specified secondary mechanistic follow-up:** the tentative-answer
  decomposition (see "Tentative-answer decomposition" below) — not a
  post-hoc discovery for Llama, because it is being declared here, before
  any Llama result exists.

Do not add further post-hoc Llama analyses until the primary and planned
outputs above are frozen.

## Phase N — post-run archive

Archive all Llama run inputs, results, and manifests; compute SHA256 for
each; copy to local storage; hash-verify the copy.

## Phase O — cross-model synthesis

Only after the Llama pilot is itself frozen: compare Qwen and Llama
descriptively. Do not pool the two models' item-level observations as if
they were independent draws from one population without an appropriate
statistical model for the model-level grouping, and do not claim
universal LLM behavior from two pilot-scale models.

## Predeclared cross-model questions

Recorded here, before any Llama outcome exists, so they cannot be
selected after seeing results:

1. Does preferred-source attribution increase committed corrective
   adoption in Llama?
2. What is Llama's `Delta_correct`?
3. Does the harmful-override direction replicate?
4. What is Llama's `Delta_harm`?
5. Does Llama show the same descriptive relationship between parametric
   margin and committed context adoption?
6. Is Llama's B:S interaction supported or unsupported?
7. Does Llama's source calibration identify the same source hierarchy as
   Qwen, or a different one?
8. Does the planned country-only sensitivity preserve the direction of
   the corrective source effect?

No expected answer is recorded for any of these. Success is not defined
as reproducing `p < 0.05`; effect direction, magnitude, uncertainty, and
a failure to replicate are all valid, reportable outcomes.

## Country-only sensitivity

Precommitted now, for the same reason it was precommitted for Qwen: the
selected KC and KW samples can differ in relation composition
(`docs/qwen_pilot_results.md` labels its own country-only check a
"planned sensitivity," not post-hoc). After Llama's own 60-item sample is
built, run the same country-only preferred-vs-dispreferred comparison if
the relevant cells contain country items. If a Llama cell has zero
country items, report the check as unavailable for that cell rather than
substituting a different relation after seeing results.

## Tentative-answer decomposition — secondary follow-up, not primary

The Qwen pilot's tentative-answer decomposition (uncertain-decision
trials broken down by tentative context / tentative memory / other) was
**post-hoc** for Qwen (`docs/qwen_pilot_results.md`, "Post-hoc exploratory
tentative-answer decomposition") — it was not designed before that
pilot's results existed.

For Llama, that phenomenon is now known in advance. This protocol
therefore classifies the same decomposition, run on Llama, as a
**pre-specified secondary mechanistic follow-up motivated by the prior
Qwen pilot** — explicitly not an original primary outcome, and not an
independent discovery, and not post-hoc for Llama, precisely because it
is being frozen in writing here before the Llama run happens. It remains
analytically separate from HOR/COR and from the primary committed
`context_adopted` outcome, exactly as it was for Qwen.

## Analysis definitions (unchanged)

- `context_adopted` remains the primary committed context-adoption
  outcome: `True` only if `Decision == "answer"` and the parsed Answer
  matches the conflicting-context answer.
- `parsed_answer_accuracy` remains the textual parsed-Answer-field
  accuracy, independent of `Decision` (`docs/decisions.md`, "Document
  frozen Qwen pilot results" — the `final_accuracy` rename).
- `abstention_rate` remains `Decision == "uncertain"`.
- Tentative answer text under `Decision: uncertain` is not merged into
  `context_adopted`.

## Real-bug discovery rule

Unless an actual implementation bug is discovered, none of the following
are modified for this replication: `PRIMARY_RELATIONS`, subject-
multiplicity logic, baseline parser semantics, baseline uncertainty
exclusion, KC/KW definitions, the parametric-margin formula, answer-
prefix scoring, margin-bin computation, foil selection, the evidence
template, C0-C4 definitions, `context_adopted`, `final_correct`,
`parsed_answer_accuracy`, source-calibration pair enumeration, AB/BA
counterbalancing, source labels, prompt versions, generation settings,
paired-comparison semantics, or the regression formula.

If a real bug affecting the Llama replication is discovered, execution
stops before fixing it, and the following is reported instead: the exact
file/function, why it is a bug, whether the frozen Qwen pilot was
affected by it, and whether fixing it would break strict cross-model
comparability. No scientific-method bug is fixed silently as part of a
preparation task.
