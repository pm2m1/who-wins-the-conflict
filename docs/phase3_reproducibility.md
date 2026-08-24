# Phase 3 — Reproducibility runbook

End-to-end: from the verified Phase 3D return to the Phase 3E results in
`runs/phase3/analysis/`. Everything below is a deterministic, offline
computation — **no model is loaded and no network access is needed.**

Regenerating the observations themselves requires a GPU and is documented
separately at the end.

---

## 0. What is and is not in this repository

Following the policy in `configs/frozen/README.md`, `data/README.md` and
`results/README.md`: small provenance records are committed; large
empirical outputs are not, and are referenced by immutable SHA256 from the
sealed manifest.

| | Location | In Git |
| --- | --- | --- |
| Frozen design | `docs/phase3_scaled_study_design.md` | ✅ |
| Sealed pre-run manifest + cohorts | `configs/phase3/freeze/` | ✅ |
| Study config | `configs/phase3/phase3_study.yaml` | ✅ |
| Prompt templates | `prompts/` | ✅ |
| Raw Phase 3D observations (4 197) | cloud-return package | ❌ hashed |
| Derived baseline artifacts | `runs/phase3/derived/` | ❌ hashed |
| Trial spec, run plan, dedup records | `runs/phase3/{derived,evidence}/` | ❌ hashed |
| Phase 3E analysis outputs | `runs/phase3/analysis/` | ❌ hashed below |

**No credential, token, private key, model weight, or Hugging Face cache
file is present anywhere in this repository or in any returned package.**
Both cloud packages were scanned for `hf_…`, `sk-…`, `ghp_…` and
private-key markers before archiving; all were clean. The runbooks use
interactive `hf auth login` and never accept a token on a command line.

---

## 1. Frozen scientific inputs

### Models — exact identities and revisions

| Key | Hugging Face id | Revision (immutable commit) | Arm |
| --- | --- | --- | --- |
| qwen | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | model-specific ENABLED |
| llama | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | model-specific ENABLED |
| mistral | `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | **common arm only** |
| gemma | `google/gemma-2-9b-it` | `11c9b309abf73637e4b6f9a3fa1e92e615547819` | **common arm only** |

Runtime for every model: **float16, unquantized**, `device_map="auto"`,
`max_memory` unset, CUDA required. Decoding: `do_sample=False`,
`num_beams=1`, `max_new_tokens=32`.

### Dataset

`akariasai/PopQA`, split `test`, revision
`098765c79ea10a2cb19c828324e33281b8336ec0`. Candidate pool: the frozen §9
primary-relation pool over `country`, `sport`, `place of birth`, `mother`.
Candidate frame digest
`e819804249e23108409bdcd9d7e3fa42f1b599cf8e2268245b7d15790f526e14`.

### Prompts and scoring

| File | SHA256 |
| --- | --- |
| `prompts/baseline.txt` | `7217f93ad201415194305de9796d7754f1ea2d8154e290b1b94444452e13756f` |
| `prompts/evidence.txt` | `839358b623b794535c7cd0315ea0d2e5ad8413be04d13b95de92c6ca36fea97b` |
| `prompts/source_calibration.txt` | `2bc6657d2ecfe29fcf875a5daa5fd1f66b1ec2ad1f68670341540888f68c5544` |

Prompt version `v1`. The evidence template varies **only** `{source}` and
`{asserted_answer}`; no other text differs between conditions.

`context_adopted` is true only when `Decision == "answer"` **and** the
parsed `Answer:` matches the conflicting context's asserted answer. Text
under `Decision: uncertain` never counts. Computed by the unchanged
Phase 2 `evaluation/classify.py`.

### Source pairs

**Common pair (frozen, identities not roles):** `a government website` (A)
vs `an anonymous online forum post` (B).

**Model-specific pairs:**

- Qwen — preferred `a government website`, dispreferred `an anonymous
  online forum post` (frozen Phase 2 pair; reused, never recalibrated).
- Llama — preferred `a government website`, dispreferred `a social media
  post` (frozen Phase 2 pair).
- **Mistral — none.** Calibration parsed 30/30 but the three
  least-preferred sources tied at 2/10 with 1:1 direct comparisons, so no
  defensible dispreferred source exists.
- **Gemma — none.** 0/30 outputs were parser-valid under the frozen strict
  parser.

Both therefore run the **common arm only** under §34, and generate no
`M1`/`M2` observations at all. Their model-specific contrasts are NOT
APPLICABLE — never measured, never null.

### Conditions

`C0` no evidence · `K1` correct+common-A · `K2` correct+common-B ·
`K3` false+common-A · `K4` false+common-B · `M1` conflicting+preferred ·
`M2` conflicting+dispreferred. Qwen/Llama run all seven; Mistral/Gemma run
`C0` and `K1`–`K4`.

### Cohorts and exclusions

- **Cohort A** — 96 fresh Qwen KW items, 32/32/32 margin strata, no
  relation quota. The 30 Phase 2 Qwen pilot KW items are excluded (§15.1);
  28 of them appeared in the screened pool and were removed.
- **Cohort B** — relation-balanced 4 × 3 grid per model × knowledge group,
  target 8/cell, minimum 6, with the §32 ladder applied.
- **Cohort C** — all-model intersection, relation-balanced, target 96,
  realized 81.
- Screening exclusions (unchanged from Phase 2): malformed baselines,
  abstentions, non-clean non-matching answers, items without a defensible
  same-relation foil, relation-ineligible and subject-multi-object items.

### Deduplication (§22)

Two planned conditions collapse into one canonical observation only when
**model key + exact revision + item id + exact rendered-prompt SHA256 +
prompt version + generation-settings fingerprint** all match. Different
item ids never collapse on identical text; different models never
deduplicate. 4 880 nominal condition slots → **4 197 unique generations**
(683 aliased). Qwen collapses 2 per item because its frozen pair *is* the
common pair; Llama 1 per item; Mistral and Gemma none.

### Analysis commitments

Declared in `configs/phase3/freeze/analysis_status.json` before generation:
43 rows, exactly one PRIMARY (Cohort A Qwen corrective), 21 declared
secondary, 6 NOT APPLICABLE, the rest exploratory/diagnostic. Effect =
paired risk difference; interval = 95% Tango matched-pair score interval;
test = exact two-sided McNemar / exact binomial; Holm within the secondary
family only.

---

## 2. Artifact digests

### Committed

| File | SHA256 |
| --- | --- |
| `configs/phase3/phase3_study.yaml` | `2b1b1f4fe21ec8a7fe7638b704cfc08af44603a837a221fe718c94e3525c12c1` |
| `configs/phase3/freeze/phase3c_pre_run_manifest.json` | `afa5a426bb88baeace13490b16ce34be85896da175654151aebd5518994fbd97` |
| `configs/phase3/freeze/analysis_status.json` | `8496428cd1217fb32f0a46ee00226a720f417f689d33448067d81b518233e921` |
| `configs/phase3/freeze/cohort_a.json` | `7d5ebdf5fa3513bfb073151f0341fd1bb58e59918e0640ffd70d83387023dcbc` |
| `configs/phase3/freeze/cohort_b.json` | `1d37170ba49de8953ea8539c4058ee1fa67bd6723c7dee3a1d7ae9186dc16187` |
| `configs/phase3/freeze/cohort_c.json` | `eb44663258a6821105fa0837a379dbfe6f78490e2cd98263b62ff0ac6edf17f0` |
| `configs/phase3/freeze/cohort_membership_map.json` | `4ef8990ef5f5252dde677f1cbde069c1e1479ccb344cefd8efca7bd21010793a` |
| `configs/phase3/freeze/final_margin_strata.json` | `3a82a34731ea181f37af1373f1de18d227bd44d8729e8f0da3f2c454ca2fdeee` |
| `configs/phase3/freeze/deduplication_map.json` | `aa30d45fc64dcb2df1218619601cedc07c04d2aa3d247cb0b36872351a283453` |

### Referenced, outside Git

| Artifact | SHA256 |
| --- | --- |
| Candidate frame | `e819804249e23108409bdcd9d7e3fa42f1b599cf8e2268245b7d15790f526e14` |
| Trial specification | `9929fb083ce4e417c9dc758556f4d2f49af8f453c3ac3f04438ac7d14db0dbb9` |
| Dedup observations | `aa093e7c674d4bd142766449011993eebc00a912058f6c235bb96bf43595b328` |
| Phase 3D run plan | `7516e2243cef354a5bb26e88e06b4be21af8fcb667c86d97345ddd1d6c1b3539` |

Per-model baseline and screening-exclusion digests are in the manifest at
`models[<key>].baseline_file_sha256` and `.exclusion_file_sha256`.

### Phase 3E outputs

| File | SHA256 |
| --- | --- |
| `phase3e_results.json` | `beeca62e97e59805df10b9fc7f5df541f79fcc946de2980dc4616b5e7374bc05` |
| `phase3e_primary.json` | `51c461c8b73a6dca9a42a752706f30498752c0c79331e3005f6112ae03a6a626` |
| `phase3e_secondary_family.json` | `01495214e76e21e4fab1ef5bb6acbb6d2d2e405796919786fbe0185d4e08ec5b` |
| `phase3e_cohort_status.json` | `2a0d90d62a04ea036a8b383ea8ebcd1bca83481bbf846cb097fa279f852da057` |
| `phase3e_diagnostics.json` | `83dbf3535f08414ab5c0cd1e570d3c930991a41436c7bfe8a7bc270c554598b5` |
| `phase3e_report.md` | `031695575b61ae3802b6b206608e634b32f07a857baa80f24222f1915227e9d4` |

---

## 3. Reproduce Phase 3E from the verified return

Requires only the Phase 3D return package and this repository. No GPU.

```bash
python -m venv .venv && . .venv/Scripts/activate   # or bin/activate
pip install -e .
```

### Step 1 — verify the return

```bash
cd <phase3d-cloud-return>
sha256sum -c CHECKSUMS.sha256 | grep -v ': OK$' || echo "ALL OK"
cd -

python -m conflict_eval.phase3 verify-evidence-return \
  --root <phase3d-cloud-return>
```

Expected:

```
gemma: OK (9 blocks, 820 records)
llama: OK (13 blocks, 1242 records)
mistral: OK (10 blocks, 945 records)
qwen: OK (12 blocks, 1190 records)
VERIFIED
```

This checks every block digest, the freeze-manifest and run-plan digests
recorded on each block, model identity and revision, dtype and
quantization, CUDA provenance, block contiguity, the observation-id
bijection against the sealed run plan, and that every record's prompt
still hashes to its recorded digest.

### Step 2 — confirm the freeze gate is open

```bash
python -m conflict_eval.phase3 gate \
  --manifest configs/phase3/freeze/phase3c_pre_run_manifest.json
```

Must end `READY = True`.

### Step 3 — run the preregistered analysis

```bash
python -m conflict_eval.phase3 analyze-3e \
  --root <phase3d-cloud-return>
```

Writes the six files in `runs/phase3/analysis/` and prints the primary
result. The digests above must reproduce exactly — the analysis is
deterministic given the same inputs (the only randomness is the
sensitivity bootstrap, seeded at 42).

Expected primary line:

```
n=96  both=56 A-only=24 B-only=0 neither=16  discordant=24
Delta=+0.2500  95% Tango CI=[+0.1741, +0.3451]  exact p=1.19209e-07
CLASSIFICATION: FULL CONFIRMATORY REPLICATION
```

### Step 4 — tests

```bash
python -m pytest -q     # 835 passed, 8 skipped
python -m ruff check .  # All checks passed
```

---

## 4. Regenerating the observations (GPU required)

Only needed to reproduce Phase 3D itself. Requires a CUDA GPU with ≥ 24 GB
VRAM (RTX 3090 class) and accepted licences for the gated Llama and Gemma
repositories.

```bash
python -m conflict_eval.phase3 build-run-plan
python -m conflict_eval.phase3 run-evidence --model qwen
python -m conflict_eval.phase3 run-evidence --model llama
python -m conflict_eval.phase3 run-evidence --model mistral
python -m conflict_eval.phase3 run-evidence --model gemma
```

`run-evidence` refuses to start unless the real-run gate opens **and** the
config still hashes to what the manifest recorded, re-renders every planned
prompt and checks it against the sealed digest, and requires every alias of
an observation to assert the same content. It is resumable: completed
checkpoints are digest-verified and skipped, and it refuses to continue
against a changed plan.

Rebuilding Phase 3C's cohorts additionally needs the screening return and
the Phase 2 Qwen pilot artifact
(`d3e6831efa6a49a7f5f6d555507462185d055c0c1d01de9254f24b3b4e518d6a`), from
which the 30-item §15.1 exclusion list is derived — never reconstructed
from memory.

### Runtime provenance of the frozen run

Both GPU stages ran on:

```
Linux 6.8.0-136-generic, glibc 2.35, x86_64
NVIDIA GeForce RTX 3090, 23.6 GiB, CUDA 12.8
Python 3.10.13 CPython, torch 2.11.0+cu128
transformers 5.15.1, tokenizers 0.22.2, datasets 5.0.1
accelerate 1.14.0, numpy 2.2.6
```

Decoding is deterministic, so the same prompt at the same revision should
give the same text. Bit-identical reproduction across different hardware is
not guaranteed and is not required: what the manifest pins is the revision,
the prompt and the decoding settings, all recorded on every observation.

---

## 5. Chain of custody

| Stage | Commit | What it froze |
| --- | --- | --- |
| 3A design | `d684f39` | the preregistered design |
| 3B implementation | `38afd32` | code + tests, no real model |
| 3C safeguards | `a514be5` | manifest/gate validation |
| 3C screening execution | `4b9ad5f` | the outcome-blind screening runner |
| **3C freeze** | **`bceab43`** | cohorts, margins, sealed §36 manifest |
| 3D preparation | `065c223` | the evidence runner and run plan |

Each GPU stage ran from a bundle of the named commit, and each returned
package carries that commit's SHA in `runtime/git-head.txt`, independently
checked on return.
