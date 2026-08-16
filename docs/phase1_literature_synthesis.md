# Phase 1 — Literature Synthesis

This document records the intellectual progression that motivated this project,
before any experimental code was written. It is a record of how the research
problem emerged from four papers, not a comprehensive literature review of
context-memory conflict in general (see `docs/research_proposal.md` for the
related-work boundary).

We do not exaggerate what any single paper proves. Each paper is read for the
narrower question it actually answers, and the "central lesson" and "open
question" for each paper are our own interpretive framing, not a claim about
the paper's stated contribution.

---

## 1. WildHallucinations

**Core question**

How factually reliable are LLMs when generating long-form information about
diverse, long-tail entities that real users actually ask about?

**Relevant findings**

- Long-tail and non-Wikipedia entities are substantially more difficult for
  evaluated LLMs than popular, well-documented entities.
- Retrieval generally improves factuality but does not eliminate
  hallucination.
- Errors can occur even when useful external information has been
  successfully retrieved.
- Web evidence itself may be contradictory, incomplete, noisy, or outdated.
- Automatic factuality evaluation can also introduce errors, so measured
  hallucination rates are themselves an approximation.

**Central lesson for this project**

    information availability != faithful information use

or equivalently:

    good retrieval != faithful generation

**Open question**

Why does a model ignore, distort, or contradict relevant retrieved evidence,
even when that evidence is available and useful?

---

## 2. HALoGEN

**Core question**

How can hallucination be measured across diverse generative tasks, and what
possible mechanisms or training-data conditions are associated with
different hallucination behaviors?

**Relevant conceptual framework**

HALoGEN proposes three descriptive categories for hallucinations found
during its error analysis. These are analytical categories used to describe
observed errors after the fact, **not proven causal mechanisms**:

- **Type A** — correct information exists in the inspected training data,
  but the generation is still incorrect (a retrieval/recall-style failure
  at generation time).
- **Type B** — incorrect or decontextualized information appears in the
  training data itself, so the model reproduces a training-data error.
- **Type C** — neither supporting correct nor incorrect evidence is
  located in the inspected data, suggesting possible overgeneralization or
  fabrication rather than reproduction of a specific source.

**Relevant findings**

- Hallucination behavior varies substantially across tasks and generators.
- Having correct information in training does not guarantee correct output
  (this is the empirical basis for the Type A category).
- Models can hallucinate in content-grounded tasks even when the relevant
  source information is explicitly present in the context window.
- Hallucination is therefore not merely a missing-knowledge problem.
- Appropriate abstention (declining to answer when unsupported) is treated
  as part of factual reliability, not a separate concern.
- Automatic verifiers are useful but imperfect, and verifier error is a
  limitation on the measured hallucination rates.

**Central lesson**

    correct context present != correct semantic processing

**Open question**

What determines whether available information actually controls the final
generation, given that its mere presence in context does not guarantee it
is used correctly?

---

## 3. The Curious Case of Factuality Finetuning

**Core question**

Can a model's own internal factuality judgments provide useful information
for improving factuality?

**Relevant findings/concepts**

- Model-internal factuality signals (e.g., hidden-state probes trained to
  predict correctness) contain information related to factual correctness
  of the model's own generations.
- Model-generated data selected according to these internal signals can
  improve factuality in the specific finetuning settings studied.
- A binary framing of "the model knows" versus "the model does not know" is
  too coarse a description of what these internal signals capture.
- Information represented internally and information expressed in the final
  generation are not necessarily equivalent — a model can carry an internal
  signal correlated with correctness while still producing an incorrect
  surface answer.

**Central lesson**

The strength of an existing parametric preference (however it is measured —
internally via probes, or behaviorally via output statistics) may affect how
readily external evidence changes model behavior. This paper studies
training-time use of internal signals; it does not test what happens at
inference time when a fixed model is confronted with evidence that
contradicts a strong existing preference.

**Open question**

At inference time, what happens when external evidence conflicts with a
strong existing parametric preference? Does preference strength predict
resistance to that evidence?

---

## 4. In Agents We Trust, but Who Do Agents Trust?

**Core question**

Does attributed source identity systematically affect which information
LLMs prioritize?

**Relevant findings/concepts**

- Models can exhibit systematic source preferences (e.g., consistently
  favoring one named information source over another across repeated
  trials).
- Source identity can influence information selection or prioritization
  independent of the specific task.
- These preferences can be predictable and measured directly, by asking a
  model to compare sources, or indirectly, by observing downstream choices.
- Source information can influence behavior independently of semantic
  content — i.e., changing only the attributed source, while holding the
  substantive claim fixed, can change model behavior.
- Source metadata is therefore potentially an active variable in reliable
  information synthesis, not an inert label.

**Central lesson**

    evidence use = function(content, source attribution, model preference, ...)

**Open question**

Does source preference affect whether an LLM abandons or preserves an
existing parametric factual preference? This paper studies source
preference in selection/ranking tasks; it does not test source preference
under conditions where source-attributed evidence directly contradicts the
model's own baseline factual answer.

---

## Cross-paper synthesis

The four papers expose different components of one information-processing
pipeline:

    WildHallucinations
        -> external information availability does not guarantee faithful use

    HALoGEN
        -> correct information in training/context does not guarantee correct output

    Factuality Finetuning
        -> models expose measurable internal factual preferences/signals

    Agents Trust
        -> source identity independently influences information selection

No individual paper studies all of these components jointly. This motivates
a pilot study that puts them in the same experimental frame:

    parametric preference
        + evidence content
        + source identity
        -> evidence selection
        -> final answer

Read `docs/research_proposal.md` for how this is narrowed into falsifiable
research questions, and `docs/phase2_research_design.md` for the exact
experimental specification. See the related-work boundary in
`docs/research_proposal.md` for why "LLMs experience knowledge conflicts" or
"source credibility can affect decisions" are not, by themselves, presented
as this project's novelty — that general phenomenon already has prior work
(e.g., Tug-of-War between Knowledge, FaithfulRAG, CARE, and related
context-memory conflict studies).
