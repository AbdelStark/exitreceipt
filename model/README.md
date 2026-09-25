---
language:
- en
license: apache-2.0
library_name: peft
pipeline_tag: text-classification
base_model: fastino/GLiNER2.5-Decide
base_model_relation: adapter
tags:
- gliner2
- lora
- peft
- agent-evaluation
- information-extraction
- synthetic-data
---

# ExitReceipt · GLiNER2.5-Decide LoRA pilot

**What it is.** A 14.7 MB PEFT LoRA adapter for
[`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide).
Given a requested outcome and a last observed state, it predicts `finished: yes/no`
and proposes `receipt` or `blocker` text spans. It was trained locally with the
[ExitReceipt](https://github.com/AbdelStark/exitreceipt) experiment. This
repository contains the adapter, not the base model.

**What the result says.** On 40 held-out, hand-authored synthetic cases, both
the pinned base model and this adapter scored **36/40 completion decisions**.
False “done” decisions fell from 2/20 to 1/20 incomplete cases, while false
“not done” decisions rose from 2/20 to 3/20 complete cases. On 16 cases with a
literal evidence annotation, typed candidate spans overlapping at least half
of the gold phrase rose from 0/16 to 9/16; exact typed matches were 3/16.
Two decisions improved and two regressed. **Completion accuracy did not
improve.** The overlap score is permissive, and candidate spans are not
independently verified receipts.

The [per-case report](https://github.com/AbdelStark/exitreceipt/blob/main/results/pilot.json),
[analysis](https://github.com/AbdelStark/exitreceipt/blob/main/docs/RESULTS.md),
and [interactive case explorer](https://abdelstark.github.io/exitreceipt/) show
every error. The result is an in-domain pilot, not a benchmark of real agents,
unseen workflows, or task completion in the world.

## Load and try it

The adapter needs `gliner2==2.0.0`, PEFT, and the base checkpoint at revision
`7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6`. The simplest supported
path is the project's CLI:

```bash
git clone https://github.com/AbdelStark/exitreceipt.git
cd exitreceipt
uv sync --locked --extra model
uv run --extra model exitreceipt predict \
  --goal "Merge the approved pull request" \
  --trace "All checks are green, but the PR remains open" \
  --adapter-repo abdelstark/exitreceipt-gliner2.5-decide-lora \
  --adapter-revision v0.1.0
```

Or load the pinned base explicitly and attach the adapter:

```python
from gliner2 import AutoExtractor
from peft import PeftModel

base = AutoExtractor.from_pretrained(
    "fastino/GLiNER2.5-Decide",
    revision="7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6",
)
model = PeftModel.from_pretrained(
    base,
    "abdelstark/exitreceipt-gliner2.5-decide-lora",
    revision="v0.1.0",
).eval()
schema = (
    model.create_schema()
    .entities(["receipt", "blocker"])
    .classification("finished", ["yes", "no"])
)
text = (
    "Goal: Merge the approved pull request\n"
    "Last observed state: All checks are green, but the PR remains open"
)
print(model.extract(text, schema, include_spans=True))
```

Use that exact input rendering and **joint schema** to reproduce the reported
comparison. This is a GLiNER2 custom extractor with a PEFT adapter; it is not
a standalone Transformers text-classification checkpoint. Downloaded base
weights are approximately 2 GB. The tagged adapter revision keeps this example
stable if the model repository changes later.

## Training and evaluation

<!-- markdownlint-disable MD013 -->
| Item | Value |
| --- | --- |
| Base checkpoint | `fastino/GLiNER2.5-Decide` at `7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6` |
| Library | `gliner2==2.0.0`; PEFT 0.21.0 |
| Corpus | 132 authored synthetic English goal/trace pairs: 72 train, 20 development, 40 test |
| Span annotations | 24 train, 8 development, 16 test; one literal phrase each |
| Text rendering | `Goal: ...\nLast observed state: ...` |
| Tasks | Single-label `finished: [yes, no]` plus `receipt`/`blocker` entities |
| Adapter | Rank 8, alpha 16; encoder, span representation, classifier |
| Optimization | 6 epochs, batch 4, seed 20260925; encoder LR 2e-5, task LR 3e-4 |
| Selection | Lowest development loss, reached at epoch 2; test excluded |
| Hardware | Local Apple Silicon MPS for this run; PyTorch backend may be nondeterministic |
| Trainable parameters | 3,833,864 of 490,277,917 reported by the installed trainer (0.78%) |
| Selected adapter SHA-256 | `f602971510dd8501ed66f979b8194a53918cfb0c426302048a821ac9d2537752` |
| Corpus SHA-256 | `c1772ef3b6989b70faa31acdbef506cb3193d943210f344639a0d1075223f565` |
| Annotation SHA-256 | `f1bc2d98dbd7ad97010b2fa5cc66c4ff73f28c13ae3159f7bc09b768f6ebe32d` |
<!-- markdownlint-enable MD013 -->

The model family's card describes a 340M-class model; the installed trainer
reports 490M parameters for its configured module. These numbers use different
counting boundaries and are reported here as given, not reconciled into a new
deployment-footprint claim.

| Held-out measure | Base | Adapter |
| --- | ---: | ---: |
| Completion accuracy | 36/40 (90%) | 36/40 (90%) |
| Macro-F1 | 0.900 | 0.900 (rounded) |
| False “done” on incomplete tasks | 2/20 | 1/20 |
| False “not done” on complete tasks | 2/20 | 3/20 |
| Typed span with ≥50% gold overlap | 0/16 | 9/16 |
| Exact typed span | 0/16 | 3/16 |

The base and adapter were evaluated on the same test inputs and the same
labels, schema, and scoring code. The included `evaluation/pilot.json` records
every input, prediction, span, and aggregate. `training/exitreceipt-run.json`
records the training config, loss history, source revision, and data hashes.
The evaluation report is also versioned in the source repo. Reproduce the
training and comparison with the
[experiment contract](https://github.com/AbdelStark/exitreceipt/blob/main/docs/EXPERIMENT.md).

## Intended use and limits

This adapter is for research, reproduction, and human review of a narrow
goal/observed-state decision format. It cannot inspect external systems,
authenticate receipts, or authorize an agent to send, merge, deploy, buy, or
mark a task complete. Do not use it as a sole completion gate in consequential
workflows.

All examples were written by the project author. Related domains occur in
train and test, and lexical cues may be easy. The sample is too small and
constructed to support a population performance claim. There are no consented
real agent traces, independent annotators, grouped workflow holdouts, latency
measurements, calibration study, or Jev comparison in this release. A typed
span may be a plausible quotation while the underlying event never happened.

## Attribution and license

The adapter and authored data are Apache-2.0 licensed. The base model and
GLiNER2 library are separate [Fastino projects](https://github.com/fastino-ai/GLiNER2)
under their respective terms. No Fastino base weights are redistributed here.
See the [base model card](https://huggingface.co/fastino/GLiNER2.5-Decide) and
the [GLiNER2 paper](https://arxiv.org/abs/2507.18546) for upstream details.
ExitReceipt is an independent experiment, not an official Fastino release.
