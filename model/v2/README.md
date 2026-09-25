---
language:
- en
license: apache-2.0
library_name: peft
pipeline_tag: text-classification
base_model: fastino/GLiNER2.5-Decide
base_model_relation: adapter
metrics:
- accuracy
- f1
tags:
- gliner2
- lora
- peft
- agent-evaluation
- outcome-prediction
- workbench
- partial-observation
---

# ExitReceipt v2 · GLiNER2.5-Decide LoRA

A **14.7 MB PEFT LoRA adapter** for [Fastino's GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide), trained with [ExitReceipt](https://github.com/AbdelStark/exitreceipt) to classify a workplace task and recorded agent actions as `finished: yes/no`. It also returns candidate `receipt` or `blocker` text spans through GLiNER2's joint schema. This repository contains adapter weights, not the base model.

**Interpret the label carefully.** V2 maps `yes` to [WorkBench](https://github.com/olly-styles/WorkBench)'s sandbox `correct` verdict, which can penalize unwanted side effects. The supplied action log omits tool returns and final sandbox state. Some verdicts cannot be inferred from that view alone. The adapter is a research aid for inspecting partial evidence, **not a verified task-completion authority**.

The root adapter at tag **`v0.2.0`** is seed `20260925`, chosen by lowest development loss [before test inference](https://github.com/AbdelStark/exitreceipt/tree/v2-pretest-lock). The other two fixed-seed adapters and their configs are in `seeds/20260926/` and `seeds/20260927/` for reproducibility; neither was promoted based on test score.

## Evaluation

The external study pairs one sandbox-successful and one unsuccessful agent run for each of 590 WorkBench tasks. All runs of a task and all paraphrases of its base template stay in one split. The test has **158 runs from 79 tasks and 10 templates unseen during training**. Classes are balanced by design, not by real-world prevalence. All models use the same input rendering and joint GLiNER2 schema.

<!-- markdownlint-disable MD013 -->
| Model | Correct / 158 | False “done” / 79 failures | Missed “done” / 79 successes | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Pinned base | 87 (55.1%) | 58 | 13 | 0.511 |
| Historical ExitReceipt v1 adapter | 79 (50.0%) | 79 | 0 | 0.333 |
| **This adapter · selected seed 20260925** | **96 (60.8%)** | **34** | **28** | **0.607** |
| V2 seed 20260926 | 100 (63.3%) | 26 | 32 | 0.632 |
| V2 seed 20260927 | 109 (69.0%) | 27 | 22 | 0.690 |
<!-- markdownlint-enable MD013 -->

The selected adapter's accuracy gain over base is **+5.7 percentage points**. A paired 95% bootstrap interval clustered by the ten held-out templates is **−6.6 to +15.8 points**, so the preregistered positive-headline rule was **not met**. The selected model corrected 30 base errors and introduced 21 new ones. It sharply reduced false “done” calls while rejecting more successful runs. The email domain regressed from 13/20 to 7/20 correct. Seed 27 did better on this test, but choosing it as the main adapter after seeing test scores would violate the pre-test selection rule. Its result is exploratory seed sensitivity.

The historical v1 adapter predicted `yes` on every external row. V2 uses much more external training data and also changes dropout and data composition; the difference does not isolate one causal factor. The v2 test has **no gold receipt/blocker spans**, so no evidence-span improvement is claimed here.

Read the [full analysis](https://github.com/AbdelStark/exitreceipt/blob/main/docs/V2_RESULTS.md), [experiment contract](https://github.com/AbdelStark/exitreceipt/blob/main/docs/V2_EXPERIMENT.md), [case explorer](https://abdelstark.github.io/exitreceipt/), and [three-seed report](evaluation/v2-robustness.json). Every test prediction is in [`evaluation/v2-selected.json`](evaluation/v2-selected.json); the historical comparator is in [`evaluation/v2-v1-comparator.json`](evaluation/v2-v1-comparator.json).

## Load and try it

The adapter requires `gliner2==2.0.0`, PEFT, and the base checkpoint at revision `7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6`. Use the project's CLI for the supported input rendering:

```bash
git clone https://github.com/AbdelStark/exitreceipt.git
cd exitreceipt
uv sync --locked --extra model
uv run exitreceipt predict \
  --goal "Cancel my first meeting on December 13" \
  --trace 'Recorded tool actions: calendar.search_events(time_min="2023-12-13"); calendar.delete_event(event_id="00000256")' \
  --adapter-repo abdelstark/exitreceipt-gliner2.5-decide-lora \
  --adapter-revision v0.2.0
```

To load through GLiNER2 and PEFT directly:

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
    revision="v0.2.0",
).eval()
schema = (
    model.create_schema()
    .entities(["receipt", "blocker"])
    .classification("finished", ["yes", "no"])
)
text = (
    "Goal: Cancel my first meeting on December 13\n"
    "Last observed state: Recorded tool actions: "
    "calendar.search_events(time_min=\"2023-12-13\"); "
    "calendar.delete_event(event_id=\"00000256\")"
)
print(model.extract(text, schema, include_spans=True))
```

This is a **GLiNER2 custom extractor with a PEFT adapter**, not a standalone `AutoModelForSequenceClassification` checkpoint. Candidate span offsets refer to the rendered input and are not independent evidence that an external action happened. The tagged historical adapter remains at [`v0.1.1`](https://huggingface.co/abdelstark/exitreceipt-gliner2.5-decide-lora/tree/v0.1.1).

For an alternate seed, attach its adapter with PEFT's `subfolder` argument (for example `subfolder="seeds/20260927"`, `revision="v0.2.0"`). It is provided to reproduce the seed study, not as a test-selected replacement for the root adapter.

## Training data and method

The [pinned WorkBench export](data/workbench-v2-manifest.json) is MIT licensed; its [copyright and terms](data/WORKBENCH_LICENSE) are included. The deterministic [builder](https://github.com/AbdelStark/exitreceipt/blob/main/scripts/build_workbench_v2.py) selected 1–6-action logs of at most 800 rendered characters, then hash-selected one successful and one failed eligible run per task. The source export is pinned to commit `49c7dfd00c03d384ec59ea57374f50b766aa5613` and SHA-256 `6035fb112bbb4fb48cd56d643898cac43c42c2d11bc6d2bb0e595d8d31ea2216`. The derived [training/evaluation cases](data/v2-cases.psv) and 24 [auxiliary span annotations](data/v2-evidence.psv) are included.

There are 838 external training rows plus 72 original authored pilot **train** rows, 184 external development rows, and 158 external test rows. No pilot dev or test row was used for v2. The model was fine-tuned locally on Apple Silicon MPS with rank-8 LoRA, alpha 16, dropout 0.1, target modules in the encoder, span representation, and classifier, task learning rate `3e-4`, batch size 4, and at most three epochs. The native trainer reported **3,833,864 trainable / 490,277,917 total parameters**. Development loss selected a checkpoint within each seed and seed `20260925` across the three fixed runs.

A six-epoch, same-seed development-only probe had worse best development loss (`4.183` versus `3.888` for the three-epoch schedule). Because the linear learning-rate schedule also changed, this is not proof that longer training universally harms performance. The [length-probe receipt](evaluation/v2-length-probe.json) and [seed lock](evaluation/v2-selection.json) were recorded without using the test split.

| Exact artifact | Identity |
| --- | --- |
| Base model revision | `7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6` |
| Selected adapter SHA-256 | `d31c4000b3610cf2a7f6effcd1f7c93730508c48a79038658e440faeb82b0646` |
| V2 cases SHA-256 | `55f9d4c018ab8fafb2b64ba0f4d688ea781788af133378d429ccdf38e7ab48ad` |
| Span annotations SHA-256 | `3e30eb0e47f7cf59424f0de9435b4ec31585f7f9685fb6397492dd34ac5b24f3` |
| Locked package set SHA-256 | `8d885dc3bf270e055d8ba5f6cdea4994f8024fedbfe3cb0f524267f96a799486` |

The `adapter_config.json` differs from the upstream PEFT export only by omitting its optional `task_type: null` field, as in the v1 release; the saved weights are unchanged. Training metadata, exact trainer configuration, and the original [`training/uv.lock`](training/uv.lock) are included. The repository's 0.2.0 lockfile changes only its own package-version entry; the training lock remains pinned here. Reproduce the scores and confidence intervals using the [repository commands](https://github.com/AbdelStark/exitreceipt#reproduce-the-v2-study).

## Intended use and limits

Use this adapter to study how open-weight fine-tuning changes judgments on **short, English, WorkBench-like agent action records**, or as a starting point for your own separately evaluated task semantics. Do not use it to automatically mark tasks complete, publish or send on an agent's behalf, enforce policy, or assert that a tool call succeeded. Read the provider or sandbox state for authoritative verification.

This dataset is synthetic workplace simulation, selected to include both success and failure for each task. It is not a representative sample of production agents. Only ten task templates are held out; domains overlap across splits. Logs omit tool outputs, and WorkBench `correct` includes task correctness and possible side effects, not just the literal requested mutation. This evaluation does not establish transfer to new organizations, unseen tools, real-world traces, or calibrated risk thresholds. No real-world safety or reliability claim follows.

## Attribution

The adapter and ExitReceipt code are [Apache-2.0 licensed](LICENSE). The derived WorkBench data retains the upstream [MIT license](data/WORKBENCH_LICENSE). The GLiNER2 base model and library are separate Fastino releases with their own terms. Please cite the [upstream GLiNER2 work](https://github.com/fastino-ai/GLiNER2) and [WorkBench](https://github.com/olly-styles/WorkBench) when using their model or benchmark.
