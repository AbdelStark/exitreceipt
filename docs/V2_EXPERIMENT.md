# V2 experiment contract: sandbox-scored action logs

**Locked before evaluating the v2 test split.** The 40-case authored pilot was
inspected during v2 design and is retired as a confirmation set. Its numbers
remain available as historical, exploratory evidence.

## Question and scope

Can a LoRA adapter improve GLiNER2.5-Decide's judgment of whether recorded
agent actions completed a workplace task? The input is the user goal plus a
compact list of recorded actions. The label is WorkBench's sandbox-scored
`correct` verdict. An action log omits tool return values and final database
state, so some labels may be impossible to infer from the supplied input. This
is a **partial-observation outcome-prediction study**, not a verified completion
service or a claim about arbitrary agent trajectories.

## Source and selection

[WorkBench](https://github.com/olly-styles/WorkBench) is MIT licensed and
provides item-level 2026 agent runs and sandbox verdicts. The source export is
pinned to commit `49c7dfd00c03d384ec59ea57374f50b766aa5613`, SHA-256
`6035fb112bbb4fb48cd56d643898cac43c42c2d11bc6d2bb0e595d8d31ea2216`.
The upstream copyright and license are preserved in
[`data/WORKBENCH_LICENSE`](../data/WORKBENCH_LICENSE).

The [builder](../scripts/build_workbench_v2.py) selects `revisited_2026`, v2
ground truth, all-tool runs with no recorded execution error. It requires 1–6
recorded actions and a rendered log of at most 800 characters. For each task
with both sandbox-successful and unsuccessful eligible runs, a fixed hash picks
one of each. Pairs with identical rendered actions are excluded. This yields
590 task pairs, 1,180 examples. The class balance is by construction and does
not represent the prevalence of successful agent runs.

The 68 eligible `base_template` groups are divided across train, development, and test
within each domain using seed `20260925`. A task and all its paraphrases and
agent runs stay in one split. The selected external examples are 838 train,
184 development, and 158 test. Training also includes the original pilot's
72 authored **train** examples, including 24 literal span annotations. No
original pilot development or test example enters v2 training or selection.
The test split is new to this project and has not been inspected through model
predictions. Exact rows, source model IDs, template IDs, and source result
files are in the [manifest](../data/workbench-v2-manifest.json).

Input rendering is fixed:

```text
Goal: <WorkBench task>
Last observed state: Recorded tool actions: <actions in observed order>
```

The original action names and argument strings are retained, except that
`.func(` is rendered as `(`. No ground-truth action, `correct` verdict, or
score is included in the model input. We do not use WorkBench's test examples
to choose hyperparameters or checkpoint.

## Training and decision rule

Both models use the same pinned base revision and joint `finished` plus
`receipt`/`blocker` schema. V2 uses the original rank-8 LoRA targets and
learning rate, with dropout 0.1, batch size 4, and at most three epochs. The
trainer selects the lowest **development loss** checkpoint per seed. Train
three seeds: `20260925`, `20260926`, `20260927`. The publishable adapter is
selected by lowest development loss across those three runs, before comparing
test predictions. The other two runs show seed sensitivity.

Primary endpoint: paired test accuracy difference (tuned minus base) on the
158 external examples. Secondary endpoints: false-complete rate on the 79
failures, true-complete recall on the 79 successes, macro-F1, per-domain
accuracy, and changed examples. Report numerator and denominator, each seed,
and a paired bootstrap confidence interval clustered by `base_template`.
This interval describes variation over the 10 held-out template groups;
it is not a guarantee on future agents or workflows. Also report the task-pair
cluster interval for comparison. No accuracy claim will be made from span
overlap on the external examples: there are no gold evidence spans in that
source.

A positive headline requires at least a five-point accuracy gain for the
development-selected seed, with a template-cluster 95% interval entirely above
zero and no rise of more than two points in false-complete rate. Otherwise the
result will be reported as mixed, neutral, or negative. The full report will
include both successes and regressions. This rule is a reporting threshold,
not an optimization target.

## Reproduction

```bash
python3 scripts/build_workbench_v2.py --verify
uv run --extra model exitreceipt check-data \
  --data data/v2-cases.psv --evidence data/v2-evidence.psv
uv run --extra model exitreceipt train \
  --data data/v2-cases.psv --evidence data/v2-evidence.psv \
  --run-dir runs/v2-20260925 --device mps --epochs 3 --batch-size 4 \
  --seed 20260925 --lora-dropout 0.1 --experiment-name exitreceipt-v2
uv run --extra model exitreceipt evaluate \
  --data data/v2-cases.psv --evidence data/v2-evidence.psv \
  --run-dir runs/v2-20260925 --split test --output results/v2-20260925.json
```

Repeat the last two commands with the other seeds. The original WorkBench
source is fetched only for `--verify`; the derived rows and source-row
manifest are committed. The base checkpoint download and local LoRA run
require substantial disk, RAM, and compute. Metal/MPS can be nondeterministic
despite fixed seeds. Model weights remain out of Git and a tagged adapter is
published separately on Hugging Face after evaluation.
