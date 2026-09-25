# Contributing to ExitReceipt

ExitReceipt is a small, evidence-led research project. Contributions that make
the experiment easier to reproduce, inspect, or falsify are especially useful.

## Good first contributions

- Add a minimal failing case for a data-validation or scoring bug.
- Improve keyboard or screen-reader use in the static explorer.
- Reproduce a result on another supported Python platform and report the exact
  model revision, device, dependency lock, and result artifact.
- Propose consented, independently annotated agent traces with a split plan
  that keeps related workflows together.

Do not submit private conversations, real tool logs, credentials, or personal
data as examples. Synthetic cases should be unambiguously labeled and should
include both the requested outcome and the last observed state.

## Local setup

```bash
uv sync --locked --extra dev
uv run --extra dev pytest -q
uv run --extra dev ruff check .
uv run exitreceipt check-data
node --check app.js
uv build
```

The checks above do not download model weights. For a model change, also use
the [`model` extra](README.md#run-the-model-locally) and record the base
revision, split hashes, training settings, device, adapter hash, and per-case
outputs. Keep large weights and run directories out of Git.

## Evaluation changes

The checked-in pilot is a fixed historical run. Do not tune on its test rows or
replace its report with a stronger-looking rerun. Propose a new evaluation
protocol before creating a new result: define the question, data source,
annotation rules, split/grouping, baselines, metrics, and selection procedure.
Report corrections and regressions together. If a PR changes the corpus or
scoring, explain whether the old numbers are still comparable.

## Pull requests

Keep a PR focused. Include the behavior or claim it changes, the command used
to verify it, and any remaining limitation. Update the README, experiment
contract, result note, or changelog when their visible claims change. A
maintainer review is required before a result is presented as a project result.

By participating, you agree to the [code of conduct](CODE_OF_CONDUCT.md).
