# ExitReceipt

### Did the agent actually finish?

A tool call is an attempt, not a receipt. ExitReceipt is an open-weight experiment in judging an agent's requested outcome against the evidence it left behind. It uses [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) for a `yes/no` decision and candidate `receipt` or `blocker` spans, then publishes what changed after local LoRA fine-tuning—including the misses.

**[Explore every prediction](https://abdelstark.github.io/exitreceipt/)** · **[Read the v2 result](docs/V2_RESULTS.md)** · **[Get the tagged adapter](https://huggingface.co/abdelstark/exitreceipt-gliner2.5-decide-lora/tree/v0.2.0)** · **[Inspect the experiment contract](docs/V2_EXPERIMENT.md)**

> **Scope:** The v2 benchmark predicts [WorkBench](https://github.com/olly-styles/WorkBench) sandbox `correct` verdicts from a task and its recorded actions. WorkBench can penalize unwanted side effects. The released action view omits tool returns and final state, so this is a partial-observation benchmark—not a live task verifier. The older 40-case authored pilot remains available in the [explorer](https://abdelstark.github.io/exitreceipt/?study=pilot) and [pilot analysis](docs/RESULTS.md).

## The measured result

V2 uses **590 paired WorkBench tasks** (one successful and one failed agent run per task), split by task template. The external test contains **158 runs across 79 tasks and 10 templates unseen during training**. The selected adapter was locked by development loss [before test inference](https://github.com/AbdelStark/exitreceipt/tree/v2-pretest-lock).

| Model on the same external test | Correct / 158 | False “done” / 79 failures | Missed “done” / 79 successes |
| --- | ---: | ---: | ---: |
| Pinned base GLiNER2.5-Decide | 87 (55.1%) | 58 | 13 |
| Historical v1 pilot adapter | 79 (50.0%) | 79 | 0 |
| **V2 adapter selected on development loss** | **96 (60.8%)** | **34** | **28** |

The selected adapter fixed 30 base errors and introduced 21 new ones. It reduced false completion calls but became more likely to reject successful runs. Its accuracy gain is **+5.7 percentage points**, with a paired 95% interval of **−6.6 to +15.8 points when resampling the 10 held-out templates**. That interval crosses zero, so the preregistered positive-headline rule was **not met**. Two further fixed seeds scored 100/158 and 109/158; they were not selected by the development rule. The [full result](docs/V2_RESULTS.md) reports all three, the email-domain regression, side-effect slices, task-level intervals, and every changed case.

![Development loss for the three fixed LoRA seeds. Checkpoint and published seed selection used development loss only.](docs/assets/v2-dev-loss.svg)

This is a real local fine-tune: the native `gliner2==2.0.0` trainer updated a rank-8 PEFT LoRA adapter with **3,833,864 trainable parameters**. The published tag contains the selected 14.7 MB adapter, the other two seed adapters as research artifacts, model card, transformed data, source manifest, training receipts, and per-case reports. It does not contain the 2 GB-class base weights.

## Inspect the cases

The [static explorer](https://abdelstark.github.io/exitreceipt/) shows base and tuned verdicts, source action logs, candidate spans, source agent and side-effect flag, filters for errors, and a link between the two runs of the same task. It reads checked-in JSON; **no model runs in the browser**. Try a [calendar search that did not delete the event](https://abdelstark.github.io/exitreceipt/?case=wb-calendar-001-no) and the [paired deletion run](https://abdelstark.github.io/exitreceipt/?case=wb-calendar-001-yes). Also inspect the [bar-chart success the adapter wrongly rejected](https://abdelstark.github.io/exitreceipt/?case=wb-analytics-002-yes).

The pilot toggle preserves the original authored-case study. Existing case links such as [TE025](https://abdelstark.github.io/exitreceipt/?case=te025) still work.

To serve the site from a clone:

```bash
python3 -m http.server 8765
# open http://localhost:8765/
```

## Reproduce the v2 study

Requires Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/). The `model` extra installs GLiNER2 and PyTorch; first use downloads the pinned base checkpoint. The published run used Apple Silicon MPS. CPU is supported for training; CUDA training is not currently wired into this project.

```bash
git clone https://github.com/AbdelStark/exitreceipt.git
cd exitreceipt
uv sync --locked --extra model --extra dev
python3 scripts/build_workbench_v2.py --verify
uv run exitreceipt check-data \
  --data data/v2-cases.psv --evidence data/v2-evidence.psv

uv run exitreceipt train \
  --data data/v2-cases.psv --evidence data/v2-evidence.psv \
  --run-dir runs/v2-20260925 --device mps --epochs 3 --batch-size 4 \
  --seed 20260925 --lora-dropout 0.1 --experiment-name exitreceipt-v2
```

Repeat `train` for seeds `20260926` and `20260927` in separate run directories. The repo includes the development-only [three-versus-six-epoch probe](results/v2-length-probe.json): the longer schedule had worse best dev loss for the one seed tested, so v2 returned to three epochs **before test inference**.

```bash
uv run python scripts/select_v2_checkpoint.py --verify
uv run exitreceipt evaluate \
  --data data/v2-cases.psv --evidence data/v2-evidence.psv \
  --run-dir runs/v2-20260925 --split test --output results/v2-20260925.json
# Repeat evaluate for the other two seeds.
uv run python scripts/summarize_v2.py
uv run --extra dev pytest -q
```

The selection verifier compares freshly trained checkpoint hashes and dev curves with the [committed pre-test lock](results/v2-selection.json); nondeterministic kernels may produce slightly different adapter bytes on another machine, even with fixed seeds. For a new experiment, choose a new run directory and create a new selection receipt rather than overwriting published outputs. The [experiment contract](docs/V2_EXPERIMENT.md) states the sampling rule, split grouping, metric definitions, decision rule, and limitations.

## Try the published model

```bash
uv run exitreceipt predict \
  --goal "Cancel my first meeting on December 13" \
  --trace 'Recorded tool actions: calendar.search_events(time_min="2023-12-13"); calendar.delete_event(event_id="00000256")' \
  --adapter-repo abdelstark/exitreceipt-gliner2.5-decide-lora \
  --adapter-revision v0.2.0
```

The output is a label plus candidate text spans with offsets. A candidate span is **not** independently checked against a calendar, mail provider, repository, or database. Load the pinned base revision and PEFT adapter directly by following the [model card](https://huggingface.co/abdelstark/exitreceipt-gliner2.5-decide-lora/tree/v0.2.0). The historical adapter remains at tag [`v0.1.1`](https://huggingface.co/abdelstark/exitreceipt-gliner2.5-decide-lora/tree/v0.1.1).

## Fine-tune for your own definition of done

`data/v2-cases.psv` shows the six-field format: `id|split|domain|label|goal|trace`. Use task- or organization-grouped splits when examples share context; include both `yes` and `no` in train, dev, and test. The rendered model input is always:

```text
Goal: <requested outcome>
Last observed state: <observed action or tool result>
```

`data/v2-evidence.psv` shows optional literal `receipt`/`blocker` span annotations. For classification-only training, pass a header-only `id|kind|evidence` file. The CLI validates duplicate IDs and texts, labels, splits, and annotation substrings. Keep private traces out of this public repository. A task-specific adapter needs its own independent evaluation; these WorkBench numbers do not transfer automatically.

## Repository map

| Path | What it contains |
| --- | --- |
| [`docs/V2_EXPERIMENT.md`](docs/V2_EXPERIMENT.md), [`docs/V2_RESULTS.md`](docs/V2_RESULTS.md) | Pre-test protocol, amendments, result, and error analysis |
| [`data/`](data/) | Versioned derived cases, source-row manifest, original pilot, and WorkBench license |
| [`src/exitreceipt/`](src/exitreceipt/) | Corpus validation, native GLiNER2 training, inference, and metrics |
| [`results/v2-selection.json`](results/v2-selection.json), [`results/v2-robustness.json`](results/v2-robustness.json) | Development lock, three-seed statistics, and per-case report links |
| [`index.html`](index.html), [`app.js`](app.js) | Static GitHub Pages case explorer |
| [`model/v2/README.md`](model/v2/README.md) | Hugging Face v2 adapter card maintained with the release |

## Contribute and cite

Reproductions, corrected annotations, independent task-family holdouts, and consented traces with final-state verification are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md); use [SECURITY.md](SECURITY.md) for private security reports. [CHANGELOG.md](CHANGELOG.md) tracks releases, and [CITATION.cff](CITATION.cff) provides citation metadata.

ExitReceipt's original code, authored cases, and site use [Apache-2.0](LICENSE). The derived WorkBench data retains its [MIT copyright and license](data/WORKBENCH_LICENSE). The base model and `gliner2` library are separate [Fastino projects](https://github.com/fastino-ai/GLiNER2) with their own license terms; no base weights are redistributed here. Please cite the upstream GLiNER2 and WorkBench work when using their model or benchmark.
