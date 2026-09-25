# Experiment contract

## Question

Can local LoRA fine-tuning of
[`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide)
improve single-label completion decisions for goal-and-trace pairs, especially
reducing false claims that incomplete tasks are finished?

The decision head is `finished: [yes, no]`. The same schema extracts candidate
`receipt` and `blocker` spans. The text format is fixed:

```text
Goal: <the requested outcome>
Last observed state: <what an agent or tool actually reported>
```

Both base and tuned models see the same text and label set. A prediction is not
a verified completion receipt. This project does not authorize an agent to
act, publish, send, or close a task.

## Dataset and split

`data/cases.psv` is the source of truth: 132 authored, synthetic English cases,
balanced in each split (72 train, 20 development, 40 test). Domains cover mail,
Git, deployment, commerce, files, compound goals, reversals, conflicts between
agent claims and provider state, and scope boundaries such as “draft” versus
“send”. Each row has a stable ID, domain, goal, last observed state, and
ground-truth label. The label follows the stated outcome and observed state,
not a model prediction.

`data/evidence.psv` marks one decisive literal phrase on 48 complex cases:
24 train, 8 development, and 16 test. Phrases are capped at eight
whitespace-separated words to fit the checkpoint's configured span width;
its tokenizer may split words differently.

Test rows are held out from training and checkpoint selection. They are
hand-written, not sampled from live agent traces. Related operational domains
appear across splits, so this is an **in-domain phrasing test**, not a test of
transfer to unseen organizations, workflows, or receipt formats. Some states
explicitly contain cues such as “sent” or “draft”. Real agent logs may be
harder.

## Procedure

1. Validate corpus IDs, exact-text duplicates, required fields, labels,
   balance, and SHA-256 hashes.
2. Load pinned base revision `7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6`
   with `gliner2==2.0.0`.
3. Train a rank-8 LoRA adapter over the encoder, span representation, and
   classifier on the 72 training rows, including 24 annotated spans. Select
   `best` by development loss only. Keep adapter weights in ignored `runs/`.
4. Run base and selected adapter through the **same joint classification and
   span schema** on the test rows once for the final pilot. Export per-case
   predictions and candidate spans.
5. Report accuracy, macro-F1, false-complete count/rate,
   false-incomplete count, and per-label metrics. On 16 annotated test rows,
   report exact typed hits, typed hits covering at least half the gold phrase,
   candidate count, and wrong-type/disjoint candidate count.

The half-span measure is permissive: overlap is not proof that the model
understood a receipt. A difference on 40 synthetic rows has high uncertainty;
no significance or broad safety claim follows.

The model card calls this a 340M-class model, while the installed trainer
reports approximately 490M total parameters for its full configured module.
The report records the trainer count without treating either number as an
independently measured deployment footprint.

## Reproduce

```bash
uv sync --locked --extra model --extra dev
uv run --extra model exitreceipt check-data
uv run --extra model exitreceipt train \
  --run-dir runs/repro --device mps --epochs 6 --batch-size 4
uv run --extra model exitreceipt evaluate \
  --run-dir runs/repro --split test --output results/repro.json
uv run --extra dev pytest -q
uv run --extra dev ruff check .
python3 -m http.server 8765
# open http://localhost:8765/
```

Use `--device cpu` where MPS is unavailable. The upstream trainer selects CPU
by default on macOS; this project explicitly moves its model to MPS after
setup. The MPS path was validated locally but is not guaranteed on every
PyTorch/macOS combination. PyTorch can still have nondeterministic kernels.
The seed, corpus hashes, checkpoint revision, package lock, device, and
settings are recorded. `runs/` and downloaded weights are excluded from Git.
The website continues to display the checked-in pilot; a new report is written
to `results/repro.json` for separate inspection.

## Next research stage

A later preregistered study can add consented agent traces, grouped holdouts,
latency and memory measurements, and a Jev comparison restricted to the shared
single-label `yes/no` contract. Comparisons must use identical examples and
labels, explicit model/endpoint revisions, and the same scoring script. No
such comparison is claimed here.
