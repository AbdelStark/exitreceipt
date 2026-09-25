# Experiment contract

## Question

Can local LoRA fine-tuning of [`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide) improve single-label completion decisions for goal-and-trace pairs, especially reducing false claims that incomplete tasks are finished?

The decision head is `finished: [yes, no]`. The same schema also extracts candidate `receipt` and `blocker` spans. The text format is fixed:

```text
Goal: <the requested outcome>
Last observed state: <what an agent or tool actually reported>
```

Both base and tuned models see the same text and label set. A model prediction is not a verified completion receipt. This project does not authorize an agent to act, publish, send, or close a task.

## Dataset and split

`data/cases.psv` is the source of truth: 132 authored, synthetic English cases, balanced in each split (72 train, 20 development, 40 test). Domains cover mail, Git, deployment, commerce, files, compound goals, reversals, conflicts between agent claims and provider state, and scope boundaries such as "draft" versus "send". Each row has a stable ID, domain, task goal, last observed state, and ground-truth label. The ground truth is assigned from the stated outcome and the observed state, not from a model. `data/evidence.psv` additionally marks one decisive literal phrase on 48 complex cases (24 train, 8 development, 16 test). Phrases are capped at eight whitespace-separated words to fit the checkpoint's configured span width; its tokenizer may split words differently.

The test split is held out from training and checkpoint selection. Rows are hand-written, not sampled from live agent traces. Related operational domains appear across splits; this is an **in-domain phrasing test**, not a test of transfer to unseen organizations, workflows, or receipt formats. Some states explicitly contain words such as "sent" or "draft". The benchmark may be easier than real agent logs.

## Procedure

1. Check corpus IDs, exact-text duplicates, required fields, labels, balance, and SHA-256.
2. Load the pinned base revision `7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6` with `gliner2==2.0.0`.
3. Train a rank-8 LoRA adapter over the encoder, span representation, and classifier on the 72 train rows, including the 24 annotated span examples. Select `best` by development loss only. Keep adapter weights in ignored `runs/`.
4. Run base and selected adapter through the **same joint classification and span schema** on the test rows exactly once for the final pilot. Export machine-readable per-case predictions and candidate spans.
5. Report accuracy, macro-F1, false-complete count/rate, false-incomplete count, and per-label metrics. For the 16 annotated test rows, report exact span-and-type hits and hits covering at least half the gold phrase, plus candidate and spurious span counts. These span measures are a deliberately small pilot; overlapping part of a phrase is not proof that the model understood the receipt. A difference on 40 synthetic rows has high uncertainty; no significance or broad safety claim follows.

The model card calls this a 340M-class model, while the installed trainer reports approximately 490M total parameters when counting its full configured module. The report quotes the model card for identity and logs the trainer count without treating either as an independently measured deploy footprint.

## Reproduce

```bash
uv sync --extra model --extra dev
uv run --extra model exitreceipt check-data
uv run --extra model exitreceipt train --run-dir runs/pilot --device mps --epochs 6 --batch-size 4
uv run --extra model exitreceipt evaluate --run-dir runs/pilot --split test --output results/pilot.json
uv run --extra dev pytest -q
uv run --extra dev ruff check .
python3 -m http.server 8765
# open http://localhost:8765/site/
```

Use `--device cpu` on non-Apple machines. The upstream trainer selects CPU by default on macOS; this project explicitly moves its model to MPS after setup. The MPS path was validated locally but is not guaranteed on every PyTorch/macOS combination. PyTorch can still have nondeterministic kernels. The pinned seed, corpus hash, checkpoint revision, package lock, device, and chosen settings are recorded. `runs/` and downloaded weights are intentionally excluded from Git.

## Next research stage

After this MVP, a separately preregistered study can add real consented agent traces, stronger grouped holdouts, latency and memory measurements, and a Jev comparison restricted to the shared single-label `yes/no` contract. Any results comparing systems must use identical examples and labels, explicit endpoint/model revisions, and the same scoring script. No such comparison is claimed here.
