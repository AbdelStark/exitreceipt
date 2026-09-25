# Doneproof

**Did the agent actually finish?**

A draft is not a send. A green pull request is not a merge. Doneproof probes
[`GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide) on
goal-and-trace pairs, fine-tunes an open-weight LoRA adapter on both completion
decisions and candidate evidence spans, and lets you inspect every base and tuned
verdict. The model makes a review decision; the tool receipt remains the evidence
of completion.

## What you can run

```bash
uv sync --extra model --extra dev
uv run --extra model doneproof check-data
uv run --extra model doneproof train --run-dir runs/pilot --device mps
uv run --extra model doneproof evaluate --run-dir runs/pilot --split test --output results/pilot.json
python3 -m http.server 8765
```

Then open `http://localhost:8765/site/`. Use `--device cpu` on non-Apple
machines. To classify one local example:

```bash
uv run --extra model doneproof predict \
  --goal "Merge the approved pull request" \
  --trace "All checks are green, but the PR remains open" \
  --adapter runs/pilot/best
```

The corpus has 72 training, 20 development, and 40 held-out synthetic cases;
48 complex cases have a manually marked decisive phrase. Training uses the
upstream GLiNER2 trainer with a rank-8 adapter. Model revision, corpus hashes,
settings, checkpoint selection, versions, per-case predictions, and extracted
spans are recorded in the report. Downloaded weights and adapters stay local under
ignored cache/`runs` paths.

**Start with the [experiment contract](docs/EXPERIMENT.md)** for the exact
limits and reproduction details. The synthetic pilot cannot establish real
agent reliability or authorize an action. A later Jev comparison will need a
separate shared-label evaluation design.

## Measured pilot

On 40 held-out authored cases, base and tuned completion accuracy were both
**36/40**. False “done” calls fell from 2 to 1, but the adapter introduced
other errors. On 16 annotated complex cases, candidate evidence spans covering
at least half the marked phrase rose from **0/16 to 9/16**; only 3/16 matched
the exact phrase and type. Read the [full result](docs/RESULTS.md) and inspect
the per-case [report](results/pilot.json) before drawing conclusions.
