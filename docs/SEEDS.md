# Three-seed sensitivity check

This supplements the [selected adapter pilot](RESULTS.md). The first seed
(`20260925`) was the original release and is the only adapter published on
Hugging Face. After inspecting its test results, two additional seeds
(`20260926`, `20260927`) were run with unchanged data, hyperparameters, base
revision, and development-loss checkpoint selection. This is a **post-hoc
check on the same 40 authored synthetic test rows**, not an independent
benchmark. All three runs are reported; none was chosen by test accuracy.

<!-- markdownlint-disable MD013 -->
| Seed | Selected dev loss | Base correct | Tuned correct | False “done” tuned | Typed half-span hits | Exact typed hits | Corrected / regressed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260925 | 5.5169 | 36/40 | 36/40 | 1/20 | 9/16 | 3/16 | 2 / 2 |
| 20260926 | 6.3223 | 36/40 | 36/40 | 0/20 | 10/16 | 3/16 | 2 / 2 |
| 20260927 | 5.7561 | 36/40 | 37/40 | 0/20 | 12/16 | 4/16 | 2 / 1 |
<!-- markdownlint-enable MD013 -->

The base model had 2/20 false “done” calls, 2/20 false “not done” calls, and
0/16 typed half-span hits. Its predictions and spans were identical in all
three report files. The two extra adapters each fixed `te025` and `te035`,
and each regressed on `te039`; seed `20260926` also regressed on `te016`.
The published seed regressed on `te039` and `te040`. No run was free of
regressions.

## Reproduce

From a clean clone with the locked model dependencies installed:

```bash
uv sync --locked --extra model --extra dev
for seed in 20260926 20260927; do
  uv run --locked --extra model exitreceipt train \
    --run-dir "runs/seed-$seed" --device mps \
    --epochs 6 --batch-size 4 --seed "$seed"
  uv run --locked --extra model exitreceipt evaluate \
    --run-dir "runs/seed-$seed" --split test \
    --output "results/seed-$seed.json"
done
uv run python scripts/summarize_seeds.py
```

Use `--device cpu` if MPS is unavailable. The training path records the
source Git commit, lockfile hash, data hashes, exact configuration, selected
adapter hash, and train/development losses. MPS/PyTorch may produce slightly
different adapter bytes on rerun. The first seed's current report was a
fresh rerun at source commit `04a8ec7`; its predictions and spans exactly
matched the originally published pilot. The two additional runs used clean
source commit `f258ddb`.

Inspect the complete [first](../results/pilot.json),
[second](../results/seed-20260926.json), and
[third](../results/seed-20260927.json) reports, or the compact
[machine-readable summary](../results/seed-robustness.json). The summary
checks identical data, test rows, base predictions, and model revision before
combining runs. It lists every corrected and regressed case ID and the SHA-256
of each report. This split shares writing style and domains with training;
repeating seeds cannot replace consented real traces, independent labels,
or grouped holdouts.
