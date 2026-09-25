# ExitReceipt pilot result

This result comes from one selected local LoRA adapter and one evaluation of
the held-out test split. The machine-readable [report](../results/pilot.json)
contains model/data hashes, versions, settings, split counts, every prediction,
and candidate spans.

The report records source commit
[`ae105bc`](https://github.com/AbdelStark/exitreceipt/commit/ae105bcd09ed275dbe7164ef40504d552b2821cf),
made before the project was renamed to ExitReceipt. That commit and its
lockfile remain in the repository history. Later naming and publication
changes did not alter the scored pilot.

| Measure | Base | Fine-tuned |
| --- | ---: | ---: |
| Completion accuracy (40 cases) | 36/40 (90%) | 36/40 (90%) |
| Macro-F1 | 0.900 | 0.900 (rounded) |
| False “done” (20 incomplete cases) | 2 | 1 |
| False “not done” (20 complete cases) | 2 | 3 |
| Typed span with ≥50% gold overlap (16 cases) | 0/16 | 9/16 |
| Exact typed span (16 cases) | 0/16 | 3/16 |
| Wrong-type or disjoint candidate spans | 0 | 3 |

The adapter changed four decisions: it corrected `te025` and `te035`, but
regressed on `te039` and `te040`. False “done” calls fell by one, while total
completion accuracy stayed flat. The tuned model returned 12 candidate spans
for 16 annotated test cases; only three exactly matched both phrase and type.
A half-span hit is a loose overlap measure, not proof of a correct explanation.

The checkpoint was chosen by development loss only. Train has 72 authored
rows and 24 span annotations; development has 20 and 8; test has 40 and 16.
Related workflow domains appear in every split. The result does not measure
real agent traces, unseen workflows, live receipts, latency, calibration, or
safety. It is not a Jev comparison.

GLiNER2.5-Decide was already strong at this completion label on the authored
data. The small adapter learned to point to some domain-specific receipt or
blocker phrases, but it also introduced one new false “done” call. The
[explorer](https://abdelstark.github.io/exitreceipt/) exposes these errors so
the model is not mistaken for a completion authority.
