# ExitReceipt pilot result

This result comes from one selected local LoRA adapter and one evaluation of
the held-out test split. The machine-readable [report](../results/pilot.json)
contains model/data hashes, versions, settings, split counts, every prediction,
and candidate spans.

The current report records source commit
[`04a8ec7`](https://github.com/AbdelStark/exitreceipt/commit/04a8ec7e1174aa70d6243d1d157ffc525e316fb0)
and adapter SHA-256
`f602971510dd8501ed66f979b8194a53918cfb0c426302048a821ac9d2537752`.
It is a fresh local rerun of the original pilot: the selected adapter bytes
changed slightly, while every base and tuned test prediction and candidate
span matched the earlier report. The original report remains in Git history.

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

The evaluated [LoRA adapter](https://huggingface.co/abdelstark/exitreceipt-gliner2.5-decide-lora)
is published separately from the base weights, with a model card and the same
per-case report. It is a reusable research artifact, not a verified task
completion service.

## Seed sensitivity on the same authored split

After the original pilot, two more seeds were trained with the same corpus,
configuration, development-loss selection rule, and test rows. This is a
**post-hoc robustness probe**, not a fresh holdout or a significance test.
No checkpoint was selected by its test score. The public model remains the
first seed's selected adapter.

<!-- markdownlint-disable MD013 -->
| Seed | Base accuracy | Tuned accuracy | False “done” | Typed ≥50% overlap | Exact typed spans |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20260925 (published adapter) | 36/40 | 36/40 | 1/20 | 9/16 | 3/16 |
| 20260926 | 36/40 | 36/40 | 0/20 | 10/16 | 3/16 |
| 20260927 | 36/40 | 37/40 | 0/20 | 12/16 | 4/16 |
<!-- markdownlint-enable MD013 -->

The base had 2/20 false “done” calls and 0/16 typed overlap in every run.
Across all three adapters, `te025` and `te035` were corrected, while `te039`
regressed. Other regressions varied by seed. These results show sensitivity
to initialization on a tiny authored test and no evidence of transfer to
unseen workflows. The [method and per-case reports](SEEDS.md) are public.
