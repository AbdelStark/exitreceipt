# ExitReceipt pilot result

The result below is from one selected local LoRA adapter and one evaluation of the held-out test split. The machine-readable [report](../results/pilot.json) contains the model/data hashes, package versions, settings, split counts, and every prediction and candidate span.

| Measure | Base | Fine-tuned |
| --- | ---: | ---: |
| Completion accuracy (40 cases) | 36/40 (90%) | 36/40 (90%) |
| Macro-F1 | 0.900 | 0.900 (rounded) |
| False “done” on 20 incomplete cases | 2 | 1 |
| False “not done” on 20 complete cases | 2 | 3 |
| Evidence phrase hit, at least half the gold span and right type (16 cases) | 0/16 | 9/16 |
| Exact evidence span and type (16 cases) | 0/16 | 3/16 |
| Candidate spans with wrong type or no gold overlap | 0 | 3 |

The adapter changed four decisions: it corrected `te025` and `te035`, but regressed on `te039` and `te040`. The net false “done” count fell by one, while the total completion accuracy did not improve. The span improvement is more pronounced, but exact boundaries remain weak; the tuned model returned 12 candidate spans for 16 annotated cases and only three exact hits. A half-span hit is a loose overlap measure, not proof of a correct explanation.

The checkpoint was chosen by development loss only. The training corpus contains 72 authored rows, with 24 span annotations; development has 20 rows and 8 span annotations; test has 40 rows and 16 span annotations. All cases are synthetic English workflow descriptions, and related domains occur in every split. The result does not measure performance on real agent traces, unseen workflows, live receipts, latency, calibration, or safety. It is not a Jev comparison.

The practical result: GLiNER2.5-Decide was already strong at this completion label on the authored data. A small adapter added the ability to point to some domain-specific receipt/blocker phrases, but it introduced one new false “done” call. The explorer exposes those errors so the model is not mistaken for a completion authority.
