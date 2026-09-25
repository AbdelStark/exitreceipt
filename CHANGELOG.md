# Changelog

## Unreleased

- Published the evaluated LoRA adapter with a full Hugging Face model card,
  training metadata, tagged weights, and per-case evaluation report.
- Added pinned Hub adapter inference and refreshed the pilot report from a
  fresh rerun that reproduced every prior test prediction and span.
- Published two further development-selected seed runs and their complete
  held-out reports as a post-hoc sensitivity check on the same synthetic split.
- Normalized the Hub adapter config by omitting PEFT's optional null task type;
  the adapter weights and inference output are unchanged.

## 0.1.0 — 2026-09-25

- Added a pinned GLiNER2.5-Decide base model and native rank-8 LoRA training
  path for joint completion classification and candidate evidence spans.
- Published the authored synthetic corpus, validation rules, and held-out
  per-case pilot report. Completion accuracy was 36/40 for both base and tuned
  models; evidence-span hits improved under the stated loose overlap measure.
- Added the static case explorer, case permalinks, CLI inference, and public
  reproduction and contribution documentation.

No adapter or upstream model weights are distributed with the repository.
