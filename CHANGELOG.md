# Changelog

## 0.2.0 — 2026-09-25

- Added the pinned, MIT-attributed WorkBench action-outcome study: 590 paired
  tasks, task-template-grouped splits, per-case source provenance, and a
  pre-test seed-selection lock.
- Fine-tuned three GLiNER2.5-Decide LoRA seeds on 910 training cases, evaluated
  all three on 158 external held-out cases, and published their complete
  reports, paired intervals, and failure analysis. The development-selected
  adapter improved from 87/158 to 96/158 correct but missed more successes;
  its template-cluster interval includes zero.
- Recorded the development-only three-versus-six-epoch probe and retained the
  three-epoch schedule before test inference.
- Published the v2 selected adapter and two alternate seed adapters with
  model cards, transformed data, training receipts, and tagged weights.
- Updated the static explorer with a v2/pilot switch, paired-task navigation,
  source-run metadata, and both false-done and missed-done counts.
- Allowed classification-only fine-tuning with a header-only evidence file.

## 0.1.0 — 2026-09-25

- Added a pinned GLiNER2.5-Decide base model and native rank-8 LoRA training
  path for joint completion classification and candidate evidence spans.
- Published the authored synthetic corpus, validation rules, and held-out
  per-case pilot report. Completion accuracy was 36/40 for both base and tuned
  models; evidence-span hits improved under the stated loose overlap measure.
- Added the static case explorer, case permalinks, CLI inference, and public
  reproduction and contribution documentation.
- Published the evaluated pilot adapter with a full Hugging Face model card,
  tagged weights, and training/evaluation receipts. Two additional seeds were
  published as a post-hoc sensitivity check on the same authored split.

No upstream base model weights are distributed with this repository.
