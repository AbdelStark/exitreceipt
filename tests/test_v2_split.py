"""Guard the external study against task and template leakage."""

import json
from collections import defaultdict
from pathlib import Path

from exitreceipt.corpus import load_cases, load_evidence, split_counts

ROOT = Path(__file__).resolve().parents[1]


def test_workbench_study_has_paired_sandbox_outcomes_and_grouped_holdout():
    cases = load_cases(ROOT / "data" / "v2-cases.psv")
    workbench = [case for case in cases if case.id.startswith("wb-")]
    manifest = json.loads((ROOT / "data" / "workbench-v2-manifest.json").read_text())
    source = {row["id"]: row for row in manifest["source_rows"]}
    assert len(workbench) == len(source) == 1180
    assert len(load_evidence(ROOT / "data" / "v2-evidence.psv", cases)) == 24
    assert split_counts(cases)["test"] == {"no": 79, "yes": 79}

    task_splits = defaultdict(set)
    task_labels = defaultdict(set)
    template_splits = defaultdict(set)
    for case in workbench:
        row = source[case.id]
        task_splits[row["task_id"]].add(case.split)
        task_labels[row["task_id"]].add(case.label)
        template_splits[row["base_template"]].add(case.split)
        assert (case.label == "yes") == row["sandbox_correct"]
        assert "Recorded tool actions:" in case.trace
    assert all(len(splits) == 1 for splits in task_splits.values())
    assert all(labels == {"yes", "no"} for labels in task_labels.values())
    assert all(len(splits) == 1 for splits in template_splits.values())
    assert len({template for template, splits in template_splits.items() if "test" in splits}) == 10
