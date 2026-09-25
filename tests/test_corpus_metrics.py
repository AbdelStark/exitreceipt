from pathlib import Path

import pytest

from exitreceipt.corpus import load_cases, load_evidence, split_counts
from exitreceipt.evidence import score_evidence
from exitreceipt.metrics import score

DATA = Path(__file__).resolve().parents[1] / "data" / "cases.psv"
EVIDENCE = DATA.with_name("evidence.psv")


def test_corpus_is_balanced_and_unique():
    cases = load_cases(DATA)
    assert len(cases) == 132
    assert split_counts(cases) == {
        "train": {"no": 36, "yes": 36},
        "dev": {"no": 10, "yes": 10},
        "test": {"no": 20, "yes": 20},
    }
    assert all(case.text.startswith("Goal: ") for case in cases)


def test_false_complete_metric_uses_incomplete_denominator():
    cases = [case for case in load_cases(DATA) if case.split == "dev"]
    predictions = {case.id: case.label for case in cases}
    predictions["dv001"] = "yes"
    result = score(cases, predictions)
    assert result["false_complete"] == 1
    assert result["false_complete_rate"] == pytest.approx(1 / 10)
    assert result["accuracy"] == pytest.approx(19 / 20)


def test_metric_rejects_partial_predictions():
    cases = [case for case in load_cases(DATA) if case.split == "dev"]
    with pytest.raises(ValueError, match="exactly"):
        score(cases, {cases[0].id: "yes"})


def test_evidence_annotations_are_literal_and_scored():
    cases = load_cases(DATA)
    evidence = load_evidence(EVIDENCE, cases)
    assert len(evidence) == 48
    case = next(case for case in cases if case.id == "te025")
    annotation = evidence[case.id]
    start = case.text.index(annotation.evidence)
    inspections = {
        case.id: {
            "spans": [
                {
                    "kind": annotation.kind,
                    "text": annotation.evidence,
                    "start": start,
                    "end": start + len(annotation.evidence),
                }
            ]
        }
    }
    assert score_evidence([case], evidence, inspections)["exact_span_and_type_hits"] == 1


def test_classification_only_corpus_accepts_header_only_evidence_file(tmp_path):
    evidence_path = tmp_path / "evidence.psv"
    evidence_path.write_text("id|kind|evidence\n", encoding="utf-8")
    assert load_evidence(evidence_path, load_cases(DATA)) == {}
