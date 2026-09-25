"""Check that the paired interval uses the same sampled rows for both models."""

from exitreceipt.statistics import paired_accuracy_interval


def test_paired_interval_is_zero_when_predictions_are_identical():
    rows = [
        {"id": "a", "truth": "yes", "base": "yes", "tuned": "yes"},
        {"id": "b", "truth": "no", "base": "yes", "tuned": "yes"},
    ]
    assert paired_accuracy_interval(rows, {"task-a": ["a"], "task-b": ["b"]}, seed=7) == [
        0.0,
        0.0,
    ]


def test_paired_interval_preserves_always_corrected_pair():
    rows = [
        {"id": "a", "truth": "yes", "base": "no", "tuned": "yes"},
        {"id": "b", "truth": "no", "base": "yes", "tuned": "no"},
    ]
    assert paired_accuracy_interval(rows, {"one": ["a", "b"]}, seed=7) == [1.0, 1.0]
