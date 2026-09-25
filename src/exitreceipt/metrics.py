"""Exact single-label metrics, with false-complete errors called out."""

from __future__ import annotations

from collections import Counter

from .corpus import LABELS, Case


def score(cases: list[Case], predictions: dict[str, str]) -> dict:
    if set(predictions) != {case.id for case in cases}:
        raise ValueError("predictions must cover exactly the selected cases")
    if any(pred not in LABELS for pred in predictions.values()):
        raise ValueError("prediction outside label set")
    pairs = Counter((case.label, predictions[case.id]) for case in cases)
    correct = sum(pairs[(label, label)] for label in LABELS)
    by_label = {}
    for label in LABELS:
        tp = pairs[(label, label)]
        fp = sum(pairs[(other, label)] for other in LABELS if other != label)
        fn = sum(pairs[(label, other)] for other in LABELS if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_label[label] = {
            "support": tp + fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    incomplete = sum(c.label == "no" for c in cases)
    false_complete = pairs[("no", "yes")]
    return {
        "n": len(cases),
        "accuracy": correct / len(cases),
        "macro_f1": sum(by_label[label]["f1"] for label in LABELS) / len(LABELS),
        "false_complete": false_complete,
        "false_complete_rate": false_complete / incomplete if incomplete else 0.0,
        "false_incomplete": pairs[("yes", "no")],
        "per_label": by_label,
        "confusion": {truth: {pred: pairs[(truth, pred)] for pred in LABELS} for truth in LABELS},
    }
