"""Evaluate whether extracted candidate spans contain annotated decisive evidence."""

from __future__ import annotations

from .corpus import Case, Evidence


def score_evidence(
    cases: list[Case], annotations: dict[str, Evidence], inspections: dict[str, dict]
) -> dict:
    selected = [case for case in cases if case.id in annotations]
    exact = overlap = spurious = candidates = 0
    for case in selected:
        annotation = annotations[case.id]
        start = case.text.index(annotation.evidence)
        end = start + len(annotation.evidence)
        spans = inspections[case.id]["spans"]
        candidates += len(spans)
        exact += any(
            span["kind"] == annotation.kind and span["start"] == start and span["end"] == end
            for span in spans
        )
        overlap += any(
            span["kind"] == annotation.kind
            and max(0, min(end, span["end"]) - max(start, span["start"])) / (end - start) >= 0.5
            for span in spans
        )
        spurious += sum(
            span["kind"] != annotation.kind or min(end, span["end"]) <= max(start, span["start"])
            for span in spans
        )
    return {
        "annotated_n": len(selected),
        "exact_span_and_type_hits": exact,
        "half_gold_span_and_type_hits": overlap,
        "candidate_spans": candidates,
        "wrong_type_or_disjoint_spans": spurious,
    }
