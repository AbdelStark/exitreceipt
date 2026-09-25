"""Paired cluster uncertainty for matched model comparisons."""

from __future__ import annotations

import random

RESAMPLES = 10_000


def paired_accuracy_interval(
    rows: list[dict], groups: dict[str, list[str]], *, seed: int
) -> list[float]:
    by_id = {row["id"]: row for row in rows}
    group_keys = sorted(groups)
    rng = random.Random(seed)
    values = []
    for _ in range(RESAMPLES):
        ids = [case_id for _ in group_keys for case_id in groups[rng.choice(group_keys)]]
        base = sum(by_id[case_id]["base"] == by_id[case_id]["truth"] for case_id in ids)
        tuned = sum(by_id[case_id]["tuned"] == by_id[case_id]["truth"] for case_id in ids)
        values.append((tuned - base) / len(ids))
    values.sort()
    return [values[int(0.025 * RESAMPLES)], values[int(0.975 * RESAMPLES) - 1]]
