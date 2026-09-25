"""Versioned synthetic cases for the agent-completion experiment."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

LABELS = ("yes", "no")
SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class Case:
    id: str
    split: str
    domain: str
    label: str
    goal: str
    trace: str

    @property
    def text(self) -> str:
        return f"Goal: {self.goal}\nLast observed state: {self.trace}"


@dataclass(frozen=True)
class Evidence:
    id: str
    kind: str
    evidence: str


def load_cases(path: Path) -> list[Case]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        fields = {"id", "split", "domain", "label", "goal", "trace"}
        if set(reader.fieldnames or ()) != fields:
            raise ValueError(f"corpus columns must be {sorted(fields)}")
        cases = [Case(**row) for row in reader]
    if not cases:
        raise ValueError("corpus is empty")
    ids: set[str] = set()
    texts: set[str] = set()
    for case in cases:
        if not all((case.id, case.domain, case.goal, case.trace)):
            raise ValueError(f"missing field in {case.id or '<unknown>'}")
        if case.id in ids:
            raise ValueError(f"duplicate id: {case.id}")
        ids.add(case.id)
        if case.text in texts:
            raise ValueError(f"duplicate goal-and-trace text: {case.id}")
        texts.add(case.text)
        if case.split not in SPLITS or case.label not in LABELS:
            raise ValueError(f"invalid split/label: {case.id}")
    for split in SPLITS:
        counts = Counter(c.label for c in cases if c.split == split)
        if min(counts.get(label, 0) for label in LABELS) < 1:
            raise ValueError(f"split {split} must contain both labels")
    return cases


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_evidence(path: Path, cases: list[Case]) -> dict[str, Evidence]:
    by_id = {case.id: case for case in cases}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        if set(reader.fieldnames or ()) != {"id", "kind", "evidence"}:
            raise ValueError("evidence columns must be id, kind, evidence")
        evidence = {}
        for row in reader:
            item = Evidence(**row)
            if item.id in evidence or item.id not in by_id:
                raise ValueError(f"duplicate or unknown evidence case: {item.id}")
            case = by_id[item.id]
            expected_kind = "receipt" if case.label == "yes" else "blocker"
            if (
                item.kind != expected_kind
                or not item.evidence
                or item.evidence not in case.trace
                or len(item.evidence.split()) > 8
            ):
                raise ValueError(f"invalid evidence annotation: {item.id}")
            evidence[item.id] = item
    if not evidence:
        raise ValueError("evidence file is empty")
    return evidence


def split_counts(cases: list[Case]) -> dict[str, dict[str, int]]:
    return {split: dict(Counter(c.label for c in cases if c.split == split)) for split in SPLITS}
