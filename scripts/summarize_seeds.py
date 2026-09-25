"""Build a deterministic, paired summary of all three synthetic pilot seeds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = [
    ROOT / "results/pilot.json",
    ROOT / "results/seed-20260926.json",
    ROOT / "results/seed-20260927.json",
]
OUTPUT = ROOT / "results/seed-robustness.json"


def summarize(paths: list[Path]) -> dict:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reference = reports[0]
    reference_rows = [
        (row["id"], row["goal"], row["trace"], row["truth"], row["base"], row["base_spans"])
        for row in reference["rows"]
    ]
    runs = []
    for path, report in zip(paths, reports, strict=True):
        for key in ("model", "revision", "data_sha256", "evidence_sha256", "split"):
            if report[key] != reference[key]:
                raise ValueError(f"mismatched {key} in {path}")
        rows = [
            (row["id"], row["goal"], row["trace"], row["truth"], row["base"], row["base_spans"])
            for row in report["rows"]
        ]
        if rows != reference_rows or report["base"] != reference["base"]:
            raise ValueError(f"base/rows mismatch in {path}")
        changed = [row for row in report["rows"] if row["base"] != row["tuned"]]
        fixed = [row["id"] for row in changed if row["base"] != row["truth"]]
        regressed = [row["id"] for row in changed if row["base"] == row["truth"]]
        runs.append(
            {
                "seed": report["training"]["seed"],
                "report": path.relative_to(ROOT).as_posix(),
                "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "adapter_sha256": report["adapter_sha256"],
                "training_code": report["training_code"],
                "best_dev_loss": report["training_best_dev_loss"],
                "accuracy": report["tuned"]["accuracy"],
                "false_complete": report["tuned"]["false_complete"],
                "false_incomplete": report["tuned"]["false_incomplete"],
                "half_gold_typed_hits": report["evidence"]["tuned"]["half_gold_span_and_type_hits"],
                "exact_typed_hits": report["evidence"]["tuned"]["exact_span_and_type_hits"],
                "fixed_case_ids": fixed,
                "regressed_case_ids": regressed,
            }
        )
    if len({run["seed"] for run in runs}) != len(runs):
        raise ValueError("duplicate training seed")
    return {
        "scope": "post-hoc seed-sensitivity check on one authored synthetic split; no transfer claim",
        "model": reference["model"],
        "revision": reference["revision"],
        "data_sha256": reference["data_sha256"],
        "evidence_sha256": reference["evidence_sha256"],
        "test_n": reference["base"]["n"],
        "annotated_test_n": reference["evidence"]["base"]["annotated_n"],
        "base": {
            "accuracy": reference["base"]["accuracy"],
            "false_complete": reference["base"]["false_complete"],
            "false_incomplete": reference["base"]["false_incomplete"],
            "half_gold_typed_hits": reference["evidence"]["base"]["half_gold_span_and_type_hits"],
            "exact_typed_hits": reference["evidence"]["base"]["exact_span_and_type_hits"],
        },
        "runs": runs,
    }


def main() -> None:
    result = summarize(REPORTS)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
