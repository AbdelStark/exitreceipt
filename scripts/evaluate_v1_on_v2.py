"""Evaluate the fixed v1 adapter on the new WorkBench test as a historical comparator."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from exitreceipt.corpus import load_cases, sha256
from exitreceipt.metrics import score
from exitreceipt.model import inspect_cases, load_model

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "abdelstark/exitreceipt-gliner2.5-decide-lora"
REVISION = "v0.1.1"


def main() -> None:
    selected = json.loads((ROOT / "results/v2-selected.json").read_text(encoding="utf-8"))
    cases = [case for case in load_cases(ROOT / "data/v2-cases.psv") if case.split == "test"]
    if [row["id"] for row in selected["rows"]] != [case.id for case in cases]:
        raise ValueError("selected report does not match versioned test rows")
    if selected["data_sha256"] != sha256(ROOT / "data/v2-cases.psv"):
        raise ValueError("selected report data hash mismatch")
    weights = Path(hf_hub_download(REPO_ID, "adapter_model.safetensors", revision=REVISION))
    inspections = inspect_cases(load_model(REPO_ID, adapter_revision=REVISION), cases)
    labels = {case_id: item["label"] for case_id, item in inspections.items()}
    selected_rows = {row["id"]: row for row in selected["rows"]}
    report = {
        "schema_version": 1,
        "study": "workbench-v2-historical-v1-adapter",
        "adapter_repo": REPO_ID,
        "adapter_revision": REVISION,
        "adapter_sha256": sha256(weights),
        "data_sha256": selected["data_sha256"],
        "base_revision": selected["revision"],
        "score": score(cases, labels),
        "rows": [
            {
                "id": case.id,
                "truth": case.label,
                "base": selected_rows[case.id]["base"],
                "v1": labels[case.id],
                "v2": selected_rows[case.id]["tuned"],
            }
            for case in cases
        ],
    }
    path = ROOT / "results/v2-v1-comparator.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["score"], indent=2))


if __name__ == "__main__":
    main()
