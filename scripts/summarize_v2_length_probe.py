"""Record the pre-test three-versus-six-epoch development comparison."""

from __future__ import annotations

import json
from pathlib import Path

from exitreceipt.corpus import sha256

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    runs = []
    for name, epochs in (("v2-20260925", 3), ("v2-longprobe-20260925", 6)):
        path = ROOT / "runs" / name / "exitreceipt-run.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if (
            metadata["config"]["seed"] != 20260925
            or metadata["config"]["epochs"] != epochs
            or metadata["config"]["lora_dropout"] != 0.1
            or metadata["data_sha256"] != sha256(ROOT / "data/v2-cases.psv")
            or metadata["code"]["tracked_changes"]
        ):
            raise ValueError(f"length probe has mismatched run metadata: {name}")
        history = metadata["training_summary"]["eval_metrics_history"]
        if len(history) != epochs:
            raise ValueError(f"missing development checkpoints in {name}")
        best = min(history, key=lambda item: item["eval_loss"])
        runs.append(
            {
                "name": name,
                "epochs": epochs,
                "best_epoch": best["epoch"] + 1,
                "best_dev_loss": best["eval_loss"],
                "dev_loss_by_epoch": [
                    {"epoch": item["epoch"] + 1, "loss": item["eval_loss"]} for item in history
                ],
                "adapter_sha256": sha256(ROOT / "runs" / name / "best/adapter_model.safetensors"),
                "training_code": metadata["code"],
            }
        )
    output = {
        "schema_version": 1,
        "scope": "development-only, single-seed training-length probe; no test results",
        "data_sha256": sha256(ROOT / "data/v2-cases.psv"),
        "base_revision": json.loads((ROOT / "runs/v2-20260925/exitreceipt-run.json").read_text())[
            "revision"
        ],
        "selected_schedule_epochs": 3,
        "runs": runs,
    }
    path = ROOT / "results/v2-length-probe.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"short": runs[0]["best_dev_loss"], "long": runs[1]["best_dev_loss"]}))


if __name__ == "__main__":
    main()
