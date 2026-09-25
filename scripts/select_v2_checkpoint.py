"""Lock the v2 seed choice from development loss before opening test predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from exitreceipt.model import MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260925, 20260926, 20260927)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    data_hash = sha256(ROOT / "data/v2-cases.psv")
    evidence_hash = sha256(ROOT / "data/v2-evidence.psv")
    runs = []
    for seed in SEEDS:
        run_dir = ROOT / "runs" / f"v2-{seed}"
        metadata = json.loads((run_dir / "exitreceipt-run.json").read_text(encoding="utf-8"))
        config = metadata["config"]
        if (
            metadata["data_sha256"] != data_hash
            or metadata["evidence_sha256"] != evidence_hash
            or metadata["revision"] != MODEL_REVISION
            or config["seed"] != seed
            or config["epochs"] != 3
            or config["batch_size"] != 4
            or config["lora_dropout"] != 0.1
            or metadata["code"]["tracked_changes"]
        ):
            raise ValueError(f"run {seed} differs from the preregistered setup")
        history = metadata["training_summary"]["eval_metrics_history"]
        if len(history) != 3:
            raise ValueError(f"expected three development checkpoints for {seed}")
        best = min(history, key=lambda item: item["eval_loss"])
        if best["eval_loss"] != metadata["training_summary"]["best_metric"]:
            raise ValueError(f"selected checkpoint for {seed} differs from dev history")
        runs.append(
            {
                "seed": seed,
                "best_dev_loss": best["eval_loss"],
                "best_epoch": best["epoch"] + 1,
                "adapter_sha256": sha256(run_dir / "best/adapter_model.safetensors"),
                "dev_loss_by_epoch": [
                    {"epoch": item["epoch"] + 1, "loss": item["eval_loss"]} for item in history
                ],
                "training_code": metadata["code"],
            }
        )
    winner = min(runs, key=lambda run: (run["best_dev_loss"], run["seed"]))
    selection = {
        "schema_version": 1,
        "selection_rule": "lowest development loss among fixed seeds; test excluded",
        "data_sha256": data_hash,
        "evidence_sha256": evidence_hash,
        "base_revision": MODEL_REVISION,
        "selected_seed": winner["seed"],
        "selected_adapter_sha256": winner["adapter_sha256"],
        "runs": runs,
    }
    path = ROOT / "results/v2-selection.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite locked seed selection: {path}")
    path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected_seed": winner["seed"], "best_dev_loss": winner["best_dev_loss"]}))


if __name__ == "__main__":
    main()
