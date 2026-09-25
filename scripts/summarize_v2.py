"""Summarize the preregistered v2 runs with paired cluster intervals."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from exitreceipt.statistics import RESAMPLES, paired_accuracy_interval

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260925, 20260926, 20260927)


def summarize(reports: list[dict], manifest: dict, selection: dict) -> dict:
    source = {row["id"]: row for row in manifest["source_rows"]}
    reference = reports[0]
    ids = [row["id"] for row in reference["rows"]]
    if len(ids) != len(set(ids)) or any(case_id not in source for case_id in ids):
        raise ValueError("report contains unknown or duplicate external case")
    if len(ids) != 158:
        raise ValueError(f"expected 158 test cases, got {len(ids)}")
    for row in reference["rows"]:
        upstream = source[row["id"]]
        if (
            row["workbench"]["task_id"] != upstream["task_id"]
            or row["workbench"]["base_template"] != upstream["base_template"]
            or row["workbench"]["unwanted_side_effects"] != upstream["unwanted_side_effects"]
        ):
            raise ValueError("report source metadata differs from pinned WorkBench manifest")
    for expected_seed, report in zip(SEEDS, reports, strict=True):
        if report["split"] != "test" or [row["id"] for row in report["rows"]] != ids:
            raise ValueError("report splits or case order differ")
        if report.get("study") != "workbench-v2-template-holdout":
            raise ValueError("report is not from the external v2 study")
        training = report["training"]
        if (
            training["seed"] != expected_seed
            or training["epochs"] != 3
            or training["batch_size"] != 4
            or training["lora_dropout"] != 0.1
        ):
            raise ValueError("report does not match preregistered v2 training")
        if report["training_code"].get("tracked_changes"):
            raise ValueError("training run used a dirty tracked checkout")
        if report["data_sha256"] != reference["data_sha256"]:
            raise ValueError("data hashes differ")
        if report["revision"] != reference["revision"]:
            raise ValueError("base model revisions differ")
        if any(
            row["truth"] != original["truth"] or row["base"] != original["base"]
            for row, original in zip(report["rows"], reference["rows"], strict=True)
        ):
            raise ValueError("gold labels or base predictions differ between seeds")

    template_groups: dict[str, list[str]] = defaultdict(list)
    task_groups: dict[str, list[str]] = defaultdict(list)
    for case_id in ids:
        template_groups[source[case_id]["base_template"]].append(case_id)
        task_groups[source[case_id]["task_id"]].append(case_id)

    runs = []
    for seed, report in zip(SEEDS, reports, strict=True):
        rows = report["rows"]
        failure_strata = {}
        for side_effects in (False, True):
            subset = [
                row
                for row in rows
                if row["truth"] == "no"
                and source[row["id"]]["unwanted_side_effects"] == side_effects
            ]
            failure_strata["with_side_effects" if side_effects else "without_side_effects"] = {
                "n": len(subset),
                "base_false_complete": sum(row["base"] == "yes" for row in subset),
                "tuned_false_complete": sum(row["tuned"] == "yes" for row in subset),
            }
        changed = [
            {
                "id": row["id"],
                "truth": row["truth"],
                "base": row["base"],
                "tuned": row["tuned"],
                "template": source[row["id"]]["base_template"],
            }
            for row in rows
            if row["base"] != row["tuned"]
        ]
        runs.append(
            {
                "seed": seed,
                "adapter_sha256": report["adapter_sha256"],
                "best_dev_loss": report["training_best_dev_loss"],
                "dev_loss_by_epoch": [
                    {"epoch": item["epoch"] + 1, "loss": item["eval_loss"]}
                    for item in report["training_eval_history"]
                ],
                "base": report["base"],
                "tuned": report["tuned"],
                "accuracy_delta": report["tuned"]["accuracy"] - report["base"]["accuracy"],
                "accuracy_delta_template_ci95": paired_accuracy_interval(
                    rows, template_groups, seed=seed + 1
                ),
                "accuracy_delta_task_ci95": paired_accuracy_interval(
                    rows, task_groups, seed=seed + 2
                ),
                "failure_strata": failure_strata,
                "changed": changed,
            }
        )
    winner = min(runs, key=lambda run: (run["best_dev_loss"], run["seed"]))
    if (
        winner["seed"] != selection["selected_seed"]
        or winner["adapter_sha256"] != selection["selected_adapter_sha256"]
        or selection["data_sha256"] != reference["data_sha256"]
    ):
        raise ValueError("test summary differs from locked development selection")
    for run, locked in zip(runs, selection["runs"], strict=True):
        if (
            run["seed"] != locked["seed"]
            or run["adapter_sha256"] != locked["adapter_sha256"]
            or run["dev_loss_by_epoch"] != locked["dev_loss_by_epoch"]
        ):
            raise ValueError("test run differs from locked development checkpoint")
    delta = winner["accuracy_delta"]
    false_done_delta = (
        winner["tuned"]["false_complete_rate"] - winner["base"]["false_complete_rate"]
    )
    headline_positive = (
        delta >= 0.05 and winner["accuracy_delta_template_ci95"][0] > 0 and false_done_delta <= 0.02
    )
    return {
        "schema_version": 1,
        "study": "workbench-v2-template-holdout",
        "source_revision": manifest["source_revision"],
        "source_sha256": manifest["source_sha256"],
        "data_sha256": reference["data_sha256"],
        "base_revision": reference["revision"],
        "test_examples": len(ids),
        "test_tasks": len(task_groups),
        "test_templates": len(template_groups),
        "selection_rule": "lowest development loss among fixed seeds",
        "selected_seed": winner["seed"],
        "headline_positive": headline_positive,
        "headline_rule": "accuracy gain >=5pp, template-cluster CI lower >0, false-complete rate rise <=2pp",
        "resamples": RESAMPLES,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "v2-robustness.json")
    parser.add_argument(
        "--selected-output", type=Path, default=ROOT / "results" / "v2-selected.json"
    )
    args = parser.parse_args()
    reports = [
        json.loads((ROOT / "results" / f"v2-{seed}.json").read_text(encoding="utf-8"))
        for seed in SEEDS
    ]
    manifest = json.loads(
        (ROOT / "data" / "workbench-v2-manifest.json").read_text(encoding="utf-8")
    )
    selection = json.loads((ROOT / "results/v2-selection.json").read_text(encoding="utf-8"))
    result = summarize(reports, manifest, selection)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected = reports[SEEDS.index(result["selected_seed"])]
    args.selected_output.write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({key: result[key] for key in ("selected_seed", "headline_positive")}, indent=2)
    )


if __name__ == "__main__":
    main()
