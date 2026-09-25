"""Build the versioned WorkBench action-outcome study from a pinned upstream export."""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

REVISION = "49c7dfd00c03d384ec59ea57374f50b766aa5613"
SOURCE_SHA256 = "6035fb112bbb4fb48cd56d643898cac43c42c2d11bc6d2bb0e595d8d31ea2216"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/olly-styles/WorkBench/{REVISION}"
    "/data/results/item_level_results.csv.gz"
)
ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("id", "split", "domain", "label", "goal", "trace")
SPLIT_SEED = 20260925


def _render_actions(raw: str) -> str | None:
    actions = ast.literal_eval(raw)
    if not isinstance(actions, list) or not 1 <= len(actions) <= 6:
        return None
    if any(not isinstance(action, str) for action in actions):
        return None
    rendered = "; ".join(action.replace(".func(", "(") for action in actions)
    if len(rendered) > 800 or "\n" in rendered or "|" in rendered:
        return None
    return "Recorded tool actions: " + rendered


def _eligible_rows(payload: bytes) -> dict[str, dict[str, list[dict]]]:
    reader = csv.DictReader(io.StringIO(gzip.decompress(payload).decode("utf-8")))
    by_task: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in reader:
        if (
            row["run_group"] != "revisited_2026"
            or row["ground_truth_version"] != "v2"
            or row["tool_selection"] != "all"
            or row["error"]
        ):
            continue
        rendered = _render_actions(row["predicted_actions"])
        if rendered is None or not row["task"].strip() or "|" in row["task"]:
            continue
        row["rendered_actions"] = rendered
        by_task[row["task_id"]][row["correct"]].append(row)
    return by_task


def _pick(rows: list[dict], task_id: str, label: str) -> dict:
    return min(
        rows,
        key=lambda row: hashlib.sha256(
            f"exitreceipt-v2/{task_id}/{label}/{row['model']}".encode()
        ).hexdigest(),
    )


def build(payload: bytes) -> tuple[list[dict], dict]:
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != SOURCE_SHA256:
        raise ValueError(f"upstream export changed: {actual_hash}")
    selected = []
    for task_id, labels in sorted(_eligible_rows(payload).items()):
        if not labels.get("True") or not labels.get("False"):
            continue
        yes = _pick(labels["True"], task_id, "yes")
        no = _pick(labels["False"], task_id, "no")
        if yes["rendered_actions"] == no["rendered_actions"]:
            continue
        if yes["task"] != no["task"] or yes["base_template"] != no["base_template"]:
            raise ValueError(f"task text/template changed across runs: {task_id}")
        selected.extend((yes, no))

    by_domain_template: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in selected:
        by_domain_template[row["domain"]][row["base_template"]].append(row)

    assignments = {}
    for domain, templates in sorted(by_domain_template.items()):
        keys = sorted(templates)
        random.Random(SPLIT_SEED + sum(map(ord, domain))).shuffle(keys)
        n = len(keys)
        dev_n = max(1, round(n * 0.15))
        test_n = max(1, round(n * 0.15))
        for index, template in enumerate(keys):
            split = "test" if index < test_n else "dev" if index < test_n + dev_n else "train"
            assignments[(domain, template)] = split

    output = []
    provenance = []
    for row in selected:
        label = "yes" if row["correct"] == "True" else "no"
        split = assignments[(row["domain"], row["base_template"])]
        case_id = f"wb-{row['task_id']}-{label}"
        output.append(
            {
                "id": case_id,
                "split": split,
                "domain": f"workbench_{row['domain']}",
                "label": label,
                "goal": row["task"].replace("\r", " ").replace("\n", " ").strip(),
                "trace": row["rendered_actions"],
            }
        )
        provenance.append(
            {
                "id": case_id,
                "task_id": row["task_id"],
                "base_template": row["base_template"],
                "model": row["model"],
                "results_file": row["results_file"],
                "sandbox_correct": row["correct"] == "True",
                "unwanted_side_effects": row["unwanted_side_effects"] == "True",
            }
        )
    output.sort(key=lambda row: row["id"])
    provenance.sort(key=lambda row: row["id"])
    counts = Counter((row["split"], row["label"]) for row in output)
    manifest = {
        "source_url": SOURCE_URL,
        "source_revision": REVISION,
        "source_sha256": SOURCE_SHA256,
        "split_seed": SPLIT_SEED,
        "selection": "one hash-selected successful and failed action log per task; 1-6 actions; rendered log <=800 characters; distinct logs",
        "split": "base_template grouped and stratified by domain; tasks never cross splits",
        "counts": {f"{split}_{label}": count for (split, label), count in sorted(counts.items())},
        "source_rows": provenance,
    }
    return output, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="check committed outputs byte-for-byte")
    args = parser.parse_args()
    payload = urllib.request.urlopen(SOURCE_URL, timeout=30).read()
    rows, manifest = build(payload)
    with (ROOT / "data" / "cases.psv").open(newline="", encoding="utf-8") as handle:
        authored_train = [row for row in csv.DictReader(handle, delimiter="|") if row["split"] == "train"]
    with (ROOT / "data" / "evidence.psv").open(newline="", encoding="utf-8") as handle:
        authored_evidence = [
            row for row in csv.DictReader(handle, delimiter="|")
            if row["id"] in {case["id"] for case in authored_train}
        ]
    manifest["auxiliary_train"] = {
        "source": "data/cases.psv and data/evidence.psv; original authored train split only",
        "cases": len(authored_train),
        "annotated_spans": len(authored_evidence),
    }

    def psv(fieldnames: tuple[str, ...], data: list[dict]) -> str:
        stream = io.StringIO(newline="")
        csv_writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="|", lineterminator="\n")
        csv_writer.writeheader()
        csv_writer.writerows(data)
        return stream.getvalue()

    outputs = {
        ROOT / "data" / "workbench-v2.psv": psv(FIELDS, rows),
        ROOT / "data" / "v2-cases.psv": psv(FIELDS, rows + authored_train),
        ROOT / "data" / "v2-evidence.psv": psv(
            ("id", "kind", "evidence"), authored_evidence
        ),
        ROOT / "data" / "workbench-v2-manifest.json": json.dumps(
            manifest, indent=2, sort_keys=True
        ) + "\n",
    }
    if args.verify:
        for path, expected in outputs.items():
            if path.read_text(encoding="utf-8") != expected:
                raise SystemExit(f"mismatch: {path}")
    else:
        for path, content in outputs.items():
            path.write_text(content, encoding="utf-8")
    print(json.dumps({"cases": len(rows), "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
