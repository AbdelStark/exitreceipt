"""Run the corpus checks, native fine-tuning, and held-out comparison."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .corpus import Case, load_cases, load_evidence, sha256, split_counts
from .evidence import score_evidence
from .metrics import score
from .model import MODEL_ID, MODEL_REVISION, inspect_cases, load_model, train_adapter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = (
    PROJECT_ROOT / "data" if (PROJECT_ROOT / "data").exists() else Path(__file__).parent / "data"
)
DEFAULT_DATA = DATA_ROOT / "cases.psv"
DEFAULT_EVIDENCE = DATA_ROOT / "evidence.psv"


def _code_provenance() -> dict:
    revision = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    changes = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    lock = PROJECT_ROOT / "uv.lock"
    return {
        "git_head": revision.stdout.strip() if revision.returncode == 0 else None,
        "tracked_changes": bool(changes.stdout.strip()) if changes.returncode == 0 else None,
        "uv_lock_sha256": sha256(lock) if lock.exists() else None,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exitreceipt")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-data", help="Validate the versioned split and print its hash")
    check.add_argument("--data", type=Path, default=DEFAULT_DATA)
    check.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)

    train = sub.add_parser("train", help="Fine-tune GLiNER2.5-Decide with LoRA")
    train.add_argument("--data", type=Path, default=DEFAULT_DATA)
    train.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=6)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    train.add_argument("--seed", type=int, default=20260925)
    train.add_argument("--lora-dropout", type=float, default=0.0)
    train.add_argument("--experiment-name", default="exitreceipt-pilot")

    evaluate = sub.add_parser("evaluate", help="Compare base and tuned models on a split")
    evaluate.add_argument("--data", type=Path, default=DEFAULT_DATA)
    evaluate.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--split", choices=["dev", "test"], default="test")
    evaluate.add_argument("--output", type=Path, required=True)

    predict = sub.add_parser("predict", help="Classify one goal and observed state locally")
    predict.add_argument("--goal", required=True)
    predict.add_argument("--trace", required=True)
    adapter_group = predict.add_mutually_exclusive_group()
    adapter_group.add_argument("--adapter", type=Path, help="Local PEFT adapter directory")
    adapter_group.add_argument("--adapter-repo", help="Hugging Face namespace/repository ID")
    predict.add_argument("--adapter-revision", help="Pinned Hub adapter commit or tag")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "predict":
        if args.adapter_revision and not args.adapter_repo:
            raise ValueError("--adapter-revision requires --adapter-repo")
        model = load_model(
            args.adapter_repo or args.adapter, adapter_revision=args.adapter_revision
        )
        case = Case("input", "test", "custom", "no", args.goal, args.trace)
        print(json.dumps(inspect_cases(model, [case])["input"], sort_keys=True))
        return 0

    cases = load_cases(args.data)
    evidence = load_evidence(args.evidence, cases)
    data_hash = sha256(args.data)
    evidence_hash = sha256(args.evidence)
    if args.command == "check-data":
        print(
            json.dumps(
                {
                    "sha256": data_hash,
                    "evidence_sha256": evidence_hash,
                    "evidence_n": len(evidence),
                    "splits": split_counts(cases),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "train":
        result = train_adapter(
            cases,
            evidence,
            args.run_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            lora_dropout=args.lora_dropout,
            experiment_name=args.experiment_name,
        )
        metadata = {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "data_sha256": data_hash,
            "evidence_sha256": evidence_hash,
            "split_counts": split_counts(cases),
            "config": result["config"],
            "training_summary": result["summary"],
            "code": _code_provenance(),
        }
        _write_json(args.run_dir / "exitreceipt-run.json", metadata)
        print(json.dumps({"run_dir": str(args.run_dir), "data_sha256": data_hash}, indent=2))
        return 0

    if args.command == "evaluate":
        run_file = args.run_dir / "exitreceipt-run.json"
        metadata = json.loads(run_file.read_text(encoding="utf-8"))
        if (
            metadata["data_sha256"] != data_hash
            or metadata["evidence_sha256"] != evidence_hash
            or metadata["revision"] != MODEL_REVISION
        ):
            raise ValueError("run metadata does not match corpus, annotations, or model revision")
        selected = [case for case in cases if case.split == args.split]
        base_inspections = inspect_cases(load_model(), selected)
        adapter = args.run_dir / "best"
        if not adapter.exists():
            raise FileNotFoundError(f"selected adapter missing: {adapter}")
        tuned_inspections = inspect_cases(load_model(adapter), selected)
        base = {id: item["label"] for id, item in base_inspections.items()}
        tuned = {id: item["label"] for id, item in tuned_inspections.items()}
        rows = [
            {
                "id": case.id,
                "domain": case.domain,
                "goal": case.goal,
                "trace": case.trace,
                "truth": case.label,
                "base": base[case.id],
                "tuned": tuned[case.id],
                "gold_evidence": evidence[case.id].__dict__ if case.id in evidence else None,
                "base_spans": base_inspections[case.id]["spans"],
                "tuned_spans": tuned_inspections[case.id]["spans"],
            }
            for case in selected
        ]
        import gliner2
        import torch

        report = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "claim_scope": "authored synthetic cases; no real-world agent validation",
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "adapter_selection": "lowest development loss; test split excluded",
            "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
            "data_sha256": data_hash,
            "evidence_sha256": evidence_hash,
            "split": args.split,
            "counts": split_counts(cases),
            "software": {
                "python": sys.version.split()[0],
                "gliner2": gliner2.__version__,
                "torch": torch.__version__,
                "platform": platform.platform(),
            },
            "training": metadata["config"],
            "training_best_dev_loss": metadata["training_summary"]["best_metric"],
            "training_code": metadata.get("code"),
            "evaluation_code": _code_provenance(),
            "base": score(selected, base),
            "tuned": score(selected, tuned),
            "evidence": {
                "base": score_evidence(selected, evidence, base_inspections),
                "tuned": score_evidence(selected, evidence, tuned_inspections),
            },
            "by_domain": {
                domain: {
                    "base": score(
                        [case for case in selected if case.domain == domain],
                        {case.id: base[case.id] for case in selected if case.domain == domain},
                    ),
                    "tuned": score(
                        [case for case in selected if case.domain == domain],
                        {case.id: tuned[case.id] for case in selected if case.domain == domain},
                    ),
                }
                for domain in sorted({case.domain for case in selected})
            },
            "rows": rows,
        }
        _write_json(args.output, report)
        print(json.dumps({"base": report["base"], "tuned": report["tuned"]}, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
