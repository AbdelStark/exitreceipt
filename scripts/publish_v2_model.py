"""Publish the development-selected v2 adapter and its evaluation receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "abdelstark/exitreceipt-gliner2.5-decide-lora"
TAG = "v0.2.0"
BASE_MODEL = "fastino/GLiNER2.5-Decide"
BASE_REVISION = "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6"
SEEDS = (20260925, 20260926, 20260927)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def release_files() -> dict[str, Path]:
    summary_path = ROOT / "results/v2-robustness.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seed = summary["selected_seed"]
    if seed not in SEEDS or [run["seed"] for run in summary["runs"]] != list(SEEDS):
        raise ValueError("unexpected v2 seed selection or run set")
    run_dir = ROOT / "runs" / f"v2-{seed}"
    weights = run_dir / "best/adapter_model.safetensors"
    report = ROOT / "results/v2-selected.json"
    card = ROOT / "model/v2/README.md"
    config = ROOT / "model/v2/adapter_config.json"
    files = {
        "README.md": card,
        "adapter_model.safetensors": weights,
        "adapter_config.json": config,
        "evaluation/v2-selected.json": report,
        "evaluation/v2-robustness.json": summary_path,
        "evaluation/v2-selection.json": ROOT / "results/v2-selection.json",
        "evaluation/v2-length-probe.json": ROOT / "results/v2-length-probe.json",
        "evaluation/pilot.json": ROOT / "results/pilot.json",
        "data/workbench-v2-manifest.json": ROOT / "data/workbench-v2-manifest.json",
        "data/workbench-v2.psv": ROOT / "data/workbench-v2.psv",
        "data/v2-cases.psv": ROOT / "data/v2-cases.psv",
        "data/v2-evidence.psv": ROOT / "data/v2-evidence.psv",
        "data/WORKBENCH_LICENSE": ROOT / "data/WORKBENCH_LICENSE",
        "training/exitreceipt-v2-run.json": run_dir / "exitreceipt-run.json",
        "training/training_config-v2.json": run_dir / "training_config.json",
        "LICENSE": ROOT / "LICENSE",
        "NOTICE": ROOT / "NOTICE",
    }
    for run_seed in SEEDS:
        files[f"evaluation/v2-{run_seed}.json"] = ROOT / "results" / f"v2-{run_seed}.json"
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    selected = json.loads(report.read_text(encoding="utf-8"))
    run = json.loads(files["training/exitreceipt-v2-run.json"].read_text(encoding="utf-8"))
    adapter_config = json.loads(config.read_text(encoding="utf-8"))
    original_config = json.loads((run_dir / "best/adapter_config.json").read_text())
    if original_config.pop("task_type", "missing") is not None:
        raise ValueError("unexpected PEFT task_type")
    if adapter_config != original_config:
        raise ValueError("release adapter config differs from selected checkpoint")
    if selected["adapter_sha256"] != digest(weights):
        raise ValueError("selected report does not match weights")
    if summary["runs"][SEEDS.index(seed)]["adapter_sha256"] != digest(weights):
        raise ValueError("summary does not match selected weights")
    if selected["training"]["seed"] != seed or selected["study"] != summary["study"]:
        raise ValueError("selected report is not the development-selected run")
    if selected["training_best_dev_loss"] != summary["runs"][SEEDS.index(seed)]["best_dev_loss"]:
        raise ValueError("development loss selection mismatch")
    if selected["training_code"] != run["code"]:
        raise ValueError("training provenance differs")
    for name, path in (
        ("data_sha256", ROOT / "data/v2-cases.psv"),
        ("evidence_sha256", ROOT / "data/v2-evidence.psv"),
    ):
        if selected[name] != run[name] or selected[name] != digest(path):
            raise ValueError(f"{name} differs from release data")
    if (
        selected["revision"] != BASE_REVISION
        or adapter_config["base_model_name_or_path"] != BASE_MODEL
    ):
        raise ValueError("base model mismatch")
    card_text = card.read_text(encoding="utf-8")
    if digest(weights) not in card_text or BASE_REVISION not in card_text:
        raise ValueError("v2 model card lacks exact adapter/base hashes")
    if "[More Information Needed]" in card_text:
        raise ValueError("model card contains template placeholders")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    files = release_files()
    print(
        json.dumps(
            {"tag": TAG, "files": {name: digest(path) for name, path in files.items()}}, indent=2
        )
    )
    if not args.publish:
        return
    api = HfApi()
    if api.whoami()["name"].lower() != REPO_ID.partition("/")[0]:
        raise RuntimeError("authenticated Hugging Face user is not the model owner")
    info = api.model_info(REPO_ID)
    if info.private:
        raise RuntimeError("expected a public model repository")
    if any(ref.name == TAG for ref in api.list_repo_refs(REPO_ID).tags):
        raise FileExistsError(f"release tag already exists: {TAG}")
    with TemporaryDirectory(prefix="exitreceipt-v2-hf-") as directory:
        stage = Path(directory)
        for name, source in files.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        commit = api.upload_folder(
            repo_id=REPO_ID,
            repo_type="model",
            folder_path=stage,
            commit_message="Publish ExitReceipt v2 WorkBench action-outcome study",
        )
        api.create_tag(REPO_ID, tag=TAG, revision=commit.oid, repo_type="model")
        for name, source in files.items():
            remote = Path(hf_hub_download(REPO_ID, name, revision=TAG))
            if digest(remote) != digest(source):
                raise ValueError(f"remote digest mismatch: {name}")
    print(json.dumps({"url": f"https://huggingface.co/{REPO_ID}/tree/{TAG}", "commit": commit.oid}))


if __name__ == "__main__":
    main()
