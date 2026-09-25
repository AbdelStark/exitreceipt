"""Publish the exact evaluated PEFT adapter, card, and receipts to Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "abdelstark/exitreceipt-gliner2.5-decide-lora"
TAG = "v0.1.1"
RUN = ROOT / "runs/pilot"
REPORT = ROOT / "results/pilot.json"
MODEL_CARD = ROOT / "model/README.md"
BASE_MODEL = "fastino/GLiNER2.5-Decide"
BASE_REVISION = "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def release_files() -> dict[str, Path]:
    """Check that the adapter and report describe the same training run."""
    weights = RUN / "best/adapter_model.safetensors"
    config = ROOT / "model/adapter_config.json"
    training = RUN / "exitreceipt-run.json"
    upstream_config = RUN / "training_config.json"
    files = {
        "README.md": MODEL_CARD,
        "adapter_model.safetensors": weights,
        "adapter_config.json": config,
        "evaluation/pilot.json": REPORT,
        "evaluation/seed-20260926.json": ROOT / "results/seed-20260926.json",
        "evaluation/seed-20260927.json": ROOT / "results/seed-20260927.json",
        "evaluation/seed-robustness.json": ROOT / "results/seed-robustness.json",
        "training/exitreceipt-run.json": training,
        "training/training_config.json": upstream_config,
        "LICENSE": ROOT / "LICENSE",
        "NOTICE": ROOT / "NOTICE",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    run = json.loads(training.read_text(encoding="utf-8"))
    adapter = json.loads(config.read_text(encoding="utf-8"))
    original_adapter = json.loads((RUN / "best/adapter_config.json").read_text(encoding="utf-8"))
    if original_adapter.pop("task_type", "missing") is not None or adapter != original_adapter:
        raise ValueError("Hub config must only omit upstream's optional null task_type")
    if report["adapter_sha256"] != digest(weights):
        raise ValueError("published report does not match selected adapter weights")
    for key in ("data_sha256", "evidence_sha256"):
        if report[key] != run[key]:
            raise ValueError(f"report/run {key} mismatch")
    if report["revision"] != run["revision"] or report["revision"] != BASE_REVISION:
        raise ValueError("base model revision mismatch")
    if report["model"] != BASE_MODEL or adapter["base_model_name_or_path"] != BASE_MODEL:
        raise ValueError("base model ID mismatch")
    if report["training_code"] != run["code"]:
        raise ValueError("report is not from the selected training run")
    for name, path in (
        ("data_sha256", ROOT / "data/cases.psv"),
        ("evidence_sha256", ROOT / "data/evidence.psv"),
    ):
        if report[name] != digest(path):
            raise ValueError(f"current {name} differs from release report")
    card = MODEL_CARD.read_text(encoding="utf-8")
    if digest(weights) not in card or BASE_REVISION not in card:
        raise ValueError("model card does not identify the exact adapter and base")
    if "[More Information Needed]" in card:
        raise ValueError("model card still contains template placeholders")
    summary = json.loads(files["evaluation/seed-robustness.json"].read_text(encoding="utf-8"))
    if [run["seed"] for run in summary["runs"]] != [20260925, 20260926, 20260927]:
        raise ValueError("seed summary has an unexpected run set")
    if summary["runs"][0]["adapter_sha256"] != digest(weights):
        raise ValueError("seed summary does not identify the published adapter")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--publish", action="store_true", help="Create and tag a new public model")
    action.add_argument(
        "--sync-supplement",
        action="store_true",
        help="Update only the card and three-seed reports on the existing model's main branch",
    )
    args = parser.parse_args()
    files = release_files()
    print(
        json.dumps(
            {
                "repo_id": REPO_ID,
                "tag": TAG,
                "files": {name: digest(path) for name, path in files.items()},
            },
            indent=2,
        )
    )
    if not args.publish and not args.sync_supplement:
        return 0

    api = HfApi()
    info = api.whoami()
    if info["name"].lower() != REPO_ID.partition("/")[0]:
        raise RuntimeError("authenticated Hugging Face user is not the model owner")
    try:
        refs = api.list_repo_refs(REPO_ID, repo_type="model")
    except RepositoryNotFoundError:
        refs = None
    tag_exists = bool(refs and any(ref.name == TAG for ref in refs.tags))
    if args.publish and tag_exists:
        raise FileExistsError(f"release tag already exists: {TAG}")
    if args.sync_supplement:
        if not tag_exists or api.model_info(REPO_ID).private:
            raise RuntimeError("expected existing public tagged model")
        tagged_weights = Path(hf_hub_download(REPO_ID, "adapter_model.safetensors", revision=TAG))
        if digest(tagged_weights) != digest(files["adapter_model.safetensors"]):
            raise ValueError("tagged Hub weights differ from the published adapter")

    with TemporaryDirectory(prefix="exitreceipt-hf-") as directory:
        stage = Path(directory)
        selected = (
            {
                name: source
                for name, source in files.items()
                if name == "README.md" or name.startswith("evaluation/seed-")
            }
            if args.sync_supplement
            else files
        )
        for name, source in selected.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        if args.publish:
            api.create_repo(REPO_ID, repo_type="model", private=False, exist_ok=True)
        commit = api.upload_folder(
            repo_id=REPO_ID,
            repo_type="model",
            folder_path=stage,
            commit_message=(
                "Document three-seed ExitReceipt sensitivity check"
                if args.sync_supplement
                else "Publish ExitReceipt GLiNER2.5-Decide LoRA pilot"
            ),
        )
        if args.publish:
            api.create_tag(REPO_ID, tag=TAG, revision=commit.oid, repo_type="model")
        for name in selected:
            revision = "main" if args.sync_supplement else TAG
            remote = Path(hf_hub_download(REPO_ID, name, revision=revision))
            if digest(remote) != digest(files[name]):
                raise ValueError(f"remote release digest mismatch: {name}")
    print(
        json.dumps(
            {
                "published": f"https://huggingface.co/{REPO_ID}",
                "commit": commit.oid,
                "tag": TAG,
                "mode": "supplement" if args.sync_supplement else "release",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
