"""Thin boundary around GLiNER2.5-Decide and its native LoRA trainer."""

from __future__ import annotations

import os
from pathlib import Path

from .corpus import LABELS, Case, Evidence

MODEL_ID = "fastino/GLiNER2.5-Decide"
MODEL_REVISION = "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6"
TASK = {"finished": list(LABELS)}
ENTITY_TYPES = ("receipt", "blocker")


def _adapter_source(adapter: Path | str, revision: str | None) -> tuple[str, dict[str, str]]:
    if isinstance(adapter, Path):
        if revision is not None:
            raise ValueError("adapter revision is only valid for a Hub repository")
        if not (adapter / "adapter_config.json").exists():
            raise FileNotFoundError(f"missing PEFT adapter in {adapter}")
        return str(adapter), {}
    if adapter.count("/") != 1 or adapter.startswith(("/", ".")):
        raise ValueError("Hub adapter must be a namespace/repository ID")
    return adapter, {"revision": revision} if revision else {}


def load_model(adapter: Path | str | None = None, *, adapter_revision: str | None = None):
    from gliner2 import AutoExtractor

    if adapter is None and adapter_revision is not None:
        raise ValueError("adapter revision requires a Hub adapter")
    source = _adapter_source(adapter, adapter_revision) if adapter is not None else None
    model = AutoExtractor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if source is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, source[0], **source[1])
    return model.eval()


def inspect_cases(model, cases: list[Case]) -> dict[str, dict]:
    schema = (
        model.create_schema().entities(list(ENTITY_TYPES)).classification("finished", list(LABELS))
    )
    inspections = {}
    for case in cases:
        result = model.extract(case.text, schema, include_spans=True)
        prediction = result.get("finished")
        if prediction not in LABELS:
            raise ValueError(f"unexpected model prediction for {case.id}: {result!r}")
        entities = result.get("entities", {})
        spans = []
        for kind in ENTITY_TYPES:
            for span in entities.get(kind, []):
                start, end = span.get("start"), span.get("end")
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or not 0 <= start < end <= len(case.text)
                ):
                    raise ValueError(f"invalid evidence offset for {case.id}: {span!r}")
                if case.text[start:end] != span.get("text"):
                    raise ValueError(f"evidence span does not match source for {case.id}: {span!r}")
                spans.append({"kind": kind, "text": span["text"], "start": start, "end": end})
        inspections[case.id] = {"label": prediction, "spans": spans}
    return inspections


def predict_cases(model, cases: list[Case]) -> dict[str, str]:
    return {id: item["label"] for id, item in inspect_cases(model, cases).items()}


def train_adapter(
    cases: list[Case],
    evidence: dict[str, Evidence],
    output_dir: Path,
    *,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
) -> dict:
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    from gliner2.training.data import Classification, InputExample
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

    if device not in {"cpu", "mps"}:
        raise ValueError("device must be cpu or mps")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty run directory: {output_dir}")

    def examples(split: str) -> list[InputExample]:
        return [
            InputExample(
                text=case.text,
                entities={evidence[case.id].kind: [evidence[case.id].evidence]}
                if case.id in evidence
                else {},
                classifications=[
                    Classification(task="finished", labels=list(LABELS), true_label=case.label)
                ],
            )
            for case in cases
            if case.split == split
        ]

    model = load_model()
    config = TrainingConfig(
        output_dir=str(output_dir),
        experiment_name="exitreceipt-pilot",
        num_epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        encoder_lr=2e-5,
        task_lr=3e-4,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_best=True,
        save_total_limit=2,
        early_stopping=False,
        use_lora=True,
        lora_r=8,
        lora_alpha=16.0,
        lora_target_modules=["encoder", "span_rep", "classifier"],
        save_adapter_only=True,
        fp16=False,
        bf16=False,
        num_workers=0,
        pin_memory=False,
        seed=seed,
        deterministic=True,
        logging_steps=5,
    )
    trainer = ExtractorTrainer(model, config)
    if device == "mps":
        # gliner2 2.0.0 defaults to CPU on macOS. Its trainer otherwise supports
        # the MPS device; keep this explicit and fail if a required op cannot run.
        trainer.device = torch.device("mps")
        trainer.model.to(trainer.device)
    summary = trainer.train(train_data=examples("train"), eval_data=examples("dev"))
    return {
        "summary": summary,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
            "seed": seed,
            "lora_r": 8,
            "lora_alpha": 16.0,
            "lora_targets": ["encoder", "span_rep", "classifier"],
            "encoder_lr": 2e-5,
            "task_lr": 3e-4,
        },
    }
