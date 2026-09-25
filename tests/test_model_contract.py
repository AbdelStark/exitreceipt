from pathlib import Path

import pytest

from exitreceipt.corpus import Case
from exitreceipt.model import _adapter_source, inspect_cases


class FakeModel:
    def __init__(self, span):
        self.span = span

    def create_schema(self):
        return self

    def entities(self, labels):
        assert labels == ["receipt", "blocker"]
        return self

    def classification(self, task, labels):
        assert task == "finished"
        assert labels == ["yes", "no"]
        return self

    def extract(self, text, schema, include_spans):
        assert schema is self and include_spans
        return {"finished": "yes", "entities": {"receipt": [self.span], "blocker": []}}


def test_joint_schema_keeps_source_offsets():
    case = Case("x", "test", "mail", "yes", "Send it.", "Provider accepted message m-1.")
    start = case.text.index("m-1")
    span = {"text": "m-1", "start": start, "end": start + 3}
    assert inspect_cases(FakeModel(span), [case])["x"] == {
        "label": "yes",
        "spans": [{"kind": "receipt", **span}],
    }


def test_joint_schema_rejects_hallucinated_span_text():
    case = Case("x", "test", "mail", "yes", "Send it.", "Provider accepted message m-1.")
    span = {"text": "m-2", "start": case.text.index("m-1"), "end": len(case.text)}
    with pytest.raises(ValueError, match="does not match source"):
        inspect_cases(FakeModel(span), [case])


def test_adapter_source_requires_explicit_local_files_or_hub_id(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="missing PEFT adapter"):
        _adapter_source(tmp_path, None)
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert _adapter_source(tmp_path, None) == (str(tmp_path), {})
    with pytest.raises(ValueError, match="only valid for a Hub"):
        _adapter_source(tmp_path, "v0.1.0")
    assert _adapter_source("abdelstark/exitreceipt-gliner2.5-decide-lora", "v0.1.0") == (
        "abdelstark/exitreceipt-gliner2.5-decide-lora",
        {"revision": "v0.1.0"},
    )
    with pytest.raises(ValueError, match="namespace/repository"):
        _adapter_source("typo", None)
