import json

import evaluate_structured
from evaluate_structured import _valid_json, _answer_text


def test_structured_output_validator_rejects_json_like_garbage():
    assert _valid_json('<ANSWER>{"class": []}') is True
    assert _valid_json('<ANSWER>{"class": []') is False
    assert _answer_text("prompt<ANSWER>value") == "value"


def test_schema_valid_counter_increments_only_for_valid_analysis(monkeypatch):
    class FakePredictor:
        def generate(self, *_args, **_kwargs):
            return '<ANSWER>{"schema_version": 1, "language": "isiXhosa"}'

    monkeypatch.setattr(
        evaluate_structured.XNLPPredictor,
        "load",
        lambda *_args, **_kwargs: FakePredictor(),
    )
    rows = [{"task": "STRUCTURE", "prompt": "test", "target": "{}"}]

    result = evaluate_structured.evaluate_checkpoint("unused", rows)

    assert result["tasks"]["STRUCTURE"]["valid_json"] == 1
    assert result["tasks"]["STRUCTURE"]["schema_valid"] == 0
