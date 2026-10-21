"""Compare baseline and structured checkpoints on identical held-out tasks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from xnlp_language import comprehend
from xnlp_language.grammar.word_analyzer import tokenize
from xnlp_trainer.inference import XNLPPredictor
from xnlp_trainer.structured_curriculum import build_curriculum


TASK_EVAL = {
    "STRUCTURE": {"max_new_tokens": 384, "metric": "schema_validity"},
    "GRAMMAR": {"max_new_tokens": 192, "metric": "schema_validity"},
    "QUESTION_ANSWER": {"max_new_tokens": 96, "metric": "answer_containment"},
    "CONTINUATION": {"max_new_tokens": 96, "metric": "nonempty_continuation"},
}


def _answer_text(output: str) -> str:
    return output.split("<ANSWER>", 1)[-1].strip()


def _valid_json(output: str) -> bool:
    text = _answer_text(output)
    if not text.startswith("{"):
        return False
    try:
        json.JSONDecoder().raw_decode(text)
        return True
    except json.JSONDecodeError:
        return False


def _json_object(output: str) -> dict[str, Any] | None:
    text = _answer_text(output)
    if not text.startswith("{"):
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def evaluate_checkpoint(checkpoint: str, rows: list[dict[str, Any]], *, limit: int = 20) -> dict[str, Any]:
    predictor = XNLPPredictor.load(checkpoint, device="cpu")
    results: dict[str, Any] = {"checkpoint": checkpoint, "tasks": {}, "samples": []}
    for row in rows[:limit]:
        task = row["task"]
        policy = TASK_EVAL.get(task, {"max_new_tokens": 96, "metric": "nonempty_continuation"})
        output = predictor.generate(row["prompt"], max_new_tokens=policy["max_new_tokens"], temperature=0.3, top_k=10, top_p=0.9, do_sample=False)
        answer = _answer_text(output)
        task_result = results["tasks"].setdefault(task, {
            "total": 0, "extended": 0, "nonempty": 0, "target_containment": 0,
            "valid_json": 0, "schema_valid": 0, "generation_budget_tokens": policy["max_new_tokens"],
            "metric": policy["metric"],
        })
        task_result["total"] += 1
        if len(tokenize(answer)) > 0:
            task_result["extended"] += 1
            task_result["nonempty"] += 1
        # Containment is retained only for extractive QA; it is not a valid
        # quality signal for free-form structure, grammar, or continuation.
        if task == "QUESTION_ANSWER" and row["target"].casefold() in answer.casefold():
            task_result["target_containment"] += 1
        parsed = _json_object(output)
        if task in {"STRUCTURE", "GRAMMAR"} and parsed is not None:
            task_result["valid_json"] += 1
            if task == "STRUCTURE":
                from xnlp_language.validation.analysis_schema import validate_analysis
                if not validate_analysis(parsed):
                    task_result["schema_valid"] += 1
        if len(results["samples"]) < 12:
            results["samples"].append({"task": task, "prompt": row["prompt"], "target": row["target"],
                                       "target_token_count": len(tokenize(row["target"])),
                                       "generation_budget_tokens": policy["max_new_tokens"], "output": output,
                                       "parsed_json": parsed is not None})
    for metrics in results["tasks"].values():
        total = max(metrics["total"], 1)
        metrics["extension_rate"] = metrics["extended"] / total
        metrics["target_containment_rate"] = metrics["target_containment"] / total
        metrics["valid_json_rate"] = metrics["valid_json"] / total
        metrics["schema_validity_rate"] = metrics["schema_valid"] / total
        metrics["nonempty_rate"] = metrics["nonempty"] / total
    return results


def compare(baseline: str, updated: str, test_corpus: str, output: str, limit: int = 20) -> dict[str, Any]:
    temp = Path(output).with_suffix(".heldout.jsonl")
    curriculum = build_curriculum(test_corpus, temp, limit=max(limit, 20))
    rows = [json.loads(line) for line in temp.read_text(encoding="utf-8").splitlines()]
    report = {
        "report_version": 2,
        "evaluation_split": str(test_corpus),
        "provenance": "corpus-derived candidates from held-out split; not expert gold",
        "baseline": evaluate_checkpoint(baseline, rows, limit=limit),
        "updated": evaluate_checkpoint(updated, rows, limit=limit),
        "grounded_qa_control": comprehend("UMqhayi ngumbongi.", "Ngubani uMqhayi?"),
        "metric_policy": {
            "STRUCTURE": "JSON parse plus canonical analysis-schema validity",
            "GRAMMAR": "JSON parse only until a reviewed grammar schema is available",
            "QUESTION_ANSWER": "target containment is retained as an extractive proxy, with grounded control reported separately",
            "CONTINUATION": "non-empty extension only; coherence and grammaticality require review",
        },
        "limitations": [
            "Structured targets are programmatically derived and are not expert grammar gold.",
            "Generation metrics do not establish grammaticality, meaning, or naturalness.",
            "The deterministic QA control is reported separately from neural generation.",
        ],
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="artifacts/xnlp_prose_pilot/model/best_model.pt")
    parser.add_argument("--updated", default="artifacts/xnlp_structured/model/structured_model_v2.pt")
    parser.add_argument("--test-corpus", default="data/processed/v2/splits/test.txt")
    parser.add_argument("--output", default="artifacts/xnlp_structured/evaluation_baseline_vs_updated.json")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.updated, args.test_corpus, args.output, args.limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
