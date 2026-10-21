"""Build provenance-linked instruction examples for isiXhosa structure and QA."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xnlp_language import understand
from xnlp_language.grammar import analyze_word
from xnlp_language.grammar.word_analyzer import tokenize


TASKS = ("STRUCTURE", "GRAMMAR", "QUESTION_ANSWER", "CONTINUATION")


def _compact_structure(analysis: dict[str, Any]) -> dict[str, Any]:
    """Keep the canonical fields while removing duplicated diagnostic payloads."""
    tokens = []
    for token in analysis.get("tokens", []):
        tokens.append({
            "surface": token.get("surface"),
            "lemma_or_root": token.get("lemma_or_root"),
            "part_of_speech": token.get("part_of_speech"),
            "morphological_segmentation": token.get("morphological_segmentation", []),
            "agreement_features": token.get("agreement_features", {}),
            "grammatical_features": token.get("grammatical_features", {}),
            "syntactic_role": token.get("syntactic_role"),
            "explanation": token.get("explanation"),
            "verification_status": token.get("verification_status"),
            "supporting_evidence": token.get("supporting_evidence", ""),
            "uncertainty": token.get("uncertainty"),
        })
    return {
        "schema_version": analysis["schema_version"],
        "original_text": analysis["original_text"],
        "language": analysis["language"],
        "tokens": tokens,
        "sentence_type": analysis["sentence_type"],
        "clauses": analysis["clauses"],
        "grammatical_relations": analysis["grammatical_relations"],
        "meaning": analysis["meaning"],
        "ambiguities": analysis["ambiguities"],
        "evidence": analysis["evidence"],
        "provenance": analysis["provenance"],
        "confidence": analysis["confidence"],
    }


def _example(task: str, prompt: str, target: str, source_id: str, source_text: str, status: str) -> dict[str, Any]:
    return {
        "dataset_version": "structured-curriculum-v2",
        "task": task,
        "prompt": f"<TASK:{task}>\n{prompt}\n<ANSWER>\n",
        "target": target,
        "source_record_id": source_id,
        "source_text": source_text,
        "provenance": {"source_record_id": source_id, "source_type": "corpus", "target_derivation": "programmatic"},
        "target_status": status,
        "annotation_status": "NOT_EXPERT_REVIEWED",
    }


def build_curriculum(corpus_path: str | Path, output_path: str | Path, *, limit: int | None = None) -> dict[str, Any]:
    """Create JSONL examples without changing the source corpus."""
    corpus_path, output_path = Path(corpus_path), Path(output_path)
    rows = [line.strip() for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    examples: list[dict[str, Any]] = []
    for index, text in enumerate(rows):
        source_id = f"curriculum_{index:06d}"
        structure = understand(text)
        compact_structure = json.dumps(_compact_structure(structure), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        examples.append(_example("STRUCTURE", text, compact_structure, source_id, text, "DERIVED_STRUCTURAL_ANALYSIS"))

        words = tokenize(text)
        if words:
            word = words[0]
            examples.append(_example(
                "GRAMMAR", f"Hlalutya eli gama: {word}",
                json.dumps(analyze_word(word), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                source_id, text, "EVIDENCE_BOUND_ANALYSIS",
            ))
            examples.append(_example(
                "QUESTION_ANSWER", f"Yintoni ekhankanyiweyo kwisivakalisi?\nUmxholo: {text}",
                word, source_id, text, "EXTRACTIVE_CANDIDATE",
            ))
        if len(words) >= 3:
            prefix = " ".join(words[: max(1, len(words) // 2)])
            examples.append(_example("CONTINUATION", prefix, text, source_id, text, "ATTESTED_CONTINUATION"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in examples:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    counts = {task: sum(item["task"] == task for item in examples) for task in TASKS}
    return {"source": str(corpus_path), "examples": len(examples), "counts": counts, "output": str(output_path)}


__all__ = ["TASKS", "build_curriculum"]
