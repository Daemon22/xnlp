"""Evidence-bound isiXhosa structure and comprehension services.

This module is deliberately conservative.  It does not invent a meaning when
the corpus-backed analyzer cannot support one.  It exposes the intermediate
representation that generation and evaluation can consume instead of treating
raw text as if it were already understood.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .grammar import analyze_text, analyze_word
from .grammar.word_analyzer import tokenize
from .validation.analysis_schema import assert_valid_analysis

_CLAUSE_BREAK = {"kwaye", "kodwa", "ngoko", "xa", "ukuba", "naxa", "kuba"}
_QUESTION_WORDS = {
    "ngubani", "yintoni", "phi", "nini", "kutheni", "njani", "zingaphi",
    "ngubani?", "yintoni?", "phi?", "nini?", "kutheni?", "njani?",
}


@dataclass(frozen=True)
class StructuredToken:
    surface: str
    normalized: str
    start: int
    end: int
    kind: str
    analysis_status: str
    analysis: dict[str, Any] = field(default_factory=dict)
    confidence: str = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class ClauseFrame:
    text: str
    token_indices: list[int]
    subjects: list[int]
    predicates: list[int]
    objects: list[int]
    connective: str | None = None
    semantic_status: str = "UNRESOLVED"


def _word_tokens(text: str) -> list[StructuredToken]:
    result: list[StructuredToken] = []
    for token in analyze_text(text)["tokens"]:
        if token["kind"] != "word":
            continue
        analysis = token["analysis"]
        result.append(StructuredToken(
            surface=token["surface"], normalized=token["surface"].casefold(),
            start=token["start"], end=token["end"], kind="word",
            analysis_status=analysis["status"], analysis=analysis.get("analysis", {}),
            confidence=analysis.get("confidence", "INSUFFICIENT_EVIDENCE"),
        ))
    return result


def _frames(text: str, tokens: list[StructuredToken]) -> list[ClauseFrame]:
    if not tokens:
        return []
    frames: list[ClauseFrame] = []
    start = 0
    for index, token in enumerate(tokens + [StructuredToken("", "", len(text), len(text), "boundary", "UNKNOWN")]):
        boundary = index == len(tokens) or token.normalized in _CLAUSE_BREAK
        if not boundary:
            continue
        indices = list(range(start, index))
        if not indices:
            start = index + 1
            continue
        predicates = [i for i in indices if tokens[i].analysis.get("type") in {"verb", "verb_pattern"}]
        nouns = [i for i in indices if tokens[i].analysis.get("type") == "noun"]
        subjects = nouns[:1]
        objects = nouns[1:]
        frames.append(ClauseFrame(
            text=" ".join(tokens[i].surface for i in indices), token_indices=indices,
            subjects=subjects, predicates=predicates, objects=objects,
            connective=tokens[index].surface if index < len(tokens) else None,
            semantic_status="PARTIAL" if predicates else "UNRESOLVED",
        ))
        start = index + 1
    return frames


def understand(text: str) -> dict[str, Any]:
    """Return a stable, inspectable structural representation of *text*."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    tokens = _word_tokens(text)
    frames = _frames(text, tokens)
    analysis = {
        "schema_version": 1,
        "original_text": text,
        "language": "isiXhosa",
        "text": text,
        "tokens": [
            {
                **asdict(token),
                "lemma_or_root": token.analysis.get("stem"),
                "part_of_speech": token.analysis.get("type"),
                "morphological_segmentation": [],
                "agreement_features": {},
                "grammatical_features": {},
                "syntactic_role": None,
                "explanation": None,
                "verification_status": token.confidence,
                "supporting_evidence": token.analysis.get("class_entry", ""),
                "uncertainty": None if token.analysis_status == "ANALYZED" else "insufficient evidence",
            }
            for token in tokens
        ],
        "sentence_type": "question" if text.rstrip().endswith("?") else "exclamation" if text.rstrip().endswith("!") else "declarative",
        "clauses": [asdict(frame) for frame in frames],
        "grammatical_relations": [
            {"subject": frame.subjects, "predicate": frame.predicates, "object": frame.objects}
            for frame in frames
        ],
        "meaning": {"status": "UNRESOLVED", "explicit_content": None, "inference": None},
        "ambiguities": [],
        "evidence": [],
        "provenance": {"source": "deterministic_corpus_bound_analyzer", "status": "OBSERVED"},
        "confidence": "PARTIAL" if tokens else "INSUFFICIENT_EVIDENCE",
        "semantic_frames": [
            {
                "predicate": tokens[frame.predicates[0]].surface if frame.predicates else None,
                "arguments": {
                    "subject": [tokens[i].surface for i in frame.subjects],
                    "object": [tokens[i].surface for i in frame.objects],
                },
                "status": frame.semantic_status,
            }
            for frame in frames
        ],
        "comprehension_boundary": {
            "mode": "evidence_bound_extraction",
            "unknown_tokens": sum(t.analysis_status == "UNKNOWN" for t in tokens),
            "semantic_claims_are_abstaining": True,
        },
    }
    assert_valid_analysis(analysis)
    return analysis


def comprehend(context: str, question: str) -> dict[str, Any]:
    """Answer by citing a supporting context sentence, or abstain.

    This is intentionally extractive and auditable.  It is a real
    comprehension path for supported text, not a fabricated neural answer.
    """
    if not isinstance(context, str) or not isinstance(question, str):
        raise TypeError("context and question must be strings")
    question_words = [w.casefold() for w in tokenize(question) if w.casefold() not in _QUESTION_WORDS]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", context) if s.strip()]
    scored = []
    for sentence in sentences:
        words = {w.casefold() for w in tokenize(sentence)}
        overlap = sum(word in words for word in question_words)
        scored.append((overlap, sentence))
    best_score, best_sentence = max(scored, default=(0, None))
    if not best_sentence or not question_words or best_score == 0:
        return {
            "status": "ABSTAIN", "answer": None, "support": [], "evidence_spans": [],
            "claim_type": "UNSUPPORTED", "inference": False,
            "confidence": "INSUFFICIENT_EVIDENCE",
        }
    start = context.find(best_sentence)
    return {
        "status": "SUPPORTED_EXTRACTION",
        "answer": best_sentence,
        "support": [best_sentence],
        "evidence_spans": [{"text": best_sentence, "start": start, "end": start + len(best_sentence)}],
        "claim_type": "EXPLICIT",
        "inference": False,
        "confidence": "SUPPORTED" if best_score >= 2 else "OBSERVED",
        "matched_question_terms": [w for w in question_words if w in best_sentence.casefold()],
    }


def validate_generation(prompt: str, generated: str) -> dict[str, Any]:
    """Score a continuation without pretending to certify grammaticality."""
    prompt_words = {w.casefold() for w in tokenize(prompt)}
    output_words = tokenize(generated)
    continuation = output_words[len(tokenize(prompt)):]
    analysis = [analyze_word(word) for word in continuation]
    known = sum(item["status"] == "ANALYZED" for item in analysis)
    repeated = len(output_words) - len(set(w.casefold() for w in output_words))
    return {
        "prompt": prompt,
        "generated": generated,
        "extended": len(output_words) > len(prompt_words),
        "continuation_token_count": len(continuation),
        "known_analysis_ratio": known / max(len(analysis), 1),
        "repetition_ratio": repeated / max(len(output_words), 1),
        "status": "REVIEW_REQUIRED",
        "limitation": "This report measures structural evidence and repetition; native-speaker review is still required for grammaticality and meaning.",
    }


__all__ = ["understand", "comprehend", "validate_generation"]
