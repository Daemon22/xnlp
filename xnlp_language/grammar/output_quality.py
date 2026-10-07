"""Evidence-backed lexical and orthographic checks for generated isiXhosa text.

This gate only certifies the checks it can support. A pass means every word
form was observed in an eligible TARGET_LANGUAGE corpus record and every
letter is present in the generated orthographic inventory. It does not prove
grammaticality, meaning, or discourse coherence.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_LANGUAGE_ROOT = Path(__file__).resolve().parents[1]
_GENERATED = _LANGUAGE_ROOT / "generated"
_WORD = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)


class FoundationUnavailableError(RuntimeError):
    """Raised when the generated, provenance-checked foundation is missing."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FoundationUnavailableError(
            f"Required XNLP foundation artifact is unavailable: {path.name}. "
            "Build the linguistic foundation before enabling the output gate."
        ) from exc
    if not isinstance(value, dict):
        raise FoundationUnavailableError(f"Invalid XNLP foundation artifact: {path.name}")
    return value


def validate_generated_text(text: str) -> dict[str, Any]:
    """Check orthographic letters and exact corpus-observed word forms.

    The compact lexicon is generated from TARGET_LANGUAGE records only.
    Unknown words are rejected as unsupported, not labelled incorrect.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    lexicon = _load_json(_GENERATED / "foundation" / "observed_lexicon.json")
    orthography = _load_json(_GENERATED / "foundation" / "orthography.json")
    forms = lexicon.get("forms")
    alphabet = orthography.get("alphabetic_characters")
    if not isinstance(forms, dict) or not isinstance(alphabet, dict):
        raise FoundationUnavailableError("XNLP foundation artifacts have an invalid format")
    lexicon_hash = lexicon.get("corpus_sha256")
    if not isinstance(lexicon_hash, str) or not lexicon_hash:
        raise FoundationUnavailableError("XNLP artifacts are missing their corpus fingerprint")
    if lexicon_hash != orthography.get("corpus_sha256"):
        raise FoundationUnavailableError("XNLP lexical and orthographic artifacts use different corpora")

    allowed_letters = set(alphabet)
    unsupported_letters = sorted({
        char.lower() for char in text
        if char.isalpha() and char.lower() not in allowed_letters
    })
    digit_offsets = [index for index, char in enumerate(text) if char.isdigit()]
    word_matches = list(_WORD.finditer(text))
    supported_forms = []
    unknown = []
    for match in word_matches:
        evidence = forms.get(match.group(0).casefold())
        if evidence is None:
            unknown.append({
                "surface": match.group(0),
                "start": match.start(),
                "end": match.end(),
            })
        else:
            supported_forms.append({
                "surface": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "lexeme_id": evidence["lexeme_id"],
                "evidence_record_ids": evidence["evidence_record_ids"],
            })
    lexical_pass = bool(word_matches) and not unknown
    orthographic_pass = not unsupported_letters and not digit_offsets

    semantic_path = _GENERATED / "semantics" / "lexical_senses.jsonl"
    try:
        semantic_rows = [
            json.loads(line)
            for line in semantic_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise FoundationUnavailableError(
            "Reviewed semantic sense data is unavailable; rebuild the foundation."
        ) from exc
    try:
        construction_rows = [
            json.loads(line)
            for line in (_GENERATED / "semantics" / "verified_constructions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise FoundationUnavailableError("Verified construction data is unavailable; rebuild the foundation.") from exc
    semantic_report = _load_json(_GENERATED / "reports" / "semantic_coverage.json")
    if semantic_report.get("verified_construction_entries") != len(construction_rows):
        raise FoundationUnavailableError("Semantic coverage report does not match the construction inventory")
    if semantic_report.get("corpus_sha256") != lexicon_hash:
        raise FoundationUnavailableError("Semantic and lexical artifacts use different corpora")
    if semantic_report.get("verified_sense_entries") != len(semantic_rows):
        raise FoundationUnavailableError("Semantic coverage report does not match the sense inventory")

    semantic_index: dict[str, list[dict[str, str]]] = {}
    for sense in semantic_rows:
        semantic_index.setdefault(sense["lemma"].casefold(), []).append({
            "sense_id": sense["sense_id"],
            "gloss": sense["gloss"],
        })
    grounded_forms = []
    ungrounded_forms = []
    for match in word_matches:
        senses = semantic_index.get(match.group(0).casefold(), [])
        item = {
            "surface": match.group(0),
            "start": match.start(),
            "end": match.end(),
        }
        if senses:
            grounded_forms.append({**item, "senses": senses})
        else:
            ungrounded_forms.append(item)

    attested_constructions = []
    text_tokens = list(_WORD.finditer(text))
    text_forms = [match.group(0).casefold() for match in text_tokens]
    for construction in construction_rows:
        construction_forms = [match.group(0).casefold() for match in _WORD.finditer(construction["surface_text"])]
        width = len(construction_forms)
        for start_index in range(len(text_forms) - width + 1):
            if text_forms[start_index:start_index + width] == construction_forms:
                attested_constructions.append({
                    "construction_id": construction["construction_id"],
                    "surface": text[text_tokens[start_index].start():text_tokens[start_index + width - 1].end()],
                    "start": text_tokens[start_index].start(),
                    "end": text_tokens[start_index + width - 1].end(),
                    "reviewed_translation": construction["translation"],
                    "evidence_record_ids": [item["record_id"] for item in construction["evidence"]],
                })
    return {
        "schema_version": 1,
        "status": "LEXICALLY_SUPPORTED" if lexical_pass and orthographic_pass else "REJECTED",
        "checks": {
            "observed_word_forms": {
                "status": "PASS" if lexical_pass else "FAIL",
                "word_count": len(word_matches),
                "supported_forms": supported_forms,
                "unknown_forms": unknown,
            },
            "orthographic_inventory": {
                "status": "PASS" if orthographic_pass else "FAIL",
                "unsupported_letters": unsupported_letters,
                "digit_offsets": digit_offsets,
            },
            "lexical_semantics": {
                "status": "PARTIAL" if ungrounded_forms else "LEXICAL_SENSES_AVAILABLE",
                "grounded_forms": grounded_forms,
                "ungrounded_forms": ungrounded_forms,
                "note": "Word glosses are not a sentence-level meaning or coherence check.",
                "coverage_summary": {
                    "status": semantic_report.get("status"),
                    "observed_lexicon_entries": semantic_report.get("observed_lexicon_entries"),
                    "verified_sense_entries": semantic_report.get("verified_sense_entries"),
                    "unique_glossed_lexemes": semantic_report.get("unique_glossed_lexemes"),
                    "coverage_percent_of_observed_lexemes": semantic_report.get(
                        "coverage_percent_of_observed_lexemes"
                    ),
                    "gaps": semantic_report.get("gaps", []),
                },
            },
            "grammar_and_meaning": {
                "status": "NOT_ASSESSED",
                "attested_construction_status": "EXACT_SPANS_FOUND" if attested_constructions else "NO_EXACT_SPAN_FOUND",
                "attested_constructions": attested_constructions,
                "reason": "The current foundation does not encode a complete semantic or grammatical validator.",
            },
        },
        "foundation": {
            "lexicon_corpus_sha256": lexicon.get("corpus_sha256"),
            "orthography_corpus_sha256": orthography.get("corpus_sha256"),
        },
    }
