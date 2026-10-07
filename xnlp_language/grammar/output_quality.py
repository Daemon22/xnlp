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
    if lexicon.get("corpus_sha256") != orthography.get("corpus_sha256"):
        raise FoundationUnavailableError("XNLP lexical and orthographic artifacts use different corpora")

    allowed_letters = set(alphabet)
    unsupported_letters = sorted({
        char.lower() for char in text
        if char.isalpha() and char.lower() not in allowed_letters
    })
    digit_offsets = [index for index, char in enumerate(text) if char.isdigit()]
    word_matches = list(_WORD.finditer(text))
    unknown = [
        {"surface": match.group(0), "start": match.start(), "end": match.end()}
        for match in word_matches
        if match.group(0).casefold() not in forms
    ]
    lexical_pass = bool(word_matches) and not unknown
    orthographic_pass = not unsupported_letters and not digit_offsets

    return {
        "schema_version": 1,
        "status": "SUPPORTED" if lexical_pass and orthographic_pass else "REJECTED",
        "checks": {
            "observed_word_forms": {
                "status": "PASS" if lexical_pass else "FAIL",
                "word_count": len(word_matches),
                "unknown_forms": unknown,
            },
            "orthographic_inventory": {
                "status": "PASS" if orthographic_pass else "FAIL",
                "unsupported_letters": unsupported_letters,
                "digit_offsets": digit_offsets,
            },
            "grammar_and_meaning": {
                "status": "NOT_ASSESSED",
                "reason": "The current foundation does not encode a complete semantic or grammatical validator.",
            },
        },
        "foundation": {
            "lexicon_corpus_sha256": lexicon.get("corpus_sha256"),
            "orthography_corpus_sha256": orthography.get("corpus_sha256"),
        },
    }
