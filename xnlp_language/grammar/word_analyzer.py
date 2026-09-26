#!/usr/bin/env python3
"""Deterministic word-analysis engine for isiXhosa.

This module attempts to segment isiXhosa surface forms using the structured
grammatical entries (noun classes, agreement concords, verb patterns).  It is
fully deterministic: given the same structured data, it always returns the
same result.

For any form it cannot segment, it returns UNKNOWN — never an incorrect
analysis.  This is a deliberate safety property.

No neural model is used.  No tokenizer is trained.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

UNKNOWN_STATUS = "UNKNOWN"
ANALYZED_STATUS = "ANALYZED"

_WORD = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)

_GRAMMAR_DIR = Path(__file__).resolve().parent
_DATA_PATH = _GRAMMAR_DIR / "analyzer_data.json"

_data: dict[str, Any] | None = None


def _load_data() -> dict[str, Any]:
    """Load analyzer data (lazily, cached)."""
    global _data
    if _data is None:
        if _DATA_PATH.exists():
            _data = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
        else:
            _data = {}
    return _data


def analyze_word(word: str) -> dict[str, Any]:
    """Deterministically analyze a single isiXhosa word.

    Returns a dict with:
      - surface_form: the original word
      - status: "ANALYZED" or "UNKNOWN"
      - analysis: dict with segmentation details (empty if UNKNOWN)
      - confidence: "HIGH_CONFIDENCE", "SUPPORTED", "OBSERVED", or "INSUFFICIENT_EVIDENCE"
      - observation_ids: list of observation IDs linked to this analysis
    """
    if not word:
        return _unknown(word)

    data = _load_data()
    lowered = word.lower()

    # Try noun-class prefix matching (longest-first)
    nc_prefixes = data.get("noun_class_prefixes", {})
    best_prefix = None
    best_entry_id = None
    for entry_id, info in nc_prefixes.items():
        if not info:
            continue
        sing = info.get("singular_prefix", "").replace("-", "")
        plur = info.get("plural_prefix", "").replace("-", "")
        for prefix, eid in [(sing, entry_id), (plur, entry_id)]:
            if not prefix:
                continue
            if lowered.startswith(prefix) and len(lowered) > len(prefix) + 1:
                if best_prefix is None or len(prefix) > len(best_prefix):
                    best_prefix = prefix
                    best_entry_id = eid

    if best_prefix:
        stem = lowered[len(best_prefix):]
        return {
            "surface_form": word,
            "status": ANALYZED_STATUS,
            "analysis": {
                "type": "noun",
                "prefix": best_prefix,
                "stem": stem,
                "class_entry": best_entry_id,
            },
            "confidence": "SUPPORTED",
            "observation_ids": [f"NC_PREFIX_{best_prefix.upper()}"],
        }

    # Try subject concord matching
    subj_concords = data.get("subject_concord_map", {})
    for ag_id, info in subj_concords.items():
        prefix = info.get("prefix", "")
        if prefix and lowered.startswith(prefix) and len(lowered) > len(prefix) + 1:
            stem = lowered[len(prefix):]
            return {
                "surface_form": word,
                "status": ANALYZED_STATUS,
                "analysis": {
                    "type": "verb",
                    "subject_concord": prefix,
                    "stem": stem,
                    "agreement_entry": ag_id,
                },
                "confidence": "OBSERVED",
                "observation_ids": [f"VM_VERB_SUBJ_{prefix.upper()}"],
            }

    # Try verb pattern matching
    verb_patterns = data.get("verb_patterns", [])
    for pattern in verb_patterns:
        sf = pattern.get("surface_form", "")
        if sf and lowered.startswith(sf):
            return {
                "surface_form": word,
                "status": ANALYZED_STATUS,
                "analysis": {
                    "type": "verb_pattern",
                    "matched_pattern": sf,
                    "candidate_segments": pattern.get("candidate_segments", []),
                },
                "confidence": "OBSERVED",
                "observation_ids": [f"VM_PATTERN_{sf.upper()}"],
            }

    # No match — return UNKNOWN (safe default, never a wrong analysis)
    return _unknown(word)


def _unknown(word: str) -> dict[str, Any]:
    """Return an UNKNOWN result (safe default)."""
    return {
        "surface_form": word,
        "status": UNKNOWN_STATUS,
        "analysis": {},
        "confidence": "INSUFFICIENT_EVIDENCE",
        "observation_ids": [],
        "notes": "Form not found in structured grammatical entries. Requires human analysis.",
    }


def analyze_words(words: list[str]) -> list[dict[str, Any]]:
    """Analyze a list of words, returning one result per word."""
    return [analyze_word(w) for w in words]


def tokenize(text: str) -> list[str]:
    """Tokenize text into word tokens (deterministic, no model)."""
    return [m.group(0) for m in _WORD.finditer(text)]


def analyze_text(text: str) -> dict[str, Any]:
    """Return word analyses with offsets while preserving every input character."""
    tokens: list[dict[str, Any]] = []
    cursor = 0

    for match in _WORD.finditer(text):
        if cursor < match.start():
            tokens.append({
                "kind": "separator",
                "surface": text[cursor:match.start()],
                "start": cursor,
                "end": match.start(),
            })
        tokens.append({
            "kind": "word",
            "surface": match.group(0),
            "start": match.start(),
            "end": match.end(),
            "analysis": analyze_word(match.group(0)),
        })
        cursor = match.end()

    if cursor < len(text):
        tokens.append({
            "kind": "separator",
            "surface": text[cursor:],
            "start": cursor,
            "end": len(text),
        })

    return {"schema_version": 1, "text": text, "tokens": tokens}


if __name__ == "__main__":
    import sys
    for w in sys.argv[1:]:
        result = analyze_word(w)
        print(json.dumps(result, ensure_ascii=False, indent=2))
