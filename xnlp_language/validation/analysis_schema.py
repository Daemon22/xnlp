"""Dependency-free validator for canonical isiXhosa sentence analyses."""
from __future__ import annotations

from typing import Any


REQUIRED_ANALYSIS_FIELDS = {
    "schema_version", "original_text", "language", "tokens", "sentence_type",
    "clauses", "grammatical_relations", "meaning", "ambiguities", "evidence",
    "provenance", "confidence",
}
REQUIRED_TOKEN_FIELDS = {
    "surface", "lemma_or_root", "part_of_speech", "morphological_segmentation",
    "agreement_features", "grammatical_features", "syntactic_role", "explanation",
    "verification_status", "supporting_evidence", "uncertainty",
}


def validate_analysis(value: Any) -> list[str]:
    """Return validation errors; an empty list means schema-valid."""
    errors: list[str] = []
    if not isinstance(value, dict):
        return ["analysis must be an object"]
    missing = sorted(REQUIRED_ANALYSIS_FIELDS - value.keys())
    errors.extend(f"missing analysis field: {field}" for field in missing)
    if value.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if value.get("language") != "isiXhosa":
        errors.append("language must be isiXhosa")
    if not isinstance(value.get("tokens"), list):
        errors.append("tokens must be a list")
    else:
        for index, token in enumerate(value["tokens"]):
            if not isinstance(token, dict):
                errors.append(f"tokens[{index}] must be an object")
                continue
            errors.extend(f"tokens[{index}] missing field: {field}" for field in sorted(REQUIRED_TOKEN_FIELDS - token.keys()))
    for field in ("clauses", "grammatical_relations", "ambiguities", "evidence"):
        if not isinstance(value.get(field), list):
            errors.append(f"{field} must be a list")
    if not isinstance(value.get("provenance"), dict):
        errors.append("provenance must be an object")
    return errors


def assert_valid_analysis(value: Any) -> None:
    errors = validate_analysis(value)
    if errors:
        raise ValueError("Invalid isiXhosa analysis: " + "; ".join(errors))


__all__ = ["validate_analysis", "assert_valid_analysis"]
