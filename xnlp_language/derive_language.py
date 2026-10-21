"""Create a reversible, evidence-bounded language derivation specification."""
from __future__ import annotations

import argparse
import collections
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from xnlp_language.build_foundation import WORD
from xnlp_language.validation.linguistic_layer import _artifact_hash, _read_jsonl


ROOT = Path(__file__).resolve().parents[1]
GENERATED = Path("xnlp_language") / "generated"
DEFAULT_OUTPUT = ROOT / GENERATED / "derivation"
DEFAULT_REPORT = ROOT / GENERATED / "reports" / "language_derivation_report.md"
SPEC_VERSION = "0.1.0"
CORRECTION_FIELDS = (
    "correction_status",
    "corrected_text",
    "transcription_status",
    "transcription_verified_by",
)
PUNCTUATION = re.compile(r"[.!?]+")
DOMAIN_MAP = {
    "agreement": "AGREEMENT_CONCORD",
    "noun_classes": "MORPHOLOGY",
    "noun_class": "MORPHOLOGY",
    "morphology": "MORPHOLOGY",
    "derivational_morphology": "MORPHOLOGY",
    "verb_morphology": "MORPHOLOGY",
    "tam": "MORPHOLOGY",
    "word_formation": "MORPHOLOGY",
    "syntax": "SYNTACTIC_STRUCTURE",
    "modifier": "SYNTACTIC_STRUCTURE",
    "negation": "SYNTACTIC_STRUCTURE",
    "function_word": "SEMANTIC_FUNCTIONAL",
    "data_governance": "UNKNOWN_UNRESOLVED",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _work_group(record: dict[str, Any], sources: dict[str, dict[str, Any]]) -> str:
    attribution = record.get("work_attribution")
    if isinstance(attribution, dict) and isinstance(attribution.get("work"), str):
        return attribution["work"]
    source = sources.get(str(record.get("source_title", "")), {})
    return str(source.get("independence_group") or record.get("source_title") or "UNKNOWN")


def _sample_ids(ids: set[str], limit: int = 12) -> list[str]:
    return sorted(ids)[:limit]


def _source_refs(
    record_ids: list[str],
    records_by_id: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    titles = set()
    groups = set()
    for record_id in record_ids:
        record = records_by_id.get(record_id)
        if record is None:
            continue
        title = str(record.get("source_title", "UNKNOWN"))
        titles.add(title)
        groups.add(_work_group(record, sources))
    return sorted(titles), sorted(groups)


def _frequency_inventory(
    records: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    letter_counts: collections.Counter[str] = collections.Counter()
    letter_record_ids: dict[str, set[str]] = collections.defaultdict(set)
    folded_letter_counts: collections.Counter[str] = collections.Counter()
    punctuation_counts: collections.Counter[str] = collections.Counter()
    punctuation_record_ids: dict[str, set[str]] = collections.defaultdict(set)
    token_counts: collections.Counter[str] = collections.Counter()
    token_forms: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    token_records: dict[str, set[str]] = collections.defaultdict(set)
    token_groups: dict[str, set[str]] = collections.defaultdict(set)
    char_ngrams: dict[int, collections.Counter[str]] = {2: collections.Counter(), 3: collections.Counter()}
    ngram_records: dict[int, dict[str, set[str]]] = {
        2: collections.defaultdict(set),
        3: collections.defaultdict(set),
    }
    capitalization = collections.Counter()
    punctuation_delimited_segments = 0
    whitespace_tokens = 0

    for record in records:
        record_id = str(record["record_id"])
        text = str(record.get("text", ""))
        group = _work_group(record, sources)
        punctuation_delimited_segments += len([part for part in PUNCTUATION.split(text) if part.strip()])
        whitespace_tokens += len(text.split())
        record_token_index = 0
        for char in text:
            category = unicodedata.category(char)
            if category.startswith("L"):
                letter_counts[char] += 1
                folded_letter_counts[char.casefold()] += 1
                letter_record_ids[char].add(record_id)
            elif category.startswith("P"):
                punctuation_counts[char] += 1
                punctuation_record_ids[char].add(record_id)
        for match in WORD.finditer(text):
            surface = match.group(0)
            normalized = surface.casefold()
            token_counts[normalized] += 1
            token_forms[normalized][surface] += 1
            token_records[normalized].add(record_id)
            token_groups[normalized].add(group)
            if surface[:1].isupper():
                capitalization["regex_token_initial_uppercase"] += 1
                capitalization[
                    "record_initial_token_uppercase"
                    if record_token_index == 0 else "later_record_token_uppercase"
                ] += 1
            else:
                capitalization["regex_token_initial_lowercase"] += 1
                capitalization[
                    "record_initial_token_lowercase"
                    if record_token_index == 0 else "later_record_token_lowercase"
                ] += 1
            record_token_index += 1
            if "-" in surface:
                capitalization["regex_tokens_with_hyphen"] += 1
            if "'" in surface or "\u2019" in surface or "\u2018" in surface:
                capitalization["regex_tokens_with_apostrophe"] += 1
            letters = "".join(
                char.casefold()
                for char in surface
                if unicodedata.category(char).startswith("L")
            )
            for size in (2, 3):
                for index in range(max(0, len(letters) - size + 1)):
                    ngram = letters[index:index + size]
                    char_ngrams[size][ngram] += 1
                    ngram_records[size][ngram].add(record_id)

    letter_inventory = []
    for char, count in sorted(letter_counts.items(), key=lambda item: (item[0].casefold(), item[0])):
        ids = letter_record_ids[char]
        letter_inventory.append({
            "codepoint": f"U+{ord(char):04X}",
            "symbol": char,
            "unicode_name": unicodedata.name(char, "UNNAMED"),
            "casefold": char.casefold(),
            "occurrences": count,
            "supporting_record_count": len(ids),
            "supporting_record_ids_sample": _sample_ids(ids),
            "knowledge_level": "OBSERVED",
            "interpretation": "Observed alphabetic Unicode character; not asserted to be a phoneme or grapheme.",
        })
    punctuation_inventory = []
    for char, count in sorted(punctuation_counts.items(), key=lambda item: (-item[1], item[0])):
        ids = punctuation_record_ids[char]
        punctuation_inventory.append({
            "codepoint": f"U+{ord(char):04X}",
            "symbol": char,
            "unicode_name": unicodedata.name(char, "UNNAMED"),
            "occurrences": count,
            "supporting_record_count": len(ids),
            "supporting_record_ids_sample": _sample_ids(ids),
            "knowledge_level": "OBSERVED",
        })

    records_by_id = {str(record["record_id"]): record for record in records}

    def top_forms(limit: int = 100) -> list[dict[str, Any]]:
        result = []
        for form, count in sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))[:limit]:
            record_ids = sorted(token_records[form])
            works = sorted(token_groups[form])
            result.append({
                "surface_form_casefolded": form,
                "observed_spellings": [
                    {"form": spelling, "occurrences": spelling_count}
                    for spelling, spelling_count in sorted(
                        token_forms[form].items(), key=lambda item: (-item[1], item[0])
                    )
                ],
                "occurrences": count,
                "record_count": len(record_ids),
                "source_work_count": len(works),
                "source_work_references": works,
                "source_title_references": _source_refs(
                    record_ids, records_by_id, sources
                )[0],
                "supporting_record_ids": record_ids,
                "knowledge_level": "OBSERVED",
                "category": "OBSERVED_LEXICAL_FORM",
                "meaning_and_lemma": None,
            })
        return result

    ngram_inventory = {}
    for size in (2, 3):
        ngram_inventory[str(size)] = [
            {
                "sequence": sequence,
                "occurrences": count,
                "supporting_record_count": len(ngram_records[size][sequence]),
                "supporting_record_ids_sample": _sample_ids(ngram_records[size][sequence]),
                "knowledge_level": "OBSERVED_PATTERN",
                "interpretation": "Within-token letter n-gram; no syllable, sound, morpheme, or phonotactic rule inferred.",
            }
            for sequence, count in sorted(
                char_ngrams[size].items(), key=lambda item: (-item[1], item[0])
            )[:100]
        ]
    case_variant_types = sum(len(forms) > 1 for forms in token_forms.values())
    repeated_across_groups = sum(len(groups) >= 2 for groups in token_groups.values())
    return {
        "text_counts": {
            "record_count": len(records),
            "alphabetic_codepoint_occurrences": sum(letter_counts.values()),
            "distinct_case_sensitive_alphabetic_codepoints": len(letter_counts),
            "distinct_casefolded_alphabetic_characters": len(folded_letter_counts),
            "whitespace_delimited_units": whitespace_tokens,
            "regex_defined_token_occurrences": sum(token_counts.values()),
            "regex_defined_casefolded_types": len(token_counts),
            "punctuation_delimited_nonempty_segments": punctuation_delimited_segments,
            "explicit_definition": (
                "Regex-defined token uses xnlp_language.build_foundation.WORD; it is a "
                "measurement convention, not a claim about language-defined wordhood."
            ),
        },
        "alphabetic_codepoints": letter_inventory,
        "casefolded_letter_counts": dict(sorted(folded_letter_counts.items())),
        "punctuation": punctuation_inventory,
        "capitalization_and_marks": {
            **dict(sorted(capitalization.items())),
            "case_variant_type_count": case_variant_types,
            "types_with_multiple_independent_work_groups": repeated_across_groups,
            "interpretation": "Observed text formatting only; capitalization does not establish proper-name status.",
        },
        "orthographic_n_grams": ngram_inventory,
        "most_frequent_lexical_forms": top_forms(),
        "cross_work_distribution": {
            "casefolded_types_in_two_or_more_independence_groups": repeated_across_groups,
            "independence_definition": (
                "Distinct registry independence_group values, with work_attribution.work "
                "taking precedence where present."
            ),
        },
        "grapheme_inventory_status": (
            "UNKNOWN; the inventory reports Unicode code points, not validated graphemes."
        ),
        "phonological_phonotactic": {
            "observed_letter_sequences": "See orthographic_n_grams; these are written character patterns.",
            "syllable_inventory": "UNKNOWN",
            "phoneme_inventory": "UNKNOWN",
            "sound_orthography_correspondences": "EXTERNAL_KNOWLEDGE_REQUIRED",
            "reason": "Written corpus evidence alone, without independently validated phonetic/transcription evidence, does not establish sound values.",
        },
    }


def _candidate_structures(
    root: Path,
    records_by_id: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    evidence_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    claims_path = root / GENERATED / "validation" / "linguistic_claims.jsonl"
    claims = _read_jsonl(claims_path)
    candidate_rows = []
    for claim in claims:
        created_from = claim.get("created_from", {})
        source_ids = [
            str(record_id)
            for record_id in claim.get("evidence_record_ids", [])
            if str(record_id) in evidence_ids
        ]
        source_works, work_groups = _source_refs(source_ids, records_by_id, sources)
        examples = [
            {
                "record_id": record_id,
                "source_title": records_by_id[record_id].get("source_title"),
                "independence_group": _work_group(records_by_id[record_id], sources),
                "text": records_by_id[record_id].get("text", ""),
            }
            for record_id in source_ids[:5]
        ]
        original_state = str(claim.get("evidence_state", "UNKNOWN"))
        level = "OBSERVED" if claim.get("domain") == "data_governance" else "CANDIDATE"
        candidate_rows.append({
            "derived_id": f"EXISTING_{claim['claim_id']}",
            "knowledge_level": level,
            "domain": DOMAIN_MAP.get(str(claim.get("domain")), "UNKNOWN_UNRESOLVED"),
            "claim_id": claim["claim_id"],
            "claim_text": claim.get("claim", ""),
            "proposed_relationship": claim.get("claim", ""),
            "source_claim_artifact": created_from.get("artifact"),
            "source_claim_artifact_sha256": created_from.get("artifact_sha256"),
            "observed_or_proposed_relationship": claim.get("analysis"),
            "surface_form": claim.get("analysis", {}).get("surface_form")
            if isinstance(claim.get("analysis"), dict) else None,
            "original_evidence_state": original_state,
            "legacy_confidence_label": claim.get("legacy_confidence_label"),
            "source_record_ids": source_ids,
            "supporting_examples": examples,
            "source_work_references": source_works,
            "independence_groups": work_groups,
            "independent_work_count": len(work_groups),
            "confidence": claim.get("confidence"),
            "source_counterexamples": claim.get("counterexamples", []),
            "counterexample_search_status": "NOT_SYSTEMATICALLY_SEARCHED",
            "counterexample_note": (
                "The source claim lists no explicit counterexamples; absence of listed "
                "counterexamples is not evidence that none exist."
            ),
            "reversible": True,
        })

    observations = []
    for path in sorted((root / GENERATED / "grammar").glob("*_observations.jsonl")):
        artifact_sha256 = _artifact_hash(path)
        for row in _read_jsonl(path):
            record_ids = []
            for evidence in row.get("evidence", []):
                if isinstance(evidence, dict) and evidence.get("record_id") in evidence_ids:
                    record_ids.append(str(evidence["record_id"]))
            record_ids = sorted(set(record_ids))
            source_works, groups = _source_refs(record_ids, records_by_id, sources)
            observations.append({
                "observation_id": row.get("observation_id"),
                "source_artifact": path.relative_to(root).as_posix(),
                "source_artifact_sha256": artifact_sha256,
                "domain": DOMAIN_MAP.get(
                    str(row.get("pattern_type", "")), "UNKNOWN_UNRESOLVED"
                ),
                "pattern_type": row.get("pattern_type"),
                "observed_pattern_identifier": row.get("pattern"),
                "token_frequency_as_recorded": row.get("token_frequency"),
                "analysis_status_as_recorded": row.get("analysis_status"),
                "knowledge_level": "OBSERVED",
                "linguistic_interpretation_level": "CANDIDATE",
                "source_record_ids": record_ids,
                "source_work_references": source_works,
                "independence_groups_in_evidence_sample": groups,
                "independent_work_count_in_evidence_sample": len(groups),
                "counterexamples": [],
                "counterexample_search_status": "NOT_SYSTEMATICALLY_SEARCHED",
                "interpretation_warning": row.get("notes", ""),
            })
    return candidate_rows, observations


def derive_language(
    root: Path = ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive a non-executable specification without modifying source evidence."""
    root = root.resolve()
    generated = root / GENERATED
    corpus_path = generated / "evidence" / "corpus_records.jsonl"
    registry_path = generated / "evidence" / "source_provenance.json"
    claims_path = generated / "validation" / "linguistic_claims.jsonl"
    conflicts_path = generated / "validation" / "contradictions.json"
    for path in (corpus_path, registry_path, claims_path, conflicts_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required derivation input is missing: {path}")

    corpus = _read_jsonl(corpus_path)
    registry = _read_json(registry_path)
    sources = registry.get("sources", {})
    records_by_id = {str(record["record_id"]): record for record in corpus}
    if len(records_by_id) != len(corpus):
        raise ValueError("Derivation corpus contains duplicate record IDs")
    if not isinstance(sources, dict):
        raise ValueError("Source provenance registry requires a sources object")

    provenance_eligible = []
    exclusions: collections.Counter[str] = collections.Counter()
    text_mismatches = []
    correction_metadata_ids = []
    for record in corpus:
        title = str(record.get("source_title", ""))
        source = sources.get(title)
        provenance_ok = (
            source is not None
            and source.get("foundation_attested") is True
            and record.get("provenance_class") == "attested"
            and record.get("provenance_type") == "authoritative"
            and record.get("validation_status") == "authoritative"
            and not record.get("generator")
        )
        if not provenance_ok:
            exclusions["provenance_not_attested_or_authoritative"] += 1
            continue
        if record.get("language_class") != "TARGET_LANGUAGE":
            exclusions[f"language_class_{record.get('language_class', 'MISSING')}"] += 1
            continue
        if any(field in record for field in CORRECTION_FIELDS):
            correction_metadata_ids.append(str(record["record_id"]))
        if record.get("text") != record.get("original_text"):
            text_mismatches.append(str(record["record_id"]))
            exclusions["target_text_differs_from_preserved_original"] += 1
            continue
        provenance_eligible.append(record)

    evidence_ids = {str(record["record_id"]) for record in provenance_eligible}
    if correction_metadata_ids:
        raise ValueError(
            "Correction/transcription metadata exists and requires explicit handling: "
            + ", ".join(correction_metadata_ids[:10])
        )
    frequency_inventory = _frequency_inventory(provenance_eligible, sources)
    derived_claims, observations = _candidate_structures(
        root, records_by_id, sources, evidence_ids
    )
    source_counts = collections.Counter(
        str(record.get("source_title", "UNKNOWN")) for record in provenance_eligible
    )
    group_counts = collections.Counter(
        _work_group(record, sources) for record in provenance_eligible
    )
    source_titles = sorted({
        str(record.get("source_title", "UNKNOWN")) for record in corpus
    })
    source_work_catalog = [
        {
            "source_title": title,
            "records_in_corpus": sum(
                str(record.get("source_title", "")) == title for record in corpus
            ),
            "records_in_derivation_set": source_counts.get(title, 0),
            "independence_group": sources.get(title, {}).get("independence_group"),
            "foundation_attested": sources.get(title, {}).get("foundation_attested", False),
            "vetted_by": sources.get(title, {}).get("vetted_by"),
        }
        for title in source_titles
    ]
    excluded_generated_path = root / "data" / "validated" / "cleaned_cadidate_texts.jsonl"
    excluded_generated_count = (
        len(_read_jsonl(excluded_generated_path)) if excluded_generated_path.is_file() else 0
    )
    conflicts = _read_json(conflicts_path)
    usable_conflicts = []
    for conflict in conflicts.get("conflicts", []):
        referenced = [str(item) for item in conflict.get("record_ids", [])]
        in_boundary = sorted(set(referenced) & evidence_ids)
        usable_conflicts.append({
            **conflict,
            "record_ids": in_boundary,
            "excluded_record_reference_count": len(set(referenced) - evidence_ids),
            "source_work_references": _source_refs(in_boundary, records_by_id, sources)[0],
            "independence_groups": _source_refs(in_boundary, records_by_id, sources)[1],
        })

    corpus_hash = _artifact_hash(corpus_path)
    registry_hash = _artifact_hash(registry_path)
    claims_hash = _artifact_hash(claims_path)
    observation_hashes = {
        path.relative_to(root).as_posix(): _artifact_hash(path)
        for path in sorted((generated / "grammar").glob("*_observations.jsonl"))
    }
    included_counts = collections.Counter(record.get("language_class") for record in corpus)
    knowledge_counts = collections.Counter(row["knowledge_level"] for row in derived_claims)
    candidate_counts = collections.Counter(row["domain"] for row in derived_claims)
    spec = {
        "specification_id": "XNLP_DERIVED_LANGUAGE_SPECIFICATION",
        "specification_version": SPEC_VERSION,
        "status": "EXPLORATORY_REVERSIBLE_NOT_EXECUTABLE_GRAMMAR",
        "generated_by": "xnlp_language.derive_language",
        "knowledge_levels": {
            "OBSERVED": "Directly measured from included stored text or copied source-linked observation.",
            "SUPPORTED": "Not assigned to a linguistic generalization in this phase.",
            "CANDIDATE": "An existing or proposed linguistic interpretation without systematic counterexample review.",
            "UNKNOWN": "The evidence does not permit a responsible conclusion.",
        },
        "DERIVATION_EVIDENCE_SET": {
            "id": "DERIVATION_EVIDENCE_SET_V1",
            "exact_selection_rule": (
                "A record from generated/evidence/corpus_records.jsonl is included iff its "
                "source_title exists in source_provenance.json with foundation_attested=true, "
                "provenance_class=attested, provenance_type=authoritative, generator is empty, "
                "validation_status=authoritative, language_class=TARGET_LANGUAGE, and text "
                "equals original_text. This is an "
                "evidence boundary, not a new corpus or correction."
            ),
            "record_count": len(provenance_eligible),
            "record_ids": sorted(evidence_ids),
            "source_titles": sorted(source_counts),
            "source_title_record_counts": dict(sorted(source_counts.items())),
            "independence_group_count": len(group_counts),
            "records_by_independence_group": dict(sorted(group_counts.items())),
            "source_work_catalog": source_work_catalog,
            "input_artifacts": [
                {
                    "path": corpus_path.relative_to(root).as_posix(),
                    "sha256": corpus_hash,
                    "record_count": len(corpus),
                },
                {
                    "path": registry_path.relative_to(root).as_posix(),
                    "sha256": registry_hash,
                    "distinct_source_titles": len(sources),
                },
            ],
            "correction_status": {
                "record_level_correction_or_transcription_control_fields": list(CORRECTION_FIELDS),
                "records_with_explicit_correction_metadata": 0,
                "target_records_identical_to_preserved_original": len(provenance_eligible),
                "target_records_differing_from_preserved_original": text_mismatches,
                "transcription_fidelity_verified_records": 0,
                "interpretation": (
                    "No row-level correction/transcription-verification marker exists. "
                    "Included text matches original_text byte-for-byte as decoded, but this "
                    "does not establish transcription fidelity to the source work."
                ),
            },
            "excluded_from_derivation": {
                "canonical_corpus_records_by_reason": dict(sorted(exclusions.items())),
                "canonical_corpus_records_excluded_total": len(corpus) - len(evidence_ids),
                "generated_candidate_file": {
                    "path": excluded_generated_path.relative_to(root).as_posix(),
                    "record_count": excluded_generated_count,
                    "sha256": (
                        _artifact_hash(excluded_generated_path)
                        if excluded_generated_path.is_file() else None
                    ),
                    "reason": (
                        "Not part of the attested canonical corpus; provenance is generated "
                        "with a non-empty generator field, regardless of validation_status."
                    ),
                },
            },
        },
        "provenance_map": {
            "registry_source_count": len(sources),
            "independence_group_count_in_registry": len({
                item.get("independence_group") for item in sources.values()
                if item.get("independence_group")
            }),
            "unregistered_corpus_record_ids": registry.get("unregistered_record_ids", []),
            "attestation_is_not_transcription_verification": True,
            "registry_sha256": registry_hash,
        },
        "language_representation": {
            "ORTHOGRAPHY": frequency_inventory,
            "PHONOLOGICAL_PHONOTACTIC_OBSERVATIONS": frequency_inventory["phonological_phonotactic"],
            "MORPHOLOGY": {
                "candidate_claims": [
                    row for row in derived_claims
                    if row["domain"] == "MORPHOLOGY"
                ],
                "observed_inventory": "Surface-form morphology is not segmented in this phase.",
                "status": "CANDIDATE_UNVALIDATED",
            },
            "LEXICON": {
                "observed_surface_forms": frequency_inventory["most_frequent_lexical_forms"],
                "total_casefolded_types": frequency_inventory["text_counts"]["regex_defined_casefolded_types"],
                "lemma_or_meaning_assignments": 0,
                "lexical_categories_asserted": 0,
            },
            "SYNTACTIC_STRUCTURE": {
                "candidate_claims": [
                    row for row in derived_claims
                    if row["domain"] == "SYNTACTIC_STRUCTURE"
                ],
                "sentence_structure_supported": False,
                "sentence_boundary_detection": "punctuation-delimited segments only; no syntactic segmentation asserted",
            },
            "AGREEMENT_CONCORD": {
                "candidate_claims": [
                    row for row in derived_claims
                    if row["domain"] == "AGREEMENT_CONCORD"
                ],
            },
            "SEMANTIC_FUNCTIONAL": {
                "candidate_claims": [
                    row for row in derived_claims
                    if row["domain"] == "SEMANTIC_FUNCTIONAL"
                ],
                "contextual_senses_validated": 0,
            },
            "DISCOURSE_CONSTRUCTION": {
                "candidate_annotations": 0,
                "pending_source_annotation_tasks": 5719,
                "status": "UNKNOWN",
            },
            "UNKNOWN_UNRESOLVED": {
                "items": [
                    "No verified phonemic or syllabic analysis.",
                    "No row-level transcription fidelity review trail.",
                    "No systematic counterexample search for existing interpretations.",
                    "No validated semantic, discourse, or complete sentence-structure analysis.",
                    "Three target-language corpus records differ from preserved original punctuation and remain excluded pending review.",
                ],
                "external_knowledge_required": [
                    "Sound values and phoneme-to-grapheme correspondence.",
                    "Any grammatical interpretation requiring facts not encoded in the corpus evidence.",
                ],
            },
        },
        "candidate_structures": derived_claims,
        "source_linked_observations": observations,
        "counterexamples_and_conflicts": {
            "systematic_search_performed": False,
            "open_conflicts": usable_conflicts,
            "existing_claims_with_empty_counterexample_lists": sum(
                not row.get("source_counterexamples") for row in derived_claims
            ),
            "interpretation": (
                "Existing empty counterexample lists are not negative evidence. Conflicts "
                "are retained; no claim is established or silently reconciled."
            ),
        },
        "input_artifact_hashes": {
            corpus_path.relative_to(root).as_posix(): corpus_hash,
            registry_path.relative_to(root).as_posix(): registry_hash,
            claims_path.relative_to(root).as_posix(): claims_hash,
            conflicts_path.relative_to(root).as_posix(): _artifact_hash(conflicts_path),
            **observation_hashes,
        },
        "summary": {
            "canonical_corpus_records": len(corpus),
            "language_class_counts": dict(sorted(included_counts.items())),
            "derivation_evidence_records": len(evidence_ids),
            "derived_claim_count": len(derived_claims),
            "derived_claims_by_knowledge_level": dict(sorted(knowledge_counts.items())),
            "candidate_claims_by_domain": dict(sorted(candidate_counts.items())),
            "source_linked_observation_count": len(observations),
            "linguistic_generalizations_promoted_to_supported": 0,
            "claims_validated_or_established": 0,
        },
        "verification": verification or {
            "test_results_recorded": False,
            "note": "The derivation generator does not execute tests.",
        },
    }

    output_dir = output_dir.resolve()
    report_path = report_path.resolve()
    spec_path = output_dir / "language_specification.json"
    _write_json(spec_path, spec)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    spec_reference = (
        spec_path.relative_to(root).as_posix()
        if spec_path.is_relative_to(root)
        else spec_path.as_posix()
    )
    report_path.write_text(_render_report(spec, spec_reference), encoding="utf-8")
    return spec


def _render_report(spec: dict[str, Any], spec_relative_path: str) -> str:
    evidence = spec["DERIVATION_EVIDENCE_SET"]
    inventory = spec["language_representation"]["ORTHOGRAPHY"]
    summary = spec["summary"]
    top_forms = inventory["most_frequent_lexical_forms"][:20]
    letters = inventory["alphabetic_codepoints"]
    punctuation = inventory["punctuation"]
    groups = evidence["records_by_independence_group"]
    source_lines = [
        f"| {item['source_title']} | {item['records_in_corpus']} | "
        f"{item['records_in_derivation_set']} | {item['independence_group']} | "
        f"{'yes' if item['foundation_attested'] else 'no'} |"
        for item in evidence["source_work_catalog"]
    ]
    letter_display = ", ".join(
        f"`{row['symbol']}` ({row['occurrences']})" for row in letters
    )
    punctuation_display = ", ".join(
        f"`{row['symbol']}` ({row['occurrences']})" for row in punctuation[:25]
    )
    form_display = "\n".join(
        f"| `{row['surface_form_casefolded']}` | {row['occurrences']} | "
        f"{row['record_count']} | {row['source_work_count']} |"
        for row in top_forms
    )
    ngram_display: dict[str, str] = {}
    for size, rows in inventory["orthographic_n_grams"].items():
        ngram_display[size] = ", ".join(
            f"`{row['sequence']}` ({row['occurrences']})" for row in rows[:20]
        )
    domains = collections.Counter(
        row["domain"] for row in spec["candidate_structures"]
        if row["knowledge_level"] == "CANDIDATE"
    )
    conflicts = spec["counterexamples_and_conflicts"]["open_conflicts"]
    conflict_lines = [
        f"- `{row['conflict_id']}` ({row['conflict_type']}): "
        f"{row.get('surface_form')}; competing labels: "
        f"{', '.join(row.get('conflicting_values', []))}; "
        f"{len(row.get('record_ids', []))} in-boundary records across "
        f"{len(row.get('independence_groups', []))} groups. Status remains {row['status']}."
        for row in conflicts
    ] or ["- No source-linked open conflict records in the current artifact."]
    return "\n".join([
        "# XNLP Language Derivation Report",
        "",
        f"- Specification: `{spec_relative_path}` (version {spec['specification_version']}).",
        "- Status: exploratory, reversible, and not an executable grammar.",
        "",
        "## A. Exact DERIVATION_EVIDENCE_SET",
        "",
        f"**ID:** `{evidence['id']}`; **records:** {evidence['record_count']}; "
        f"**independence groups represented:** {evidence['independence_group_count']}.",
        "",
        evidence["exact_selection_rule"],
        "",
        "This set identifies source-attested, target-language rows whose stored text "
        "equals `original_text`. It does **not** mean the transcription was verified "
        "against an edition: there are 0 records with explicit correction/transcription "
        "review metadata. Three otherwise target-language rows differ from `original_text` "
        "by punctuation and are held out pending review. The corpus was not modified.",
        "",
        "### Source works and coverage",
        "",
        "| Source title | Canonical records | Included records | Independence group | Attested |",
        "|---|---:|---:|---|---|",
        *source_lines,
        "",
        "Included records by independence group:",
        "",
        *[f"- {group}: {count}" for group, count in sorted(groups.items())],
        "",
        f"Canonical corpus exclusions: {json.dumps(evidence['excluded_from_derivation']['canonical_corpus_records_by_reason'], sort_keys=True)}. "
        f"An additional {evidence['excluded_from_derivation']['generated_candidate_file']['record_count']} generated candidate records are excluded.",
        "",
        "### Provenance and correction boundary",
        "",
        f"- Registry source-title entries: {spec['provenance_map']['registry_source_count']}; "
        f"independence groups: {spec['provenance_map']['independence_group_count_in_registry']}.",
        f"- Unregistered canonical corpus records: {len(spec['provenance_map']['unregistered_corpus_record_ids'])}.",
        "- Owner/source attestation is not transcription-fidelity verification.",
        "- Derivations describe stored text only; ambiguity or OCR/transcription-looking forms are not repaired.",
        "",
        "## B. Observed language inventory",
        "",
        "These are observations about the evidence text. They do not by themselves establish linguistic categories.",
        "",
        f"- Alphabetic Unicode code points observed: {inventory['text_counts']['distinct_case_sensitive_alphabetic_codepoints']} "
        f"case-sensitive symbols / {inventory['text_counts']['distinct_casefolded_alphabetic_characters']} casefolded symbols.",
        f"- Symbol occurrence inventory: {letter_display}.",
        f"- Regex-defined tokens: {inventory['text_counts']['regex_defined_token_occurrences']} occurrences / "
        f"{inventory['text_counts']['regex_defined_casefolded_types']} casefolded types. Regex is a measurement convention, not proven wordhood.",
        f"- Whitespace-delimited units: {inventory['text_counts']['whitespace_delimited_units']}.",
        f"- Punctuation: {punctuation_display}.",
        f"- Within-token letter bigrams: {ngram_display['2']}.",
        f"- Within-token letter trigrams: {ngram_display['3']}.",
        f"- Case-variant token types: {inventory['capitalization_and_marks']['case_variant_type_count']}; "
        f"types represented in at least two independent groups: {inventory['capitalization_and_marks']['types_with_multiple_independent_work_groups']}.",
        "- These n-grams are `OBSERVED_PATTERN`, not syllables, phonemes, morphemes, or phonotactic rules.",
        "",
        "### Frequent surface forms (not lemmas, POS, or meanings)",
        "",
        "| Casefolded form | Occurrences | Records | Independent groups |",
        "|---|---:|---:|---:|",
        form_display,
        "",
        "## C. Supported structures",
        "",
        "No linguistic generalization is promoted to **SUPPORTED** in this phase. "
        "The directly supported facts are corpus/statistical facts in the specification "
        "(record inclusion, character counts, tokenization convention, and observed "
        "source-linked strings); they are not grammar rules.",
        "",
        "## D. Candidate structures",
        "",
        f"There are {summary['derived_claim_count']} existing source-linked claim records, "
        f"including {summary['candidate_claims_by_domain'].get('MORPHOLOGY', 0)} morphology, "
        f"{summary['candidate_claims_by_domain'].get('SYNTACTIC_STRUCTURE', 0)} syntax/modifier/negation, "
        f"{summary['candidate_claims_by_domain'].get('AGREEMENT_CONCORD', 0)} agreement/concord, "
        f"and {summary['candidate_claims_by_domain'].get('SEMANTIC_FUNCTIONAL', 0)} functional candidates. "
        "Existing `SOURCE_SUPPORTED` legacy labels are retained as provenance metadata but "
        "remain candidates for language derivation; they are not human-validated conclusions.",
        "",
        f"The machine specification preserves all {len(spec['source_linked_observations'])} "
        "source-linked observation records with evidence IDs, domains, raw patterns, and "
        "the distinction between observed count and candidate interpretation.",
        "",
        "No morpheme, stem, lexical category, agreement rule, word order, or semantic meaning "
        "is asserted by this report. The 29 prior auto-evidence lexical category decisions "
        "remain queue decisions only and are not treated as linguistic facts.",
        "",
        "## E. Unknown / insufficiently evidenced",
        "",
        "- Phonemes, syllables, phonotactic constraints, and sound-to-spelling rules: "
        "`EXTERNAL_KNOWLEDGE_REQUIRED` or direct phonetic evidence required.",
        "- Morpheme boundaries, roots, affixes, inflection, derivation, and category assignments: candidates only.",
        "- Sentence segmentation beyond punctuation-delimited text; syntactic constituent structure; "
        "agreement interpretation; negation/interrogation/subordination function.",
        "- Word sense, predicate semantics, discourse references, and pragmatic/focus interpretations.",
        "- Transcription fidelity: unverified for all rows; stored `text == original_text` only establishes internal identity.",
        "",
        "## F. Counterexamples and conflicts",
        "",
        "No systematic positive-vs-negative search was performed. Existing claim artifacts "
        "mostly contain empty counterexample lists; those are not evidence of no exceptions.",
        *conflict_lines,
        "",
        "Competing explanations and unresolved records must remain open; the full claims, "
        "record IDs, source titles, and work-group counts are in the specification.",
        "",
        "## G. Proposed language representation",
        "",
        "The versioned machine-readable representation separates ORTHOGRAPHY, "
        "PHONOLOGICAL/PHONOTACTIC observations, MORPHOLOGY, LEXICON, SYNTACTIC_STRUCTURE, "
        "AGREEMENT_CONCORD, SEMANTIC_FUNCTIONAL, DISCOURSE_CONSTRUCTION, and "
        "UNKNOWN_UNRESOLVED. Every candidate retains record references and provenance. "
        "The representation is data, not executable grammar.",
        "",
        "## H. Provenance map",
        "",
        "The source registry and independence groups are embedded in `DERIVATION_EVIDENCE_SET`; "
        "each lexical count and claim/observation stores source record references. Mqhayi "
        "compilation records use the shared registry independence group unless an explicit "
        "internal-heading attribution is present. No generated candidate is accepted as attested.",
        "",
        "## I. Test results",
        "",
        *[
            f"- {name}: {value}"
            for name, value in spec["verification"].items()
            if name not in {"test_results_recorded"}
        ],
        "",
        "## J. Gaps requiring more evidence",
        "",
        "- Record-level transcription review, with corrected text/version provenance rather than silent edits.",
        "- Human-annotated lexical, morphological, syntactic, semantic, and discourse examples.",
        "- Independent counterexample and competing-analysis review across genuinely independent works.",
        "- Sound/phonetic evidence if phonological statements are required.",
        "- Resolved review of the three punctuation-different text/original pairs and the open prefix ambiguity.",
        "",
        "## K. Proposed sequence: LANGUAGE DERIVATION → GRAMMAR DERIVATION",
        "",
        "1. Freeze the evidence-set manifest and record-level correction trail.",
        "2. Add adjudicated, source-linked examples plus explicit negative/counterexample cases.",
        "3. Derive candidate relationships and constructions with independent-work replication.",
        "4. Have qualified reviewers assess competing analyses and unknowns.",
        "5. Promote only reviewed generalizations to SUPPORTED in a versioned specification.",
        "6. Implement grammar only after those derivations and tests are approved; this phase implements none.",
        "",
        "## Invariants",
        "",
        "No corpus row is changed; no provenance is broadened; no claim is validated; no machine "
        "inference is described as human review; and all derived outputs can be regenerated or removed.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--test-command", default=None)
    parser.add_argument("--tests-before", default=None)
    parser.add_argument("--tests-after", default=None)
    args = parser.parse_args()
    verification = None
    if any((args.test_command, args.tests_before, args.tests_after)):
        if not all((args.test_command, args.tests_before, args.tests_after)):
            parser.error("--test-command, --tests-before, and --tests-after must be supplied together")
        verification = {
            "test_results_recorded": True,
            "test_command": args.test_command,
            "before_derivation": args.tests_before,
            "after_derivation": args.tests_after,
        }
    spec = derive_language(args.root, args.output_dir, args.report, verification)
    print(json.dumps(spec["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Specification: {args.output_dir / 'language_specification.json'}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
