"""Build a conservative, provenance-linked formal grammar audit."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from xnlp_language.build_foundation import WORD

ROOT = Path(__file__).resolve().parents[1]
GENERATED = Path("xnlp_language") / "generated"
DEFAULT_OUTPUT = ROOT / GENERATED / "grammar_derivation"
DEFAULT_REPORT = DEFAULT_OUTPUT / "grammar_derivation_report.md"
GRAMMAR_DIR = GENERATED / "grammar"
SPECIFICATION_VERSION = "0.1.0"
CLAIM_PATH = GENERATED / "derivation" / "language_specification.json"
DERIVATION_REPORT_PATH = GENERATED / "reports" / "language_derivation_report.md"
PROVENANCE_PATH = GENERATED / "evidence" / "source_provenance.json"
CORPUS_PATH = GENERATED / "evidence" / "corpus_records.jsonl"
BUNDLE_PATH = GENERATED / "foundation_bundle.json"
LEGACY_STRUCTURE_FILES = (
    "grammar_rules.json",
    "noun_classes.jsonl",
    "agreement_system.jsonl",
    "verb_morphology.jsonl",
    "derivational_morphology.jsonl",
)
QUESTION_LAYERS = (
    "ORTHOGRAPHIC_TOKEN_STRUCTURE",
    "MORPHOLOGY",
    "LEXICAL_CATEGORIES",
    "AGREEMENT_CONCORD",
    "NOUN_CLASS_LIKE_SYSTEMS",
    "VERB_STRUCTURE",
    "PHRASE_STRUCTURE",
    "CLAUSE_STRUCTURE",
    "SENTENCE_CONSTRUCTION",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _token_sequence(value: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in WORD.finditer(value))


def _collect_record_ids(value: Any) -> set[str]:
    record_fields = {
        "record_id",
        "source_record_id",
        "source_record_ids",
        "evidence_record_ids",
        "evidence_records",
        "record_ids",
    }
    negative_fields = {"counterexamples", "exceptions", "negative_evidence"}
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in negative_fields:
                continue
            if key in record_fields:
                if isinstance(child, str):
                    found.add(child)
                else:
                    found.update(_collect_record_ids(child))
            else:
                found.update(_collect_record_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_collect_record_ids(child))
    return found


def _eligible(record: dict[str, Any], source_registry: dict[str, Any]) -> bool:
    source = source_registry.get(str(record.get("source_title", "")), {})
    return (
        source.get("foundation_attested") is True
        and record.get("provenance_class") == "attested"
        and record.get("provenance_type") == "authoritative"
        and not record.get("generator")
        and record.get("validation_status") == "authoritative"
        and record.get("language_class") == "TARGET_LANGUAGE"
        and record.get("text") == record.get("original_text")
    )


def _load_inputs(root: Path) -> dict[str, Any]:
    derivation_path = root / CLAIM_PATH
    derivation = _read_json(derivation_path)
    evidence_set = derivation["DERIVATION_EVIDENCE_SET"]
    expected_ids = set(evidence_set["record_ids"])
    corpus_records = _read_jsonl(root / CORPUS_PATH)
    records_by_id = {str(record["record_id"]): record for record in corpus_records}
    if len(records_by_id) != len(corpus_records):
        raise ValueError("Canonical corpus contains duplicate record IDs")
    provenance = _read_json(root / PROVENANCE_PATH)
    sources = provenance.get("sources", {})
    attested_sources = {
        title: item for title, item in sources.items()
        if item.get("foundation_attested") is True
    }
    corpus_evidence = {
        record_id: records_by_id[record_id]
        for record_id in expected_ids
        if record_id in records_by_id
    }
    missing_ids = sorted(expected_ids - set(records_by_id))
    if missing_ids:
        raise ValueError(f"Derivation evidence IDs missing from corpus: {missing_ids[:5]}")
    invalid_ids = sorted(
        record_id for record_id, record in corpus_evidence.items()
        if not _eligible(record, attested_sources)
    )
    if invalid_ids:
        raise ValueError(
            "Frozen derivation evidence contains records that no longer meet "
            f"the attested evidence boundary: {invalid_ids[:5]}"
        )

    bundle = _read_json(root / BUNDLE_PATH)
    if bundle.get("bundle_id") != "XNLP_ISIXHOSA_EVIDENCE_FOUNDATION_BUNDLE_V1":
        raise ValueError("Unexpected foundation bundle identity")
    bundle_derivation = bundle.get("language_derivation") or {}
    if bundle_derivation.get("evidence_set_id") != evidence_set.get("id"):
        raise ValueError("Foundation bundle and derivation evidence set disagree")

    raw_observations: dict[str, dict[str, Any]] = {}
    observation_files = sorted((root / GRAMMAR_DIR).glob("*_observations.jsonl"))
    for path in observation_files:
        for observation in _read_jsonl(path):
            observation_id = str(observation.get("observation_id", ""))
            if observation_id in raw_observations:
                raise ValueError(f"Duplicate grammar observation ID: {observation_id}")
            raw_observations[observation_id] = {
                **observation,
                "_artifact": path.relative_to(root).as_posix(),
                "_artifact_sha256": _sha256(path),
            }
    conflicts = _read_jsonl(root / GRAMMAR_DIR / "linguistic_conflicts.jsonl")

    legacy_structures: list[dict[str, Any]] = []
    for filename in LEGACY_STRUCTURE_FILES:
        path = root / GRAMMAR_DIR / filename
        content = _read_json(path) if path.suffix == ".json" else _read_jsonl(path)
        if not isinstance(content, list):
            raise ValueError(f"Expected a list of grammar structures in {path}")
        for index, structure in enumerate(content):
            legacy_structures.append({
                "source_artifact": path.relative_to(root).as_posix(),
                "source_artifact_sha256": _sha256(path),
                "source_index": index,
                "structure": structure,
            })

    input_hashes = dict(derivation.get("input_artifact_hashes", {}))
    input_hashes[CLAIM_PATH.as_posix()] = _sha256(derivation_path)
    input_hashes[DERIVATION_REPORT_PATH.as_posix()] = _sha256(
        root / DERIVATION_REPORT_PATH
    )
    for relative in (
        PROVENANCE_PATH,
        CORPUS_PATH,
        *(GRAMMAR_DIR / filename for filename in LEGACY_STRUCTURE_FILES),
        *(path.relative_to(root) for path in observation_files),
    ):
        input_hashes[relative.as_posix()] = _sha256(root / relative)
    stale_hashes = [
        name for name, expected_hash in derivation.get("input_artifact_hashes", {}).items()
        if not (root / name).is_file() or _sha256(root / name) != expected_hash
    ]
    if stale_hashes:
        raise ValueError(
            "Language derivation input hashes do not match current evidence: "
            + ", ".join(stale_hashes[:8])
        )

    return {
        "derivation": derivation,
        "evidence_set": evidence_set,
        "corpus_records": corpus_records,
        "records_by_id": records_by_id,
        "corpus_evidence": corpus_evidence,
        "provenance": provenance,
        "sources": sources,
        "attested_sources": attested_sources,
        "raw_observations": raw_observations,
        "conflicts": conflicts,
        "legacy_structures": legacy_structures,
        "input_hashes": dict(sorted(input_hashes.items())),
        "bundle": bundle,
    }


def _record_metadata(
    record_id: str,
    records_by_id: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    evidence_ids: set[str],
) -> dict[str, Any]:
    record = records_by_id.get(record_id)
    if record is None:
        return {
            "record_id": record_id,
            "evidence_status": "RECORD_NOT_FOUND",
            "source_title": None,
            "source_author": "NOT_RECORDED",
            "source_work": None,
            "source_section": None,
        }
    source_title = str(record.get("source_title", "UNKNOWN"))
    source = sources.get(source_title, {})
    attribution = record.get("work_attribution")
    source_work = (
        attribution.get("work")
        if isinstance(attribution, dict)
        else source.get("independence_group") or source_title
    )
    return {
        "record_id": record_id,
        "evidence_status": (
            "IN_DERIVATION_EVIDENCE_SET"
            if record_id in evidence_ids else "OUTSIDE_DERIVATION_EVIDENCE_SET"
        ),
        "source_title": source_title,
        "source_author": record.get("source_author", "NOT_RECORDED"),
        "source_work": source_work,
        "source_section": record.get("source_section"),
        "independence_group": (
            source_work or source.get("independence_group") or source_title
        ),
        "provenance_class": record.get("provenance_class"),
        "provenance_type": record.get("provenance_type"),
        "foundation_attested": source.get("foundation_attested") is True,
    }


def _candidate_observation_ids(
    candidate: dict[str, Any],
    raw_observations: dict[str, dict[str, Any]],
) -> list[str]:
    recorded_ids = candidate.get("observation_ids", [])
    selected = {
        str(observation_id)
        for observation_id in recorded_ids
        if str(observation_id) in raw_observations
    }
    claim_id = str(candidate.get("claim_id", ""))
    selected.update(
        observation_id for observation_id in raw_observations
        if claim_id.endswith(observation_id)
    )
    source_record_ids = set(candidate.get("source_record_ids", []))
    source_artifact = str(candidate.get("source_claim_artifact", ""))
    overlaps: list[tuple[int, str]] = []
    for observation_id, observation in raw_observations.items():
        if observation["_artifact"] != source_artifact:
            continue
        overlap = source_record_ids.intersection(
            item.get("record_id")
            for item in observation.get("evidence", [])
            if isinstance(item, dict)
        )
        if overlap:
            overlaps.append((len(overlap), observation_id))
    if overlaps:
        maximum = max(count for count, _ in overlaps)
        selected.update(
            observation_id for count, observation_id in overlaps
            if count == maximum
        )
    return sorted(selected)


def _surface_search_index(
    records: dict[str, dict[str, Any]],
    max_form_tokens: int = 4,
) -> dict[tuple[str, ...], dict[str, Any]]:
    index: dict[tuple[str, ...], dict[str, Any]] = {}
    for record_id, record in records.items():
        tokens = _token_sequence(str(record.get("text", "")))
        for size in range(1, min(max_form_tokens, len(tokens)) + 1):
            for start in range(len(tokens) - size + 1):
                form = tokens[start:start + size]
                entry = index.setdefault(
                    form,
                    {"occurrences": 0, "record_ids": set()},
                )
                entry["occurrences"] += 1
                entry["record_ids"].add(record_id)
    return index


def _observation_cues(
    observation_ids: list[str],
    raw_observations: dict[str, dict[str, Any]],
    evidence_ids: set[str],
) -> list[dict[str, Any]]:
    cues: dict[tuple[str, str], dict[str, Any]] = {}
    for observation_id in observation_ids:
        observation = raw_observations[observation_id]
        for evidence in observation.get("evidence", []):
            if not isinstance(evidence, dict):
                continue
            record_id = str(evidence.get("record_id", ""))
            form = evidence.get("surface_form")
            if record_id not in evidence_ids or not isinstance(form, str) or not form:
                continue
            key = (observation_id, form)
            entry = cues.setdefault(key, {
                "observation_id": observation_id,
                "surface_form": form,
                "supporting_record_ids": set(),
                "character_span_checks": [],
            })
            entry["supporting_record_ids"].add(record_id)
            entry["character_span_checks"].append({
                "record_id": record_id,
                "span_matches_surface": (
                    isinstance(evidence.get("character_span"), list)
                    and len(evidence["character_span"]) == 2
                    and evidence["character_span"][0] >= 0
                    and evidence["character_span"][1] <= len(evidence.get("context", ""))
                    and evidence["context"][
                        evidence["character_span"][0]:evidence["character_span"][1]
                    ].casefold() == form.casefold()
                ),
            })
    return [
        {
            **cue,
            "supporting_record_ids": sorted(cue["supporting_record_ids"]),
            "character_span_checks": sorted(
                cue["character_span_checks"],
                key=lambda item: (item["record_id"], item["span_matches_surface"]),
            ),
        }
        for _, cue in sorted(cues.items())
    ]


def _independence(
    in_boundary_ids: list[str],
    records_by_id: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_records = [
        records_by_id[record_id]
        for record_id in in_boundary_ids
        if record_id in records_by_id
    ]
    titles = sorted({str(record.get("source_title", "UNKNOWN")) for record in source_records})
    groups = set()
    sections = set()
    for record in source_records:
        title = str(record.get("source_title", "UNKNOWN"))
        attribution = record.get("work_attribution")
        source = sources.get(title, {})
        groups.add(
            attribution.get("work")
            if isinstance(attribution, dict)
            else source.get("independence_group") or title
        )
        if record.get("source_section"):
            sections.add((title, str(record["source_section"])))
    if len(source_records) <= 1:
        strongest = "SAME_RECORD"
    elif len(groups) <= 1:
        strongest = "SAME_WORK"
    else:
        strongest = "DIFFERENT_WORK"
    if source_records and all(not record.get("source_author") for record in source_records):
        author_assessment = "AUTHOR_METADATA_NOT_RECORDED"
    else:
        author_assessment = "AUTHORS_NOT_INDEPENDENTLY_VERIFIED"
    return {
        "cited_record_count": len(set(in_boundary_ids)),
        "source_titles": titles,
        "independence_groups": sorted(groups),
        "independence_group_count": len(groups),
        "source_section_count": len(sections),
        "strongest_observed_level": strongest,
        "same_record": len(set(in_boundary_ids)) == 1,
        "same_section": len(sections) == 1,
        "same_work": len(groups) == 1,
        "different_work": len(groups) > 1,
        "same_author": "UNKNOWN",
        "different_author": "UNKNOWN",
        "author_attribution_status": author_assessment,
    }


def _make_candidate(
    *,
    candidate_id: str,
    source_kind: str,
    source_artifact: str,
    source_artifact_sha256: str,
    raw_status: Any,
    assertion: str,
    domain: str,
    cited_ids: list[str],
    observation_ids: list[str],
    observations: dict[str, dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    evidence_ids: set[str],
    search_index: dict[tuple[str, ...], dict[str, Any]],
    existing_counterexamples: list[Any],
    confidence_as_recorded: Any,
    raw_data: dict[str, Any],
) -> dict[str, Any]:
    all_cited_ids = sorted({str(record_id) for record_id in cited_ids if record_id})
    for observation_id in observation_ids:
        observation = observations.get(observation_id)
        if observation is not None:
            all_cited_ids = sorted(
                set(all_cited_ids).union(_collect_record_ids(observation))
            )
    in_boundary_ids = sorted(set(all_cited_ids).intersection(evidence_ids))
    excluded_ids = sorted(set(all_cited_ids) - evidence_ids)
    if domain in {"DATA_GOVERNANCE", "UNKNOWN_UNRESOLVED"}:
        disposition = "MOVE_TO_UNKNOWN"
        status = "UNKNOWN"
        rationale = (
            "The available evidence does not justify a linguistic analysis; "
            "the item remains unknown rather than being promoted or rejected."
        )
    else:
        disposition = "RETAIN_AS_CANDIDATE"
        status = "CANDIDATE"
        rationale = (
            "Promotion criteria are not met: source observation interpretations "
            "are not independently adjudicated, and grammatical counterexample "
            "labels are unavailable."
        )
    related_observations = [
        observation_id
        for observation_id in observation_ids
        if observation_id in observations
    ]
    surface_cues = _observation_cues(
        related_observations,
        observations,
        evidence_ids,
    )
    cue_searches = []
    for cue in surface_cues:
        token_form = _token_sequence(cue["surface_form"])
        if not token_form:
            result = {
                "observation_id": cue["observation_id"],
                "surface_form": cue["surface_form"],
                "search_status": "NO_SEARCHABLE_TOKENS",
                "occurrence_count": 0,
                "record_ids": [],
            }
        elif len(token_form) > 4:
            result = {
                "observation_id": cue["observation_id"],
                "surface_form": cue["surface_form"],
                "search_status": "FORM_EXCEEDS_SEARCH_WINDOW",
                "occurrence_count": None,
                "record_ids": [],
            }
        else:
            found = search_index.get(token_form, {"occurrences": 0, "record_ids": set()})
            search_id = "FORM_" + hashlib.sha256(
                " ".join(token_form).encode("utf-8")
            ).hexdigest()[:16]
            result = {
                "search_id": search_id,
                "observation_id": cue["observation_id"],
                "surface_form": cue["surface_form"],
                "search_status": "FULL_DERIVATION_SET_EXACT_TOKEN_SEQUENCE_SCAN",
                "occurrence_count": found["occurrences"],
                "record_count": len(found["record_ids"]),
                "matching_record_ids_sha256": hashlib.sha256(
                    "\n".join(sorted(found["record_ids"])).encode("utf-8")
                ).hexdigest(),
                "candidate_support_record_ids": cue["supporting_record_ids"],
            }
        cue_searches.append(result)
    candidate_evidence = [
        _record_metadata(record_id, records_by_id, sources, evidence_ids)
        for record_id in all_cited_ids
    ]
    counterexamples = [
        str(record_id)
        for record_id in existing_counterexamples
        if str(record_id) in evidence_ids
    ]
    if not surface_cues:
        search_status = "NO_LINKED_ATTESTED_SURFACE_CUE"
    elif any(item["search_status"] == "FORM_EXCEEDS_SEARCH_WINDOW" for item in cue_searches):
        search_status = "PARTIAL_FULL_CORPUS_SURFACE_SCAN"
    else:
        search_status = "FULL_CORPUS_SURFACE_SCAN_NO_GRAMMATICAL_COUNTEREXAMPLE_LABELS"
    return {
        "rule_id": candidate_id,
        "rule_type": domain,
        "source_kind": source_kind,
        "status": status,
        "disposition": disposition,
        "disposition_rationale": rationale,
        "source_status_as_recorded": raw_status,
        "input_structure": None,
        "proposed_relationship": assertion,
        "context": "NOT_SPECIFIED_BY_ADJUDICATED_EVIDENCE",
        "confidence": {
            "assessment": "NOT_ASSESSED",
            "legacy_label": confidence_as_recorded,
        },
        "external_knowledge_dependency": "POSSIBLE_UNRESOLVED_NOT_IMPORTED",
        "source_artifact": source_artifact,
        "source_artifact_sha256": source_artifact_sha256,
        "source_record_ids_as_recorded": all_cited_ids,
        "evidence_record_ids": in_boundary_ids,
        "out_of_boundary_record_ids": excluded_ids,
        "observation_ids": related_observations,
        "independence": _independence(in_boundary_ids, records_by_id, sources),
        "counterexamples": {
            "source_listed_counterexamples_as_recorded": existing_counterexamples,
            "source_listed_record_ids_in_boundary": counterexamples,
            "surface_cue_searches": cue_searches,
            "search_status": search_status,
            "grammatical_counterexample_status": (
                "REQUIRES_HUMAN_ADJUDICATION"
            ),
            "limitation": (
                "Exact surface-form retrieval across the complete frozen evidence "
                "set is not a grammatical counterexample search. No corpus-wide "
                "positive/negative grammatical annotation is available."
            ),
        },
        "competing_analyses": [],
        "provenance": {
            "derivation_evidence_set_id": "DERIVATION_EVIDENCE_SET_V1",
            "source_title_ids": sorted({
                item["source_title"] for item in candidate_evidence
                if item["source_title"]
            }),
            "evidence_record_ids": in_boundary_ids,
        },
        "unresolved_questions": [
            "Does the observed surface pattern have the proposed grammatical function?",
            "What corpus-wide counterexamples and competing analyses occur?",
        ],
        "source_data_as_recorded": raw_data,
    }


def _grammar_inventory(inputs: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence_ids = set(inputs["evidence_set"]["record_ids"])
    raw_observations = inputs["raw_observations"]
    records_by_id = inputs["records_by_id"]
    sources = inputs["sources"]
    search_index = _surface_search_index(inputs["corpus_evidence"])
    rules: list[dict[str, Any]] = []
    claim_observations = inputs["derivation"]["source_linked_observations"]
    for candidate in inputs["derivation"]["candidate_structures"]:
        observation_ids = _candidate_observation_ids(candidate, raw_observations)
        claim_artifact = str(candidate.get("source_claim_artifact", ""))
        matching_observation_refs = [
            observation for observation in claim_observations
            if observation.get("source_artifact") == claim_artifact
            and observation.get("observation_id") in observation_ids
        ]
        rules.append(_make_candidate(
            candidate_id=str(candidate["derived_id"]),
            source_kind="DERIVED_LANGUAGE_CANDIDATE",
            source_artifact=claim_artifact,
            source_artifact_sha256=str(candidate.get("source_claim_artifact_sha256", "")),
            raw_status=candidate.get("knowledge_level"),
            assertion=str(
                candidate.get("proposed_relationship")
                or candidate.get("claim_text")
                or candidate.get("claim_id")
            ),
            domain=str(candidate.get("domain", "UNKNOWN_UNRESOLVED")),
            cited_ids=list(candidate.get("source_record_ids", [])),
            observation_ids=observation_ids,
            observations=raw_observations,
            records_by_id=records_by_id,
            sources=sources,
            evidence_ids=evidence_ids,
            search_index=search_index,
            existing_counterexamples=list(candidate.get("source_counterexamples", [])),
            confidence_as_recorded=candidate.get("confidence"),
            raw_data={
                "claim_id": candidate.get("claim_id"),
                "source_claim_artifact": claim_artifact,
                "source_work_references": candidate.get("source_work_references", []),
                "interpretation": candidate.get("observed_or_proposed_relationship"),
                "observation_links_by_record_overlap": [
                    item["observation_id"] for item in matching_observation_refs
                ],
            },
        ))

    for item in inputs["legacy_structures"]:
        structure = item["structure"]
        source_artifact = item["source_artifact"]
        domain = str(structure.get("category", structure.get("domain", "UNKNOWN")))
        if (
            source_artifact.endswith("grammar_rules.json")
            and domain.casefold() == "orthography"
            and "data-governance" in str(structure.get("notes", "")).casefold()
        ):
            domain = "DATA_GOVERNANCE"
        entry_id = str(
            structure.get("entry_id")
            or structure.get("rule_id")
            or f"{Path(source_artifact).stem}_{item['source_index']}"
        )
        cited_ids = structure.get(
            "source_record_ids",
            structure.get(
                "evidence_records",
                structure.get("record_ids", []),
            ),
        )
        cited_ids = sorted(
            set(
                str(record_id)
                for record_id in cited_ids
                if isinstance(record_id, str)
            ).union(_collect_record_ids(structure))
        ) if isinstance(cited_ids, list) else sorted(_collect_record_ids(structure))
        observations = structure.get("observation_ids", [])
        if not observations:
            observations = [
                observation_id for observation_id in raw_observations
                if entry_id.endswith(observation_id)
            ]
        assertion = str(
            structure.get("claim")
            or structure.get("rule")
            or structure.get("analysis")
            or structure.get("phenomenon")
            or entry_id
        )
        rules.append(_make_candidate(
            candidate_id=f"LEGACY_{entry_id}",
            source_kind="LEGACY_STRUCTURED_GRAMMAR_ENTRY",
            source_artifact=source_artifact,
            source_artifact_sha256=item["source_artifact_sha256"],
            raw_status=structure.get("review_status", structure.get("status")),
            assertion=assertion,
            domain=domain,
            cited_ids=list(cited_ids) if isinstance(cited_ids, list) else [],
            observation_ids=[
                str(observation_id) for observation_id in observations
                if str(observation_id) in raw_observations
            ],
            observations=raw_observations,
            records_by_id=records_by_id,
            sources=sources,
            evidence_ids=evidence_ids,
            search_index=search_index,
            existing_counterexamples=list(structure.get("counterexamples", []))
            + list(structure.get("exceptions", [])),
            confidence_as_recorded=structure.get("confidence"),
            raw_data={
                "entry_id": entry_id,
                "review_status": structure.get("review_status", structure.get("status")),
                "notes": structure.get("notes"),
                "observation_ids_as_recorded": structure.get("observation_ids", []),
            },
        ))

    for rule in rules:
        if rule["rule_type"] in {"DATA_GOVERNANCE", "UNKNOWN_UNRESOLVED"}:
            continue
        cited_ids = set(rule["evidence_record_ids"])
        for conflict in inputs["conflicts"]:
            conflict_ids = set(conflict.get("record_ids", []))
            overlap = sorted(cited_ids.intersection(conflict_ids))
            if not overlap:
                continue
            rule["competing_analyses"].append({
                "conflict_id": conflict.get("conflict_id"),
                "conflict_type": conflict.get("conflict_type"),
                "conflicting_values": conflict.get("conflicting_values", []),
                "status": conflict.get("status"),
                "evidence_record_ids": overlap,
                "interpretation": (
                    "Overlapping source records belong to an open conflict; this "
                    "does not assign any competing analysis to the rule."
                ),
            })

    observation_inventory = []
    for observation in inputs["derivation"]["source_linked_observations"]:
        source_ids = sorted(set(observation.get("source_record_ids", [])))
        in_boundary = sorted(set(source_ids).intersection(evidence_ids))
        observation_inventory.append({
            "observation_id": observation["observation_id"],
            "pattern_type": observation.get("pattern_type"),
            "domain": observation.get("domain"),
            "observed_pattern_identifier": observation.get("observed_pattern_identifier"),
            "status": "OBSERVATION",
            "interpretation_status_as_recorded": observation.get(
                "linguistic_interpretation_level"
            ),
            "source_artifact": observation.get("source_artifact"),
            "source_artifact_sha256": observation.get("source_artifact_sha256"),
            "source_record_ids_as_recorded": source_ids,
            "evidence_record_ids": in_boundary,
            "out_of_boundary_record_ids": sorted(set(source_ids) - evidence_ids),
            "independence": _independence(in_boundary, records_by_id, sources),
            "interpretation_warning": observation.get("interpretation_warning"),
            "counterexample_search_status": "NOT_SYSTEMATICALLY_SEARCHED",
        })
    return rules, observation_inventory


def _candidate_disposition_report(rules: list[dict[str, Any]]) -> dict[str, Any]:
    counts = collections.Counter(rule["disposition"] for rule in rules)
    for disposition in (
        "PROMOTE_TO_SUPPORTED",
        "RETAIN_AS_CANDIDATE",
        "MOVE_TO_UNKNOWN",
        "REJECT",
    ):
        counts.setdefault(disposition, 0)
    by_kind: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for rule in rules:
        by_kind[rule["source_kind"]][rule["disposition"]] += 1
    return {
        "report_id": "XNLP_GRAMMAR_CANDIDATE_DISPOSITIONS_V1",
        "total_items": len(rules),
        "counts": dict(sorted(counts.items())),
        "counts_by_source_kind": {
            kind: dict(sorted(values.items()))
            for kind, values in sorted(by_kind.items())
        },
        "dispositions": [
            {
                "rule_id": rule["rule_id"],
                "source_kind": rule["source_kind"],
                "source_artifact": rule["source_artifact"],
                "source_status_as_recorded": rule["source_status_as_recorded"],
                "disposition": rule["disposition"],
                "status": rule["status"],
                "rationale": rule["disposition_rationale"],
                "evidence_record_ids": rule["evidence_record_ids"],
                "out_of_boundary_record_ids": rule["out_of_boundary_record_ids"],
            }
            for rule in rules
        ],
    }


def _counterexample_report(
    rules: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    evidence_record_count: int,
    search_index: dict[tuple[str, ...], dict[str, Any]],
) -> dict[str, Any]:
    unique_searches: dict[str, dict[str, Any]] = {}
    for rule in rules:
        for search in rule["counterexamples"]["surface_cue_searches"]:
            search_id = search.get("search_id")
            if not search_id or search_id in unique_searches:
                continue
            token_form = _token_sequence(search["surface_form"])
            found = search_index.get(token_form, {"occurrences": 0, "record_ids": set()})
            unique_searches[search_id] = {
                "search_id": search_id,
                "surface_form": search["surface_form"],
                "occurrence_count": found["occurrences"],
                "matching_record_count": len(found["record_ids"]),
                "matched_record_ids": sorted(found["record_ids"]),
            }
    return {
        "report_id": "XNLP_GRAMMAR_COUNTEREXAMPLE_SEARCH_V1",
        "evidence_set_id": "DERIVATION_EVIDENCE_SET_V1",
        "full_evidence_set_record_count_scanned": evidence_record_count,
        "search_scope": (
            "All exact token sequences explicitly evidenced in linked grammar "
            "observation examples were searched across every record in the frozen "
            "derivation evidence set. The search is lexical retrieval only."
        ),
        "rule_searches": [
            {
                "rule_id": rule["rule_id"],
                "search_status": rule["counterexamples"]["search_status"],
                "surface_cue_searches": rule["counterexamples"]["surface_cue_searches"],
                "source_listed_counterexamples": rule["counterexamples"][
                    "source_listed_record_ids_in_boundary"
                ],
                "grammatical_counterexample_status": rule["counterexamples"][
                    "grammatical_counterexample_status"
                ],
            }
            for rule in rules
        ],
        "unique_surface_form_searches": [
            unique_searches[search_id] for search_id in sorted(unique_searches)
        ],
        "observation_searches": [
            {
                "observation_id": observation["observation_id"],
                "search_status": "SEE_LINKED_RULE_SEARCHES",
                "evidence_record_ids": observation["evidence_record_ids"],
            }
            for observation in observations
        ],
        "global_result": (
            "No grammatical rule has a complete counterexample search because the "
            "corpus has no exhaustive positive/negative grammatical annotations. "
            "Surface-form scans do not establish a rule or its counterexamples."
        ),
        "required_next_action": "REQUIRES HUMAN ADJUDICATION",
    }


def _unresolved_questions(rules: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = {
        "ORTHOGRAPHIC_TOKEN_STRUCTURE": "Which recurring whitespace or punctuation patterns correspond to language-defined word and boundary units?",
        "MORPHOLOGY": "Which recurring surface sequences are independently validated morphemes, and what distributional/function evidence supports each segmentation?",
        "LEXICAL_CATEGORIES": "Which forms exhibit reproducible distributional behavior supporting lexical-category labels?",
        "AGREEMENT_CONCORD": "Do recurring forms show agreement relationships across independently annotated contexts?",
        "NOUN_CLASS_LIKE_SYSTEMS": "Do the observed prefix and pairing candidates form a supported system, with what exceptions and agreement behavior?",
        "VERB_STRUCTURE": "Which candidate segments, alternations, and clause roles are supported by attested positive and negative examples?",
        "PHRASE_STRUCTURE": "Which recurring category combinations form evidence-supported phrases?",
        "CLAUSE_STRUCTURE": "What constituent orders and argument relationships are present, and in which constructions?",
        "SENTENCE_CONSTRUCTION": "Which complete constructions recur with independently adjudicated form and function?",
    }
    linked = collections.defaultdict(list)
    for rule in rules:
        layer = rule["rule_type"]
        if layer in {"MORPHOLOGY", "noun_class", "agreement"}:
            linked["MORPHOLOGY" if layer == "MORPHOLOGY" else (
                "NOUN_CLASS_LIKE_SYSTEMS" if layer == "noun_class" else "AGREEMENT_CONCORD"
            )].append(rule["rule_id"])
        elif layer in {"AGREEMENT_CONCORD"}:
            linked["AGREEMENT_CONCORD"].append(rule["rule_id"])
        elif layer in {"SYNTACTIC_STRUCTURE", "syntax", "modifier", "negation"}:
            linked["CLAUSE_STRUCTURE"].append(rule["rule_id"])
            linked["PHRASE_STRUCTURE"].append(rule["rule_id"])
        elif layer == "SEMANTIC_FUNCTIONAL":
            linked["LEXICAL_CATEGORIES"].append(rule["rule_id"])
    entries = []
    for layer in QUESTION_LAYERS:
        entries.append({
            "question_id": f"OPEN_{layer}_001",
            "layer": layer,
            "question": prompts[layer],
            "status": "UNKNOWN",
            "required_action": "REQUIRES HUMAN ADJUDICATION",
            "linked_candidate_rule_ids": sorted(set(linked.get(layer, []))),
        })
    entries.append({
        "question_id": "OPEN_CONFLICT_A_PREFIX_001",
        "layer": "MORPHOLOGY_AND_CLAUSE_STRUCTURE",
        "question": "How should the attested surface form a- be analyzed in its conflicting contexts?",
        "status": "CONTRADICTED_OR_UNRESOLVED",
        "required_action": "REQUIRES HUMAN ADJUDICATION",
        "competing_analyses": [
            "subject_concord_class_6",
            "noun_class_prefix_class_6",
            "negation_marker",
        ],
        "linked_candidate_rule_ids": sorted(
            rule["rule_id"]
            for rule in rules
            if any(
                analysis.get("conflict_id") == "CONFLICT_LINGUISTIC_001"
                for analysis in rule.get("competing_analyses", [])
            )
        ),
    })
    entries.append({
        "question_id": "OPEN_TRANSCRIPTION_FIDELITY_001",
        "layer": "PROVENANCE",
        "question": "Do stored records faithfully represent their source editions?",
        "status": "UNKNOWN",
        "required_action": "REQUIRES ADDITIONAL EVIDENCE",
        "linked_candidate_rule_ids": [],
    })
    entries.append({
        "question_id": "OPEN_PHONOLOGICAL_EVIDENCE_001",
        "layer": "PHONOLOGICAL_PHONOTACTIC",
        "question": "What phonological or phonotactic generalizations, if any, follow from direct evidence?",
        "status": "UNKNOWN",
        "required_action": "EXTERNAL_KNOWLEDGE_REQUIRED",
        "evidence_note": (
            "No external linguistic sources or assumptions are imported; written "
            "character sequences alone do not establish sound values."
        ),
        "linked_candidate_rule_ids": [],
    })
    entries.append({
        "question_id": "OPEN_AUTHOR_INDEPENDENCE_001",
        "layer": "PROVENANCE",
        "question": "Are the cited works independent by author as well as by source title?",
        "status": "UNKNOWN",
        "required_action": "REQUIRES ADDITIONAL EVIDENCE",
        "linked_candidate_rule_ids": [],
    })
    entries.append({
        "question_id": "OPEN_COUNTEREXAMPLE_ANNOTATION_001",
        "layer": "ALL_GRAMMAR_LAYERS",
        "question": "Which corpus-wide contexts are positive, negative, exceptional, or competing evidence for each candidate?",
        "status": "UNKNOWN",
        "required_action": "REQUIRES HUMAN ADJUDICATION",
        "linked_candidate_rule_ids": [],
    })
    return {
        "report_id": "XNLP_UNRESOLVED_GRAMMAR_QUESTIONS_V1",
        "questions": entries,
    }


def _coverage_report(
    rules: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    orthography: dict[str, Any],
) -> dict[str, Any]:
    by_domain = collections.Counter(rule["rule_type"] for rule in rules)
    by_status = collections.Counter(rule["status"] for rule in rules)
    layers = []
    domain_map = {
        "ORTHOGRAPHIC_TOKEN_STRUCTURE": {"orthography"},
        "MORPHOLOGY": {"MORPHOLOGY", "morphology", "verb_morphology", "derivational_morphology"},
        "LEXICAL_CATEGORIES": {"SEMANTIC_FUNCTIONAL", "function_word"},
        "AGREEMENT_CONCORD": {"AGREEMENT_CONCORD", "agreement"},
        "NOUN_CLASS_LIKE_SYSTEMS": {"noun_class"},
        "VERB_STRUCTURE": {"MORPHOLOGY", "verb_morphology"},
        "PHRASE_STRUCTURE": {"SYNTACTIC_STRUCTURE", "syntax", "modifier"},
        "CLAUSE_STRUCTURE": {"SYNTACTIC_STRUCTURE", "syntax", "negation"},
        "SENTENCE_CONSTRUCTION": {"SYNTACTIC_STRUCTURE", "syntax"},
    }
    for layer in QUESTION_LAYERS:
        domains = domain_map[layer]
        linked_rules = sum(count for domain, count in by_domain.items() if domain in domains)
        linked_observations = sum(
            1 for observation in observations
            if observation.get("domain") in domains
        )
        if layer == "ORTHOGRAPHIC_TOKEN_STRUCTURE":
            linked_observations = (
                len(orthography.get("alphabetic_codepoints", []))
                + len(orthography.get("punctuation", []))
                + sum(
                    len(orthography.get("orthographic_n_grams", {}).get(size, []))
                    for size in ("2", "3")
                )
                + len(orthography.get("most_frequent_lexical_forms", []))
            )
        layers.append({
            "layer": layer,
            "linked_candidate_item_count": linked_rules,
            "linked_observation_count": linked_observations,
            "supported_grammar_rule_count": 0,
            "coverage_status": (
                "OBSERVATIONS_AND_CANDIDATES_ONLY"
                if linked_rules or linked_observations else "NO_DERIVED_COVERAGE"
            ),
            "limitations": [
                "No corpus-wide grammatical counterexample annotations.",
                "No generalization promoted to SUPPORTED.",
            ],
        })
    return {
        "report_id": "XNLP_GRAMMAR_COVERAGE_V1",
        "rule_item_count": len(rules),
        "rule_item_counts_by_status": dict(sorted(by_status.items())),
        "rule_item_counts_by_domain": dict(sorted(by_domain.items())),
        "observation_count": len(observations),
        "supported_grammar_rule_count": 0,
        "layers": layers,
    }


def _independence_report(rules: list[dict[str, Any]]) -> dict[str, Any]:
    level_counts = collections.Counter(
        rule["independence"]["strongest_observed_level"] for rule in rules
    )
    return {
        "report_id": "XNLP_GRAMMAR_INDEPENDENCE_V1",
        "rules": [
            {
                "rule_id": rule["rule_id"],
                "evidence_record_ids": rule["evidence_record_ids"],
                "source_titles": rule["independence"]["source_titles"],
                "independence_groups": rule["independence"]["independence_groups"],
                "source_section_count": rule["independence"]["source_section_count"],
                "strongest_observed_level": rule["independence"]["strongest_observed_level"],
                "same_author": rule["independence"]["same_author"],
                "different_author": rule["independence"]["different_author"],
                "author_attribution_status": rule["independence"]["author_attribution_status"],
            }
            for rule in rules
        ],
        "counts_by_strongest_observed_level": dict(sorted(level_counts.items())),
        "author_independence_limit": (
            "Corpus and source registry do not record verified source-author identities "
            "for these records; same-author and different-author claims remain UNKNOWN."
        ),
    }


def _provenance_map(
    rules: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    evidence_ids: set[str],
) -> dict[str, Any]:
    record_to_rules: dict[str, list[str]] = collections.defaultdict(list)
    rule_to_evidence = {}
    for rule in rules:
        rule_id = rule["rule_id"]
        rule_to_evidence[rule_id] = {
            "source_artifact": rule["source_artifact"],
            "source_artifact_sha256": rule["source_artifact_sha256"],
            "observation_ids": rule["observation_ids"],
            "evidence_record_ids": rule["evidence_record_ids"],
            "out_of_boundary_record_ids": rule["out_of_boundary_record_ids"],
        }
        for record_id in rule["evidence_record_ids"]:
            record_to_rules[record_id].append(rule_id)
    return {
        "report_id": "XNLP_GRAMMAR_PROVENANCE_MAP_V1",
        "evidence_set_id": "DERIVATION_EVIDENCE_SET_V1",
        "rule_to_evidence": rule_to_evidence,
        "records": {
            record_id: _record_metadata(record_id, records_by_id, sources, evidence_ids)
            for record_id in sorted(record_to_rules)
        },
        "record_to_rules": {
            record_id: sorted(rule_ids)
            for record_id, rule_ids in sorted(record_to_rules.items())
        },
        "source_author_policy": (
            "Source author is reported only when explicitly present in the corpus "
            "record; otherwise NOT_RECORDED. No author identity is inferred from titles."
        ),
    }


def _render_report(
    specification: dict[str, Any],
    dispositions: dict[str, Any],
    coverage: dict[str, Any],
    counterexamples: dict[str, Any],
    unresolved: dict[str, Any],
    independence: dict[str, Any],
    verification: dict[str, Any] | None,
) -> str:
    summary = specification["summary"]
    lines = [
        "# XNLP Formal Grammar Derivation Report",
        "",
        f"- Specification: `grammar_specification.json` (version {SPECIFICATION_VERSION}).",
        "- Scope: evidence-bounded, reviewable, non-executable grammar audit.",
        "- No corpus records, provenance labels, or existing observations were changed.",
        "",
        "## A. Existing candidate inventory",
        "",
        f"- Derived candidate structures audited: {summary['derived_candidate_structure_count']}.",
        f"- Legacy structured grammar entries audited: {summary['legacy_structure_count']}.",
        f"- Source-linked observations preserved: {summary['source_linked_observation_count']}.",
        f"- Total disposition items: {dispositions['total_items']}.",
        "- Existing legacy `SUPPORTED` labels are recorded as-is but are not treated as promoted linguistic rules.",
        "",
        "### Candidate dispositions",
        "",
        "| Disposition | Count |",
        "|---|---:|",
        *[
            f"| {name} | {count} |"
            for name, count in sorted(dispositions["counts"].items())
        ],
        "",
        "No grammatical generalization met the promotion gate. Items with an explicit data-governance scope are kept UNKNOWN/out-of-grammar; linguistic analyses remain candidates pending adjudication.",
        "",
        "## B. Candidate dispositions and C. promoted supported rules",
        "",
        f"- Promoted supported grammar rules: **{summary['supported_grammar_rule_count']}**.",
        "- The specification contains no executable rule and no rule with status SUPPORTED.",
        "- Candidates remain candidates when interpretation, full counterexample search, or author independence is unresolved.",
        "",
        "## D. Rules retained as candidates and E. rejected rules",
        "",
        f"- Retained candidate analyses: {dispositions['counts'].get('RETAIN_AS_CANDIDATE', 0)}.",
        f"- Rejected analyses: {dispositions['counts'].get('REJECT', 0)}.",
        "- No source-backed candidate was rejected solely from absence of evidence; rejection requires a contradictory evidential basis.",
        "",
        "## F. Unknown grammatical areas",
        "",
        *[
            f"- {item['layer']}: {item['question']} **{item['required_action']}**."
            for item in unresolved["questions"]
        ],
        "",
        "## G–M. Morphology, agreement, nominal, verbal, phrase, clause, and construction structures",
        "",
        "No grammatical rule was promoted in these layers. Exact counts of linked observations and candidate items are in `grammar_coverage.json`. Existing labels, interpretations, and examples remain source metadata—not independent validation.",
        "",
        "## N. Counterexamples",
        "",
        f"- The frozen evidence set was scanned in full ({counterexamples['full_evidence_set_record_count_scanned']} records) for exact token sequences explicitly present in linked source-observation examples.",
        "- This is exhaustive surface-form retrieval for those cues, not a grammatical counterexample search. The corpus does not provide exhaustive positive/negative grammatical labels; all candidate rules require human counterexample adjudication.",
        "- Existing listed counterexamples are preserved when their record IDs are inside the evidence set; no absence of counterexamples is inferred from empty legacy lists.",
        "",
        "## O. Competing analyses",
        "",
        "The open `a-` conflict is preserved with the competing analyses subject concord, noun-prefix, and negation marker; it remains unresolved. Full conflict records are included in the specification.",
        "",
        "## P. Independence evidence",
        "",
        f"- Rules with cited evidence from different work groups: {independence['counts_by_strongest_observed_level'].get('DIFFERENT_WORK', 0)}.",
        f"- Rules with evidence limited to one work group: {independence['counts_by_strongest_observed_level'].get('SAME_WORK', 0)}.",
        f"- Rules with one cited record: {independence['counts_by_strongest_observed_level'].get('SAME_RECORD', 0)}.",
        "- Author independence is UNKNOWN because verified author metadata is not recorded; source-title plurality is not counted as author independence.",
        "",
        "## Q. Provenance coverage",
        "",
        f"- Evidence set: `{specification['evidence_set']['id']}`, {specification['evidence_set']['record_count']} records.",
        f"- Source author metadata recorded on cited records: {summary['records_with_source_author_metadata']}.",
        f"- Out-of-boundary candidate citations excluded from evidence: {summary['out_of_boundary_citation_count']}.",
        "- Every in-boundary rule reference resolves to a source record; authorship and transcription fidelity were not inferred.",
        "",
        "## R. Grammar coverage",
        "",
        "| Layer | Candidate items | Observations | Supported rules | Coverage |",
        "|---|---:|---:|---:|---|",
        *[
            f"| {layer['layer']} | {layer['linked_candidate_item_count']} | {layer['linked_observation_count']} | 0 | {layer['coverage_status']} |"
            for layer in coverage["layers"]
        ],
        "",
        "## S. Human-adjudication requirements",
        "",
        "- Review proposed interpretations and lexical/morphological segments against the linked, attested record contexts.",
        "- Annotate positive, negative, competing, and counterexample contexts across the full evidence set.",
        "- Record validated source work and author relationships; do not infer them from title strings.",
        "- Review the unresolved `a-` conflict and transcription fidelity.",
        "",
        "## T. Test results",
        "",
        *(
            [
                f"- Command: `{verification['test_command']}`.",
                f"- Result: {verification['test_results']}.",
            ]
            if verification else
            ["- Verification metadata is not recorded; run the grammar derivation test suite."]
        ),
        "- Tests assert evidence boundaries, trace resolution, no silent candidate promotion, corpus non-mutation, and deterministic regeneration.",
        "",
        "## U. Exact artifacts changed",
        "",
        "- `xnlp_language/derive_grammar.py` — evidence-bounded grammar audit generator.",
        "- `xnlp_language/build_bundle.py` — direct bundle reference to the grammar audit.",
        "- `tests/test_grammar_derivation.py` — grammar audit validation tests.",
        "- `tests/test_foundation_bundle.py` — bundle catalog assertions.",
        "- `xnlp_language/README.md` — audit regeneration and evidence limits.",
        "",
        "## V. Exact artifacts created",
        "",
        *[
            f"- `xnlp_language/generated/grammar_derivation/{name}`"
            for name in (
                "grammar_specification.json",
                "candidate_dispositions.json",
                "counterexample_report.json",
                "unresolved_grammar_questions.json",
                "grammar_coverage.json",
                "independence_report.json",
                "provenance_map.json",
                "grammar_derivation_report.md",
            )
        ],
        "",
        "## W. Next phase required",
        "",
        "Formal grammar validation with human-adjudicated examples, negative evidence, and independently reviewed provenance is required before executable grammar implementation.",
        "",
        "## Invariants",
        "",
        "No grammar engine, parser, generator, tagger, grammar checker, training run, or model integration was created. No unsupported language fact was made authoritative.",
        "",
    ]
    return "\n".join(lines)


def derive_grammar(
    root: Path = ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a frozen-evidence grammar audit without editing its inputs."""
    root = root.resolve()
    output_dir = output_dir.resolve()
    report_path = report_path.resolve()
    inputs = _load_inputs(root)
    rules, observations = _grammar_inventory(inputs)
    dispositions = _candidate_disposition_report(rules)
    counterexamples = _counterexample_report(
        rules,
        observations,
        len(inputs["evidence_set"]["record_ids"]),
        _surface_search_index(inputs["corpus_evidence"]),
    )
    unresolved = _unresolved_questions(rules)
    orthography = inputs["derivation"]["language_representation"]["ORTHOGRAPHY"]
    coverage = _coverage_report(rules, observations, orthography)
    independence = _independence_report(rules)
    provenance_map = _provenance_map(
        rules,
        inputs["records_by_id"],
        inputs["sources"],
        set(inputs["evidence_set"]["record_ids"]),
    )
    all_evidence_refs = {
        record_id for rule in rules for record_id in rule["evidence_record_ids"]
    }
    records_with_author = sum(
        1 for record_id in all_evidence_refs
        if inputs["records_by_id"][record_id].get("source_author")
    )
    summary = {
        "derived_candidate_structure_count": len(inputs["derivation"]["candidate_structures"]),
        "legacy_structure_count": len(inputs["legacy_structures"]),
        "source_linked_observation_count": len(observations),
        "grammar_rule_item_count": len(rules),
        "grammar_rule_item_counts_by_status": dict(
            sorted(collections.Counter(rule["status"] for rule in rules).items())
        ),
        "candidate_disposition_counts": dispositions["counts"],
        "supported_grammar_rule_count": 0,
        "rejected_rule_count": dispositions["counts"].get("REJECT", 0),
        "claims_validated_or_established": 0,
        "derivation_evidence_record_count": len(inputs["evidence_set"]["record_ids"]),
        "unique_cited_evidence_record_count": len(all_evidence_refs),
        "out_of_boundary_citation_count": sum(
            len(rule["out_of_boundary_record_ids"]) for rule in rules
        ),
        "records_with_source_author_metadata": records_with_author,
        "full_derivation_evidence_set_counterexample_surface_scan": True,
    }
    specification = {
        "specification_id": "XNLP_FORMAL_GRAMMAR_SPECIFICATION_V1",
        "specification_version": SPECIFICATION_VERSION,
        "status": "REVIEWABLE_NON_EXECUTABLE_EVIDENCE_AUDIT",
        "evidence_set": {
            "id": inputs["evidence_set"]["id"],
            "record_count": inputs["evidence_set"]["record_count"],
            "manifest_sha256": _sha256(root / CLAIM_PATH),
            "record_ids": inputs["evidence_set"]["record_ids"],
            "eligibility_preserved_from_derivation": True,
            "provenance_registry_sha256": _sha256(root / PROVENANCE_PATH),
            "source_author_fidelity_note": (
                "Source author metadata and transcription fidelity are not inferred."
            ),
        },
        "knowledge_levels": [
            "OBSERVATION",
            "SUPPORTED_GENERALIZATION",
            "CANDIDATE_GENERALIZATION",
            "UNKNOWN",
            "CONTRADICTED_REJECTED",
        ],
        "promotion_policy": {
            "supported_rule_count": 0,
            "promotions_performed": False,
            "required_checks": [
                "valid source observations",
                "intact provenance",
                "no unsupported external knowledge",
                "recurring evidence",
                "full evidence-set counterexample analysis",
                "competing analyses considered",
                "independence recorded",
                "narrow evidence-bounded formulation",
                "reproducibility",
                "traceable reviewer evidence",
            ],
            "failure_action": "DO_NOT_PROMOTE",
        },
        "input_artifact_hashes": inputs["input_hashes"],
        "rules": rules,
        "observations": observations,
        "upstream_orthographic_observations": {
            "source_specification_id": inputs["derivation"]["specification_id"],
            "source_artifact": CLAIM_PATH.as_posix(),
            "source_artifact_sha256": _sha256(root / CLAIM_PATH),
            "evidence_set_id": inputs["evidence_set"]["id"],
            "inventory": orthography,
            "interpretation_boundary": (
                "Reused from the completed language derivation; no corpus "
                "re-derivation or linguistic interpretation is performed here."
            ),
        },
        "conflicts": [
            {
                **conflict,
                "record_ids": [
                    record_id for record_id in conflict.get("record_ids", [])
                    if record_id in set(inputs["evidence_set"]["record_ids"])
                ],
            }
            for conflict in inputs["conflicts"]
            if any(
                record_id in set(inputs["evidence_set"]["record_ids"])
                for record_id in conflict.get("record_ids", [])
            )
        ],
        "summary": summary,
        "verification": verification or {
            "test_results_recorded": False,
            "note": "This generation command does not execute tests.",
        },
        "bundle_reference": {
            "bundle_id": inputs["bundle"]["bundle_id"],
            "bundle_version": inputs["bundle"].get("bundle_version"),
            "bundle_hash_omitted_to_avoid_catalog_hash_cycle": True,
        },
        "artifacts": {
            "candidate_dispositions": "candidate_dispositions.json",
            "counterexample_report": "counterexample_report.json",
            "unresolved_questions": "unresolved_grammar_questions.json",
            "coverage": "grammar_coverage.json",
            "independence": "independence_report.json",
            "provenance_map": "provenance_map.json",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "grammar_specification.json", specification)
    _write_json(output_dir / "candidate_dispositions.json", dispositions)
    _write_json(output_dir / "counterexample_report.json", counterexamples)
    _write_json(output_dir / "unresolved_grammar_questions.json", unresolved)
    _write_json(output_dir / "grammar_coverage.json", coverage)
    _write_json(output_dir / "independence_report.json", independence)
    _write_json(output_dir / "provenance_map.json", provenance_map)
    _write_text(
        report_path,
        _render_report(
            specification,
            dispositions,
            coverage,
            counterexamples,
            unresolved,
            independence,
            verification,
        ),
    )
    return specification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--test-command", default=None)
    parser.add_argument("--test-results", default=None)
    args = parser.parse_args()
    if bool(args.test_command) != bool(args.test_results):
        parser.error("--test-command and --test-results must be supplied together")
    verification = (
        {
            "test_results_recorded": True,
            "test_command": args.test_command,
            "test_results": args.test_results,
        }
        if args.test_command else None
    )
    specification = derive_grammar(
        args.root,
        args.output_dir,
        args.report,
        verification,
    )
    print(json.dumps(specification["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Grammar specification: {args.output_dir / 'grammar_specification.json'}")
    print(f"Grammar report: {args.report}")


if __name__ == "__main__":
    main()
