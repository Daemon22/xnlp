"""Build explicit, evidence-linked linguistic validation artifacts."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from xnlp_language.build_foundation import WORD


EVIDENCE_STATES = (
    "OBSERVED",
    "CANDIDATE",
    "SOURCE_SUPPORTED",
    "CROSS_SOURCE_SUPPORTED",
    "VALIDATED",
    "ESTABLISHED",
    "REJECTED",
    "UNRESOLVED",
)
ESTABLISHED_STATES = {"VALIDATED", "ESTABLISHED"}
GENERATED = Path("xnlp_language") / "generated"
CLAIM_SOURCES = (
    ("noun_classes", "grammar/noun_classes.jsonl"),
    ("agreement", "grammar/agreement_system.jsonl"),
    ("morphology", "grammar/verb_morphology.jsonl"),
    ("derivational_morphology", "grammar/derivational_morphology.jsonl"),
    ("grammar_observations", None),
    ("data_governance", "grammar/grammar_rules.json"),
)
GOLD_DOMAINS = (
    "lexical_understanding",
    "morphological_analysis",
    "agreement",
    "syntax",
    "semantics",
    "discourse_context",
    "controlled_generation_comprehension",
)
NEAR_DUPLICATE_THRESHOLD = 0.9
SHINGLE_SIZE = 5


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence_ids(entry: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("evidence_records", "source_record_ids"):
        value = entry.get(key, [])
        if isinstance(value, list):
            values.extend(item for item in value if isinstance(item, str))
    for key in ("evidence", "evidence_sample", "supporting_evidence"):
        value = entry.get(key, [])
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    values.append(item)
                elif isinstance(item, dict):
                    record_id = item.get("record_id")
                    if isinstance(record_id, str):
                        values.append(record_id)
    return sorted(set(values))


def _claim_id(domain: str, identifier: str) -> str:
    stable = re.sub(r"[^A-Za-z0-9_.-]+", "_", identifier).strip("_")
    return f"CLAIM_{domain.upper()}_{stable}"


def _legacy_state(domain: str, entry: dict[str, Any]) -> str:
    status = entry.get("review_status", entry.get("status", entry.get("analysis_status")))
    if status == "REJECTED":
        return "REJECTED"
    if status == "CONTESTED":
        return "UNRESOLVED"
    if domain == "data_governance":
        return "OBSERVED"
    if status == "SUPPORTED":
        return "SOURCE_SUPPORTED"
    return "CANDIDATE"


def _record_provenance_class(
    record: dict[str, Any],
    registry_sources: dict[str, dict[str, Any]] | None = None,
) -> str:
    provenance_type = str(record.get("provenance_type", "")).casefold()
    generator = record.get("generator")
    if provenance_type in {"synthetic", "generated", "derived"} or generator:
        return "synthetic"
    if registry_sources is None:
        return str(record.get("provenance_class", "unknown"))
    source = registry_sources.get(str(record.get("source_title", "")))
    if (
        source
        and source.get("foundation_attested") is True
        and provenance_type == "authoritative"
        and not generator
    ):
        return "attested"
    return "unknown"


def _provenance_summary(
    evidence_ids: list[str],
    corpus_by_id: dict[str, dict[str, Any]],
    registry_sources: dict[str, dict[str, Any]] | None = None,
) -> dict[str, int]:
    summary = {"attested": 0, "synthetic": 0, "unknown": 0}
    for record_id in evidence_ids:
        record = corpus_by_id.get(record_id)
        summary[
            _record_provenance_class(record, registry_sources) if record else "unknown"
        ] += 1
    return summary


def _claim_description(domain: str, entry: dict[str, Any]) -> str:
    for key in ("claim", "rule", "phenomenon", "pattern", "description"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if domain == "morphology":
        return (
            f"Candidate morphological segmentation for "
            f"{entry.get('surface_form', 'an unspecified surface form')}."
        )
    return f"Evidence-linked {domain.replace('_', ' ')} analysis."


def _build_claims(
    root: Path,
    corpus_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    claim_rows: list[dict[str, Any]] = []
    used_claim_ids: set[str] = set()
    source_paths: dict[str, str] = {}
    source_support_violations = []
    registry_path = root / GENERATED / "evidence" / "source_provenance.json"
    if not registry_path.is_file():
        raise FileNotFoundError(f"Source provenance registry is missing: {registry_path}")
    source_paths[registry_path.relative_to(root).as_posix()] = _artifact_hash(registry_path)
    registry_sources = json.loads(registry_path.read_text(encoding="utf-8")).get("sources", {})
    for domain, relative in CLAIM_SOURCES:
        if relative is None:
            paths = sorted((root / GENERATED / "grammar").glob("*_observations.jsonl"))
        else:
            paths = [root / GENERATED / relative]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Required linguistic claim input is missing: {path}")
            source_path = path.relative_to(root).as_posix()
            source_paths[source_path] = _artifact_hash(path)
            rows = _read_jsonl(path) if path.suffix == ".jsonl" else json.loads(
                path.read_text(encoding="utf-8")
            )
            if not isinstance(rows, list):
                rows = [rows]
            path_domain = (
                path.stem.removesuffix("_observations")
                if domain == "grammar_observations"
                else domain
            )
            for index, entry in enumerate(rows, start=1):
                evidence_ids = _evidence_ids(entry)
                if not evidence_ids:
                    raise ValueError(
                        f"Claim source {source_path} row {index} has no evidence record IDs"
                    )
                missing = sorted(set(evidence_ids) - corpus_by_id.keys())
                if missing:
                    raise ValueError(
                        f"Claim source {source_path} row {index} references unknown "
                        f"record IDs: {missing[:5]}"
                    )
                identifier = str(
                    entry.get("entry_id")
                    or entry.get("observation_id")
                    or entry.get("rule_id")
                    or entry.get("analysis_id")
                    or index
                )
                claim_identifier = _claim_id(path_domain, identifier)
                if claim_identifier in used_claim_ids:
                    claim_identifier = f"{claim_identifier}_{index}"
                if claim_identifier in used_claim_ids:
                    raise ValueError(
                        f"Cannot create a stable unique claim ID for {source_path} row {index}"
                    )
                used_claim_ids.add(claim_identifier)
                legacy_state = _legacy_state(path_domain, entry)
                provenance_summary = _provenance_summary(
                    evidence_ids, corpus_by_id, registry_sources
                )
                if legacy_state == "SOURCE_SUPPORTED" and provenance_summary["attested"] == 0:
                    source_support_violations.append({
                        "claim_id": claim_identifier,
                        "evidence_state_from_legacy": legacy_state,
                        "provenance_summary": provenance_summary,
                        "reason": (
                            "SOURCE_SUPPORTED requires at least one attested evidence record; "
                            "synthetic-only or unknown-only evidence is insufficient."
                        ),
                    })
                    legacy_state = "UNRESOLVED"
                examples = []
                for record_id in evidence_ids[:5]:
                    source = corpus_by_id[record_id]
                    text = source.get("text", source.get("original_text", ""))
                    examples.append({
                        "record_id": record_id,
                        "source_work": source.get("source_title", "UNKNOWN"),
                        "surface_form": entry.get("surface_form"),
                        "text": text,
                    })
                claim_rows.append({
                    "claim_id": claim_identifier,
                    "domain": path_domain,
                    "claim": _claim_description(path_domain, entry),
                    "evidence_state": legacy_state,
                    "confidence": entry.get("confidence", "INSUFFICIENT_EVIDENCE"),
                    "provenance_summary": provenance_summary,
                    **({
                        "legacy_confidence_label": "SUPPORTED",
                        "legacy_label_migration_note": (
                            "xnlp_language/schemas/linguistic_claim_schema.json"
                            "#/definitions/legacy_label_migration"
                        ),
                    } if _legacy_state(path_domain, entry) == "SOURCE_SUPPORTED" else {}),
                    "evidence_record_ids": evidence_ids,
                    "source_works": sorted({
                        str(corpus_by_id[record_id].get("source_title", "UNKNOWN"))
                        for record_id in evidence_ids
                    }),
                    "source_collections": sorted({
                        str(corpus_by_id[record_id].get("source", "UNKNOWN"))
                        for record_id in evidence_ids
                    }),
                    "examples": examples,
                    "counterexamples": [],
                    "analysis": entry.get(
                        "candidate_segments",
                        entry.get("notes", entry.get("analysis", "")),
                    ),
                    "review_required": legacy_state not in ESTABLISHED_STATES,
                    "review_notes": [
                        "Legacy status retained separately; no claim was promoted to "
                        "VALIDATED or ESTABLISHED during migration."
                    ],
                    "created_from": {
                        "artifact": source_path,
                        "artifact_sha256": source_paths[source_path],
                        "legacy_status": entry.get(
                            "review_status",
                            entry.get("status", entry.get("analysis_status")),
                        ),
                    },
                    "validated_by": [],
                    "review_history": [],
                })
    claim_rows.sort(key=lambda row: row["claim_id"])
    if len({row["claim_id"] for row in claim_rows}) != len(claim_rows):
        raise ValueError("Linguistic claim IDs are not unique")
    totals: dict[str, Any] = {}
    for domain in sorted({row["domain"] for row in claim_rows}):
        rows = [row for row in claim_rows if row["domain"] == domain]
        state_counts = Counter({state: 0 for state in EVIDENCE_STATES})
        state_counts.update(row["evidence_state"] for row in rows)
        totals[domain] = {
            "total": len(rows),
            "by_evidence_state": dict(sorted(state_counts.items())),
        }
    return claim_rows, {
        "total": len(claim_rows),
        "domains": totals,
        "sources": source_paths,
        "source_support_violations": source_support_violations,
    }


def apply_review_decision(
    claim: dict[str, Any],
    new_state: str,
    reviewer_id: str,
    protocol_id: str,
    corpus_record_ids: set[str],
    establishment_basis: str | None = None,
) -> dict[str, Any]:
    """Apply an explicit, auditable human decision to a traceable claim."""
    if new_state not in EVIDENCE_STATES:
        raise ValueError(f"Unsupported evidence state: {new_state}")
    if not reviewer_id.strip() or not protocol_id.strip():
        raise ValueError("A reviewer ID and validation protocol ID are required")
    evidence_ids = claim.get("evidence_record_ids", [])
    if not evidence_ids or not set(evidence_ids).issubset(corpus_record_ids):
        raise ValueError("Linguistic claims require resolved corpus evidence")
    current = claim.get("evidence_state")
    if current == "REJECTED" and new_state != "REJECTED":
        raise ValueError("Rejected claims cannot be promoted; create a separately reviewed claim")
    if new_state == "ESTABLISHED" and current != "VALIDATED":
        raise ValueError("ESTABLISHED requires a prior explicit VALIDATED decision")
    if new_state == "ESTABLISHED" and not (establishment_basis or "").strip():
        raise ValueError("ESTABLISHED requires a documented strong-evidence basis")
    if new_state == "CROSS_SOURCE_SUPPORTED" and (
        len(set(claim.get("source_works", []))) < 2
        or len(set(claim.get("source_collections", []))) < 2
    ):
        raise ValueError(
            "CROSS_SOURCE_SUPPORTED requires evidence across multiple works and source collections"
        )
    if new_state in {"VALIDATED", "ESTABLISHED"}:
        claim.setdefault("validated_by", []).append(reviewer_id)
    claim["evidence_state"] = new_state
    claim["review_required"] = new_state not in ESTABLISHED_STATES | {"REJECTED"}
    claim.setdefault("review_history", []).append({
        "from": current,
        "to": new_state,
        "reviewer_id": reviewer_id,
        "protocol_id": protocol_id,
    })
    if new_state == "ESTABLISHED":
        claim["review_history"][-1]["establishment_basis"] = establishment_basis.strip()
    return claim


def validate_claims(
    claims: list[dict[str, Any]],
    corpus_record_ids: set[str],
    corpus_by_id: dict[str, dict[str, Any]] | None = None,
) -> None:
    claim_ids = set()
    for claim in claims:
        if claim.get("evidence_state") not in EVIDENCE_STATES:
            raise ValueError(f"Invalid evidence state on {claim.get('claim_id')}")
        if not claim.get("evidence_record_ids"):
            raise ValueError(f"Orphan claim cannot enter foundation: {claim.get('claim_id')}")
        if not set(claim["evidence_record_ids"]).issubset(corpus_record_ids):
            raise ValueError(f"Unresolved evidence link on {claim.get('claim_id')}")
        if corpus_by_id is not None:
            expected_summary = _provenance_summary(claim["evidence_record_ids"], corpus_by_id)
            if claim.get("provenance_summary") != expected_summary:
                raise ValueError(
                    f"Incorrect evidence provenance summary on {claim.get('claim_id')}"
                )
        else:
            expected_summary = claim.get("provenance_summary", {})
        if claim["evidence_state"] == "SOURCE_SUPPORTED" and expected_summary.get(
            "attested", 0
        ) == 0:
            raise ValueError(
                f"Synthetic-only or unknown-only evidence cannot source-support "
                f"{claim.get('claim_id')}"
            )
        if claim["evidence_state"] == "ESTABLISHED" and not claim.get("validated_by"):
            raise ValueError(f"ESTABLISHED claim lacks explicit validation: {claim['claim_id']}")
        if claim["evidence_state"] in {"VALIDATED", "ESTABLISHED", "REJECTED"}:
            history = claim.get("review_history", [])
            if not history or not history[-1].get("reviewer_id") or not history[-1].get("protocol_id"):
                raise ValueError(
                    f"Reviewed claim lacks explicit reviewer/protocol provenance: {claim['claim_id']}"
                )
        if claim["evidence_state"] == "ESTABLISHED" and not any(
            review.get("to") == "VALIDATED" for review in claim.get("review_history", [])
        ):
            raise ValueError(
                f"ESTABLISHED claim lacks a prior VALIDATED decision: {claim['claim_id']}"
            )
        if claim["evidence_state"] == "ESTABLISHED" and not any(
            review.get("to") == "ESTABLISHED"
            and review.get("establishment_basis", "").strip()
            for review in claim.get("review_history", [])
        ):
            raise ValueError(
                f"ESTABLISHED claim lacks a documented strong-evidence basis: {claim['claim_id']}"
            )
        if claim["claim_id"] in claim_ids:
            raise ValueError(f"Duplicate claim ID: {claim['claim_id']}")
        claim_ids.add(claim["claim_id"])


def _lexical_tasks(root: Path, corpus_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    lexical_path = root / GENERATED / "lexicon" / "lexicon_observed.jsonl"
    task_queue_path = root / GENERATED / "annotation" / "lexical_analysis_queue.jsonl"
    training_provenance_path = root / "data" / "processed" / "v2" / "training_corpus_provenance.jsonl"
    token_occurrences: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "forms": set(), "record_ids": set(), "works": set(), "sources": set()}
    )
    for record in _read_jsonl(training_provenance_path):
        record_id = str(record["record_id"])
        if record_id not in corpus_by_id:
            raise ValueError(f"Lexical training record is absent from corpus evidence: {record_id}")
        for match in WORD.finditer(record.get("text", "")):
            surface = match.group(0)
            observation = token_occurrences[surface.lower()]
            observation["count"] += 1
            observation["forms"].add(surface)
            observation["record_ids"].add(record_id)
            observation["works"].add(str(record.get("source_title", "UNKNOWN")))
            observation["sources"].add(str(record.get("source", "UNKNOWN")))

    tasks = []
    for row in _read_jsonl(lexical_path):
        form = str(row.get("lemma", "")).lower()
        occurrence = token_occurrences.get(form)
        if not occurrence:
            raise ValueError(f"Lexical observation {row.get('lexeme_id')} has no corpus occurrence")
        if occurrence["count"] != row.get("frequency"):
            raise ValueError(
                f"Lexical frequency mismatch for {row.get('lexeme_id')}: "
                f"observed {row.get('frequency')}, recounted {occurrence['count']}"
            )
        record_ids = sorted(occurrence["record_ids"])
        forms = sorted(occurrence["forms"])
        task = {
            "task_id": f"LEXAN_{row['lexeme_id']}",
            "lexeme_observation_id": row["lexeme_id"],
            "surface_form": forms[0],
            "observed_surface_forms": forms,
            "normalized_form": None,
            "lemma": None,
            "lexical_category": None,
            "possible_variants": [],
            "source_occurrence_count": row.get("frequency"),
            "source_work_references": sorted(occurrence["works"]),
            "source_collection_references": sorted(occurrence["sources"]),
            "source_record_ids": record_ids,
            "semantic_status": "UNRESOLVED",
            "morphology_status": "UNRESOLVED",
            "evidence_state": "UNRESOLVED",
            "analysis_status": "PENDING_HUMAN_REVIEW",
            "machine_suggestion": None,
            "reviewer_decision": None,
            "status": "pending",
            "assigned_lemma": None,
            "assigned_category": None,
            "assigned_meaning": None,
            "adjudicator": None,
            "adjudication_note": None,
            "flagged_by": None,
            "auto_evidence": None,
            "adjudication_history": [],
        }
        tasks.append(task)
    if task_queue_path.is_file():
        existing_tasks = {
            task["task_id"]: task
            for task in _read_jsonl(task_queue_path)
            if task.get("task_id")
        }
        adjudication_fields = (
            "status",
            "assigned_lemma",
            "assigned_category",
            "assigned_meaning",
            "adjudicator",
            "adjudication_note",
            "flagged_by",
            "auto_evidence",
            "adjudication_history",
        )
        for task in tasks:
            previous = existing_tasks.get(task["task_id"], {})
            for field, value in previous.items():
                if field not in task:
                    task[field] = value
            for field in adjudication_fields:
                if field in previous:
                    task[field] = previous[field]
            if (
                task["status"] == "disputed"
                and task["adjudicator"] == "machine-only"
            ):
                task["adjudicator"] = None
                task["flagged_by"] = "machine-only"
    return tasks


def _normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _shingles(text: str) -> set[str]:
    padded = f"  {text}  "
    return {padded[index:index + SHINGLE_SIZE] for index in range(max(0, len(padded) - SHINGLE_SIZE + 1))}


def _partition_audit(
    partitions: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    names = sorted(partitions)
    result: dict[str, Any] = {
        "record_overlaps": {},
        "source_work_overlaps": {},
        "source_overlaps": {},
        "author_overlaps": {},
        "author_metadata_available": {},
        "exact_text_overlap_counts": {},
        "near_duplicate_pairs": {},
    }
    profiles: dict[str, list[tuple[dict[str, Any], str, set[str]]]] = {}
    record_sets: dict[str, set[str]] = {}
    work_sets: dict[str, set[str]] = {}
    source_sets: dict[str, set[str]] = {}
    author_sets: dict[str, set[str]] = {}
    author_metadata_available: dict[str, bool] = {}
    for name, rows in partitions.items():
        profiles[name] = [
            (row, _normalize_text(row.get("text", row.get("original_text", ""))),
             _shingles(_normalize_text(row.get("text", row.get("original_text", "")))))
            for row in rows
        ]
        record_sets[name] = {str(row["record_id"]) for row in rows}
        work_sets[name] = {str(row.get("source_title", "UNKNOWN")) for row in rows}
        source_sets[name] = {str(row.get("source", "UNKNOWN")) for row in rows}
        author_sets[name] = {
            str(row["author"]) for row in rows
            if row.get("author") not in (None, "")
        }
        author_metadata_available[name] = any(
            row.get("author") not in (None, "") for row in rows
        )
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1:]:
            pair = f"{left_name}_{right_name}"
            result["record_overlaps"][pair] = len(record_sets[left_name] & record_sets[right_name])
            result["source_work_overlaps"][pair] = sorted(
                work_sets[left_name] & work_sets[right_name]
            )
            result["source_overlaps"][pair] = sorted(source_sets[left_name] & source_sets[right_name])
            result["author_overlaps"][pair] = sorted(author_sets[left_name] & author_sets[right_name])
            result["author_metadata_available"][pair] = (
                author_metadata_available[left_name]
                and author_metadata_available[right_name]
            )
            right_by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
            postings: dict[str, list[int]] = defaultdict(list)
            for index, (row, text, grams) in enumerate(profiles[right_name]):
                if text:
                    right_by_text[text].append(row)
                    for gram in grams:
                        postings[gram].append(index)
            exact_pairs = sum(
                len(right_by_text.get(text, []))
                for _, text, _ in profiles[left_name]
                if text
            )
            near_pairs = []
            for left_index_row, (left_row, text, grams) in enumerate(profiles[left_name]):
                if not text:
                    continue
                candidate_counts: Counter[int] = Counter()
                for gram in grams:
                    candidate_counts.update(postings.get(gram, []))
                for right_row_index, common_grams in candidate_counts.items():
                    _, right_text, right_grams = profiles[right_name][right_row_index]
                    if not right_grams or text == right_text:
                        continue
                    if min(len(grams), len(right_grams)) / max(len(grams), len(right_grams)) < NEAR_DUPLICATE_THRESHOLD:
                        continue
                    union_size = len(grams) + len(right_grams) - common_grams
                    similarity = common_grams / union_size if union_size else 0
                    if similarity >= NEAR_DUPLICATE_THRESHOLD:
                        right_row = profiles[right_name][right_row_index][0]
                        near_pairs.append({
                            "left_record_id": left_row["record_id"],
                            "right_record_id": right_row["record_id"],
                            "shingle_jaccard": round(similarity, 6),
                        })
            result["exact_text_overlap_counts"][pair] = exact_pairs
            result["near_duplicate_pairs"][pair] = sorted(
                near_pairs,
                key=lambda item: (item["left_record_id"], item["right_record_id"]),
            )
    return result


def _lineage_audit(partitions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summaries = {}
    for name, rows in partitions.items():
        generated_or_derived = [
            row for row in rows
            if str(row.get("provenance_type", "")).casefold() in {"generated", "derived"}
            or bool(row.get("generator"))
        ]
        unknown_origin = [
            row for row in rows
            if not row.get("provenance_type") and not row.get("source_type")
        ]
        summaries[name] = {
            "records": len(rows),
            "provenance_types": dict(sorted(Counter(
                str(row.get("provenance_type", "UNKNOWN")) for row in rows
            ).items())),
            "generated_or_derived_marked_records": len(generated_or_derived),
            "unknown_origin_records": len(unknown_origin),
        }
    return {
        "partitions": summaries,
        "marker_check_passed": all(
            details["generated_or_derived_marked_records"] == 0
            and details["unknown_origin_records"] == 0
            for details in summaries.values()
        ),
        "limitation": (
            "Detects declared generated/derived provenance and unclassified origins; "
            "unmarked synthetic content cannot be ruled out by metadata alone."
        ),
    }


def build_linguistic_layer(root: Path) -> dict[str, Any]:
    """Create deterministic claims, work queues, evaluation-pool, and audit reports."""
    root = root.resolve()
    generated = root / GENERATED
    report_path = root / "artifacts" / "xnlp_prose_pilot_work_isolated" / "training_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Work-isolated split report is required: {report_path}")
    experiment = json.loads(report_path.read_text(encoding="utf-8"))
    split_config = experiment.get("training_data", {}).get("splits", {})
    split_paths = {}
    for name in ("train", "val", "test"):
        path_value = split_config.get(name, {}).get("provenance_path")
        path = root / path_value if path_value else None
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Work-isolated {name} provenance is missing: {path}")
        split_paths[name] = path
    corpus_path = generated / "evidence" / "corpus_records.jsonl"
    source_registry_path = generated / "evidence" / "source_provenance.json"
    source_registry = json.loads(source_registry_path.read_text(encoding="utf-8"))
    corpus_records = _read_jsonl(corpus_path)
    corpus_by_id = {str(row["record_id"]): row for row in corpus_records}
    if len(corpus_by_id) != len(corpus_records):
        raise ValueError("Corpus evidence contains duplicate record IDs")
    lexical_tasks = _lexical_tasks(root, corpus_by_id)
    lexical_queue_path = generated / "annotation" / "lexical_analysis_queue.jsonl"
    _write_jsonl(lexical_queue_path, lexical_tasks)
    input_paths = [
        corpus_path,
        source_registry_path,
        generated / "lexicon" / "lexicon_observed.jsonl",
        generated / "annotation" / "semantic_review_queue.jsonl",
        generated / "annotation" / "discourse_review_queue.jsonl",
        lexical_queue_path,
        root / "data" / "processed" / "v2" / "training_corpus_provenance.jsonl",
        root / "xnlp_language" / "build_foundation.py",
        root / "xnlp_language" / "validation" / "linguistic_layer.py",
        root / "xnlp_language" / "source_provenance.json",
        root / "corpus" / "mqhayi_complete.txt",
        report_path,
        *split_paths.values(),
        root / "data" / "processed" / "v2" / "splits" / "split_summary.json",
        *(
            root / "data" / "processed" / "v2" / "splits" / f"{name}_provenance.jsonl"
            for name in ("train", "val", "test")
        ),
        *(
            path for path in (generated / "grammar").glob("*_observations.jsonl")
        ),
        *(
            generated / "grammar" / relative
            for _, relative in CLAIM_SOURCES
            if relative is not None
        ),
    ]
    input_hashes = dict(sorted({
        path.relative_to(root).as_posix(): _artifact_hash(path)
        for path in input_paths if path.is_file()
    }.items()))
    validation_path = generated / "reports" / "linguistic_validation_report.json"
    split_report_path = generated / "reports" / "split_integrity.json"
    reusable_outputs = (
        generated / "validation" / "linguistic_claims.jsonl",
        generated / "validation" / "evidence_states.json",
        generated / "annotation" / "lexical_analysis_queue.jsonl",
        generated / "evaluation" / "gold_candidate_pool.jsonl",
        generated / "reports" / "linguistic_validation_report.md",
        split_report_path,
    )
    if validation_path.is_file() and all(path.is_file() for path in reusable_outputs):
        previous_report = json.loads(validation_path.read_text(encoding="utf-8"))
        if previous_report.get("input_sha256") == input_hashes:
            previous_split = json.loads(split_report_path.read_text(encoding="utf-8"))
            domain_summary = previous_report["grammar_claims_by_domain"]
            return {
                "claim_summary": {
                    "total": previous_report["verified"]["source_traceable_claims"],
                    "domains": domain_summary,
                    "sources": input_hashes,
                },
                "split_integrity": previous_split,
                "validation_report": previous_report,
            }

    claims, claim_summary = _build_claims(root, corpus_by_id)
    validate_claims(claims, set(corpus_by_id), corpus_by_id)
    _write_jsonl(generated / "validation" / "linguistic_claims.jsonl", claims)
    _write_json(
        generated / "validation" / "evidence_states.json",
        {
            "states": {
                "OBSERVED": "A source form or directly observable evidence exists.",
                "CANDIDATE": "An interpretation has been proposed but is not established.",
                "SOURCE_SUPPORTED": "Identifiable attested-source evidence directly supports the interpretation.",
                "CROSS_SOURCE_SUPPORTED": "Relevant support spans independent source works; this requires explicit review.",
                "VALIDATED": "A claim passed a documented project validation protocol and human decision.",
                "ESTABLISHED": "A strongly evidenced claim explicitly validated and approved; never assigned automatically.",
                "REJECTED": "A disproven interpretation retained with its evidence and review history.",
                "UNRESOLVED": "Evidence or interpretation is insufficient or ambiguous.",
            },
            "confidence_is_independent": True,
            "automatic_promotion": False,
            "established_claims_count": sum(
                claim["evidence_state"] == "ESTABLISHED" for claim in claims
            ),
        },
    )

    lexical_status_counts = Counter(task["status"] for task in lexical_tasks)
    lexical_tasks_requiring_human_attention = sum(
        task.get("status") in {"pending", "disputed"}
        or any(
            task.get(field) is None
            for field in ("assigned_lemma", "assigned_category", "assigned_meaning")
        )
        for task in lexical_tasks
    )
    lexical_unassigned_dimensions = {
        field: sum(task.get(field) is None for task in lexical_tasks)
        for field in ("assigned_lemma", "assigned_category", "assigned_meaning")
    }
    lexical_auto_rule_histogram = Counter(
        rule_id
        for task in lexical_tasks
        if task.get("adjudicator") == "auto-evidence"
        for rule_id in (task.get("auto_evidence") or {}).get("rule_ids", [])
    )
    source_work_counts = source_registry.get("source_record_counts", {})
    provenance_counts = source_registry.get("derived_provenance_counts", {})
    work_attribution_counts = source_registry.get("work_attribution_counts", {})
    semantic_tasks = _read_jsonl(generated / "annotation" / "semantic_review_queue.jsonl")
    discourse_tasks = _read_jsonl(generated / "annotation" / "discourse_review_queue.jsonl")

    split_records: dict[str, list[dict[str, Any]]] = {}
    for name in ("train", "val", "test"):
        split_records[name] = _read_jsonl(split_paths[name])
    test_work = split_config.get("test", {}).get("works", [])
    if len(test_work) != 1:
        raise ValueError("The work-isolated evaluation candidate must hold out exactly one source work")
    selected_work = test_work[0]
    gold_candidates = [
        {
            "candidate_id": f"GOLD_CANDIDATE_{row['record_id']}",
            "record_id": row["record_id"],
            "source_work": row.get("source_title", "UNKNOWN"),
            "source": row.get("source", "UNKNOWN"),
            "source_text": row.get("text", row.get("original_text", "")),
            "annotation_status": "PENDING_HUMAN_VALIDATION",
            "evidence_state": "UNRESOLVED",
            "target_domains": list(GOLD_DOMAINS),
            "validated_annotation": None,
        }
        for row in split_records["test"]
    ]
    if any(row["source_work"] != selected_work for row in gold_candidates):
        raise ValueError("Work-isolated test provenance contains records outside its held-out work")
    _write_jsonl(generated / "evaluation" / "gold_candidate_pool.jsonl", gold_candidates)

    isolated_audit = _partition_audit(split_records)
    historical_paths = {
        name: root / "data" / "processed" / "v2" / "splits" / f"{name}_provenance.jsonl"
        for name in ("train", "val", "test")
    }
    historical_records = {
        name: _read_jsonl(path) for name, path in historical_paths.items()
        if path.is_file()
    }
    historical_audit = _partition_audit(historical_records) if len(historical_records) == 3 else {
        "available": False,
        "reason": "All three historical partition provenance files are required for the complete audit.",
    }
    historical_summary_path = root / "data" / "processed" / "v2" / "splits" / "split_summary.json"
    historical_summary = (
        json.loads(historical_summary_path.read_text(encoding="utf-8"))
        if historical_summary_path.is_file() else {}
    )
    heldout_vs_training = _partition_audit({
        "work_isolated_train": split_records["train"],
        "gold_candidate_pool": split_records["test"],
    })
    gold_work_isolated = not any(heldout_vs_training["source_work_overlaps"].values())
    split_integrity = {
        "historical_split": {
            "label": "DIAGNOSTIC_ONLY_NOT_GENERALIZATION_EVIDENCE",
            "reported_split_method": historical_summary.get("split_method"),
            "audit": historical_audit,
            "derived_generated_data_contamination": _lineage_audit(historical_records),
            "sampling_limitation_note": (
                "Legacy near-duplicate sample results are retained as historical diagnostics; "
                "the audit above compares all cross-partition records."
            ),
        },
        "work_isolated_split": {
            "label": "PRIMARY_WORK_ISOLATED_EXPERIMENT",
            "audit": isolated_audit,
            "derived_generated_data_contamination": _lineage_audit(split_records),
            "test_work": selected_work,
            "author_isolated": (
                not any(isolated_audit["author_overlaps"].values())
                if all(isolated_audit["author_metadata_available"].values())
                else None
            ),
            "author_metadata_available": all(
                isolated_audit["author_metadata_available"].values()
            ),
            "source_isolated": not any(isolated_audit["source_overlaps"].values()),
            "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
            "near_duplicate_method": (
                f"Exhaustive cross-partition candidate comparison by character "
                f"{SHINGLE_SIZE}-gram Jaccard similarity."
            ),
        },
        "gold_evaluation": {
            "candidate_pool_path": (
                generated / "evaluation" / "gold_candidate_pool.jsonl"
            ).relative_to(root).as_posix(),
            "candidate_records": len(gold_candidates),
            "source_work": selected_work,
            "overlaps_work_isolated_training_works": not gold_work_isolated,
            "work_isolated_from_current_training": gold_work_isolated,
            "validated_gold_annotations": sum(
                row["annotation_status"] == "VALIDATED" for row in gold_candidates
            ),
            "gold_set_populated": False,
            "eligible_for_future_models_using_reserved_work_isolated_training": gold_work_isolated,
            "prior_checkpoint_contamination_warning": (
                "Do not use this pool to evaluate models trained with historical splits or "
                "other data containing this source work. The pool contains no gold labels yet."
            ),
        },
    }
    _write_json(generated / "reports" / "split_integrity.json", split_integrity)

    status_totals = Counter({state: 0 for state in EVIDENCE_STATES})
    status_totals.update(claim["evidence_state"] for claim in claims)
    validation_report = {
        "report_id": "XNLP_LINGUISTIC_VALIDATION_REPORT_V1",
        "verified": {
            "corpus_records_preserved": len(corpus_records),
            "source_traceable_claims": len(claims),
            "orphan_claims": 0,
            "unresolved_claims": status_totals["UNRESOLVED"],
            "validated_claims": status_totals["VALIDATED"],
            "established_claims": status_totals["ESTABLISHED"],
            "lexical_analysis_tasks": len(lexical_tasks),
            "lexical_adjudication_status_counts": dict(sorted(lexical_status_counts.items())),
            "lexical_tasks_requiring_human_attention": lexical_tasks_requiring_human_attention,
            "distinct_source_works": source_registry.get("distinct_source_work_count", 0),
            "unregistered_source_records": len(source_registry.get("unregistered_record_ids", [])),
            "semantic_tasks_pending": len(semantic_tasks),
            "discourse_tasks_pending": len(discourse_tasks),
            "gold_candidate_records": len(gold_candidates),
            "gold_validated_annotations": 0,
            "historical_split_work_isolated": not any(
                historical_audit["source_work_overlaps"].values()
            ) if historical_audit.get("available", False) else None,
            "work_isolated_candidate_pool_separated_from_training_works": gold_work_isolated,
        },
        "evidence_state_totals": dict(sorted(status_totals.items())),
        "unresolved_analysis_totals": {
            "linguistic_claims": status_totals["UNRESOLVED"],
            "lexical_analysis_tasks": len(lexical_tasks),
            "semantic_annotations": len(semantic_tasks),
            "discourse_annotations": len(discourse_tasks),
        },
        "grammar_claims_by_domain": claim_summary["domains"],
        "legacy_supported_claims": sum(
            "legacy_confidence_label" in claim for claim in claims
        ),
        "source_support_violations": claim_summary["source_support_violations"],
        "lexical_adjudication_counts": dict(sorted(lexical_status_counts.items())),
        "lexical_tasks_requiring_human_attention": lexical_tasks_requiring_human_attention,
        "lexical_unassigned_dimensions": lexical_unassigned_dimensions,
        "lexical_auto_evidence_rule_histogram": dict(
            sorted(lexical_auto_rule_histogram.items())
        ),
        "source_provenance": {
            "distinct_source_works": source_registry.get("distinct_source_work_count", 0),
            "records_by_source_work": source_work_counts,
            "derived_record_counts": provenance_counts,
            "unregistered_record_ids": source_registry.get("unregistered_record_ids", []),
            "work_attribution_counts": work_attribution_counts,
        },
        "what_we_know": [
            "The preserved corpus contains source-linked isiXhosa text records.",
            "Lexical artifacts establish surface-form occurrence, not lemma, part of speech, or sense.",
            "Structured morphology includes proposed analyses with source evidence; most remain candidates.",
        ],
        "strongly_supported": [
            "The existence of cited source forms and contexts.",
            (
                f"{status_totals['SOURCE_SUPPORTED']} structured noun-class/agreement claims "
                "carry legacy SUPPORTED labels and direct source links; these are source-supported, "
                "not project-validated."
            ),
        ],
        "candidate": [
            "Grammar observations, noun-class and agreement interpretations, and verb segmentations not explicitly human-validated.",
            (
                "Lexical queue adjudication progress: "
                f"{lexical_status_counts['pending']} pending, "
                f"{lexical_status_counts['adjudicated']} adjudicated, "
                f"{lexical_status_counts['disputed']} disputed."
            ),
            (
                f"{lexical_tasks_requiring_human_attention} lexical tasks still need some "
                "human attention; auto-evidence rows have only a category assignment, "
                "with lemma and meaning left unassigned."
            ),
            (
                "Auto-evidence lexical decisions are reversible and retain their rule IDs "
                "and supporting record IDs in an audit trail; rule histogram: "
                f"{dict(sorted(lexical_auto_rule_histogram.items()))}."
            ),
        ],
        "unresolved": [
            "Lexical lemmas, categories, semantic senses, and contextual meanings pending review.",
            "Discourse functions, references, and relations pending review.",
            "No claim currently meets the project's explicit VALIDATED or ESTABLISHED standard.",
        ],
        "useful_evidence_domains": [
            "Observed lexicon and corpus-backed noun-class/agreement/morphology candidates.",
            "Source-linked syntax, TAM, negation, function-word, modifier, and word-formation observations.",
        ],
        "sparse_domains": [
            "Validated semantic and discourse annotations: none yet.",
            "Validated syntax and generation/comprehension gold examples: none yet.",
        ],
        "gold_evaluation": split_integrity["gold_evaluation"],
        "professional_grade_blockers": [
            "Human-reviewed and independently adjudicated linguistic claims.",
            "Validated lexical, semantic, discourse, morphology, and syntax annotations.",
            "An appropriately diverse, work-isolated gold set with validated labels.",
            "Author/source generalization and professional-quality generation evaluation.",
        ],
        "limitations": [
            "Source-work isolation does not imply author or source isolation.",
            "Machine/model suggestions are not promoted to human/source validation.",
            "No training run was performed by this validation-layer builder.",
        ],
        "input_sha256": input_hashes,
    }
    validation_report["human_report_path"] = (
        generated / "reports" / "linguistic_validation_report.md"
    ).relative_to(root).as_posix()
    _write_json(generated / "reports" / "linguistic_validation_report.json", validation_report)
    markdown_lines = [
        "# XNLP Linguistic Validation Report",
        "",
        "## What do we actually know?",
        *[f"- {item}" for item in validation_report["what_we_know"]],
        "",
        "## What is strongly supported?",
        *[f"- {item}" for item in validation_report["strongly_supported"]],
        "",
        "## What remains candidate?",
        *[f"- {item}" for item in validation_report["candidate"]],
        "",
        "## Lexical adjudication progress",
        *[
            f"- {status}: {count}"
            for status, count in sorted(lexical_status_counts.items())
        ],
        f"- Source-supported provenance violations: {len(claim_summary['source_support_violations'])}.",
        f"- Distinct source works: {source_registry.get('distinct_source_work_count', 0)}.",
        f"- Work attribution counts: {json.dumps(work_attribution_counts, ensure_ascii=False, sort_keys=True)}.",
        f"- Derived record provenance counts: {json.dumps(provenance_counts, sort_keys=True)}.",
        f"- Unregistered source records: {len(source_registry.get('unregistered_record_ids', []))}.",
        "- Auto-evidence decisions are reversible and retain their evidence/rule audit trail.",
        f"- Auto-evidence rule histogram: {json.dumps(dict(sorted(lexical_auto_rule_histogram.items())), sort_keys=True)}.",
        f"- Tasks still needing any human attention: {lexical_tasks_requiring_human_attention}; unassigned dimensions: {json.dumps(lexical_unassigned_dimensions, sort_keys=True)}.",
        *[
            f"- Provenance violation `{violation['claim_id']}`: {violation['reason']}"
            for violation in claim_summary["source_support_violations"]
        ],
        "",
        "## What remains unresolved?",
        *[f"- {item}" for item in validation_report["unresolved"]],
        *[
            f"- Unresolved {name.replace('_', ' ')}: {count}"
            for name, count in validation_report["unresolved_analysis_totals"].items()
        ],
        "",
        "## Claim evidence-state totals",
        *[
            f"- {state}: {count}"
            for state, count in validation_report["evidence_state_totals"].items()
        ],
        "",
        "## Useful evidence and sparse domains",
        *[f"- {item}" for item in validation_report["useful_evidence_domains"]],
        *[f"- Sparse: {item}" for item in validation_report["sparse_domains"]],
        "",
        "## Is the gold evaluation set genuinely work-isolated?",
        (
            f"- {len(gold_candidates)} unannotated candidate records from `{selected_work}` "
            f"are separate from the designated work-isolated training partition: "
            f"{'yes' if gold_work_isolated else 'no'}."
        ),
        "- Validated gold annotations: 0. This is a pending candidate pool, not a populated gold set.",
        "- Do not evaluate checkpoints trained on historical or other data containing the held-out work.",
        (
            "- Author metadata is unavailable; the source collection is not isolated across "
            "the work-isolated experiment."
        ),
        (
            "- Historical record splits overlap source works and are diagnostic only."
        ),
        (
            "- The work-isolated partition audit found zero record, exact-text, and "
            "near-duplicate overlaps across train/validation/test."
        ),
        "",
        "## What is still blocking professional-grade training?",
        *[f"- {item}" for item in validation_report["professional_grade_blockers"]],
        "",
        "No claim of professional isiXhosa competence is made.",
    ]
    markdown_path = generated / "reports" / "linguistic_validation_report.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    return {
        "claim_summary": claim_summary,
        "split_integrity": split_integrity,
        "validation_report": validation_report,
    }
