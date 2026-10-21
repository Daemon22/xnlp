#!/usr/bin/env python3
"""Build a compact, provenance-linked index of the complete XNLP foundation."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from xnlp_language.validation.linguistic_layer import build_linguistic_layer

ROOT = Path(__file__).resolve().parents[1]
GENERATED = Path("xnlp_language/generated")
BUNDLE_PATH = GENERATED / "foundation_bundle.json"

CORE_GRAMMAR_FILES = {
    "grammar_rules": "grammar/grammar_rules.json",
    "noun_classes": "grammar/noun_classes.jsonl",
    "agreement": "grammar/agreement_system.jsonl",
    "verb_morphology": "grammar/verb_morphology.jsonl",
    "derivational_morphology": "grammar/derivational_morphology.jsonl",
}
OBSERVATION_GLOB = "grammar/*_observations.jsonl"
REVIEW_FILES = {
    "general_linguistic": "annotation/linguistic_review_queue.jsonl",
    "lexical_analysis": "annotation/lexical_analysis_queue.jsonl",
    "noun_class": "grammar/noun_class_review_queue.jsonl",
    "agreement": "grammar/agreement_review_queue.jsonl",
    "verb_analysis": "grammar/verb_analysis_queue.jsonl",
    "semantic": "annotation/semantic_review_queue.jsonl",
    "discourse": "annotation/discourse_review_queue.jsonl",
}
EVALUATION_REPORTS = (
    "data/reports/mqhayi_checkpoint_verification.json",
    "data/reports/mqhayi_work_isolated_comparison.json",
)
EXPERIMENT_REPORTS = (
    "artifacts/xnlp_prose_pilot/training_report.json",
    "artifacts/xnlp_prose_pilot_linebreaks/training_report.json",
    "artifacts/xnlp_prose_pilot_packed_context/training_report.json",
    "artifacts/xnlp_prose_pilot_work_isolated/training_report.json",
)
LINKED_INPUTS = (
    "data/processed/v2/corpus_manifest.json",
    "data/processed/v2/splits/split_summary.json",
    "xnlp_language/schemas/semantic_frame_schema.json",
    "xnlp_language/schemas/discourse_annotation_schema.json",
    "xnlp_language/schemas/linguistic_claim_schema.json",
    "xnlp_language/schemas/lexical_analysis_task_schema.json",
    "xnlp_language/schemas/morphology_schema.json",
    "xnlp_language/annotation/adjudicate_lexical.py",
    "train_prose_pilot.py",
    "evaluate_mqhayi_checkpoint.py",
    "compare_mqhayi_checkpoints.py",
    "xnlp_trainer/data.py",
    "core_llm/tokenizer.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_structured(path: Path) -> Any:
    if path.suffix == ".jsonl":
        return _load_jsonl(path)
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return path.read_text(encoding="utf-8")


def _artifact_entry(root: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    entry: dict[str, Any] = {
        "path": relative,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".jsonl":
        entry["records"] = sum(
            1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    elif path.suffix == ".json":
        content = json.loads(path.read_text(encoding="utf-8"))
        entry["records"] = len(content) if isinstance(content, list) else None
    return entry


def _walk_record_references(value: Any) -> Iterable[str]:
    record_keys = {
        "record_id",
        "source_record_id",
        "source_record_ids",
        "evidence_record_ids",
        "evidence_records",
        "record_ids",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if key in record_keys:
                if isinstance(child, str):
                    yield child
                elif isinstance(child, list):
                    yield from (item for item in child if isinstance(item, str))
            yield from _walk_record_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_record_references(child)


def _summarize_report(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file():
        return {"path": relative_path, "available": False}
    report = json.loads(path.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "path": relative_path,
        "available": True,
        "sha256": _sha256(path),
    }
    if "experiment" in report:
        summary.update({
            "experiment": report.get("experiment"),
            "training_status": report.get("training_status"),
            "preset": report.get("preset"),
            "parameters": report.get("parameters"),
            "best_epoch": report.get("best_epoch"),
            "best_validation_loss": report.get("best_validation_loss"),
            "validation_perplexity": report.get("validation_perplexity"),
            "checkpoint_sha256": report.get("checkpoint_sha256"),
            "release_note": report.get("release_note"),
        })
        split_data = report.get("training_data", {})
        if "split_policy" in split_data:
            summary["split_policy"] = split_data["split_policy"]
        if "splits" in split_data:
            summary["splits"] = {
                name: {
                    key: split.get(key)
                    for key in ("records", "works", "text_sha256")
                }
                for name, split in split_data["splits"].items()
            }
        evaluation = report.get("mqhayi_holdout_evaluation", {})
        selection = evaluation.get("selection", {})
        summary["heldout_evaluation"] = {
            "source": selection.get("source"),
            "records": selection.get("sample_size"),
            "test_work": split_data.get("test_work"),
            "exact_overlaps_excluded": selection.get(
                "exact_training_text_overlaps_excluded"
            ),
            "tokenizer_round_trip": evaluation.get("tokenizer_round_trip", {}).get(
                "passed"
            ),
            "next_token_loss": evaluation.get(
                "next_token_evaluation", {}
            ).get("cross_entropy_loss"),
            "perplexity": evaluation.get(
                "next_token_evaluation", {}
            ).get("perplexity"),
        }
    else:
        summary.update({
            "experiment": report.get("experiment"),
            "interpretation": report.get("interpretation"),
            "record_count": len(report.get("shared_sample_record_ids", [])),
            "evaluations": {
                name: {
                    "checkpoint_sha256": result.get("checkpoint_sha256"),
                    "records": result.get("selection", {}).get("sample_size"),
                    "exact_overlaps_excluded": result.get("selection", {}).get(
                        "exact_training_text_overlaps_excluded"
                    ),
                    "tokenizer_round_trip": result.get(
                        "tokenizer_round_trip", {}
                    ).get("passed"),
                    "next_token_loss": result.get(
                        "next_token_evaluation", {}
                    ).get("cross_entropy_loss"),
                    "perplexity": result.get(
                        "next_token_evaluation", {}
                    ).get("perplexity"),
                }
                for name, result in report.get("evaluations", {}).items()
            },
        })
        if "next_token_evaluation" in report:
            summary["checkpoint_evaluation"] = {
                "checkpoint_sha256": report.get("checkpoint_sha256"),
                "checkpoint_epoch": report.get("checkpoint_epoch"),
                "records": report.get("selection", {}).get("sample_size"),
                "exact_overlaps_excluded": report.get("selection", {}).get(
                    "exact_training_text_overlaps_excluded"
                ),
                "tokenizer_round_trip": report.get(
                    "tokenizer_round_trip", {}
                ).get("passed"),
                "next_token_loss": report.get(
                    "next_token_evaluation", {}
                ).get("cross_entropy_loss"),
                "perplexity": report.get(
                    "next_token_evaluation", {}
                ).get("perplexity"),
            }
    return summary


def build_foundation_bundle(
    root: Path = ROOT,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Build an auditable bundle index without copying large source datasets."""
    root = root.resolve()
    generated_dir = root / GENERATED
    manifest_path = generated_dir / "foundation_manifest.json"
    corpus_path = generated_dir / "evidence" / "corpus_records.jsonl"
    coverage_path = generated_dir / "reports" / "structural_coverage.json"
    required = (manifest_path, corpus_path, coverage_path)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required foundation inputs are missing: "
            + ", ".join(str(path) for path in missing)
        )

    linguistic_layer = build_linguistic_layer(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_records = _load_jsonl(corpus_path)
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    corpus_ids = {record["record_id"] for record in corpus_records}
    if len(corpus_ids) != len(corpus_records):
        raise ValueError("Corpus evidence contains duplicate record_id values")

    corpus_sources = Counter(record.get("source", "UNKNOWN") for record in corpus_records)
    language_classes = Counter(
        record.get("language_class", "UNKNOWN") for record in corpus_records
    )
    orthographies = Counter(
        record.get("orthography", "UNKNOWN") for record in corpus_records
    )
    source_works = Counter(
        record.get("source_title", "UNKNOWN") for record in corpus_records
    )
    provenance_path = generated_dir / "evidence" / "source_provenance.json"
    source_provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_file() else {}
    )
    derivation_spec_path = generated_dir / "derivation" / "language_specification.json"
    derivation_report_path = generated_dir / "reports" / "language_derivation_report.md"
    derivation_spec = (
        json.loads(derivation_spec_path.read_text(encoding="utf-8"))
        if derivation_spec_path.is_file() else None
    )
    grammar_spec_path = generated_dir / "grammar_derivation" / "grammar_specification.json"
    grammar_report_path = generated_dir / "grammar_derivation" / "grammar_derivation_report.md"
    grammar_spec = (
        json.loads(grammar_spec_path.read_text(encoding="utf-8"))
        if grammar_spec_path.is_file() else None
    )
    corpus_text_field = (
        "text" if corpus_records and "text" in corpus_records[0] else "original_text"
    )
    if any(corpus_text_field not in record for record in corpus_records):
        raise ValueError(f"Corpus records do not consistently contain {corpus_text_field}")

    generated_files = sorted(
        path for path in generated_dir.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != BUNDLE_PATH.as_posix()
    )
    artifact_catalog = [
        _artifact_entry(root, path) for path in generated_files
    ]
    catalog_by_path = {entry["path"]: entry for entry in artifact_catalog}
    evidence_references = Counter()
    structured_grammar = {}
    for name, relative in CORE_GRAMMAR_FILES.items():
        artifact_path = GENERATED / relative
        path = root / artifact_path
        if not path.is_file():
            raise FileNotFoundError(f"Required grammar artifact is missing: {path}")
        data = _load_structured(path)
        entries = data if isinstance(data, list) else [data]
        structured_grammar[name] = {
            "artifact": catalog_by_path[artifact_path.as_posix()],
            "entries": data,
        }
        for entry in entries:
            evidence_references.update(_walk_record_references(entry))

    observation_files = sorted(generated_dir.glob(OBSERVATION_GLOB))
    observations = {}
    for path in observation_files:
        relative = path.relative_to(root).as_posix()
        entries = _load_jsonl(path)
        observations[relative] = {
            "artifact": catalog_by_path[relative],
            "entries": entries,
        }
        for entry in entries:
            evidence_references.update(_walk_record_references(entry))

    review_work = {}
    for name, relative in REVIEW_FILES.items():
        path = generated_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required review queue is missing: {path}")
        rows = _load_jsonl(path)
        empty_analysis = sum(
            not row.get("proposed_analysis")
            or row.get("proposed_analysis") in ({}, [], "")
            for row in rows
        )
        review_work[name] = {
            "artifact": catalog_by_path[path.relative_to(root).as_posix()],
            "tasks": len(rows),
            "unanalysed_tasks": empty_analysis,
        }
        for row in rows:
            evidence_references.update(_walk_record_references(row))

    unresolved_references = sorted(
        reference for reference in evidence_references if reference not in corpus_ids
    )
    domain_coverage = []
    for domain in coverage.get("domains", []):
        domain_coverage.append({
            "domain": domain.get("domain"),
            "entries": domain.get("entries"),
            "coverage_status": domain.get("coverage_status"),
            "confidence": domain.get("confidence"),
            "review_tasks": domain.get("review_tasks"),
            "notes": domain.get("notes"),
        })

    split_summary_path = root / "data" / "processed" / "v2" / "splits" / "split_summary.json"
    historical_split = {"available": False}
    if split_summary_path.is_file():
        legacy_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
        by_work = {
            name: set(legacy_summary.get(f"by_work_{name}", {}))
            for name in ("train", "val", "test")
        }
        work_overlaps = {
            "train_validation": sorted(by_work["train"] & by_work["val"]),
            "train_test": sorted(by_work["train"] & by_work["test"]),
            "validation_test": sorted(by_work["val"] & by_work["test"]),
        }
        historical_split = {
            "available": True,
            "path": split_summary_path.relative_to(root).as_posix(),
            "reported_method": legacy_summary.get("split_method"),
            "whole_work_isolation_verified": not any(work_overlaps.values()),
            "work_overlaps": work_overlaps,
            "warning": (
                "The historical split places source works in multiple partitions; "
                "its reported work-level isolation must not be relied on."
            ) if any(work_overlaps.values()) else None,
        }

    eval_reports = {
        path: _summarize_report(root, path)
        for path in EVALUATION_REPORTS
    }
    experiment_reports = {
        path: _summarize_report(root, path)
        for path in EXPERIMENT_REPORTS
    }
    linked_inputs = []
    for relative_path in LINKED_INPUTS:
        path = root / relative_path
        if path.is_file():
            linked_inputs.append(_artifact_entry(root, path))
        else:
            linked_inputs.append({"path": relative_path, "available": False})
    artifact_catalog.extend(
        item for item in linked_inputs
        if item.get("available", True) and item.get("path") not in catalog_by_path
    )
    lexical_catalog = catalog_by_path.get(
        (GENERATED / "lexicon" / "lexicon_observed.jsonl").as_posix()
    )
    semantic_review = review_work["semantic"]
    discourse_review = review_work["discourse"]
    lexical_review = review_work["lexical_analysis"]
    claim_summary = linguistic_layer["claim_summary"]
    validation_report = linguistic_layer["validation_report"]
    bundle = {
        "bundle_id": "XNLP_ISIXHOSA_EVIDENCE_FOUNDATION_BUNDLE_V1",
        "bundle_version": "1.0.0",
        "purpose": (
            "A single, traceable entry point joining preserved corpus evidence, "
            "observed lexicon, structured grammar proposals, annotation work, "
            "validation, and model experiments."
        ),
        "release_state": {
            "decision": manifest.get("decision"),
            "model_training": manifest.get("model_training"),
            "established_grammar_rule_count": validation_report[
                "evidence_state_totals"
            ].get("ESTABLISHED", 0),
            "warning": (
                "The corpus supplies strong examples, not a complete grammar by itself. "
                "Current structured grammar entries remain supported, observed, "
                "provisional, or under review; no entry is represented here as "
                "peer-reviewed established grammar."
            ),
        },
        "source_corpus": {
            "manifest_path": "data/processed/v2/corpus_manifest.json",
            "manifest": json.loads(
                (root / "data" / "processed" / "v2" / "corpus_manifest.json")
                .read_text(encoding="utf-8")
            ),
            "foundation_corpus_sha256": manifest.get("corpus_sha256"),
            "records_preserved": len(corpus_records),
            "eligible_analysis_records": manifest.get("analysis_records"),
            "records_by_source": dict(sorted(corpus_sources.items())),
            "records_by_language_class": dict(sorted(language_classes.items())),
            "records_by_orthography": dict(sorted(orthographies.items())),
            "source_work_count": len(source_works),
            "source_provenance_registry": (
                catalog_by_path[provenance_path.relative_to(root).as_posix()]
                if provenance_path.is_file()
                else None
            ),
            "distinct_source_work_count": source_provenance.get(
                "distinct_source_work_count", len(source_works)
            ),
            "records_by_source_work": source_provenance.get(
                "source_record_counts", dict(sorted(source_works.items()))
            ),
            "derived_provenance_counts": source_provenance.get(
                "derived_provenance_counts", {}
            ),
            "unregistered_source_record_count": len(
                source_provenance.get("unregistered_record_ids", [])
            ),
            "work_attribution_counts": source_provenance.get(
                "work_attribution_counts", {}
            ),
            "largest_source_works": [
                {"source_title": title, "records": count}
                for title, count in source_works.most_common(12)
            ],
            "evidence_artifact": catalog_by_path[corpus_path.relative_to(root).as_posix()],
            "text_field": corpus_text_field,
            "preservation_policy": "Original source text and orthography are retained unchanged.",
        },
        "language_derivation": (
            {
                "status": derivation_spec.get("status"),
                "specification_version": derivation_spec.get("specification_version"),
                "evidence_set_id": derivation_spec["DERIVATION_EVIDENCE_SET"]["id"],
                "evidence_record_count": derivation_spec["DERIVATION_EVIDENCE_SET"]["record_count"],
                "specification_artifact": catalog_by_path[
                    derivation_spec_path.relative_to(root).as_posix()
                ],
                "human_report_artifact": (
                    catalog_by_path[derivation_report_path.relative_to(root).as_posix()]
                    if derivation_report_path.is_file() else None
                ),
                "linguistic_generalizations_promoted_to_supported": (
                    derivation_spec["summary"]["linguistic_generalizations_promoted_to_supported"]
                ),
            }
            if derivation_spec is not None else None
        ),
        "formal_grammar_derivation": (
            {
                "status": grammar_spec.get("status"),
                "specification_version": grammar_spec.get("specification_version"),
                "evidence_set_id": grammar_spec["evidence_set"]["id"],
                "evidence_record_count": grammar_spec["evidence_set"]["record_count"],
                "supported_grammar_rule_count": grammar_spec["summary"][
                    "supported_grammar_rule_count"
                ],
                "specification_artifact": catalog_by_path[
                    grammar_spec_path.relative_to(root).as_posix()
                ],
                "human_report_artifact": catalog_by_path[
                    grammar_report_path.relative_to(root).as_posix()
                ],
            }
            if grammar_spec is not None else None
        ),
        "lexicon": {
            "status": "OBSERVED_UNANALYSED",
            "entries": lexical_catalog.get("records") if lexical_catalog else 0,
            "artifact": lexical_catalog,
            "warning": (
                "Lexical entries are surface observations; lemma, part of speech, "
                "and semantic analysis remain unassigned unless an individual record "
                "explicitly says otherwise."
            ),
        },
        "grammar": {
            "domain_coverage": domain_coverage,
            "structured_entries": structured_grammar,
            "observations": observations,
            "open_conflicts": manifest.get("conflicts_open"),
            "evidence_link_audit": {
                "unique_cited_record_ids": len(evidence_references),
                "resolved_record_ids": len(evidence_references) - len(unresolved_references),
                "unresolved_record_ids": unresolved_references,
                "status": "PASS" if not unresolved_references else "REVIEW_REQUIRED",
            },
        },
        "linguistic_validation": {
            "evidence_states": json.loads(
                (generated_dir / "validation" / "evidence_states.json")
                .read_text(encoding="utf-8")
            ),
            "claims": {
                "total": claim_summary["total"],
                "domains": claim_summary["domains"],
                "artifact": catalog_by_path[
                    (GENERATED / "validation" / "linguistic_claims.jsonl").as_posix()
                ],
                "schema": "xnlp_language/schemas/linguistic_claim_schema.json",
            },
            "lexical_analysis": {
                "tasks": lexical_review["tasks"],
                "pending_human_review": lexical_review["tasks"],
                "lemma_or_category_auto_assigned": False,
                "artifact": catalog_by_path[
                    (GENERATED / "annotation" / "lexical_analysis_queue.jsonl").as_posix()
                ],
                "schema": "xnlp_language/schemas/lexical_analysis_task_schema.json",
            },
            "morphology": {
                "candidate_claims": sum(
                    count["total"]
                    for domain, count in claim_summary["domains"].items()
                    if domain in {"morphology", "derivational_morphology"}
                ),
                "analyses_require_human_validation": True,
                "alternatives_preserved_in_source_artifacts": True,
            },
            "unresolved_analysis_totals": validation_report[
                "unresolved_analysis_totals"
            ],
            "evaluation": {
                "candidate_pool": catalog_by_path[
                    (GENERATED / "evaluation" / "gold_candidate_pool.jsonl").as_posix()
                ],
                "gold_validated_annotations": validation_report["verified"][
                    "gold_validated_annotations"
                ],
                "gold_set_populated": False,
            },
            "split_integrity": catalog_by_path[
                (GENERATED / "reports" / "split_integrity.json").as_posix()
            ],
            "validation_report": {
                **validation_report,
                "machine_readable_artifact": catalog_by_path[
                    (GENERATED / "reports" / "linguistic_validation_report.json").as_posix()
                ],
                "human_readable_artifact": catalog_by_path[
                    (GENERATED / "reports" / "linguistic_validation_report.md").as_posix()
                ],
            },
            "status": (
                "PASS_WITH_UNRESOLVED_REVIEW_WORK"
                if not unresolved_references
                else "REVIEW_REQUIRED"
            ),
        },
        "annotation_work": {
            "tasks_total": manifest.get("review_tasks_total"),
            "semantic": {
                **review_work["semantic"],
                "schema": "xnlp_language/schemas/semantic_frame_schema.json",
                "schema_artifact": next(
                    (item for item in linked_inputs
                     if item["path"] == "xnlp_language/schemas/semantic_frame_schema.json"),
                    {"path": "xnlp_language/schemas/semantic_frame_schema.json", "available": False},
                ),
                "all_analysis_fields_empty": (
                    semantic_review["unanalysed_tasks"] == semantic_review["tasks"]
                ),
            },
            "lexical_analysis": {
                **lexical_review,
                "all_lemmas_unassigned": True,
            },
            "discourse": {
                **review_work["discourse"],
                "schema": "xnlp_language/schemas/discourse_annotation_schema.json",
                "schema_artifact": next(
                    (item for item in linked_inputs
                     if item["path"] == "xnlp_language/schemas/discourse_annotation_schema.json"),
                    {"path": "xnlp_language/schemas/discourse_annotation_schema.json", "available": False},
                ),
                "all_analysis_fields_empty": (
                    discourse_review["unanalysed_tasks"] == discourse_review["tasks"]
                ),
            },
            "queues": review_work,
        },
        "split_integrity": {
            "historical_record_split": historical_split,
            "work_isolated_experiment": experiment_reports[
                "artifacts/xnlp_prose_pilot_work_isolated/training_report.json"
            ],
        },
        "model_experiments": {
            "training_runs": experiment_reports,
            "evaluations": eval_reports,
            "model_checkpoint_binaries_included": False,
            "quality_boundary": (
                "Training loss, tokenizer round trips, and next-token metrics do not "
                "establish grammaticality or professional writing quality."
            ),
        },
        "artifact_catalog": artifact_catalog,
        "linked_inputs": linked_inputs,
        "license_and_rights": {
            "manifest_source_tiers": "data/processed/v2/corpus_manifest.json",
            "note": (
                "Preserve each source's stated license and attribution. The user has "
                "reported having additional model-training rights; this bundle does "
                "not independently verify those rights or broaden reuse permissions."
            ),
        },
    }
    output_path = output_path or (root / BUNDLE_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / BUNDLE_PATH)
    args = parser.parse_args()
    bundle = build_foundation_bundle(output_path=args.output)
    print(f"Foundation bundle: {args.output}")
    print(f"Preserved records: {bundle['source_corpus']['records_preserved']}")
    print(f"Observed lexicon entries: {bundle['lexicon']['entries']}")
    print(f"Grammar domains: {len(bundle['grammar']['domain_coverage'])}")
    print(f"Artifact catalog entries: {len(bundle['artifact_catalog'])}")
    print(
        "Evidence link audit: "
        f"{bundle['grammar']['evidence_link_audit']['status']} "
        f"({len(bundle['grammar']['evidence_link_audit']['unresolved_record_ids'])} unresolved)"
    )


if __name__ == "__main__":
    main()
