#!/usr/bin/env python3
"""Dependency-free integrity and provenance checks for generated foundation data."""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "xnlp_language" / "generated"

def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def main() -> None:
    schema_paths = sorted((ROOT / "xnlp_language" / "schemas").glob("*.json"))
    assert schema_paths, "no schema assets"
    for schema_path in schema_paths:
        assert isinstance(json.loads(schema_path.read_text(encoding="utf-8")), dict), f"invalid schema: {schema_path.name}"
    records = rows(GENERATED / "evidence" / "corpus_records.jsonl")
    record_ids = {r["record_id"] for r in records}
    assert records, "no corpus records"
    assert all(r["source"] in {"mqhayi", "masikhanyise"} for r in records), "non-principal source"
    assert all(r["provenance_type"] == r["validation_status"] == "authoritative" for r in records), "unverified record"
    assert len(record_ids) == len(records), "duplicate record ID"
    registry = json.loads((GENERATED / "validation" / "language_integrity_registry.json").read_text(encoding="utf-8"))
    assert len(registry) == len(records), "language-integrity registry does not preserve every record"
    source_by_id = {record["record_id"]: record for record in records}
    for item in registry:
        record = source_by_id[item["record_id"]]
        assert item["original_text_preserved"] is True
        import hashlib
        assert item["original_text_sha256"] == hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
        assert item["preservation_action"] == "RETAIN_UNCHANGED"
        assert (item["automated_linguistic_analysis"] == "INCLUDED") == (item["language_class"] == "TARGET_LANGUAGE")
    analysis_ids = {item["record_id"] for item in registry if item["automated_linguistic_analysis"] == "INCLUDED"}
    lexicon = rows(GENERATED / "lexicon" / "lexicon_observed.jsonl")
    for entry in lexicon:
        assert entry["analysis_status"] == "UNANALYZED"
        assert entry["evidence"] and all(e["record_id"] in analysis_ids for e in entry["evidence"])
    observed_lexicon = json.loads(
        (GENERATED / "foundation" / "observed_lexicon.json").read_text(encoding="utf-8")
    )
    observed_forms = observed_lexicon["forms"]
    assert observed_lexicon["analysis_records"] == len(analysis_ids)
    assert observed_lexicon["form_count"] == len(observed_forms)
    for entry in lexicon:
        entry_record_ids = {e["record_id"] for e in entry["evidence"]}
        for surface in {entry["lemma"], *entry["surface_forms"]}:
            indexed = observed_forms[surface.casefold()]
            assert indexed["lexeme_id"] == entry["lexeme_id"]
            assert set(indexed["evidence_record_ids"]).issubset(analysis_ids)
            assert set(indexed["evidence_record_ids"]).issubset(entry_record_ids)
    noun = json.loads((GENERATED / "foundation" / "noun_class_observations.json").read_text(encoding="utf-8"))
    for entry in noun:
        assert entry["analysis_status"] == "PROVISIONAL"
        assert entry["evidence"] and all(e["record_id"] in analysis_ids for e in entry["evidence"])
    review_tasks = rows(GENERATED / "annotation" / "linguistic_review_queue.jsonl")
    assert review_tasks, "no linguistic review tasks"
    for task in review_tasks:
        assert task["status"] == "PENDING_HUMAN_REVIEW"
        assert task["proposed_analysis"] is None
        assert task["confidence"] == "INSUFFICIENT_EVIDENCE"
        assert task["source_record_ids"] and set(task["source_record_ids"]).issubset(analysis_ids)
    contradictions = json.loads((GENERATED / "validation" / "contradictions.json").read_text(encoding="utf-8"))
    conflict_items = contradictions.get("items", contradictions.get("conflicts", []))
    for item in conflict_items:
        assert item["status"] == "OPEN"
        assert all(record_id in record_ids for record_id in item["record_ids"])
    manifest = json.loads((GENERATED / "foundation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_training"] == "FROZEN"
    semantic_senses = rows(GENERATED / "semantics" / "lexical_senses.jsonl")
    semantic_relations = rows(GENERATED / "semantics" / "semantic_relations.jsonl")
    semantic_profiles = json.loads(
        (GENERATED / "semantics" / "noun_class_profiles.json").read_text(encoding="utf-8")
    )["profiles"]
    semantic_report = json.loads(
        (GENERATED / "reports" / "semantic_coverage.json").read_text(encoding="utf-8")
    )
    assert semantic_report["corpus_sha256"] == manifest["corpus_sha256"]
    assert semantic_report["verified_sense_entries"] == len(semantic_senses)
    assert semantic_report["semantic_relation_entries"] == len(semantic_relations)
    assert semantic_report["noun_class_profiles"] == len(semantic_profiles)
    assert semantic_report["status"] in {"PARTIAL", "INSUFFICIENT_EVIDENCE"}
    sense_ids = {sense["sense_id"] for sense in semantic_senses}
    assert len(sense_ids) == len(semantic_senses), "duplicate semantic sense ID"
    for sense in semantic_senses:
        assert sense["review_status"] == "HUMAN_REVIEWED"
        assert sense["gloss"] and sense["evidence"]
        for evidence_item in sense["evidence"]:
            record = source_by_id[evidence_item["record_id"]]
            assert record["record_id"] in analysis_ids
            assert evidence_item["source_work"] == record["source"]
            assert evidence_item["surface_form"].casefold() == sense["lemma"].casefold()
            assert evidence_item["surface_form"].casefold() in {
                match.group(0).casefold()
                for match in re.finditer(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", record["text"])
            }
    for profile in semantic_profiles:
        assert profile["review_status"] == "HUMAN_REVIEWED"
        assert profile["semantic_tendencies"]
        assert "do not entail" in profile["scope_note"]
    coverage_domains = {
        item["domain"]: item for item in json.loads(
            (GENERATED / "reports" / "structural_coverage.json").read_text(encoding="utf-8")
        )["domains"]
    }
    assert coverage_domains["semantics"]["entries"] == len(semantic_senses)
    expected_semantic_status = "PARTIAL" if semantic_senses else "INSUFFICIENT_EVIDENCE"
    assert coverage_domains["semantics"]["coverage_status"] == expected_semantic_status
    assert manifest["analysis_records"] == len(analysis_ids)
    assert manifest["decision"].startswith("FOUNDATION INCOMPLETE")

    # Validate structured grammatical system artifacts
    nc_classes = rows(GENERATED / "grammar" / "noun_classes.jsonl")
    assert nc_classes, "no structured noun-class entries"
    for entry in nc_classes:
        assert entry["review_status"] in {"OBSERVED", "UNDER_REVIEW", "SUPPORTED", "CONTESTED", "REJECTED"}
        assert entry["confidence"] in {"OBSERVED", "SUPPORTED", "HIGH_CONFIDENCE",
                                       "PROVISIONAL", "INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"}
        assert entry["review_status"] != "ESTABLISHED" or entry["observation_ids"]
        assert entry["evidence_records"]

    ag_entries = rows(GENERATED / "grammar" / "agreement_system.jsonl")
    assert ag_entries, "no structured agreement entries"
    assert all(e["evidence_records"] for e in ag_entries)

    vm_entries = rows(GENERATED / "grammar" / "verb_morphology.jsonl")
    assert vm_entries, "no structured verb morphology entries"
    for entry in vm_entries:
        assert entry["candidate_segments"]
        assert entry["supporting_evidence"]

    der_entries = rows(GENERATED / "grammar" / "derivational_morphology.jsonl")
    assert der_entries, "no structured derivational morphology entries"

    nc_review = rows(GENERATED / "grammar" / "noun_class_review_queue.jsonl")
    assert nc_review, "no noun-class review tasks"

    ag_review = rows(GENERATED / "grammar" / "agreement_review_queue.jsonl")
    assert ag_review, "no agreement review tasks"

    vm_queue = rows(GENERATED / "grammar" / "verb_analysis_queue.jsonl")
    assert vm_queue, "no verb analysis queue tasks"

    assert (GENERATED / "grammar" / "evidence_graph.json").exists(), "no evidence graph"
    assert (GENERATED / "grammar" / "linguistic_conflicts.jsonl").exists(), "no linguistic conflicts"
    assert (GENERATED / "reports" / "structural_coverage.json").exists(), "no coverage report"
    assert (GENERATED / "foundation" / "isiXhosa_grammar_state.md").exists(), "no grammar report"
    assert (GENERATED / "reports" / "structural_gate_report.md").exists(), "no structural gate report"
    gate_report = (GENERATED / "reports" / "structural_gate_report.md").read_text(encoding="utf-8")
    assert gate_report.rstrip().endswith("FOUNDATION INSUFFICIENT — CONTINUE STRUCTURAL ANALYSIS"), \
        "structural gate must end with the required decision string"

    word_analyzer_dir = ROOT / "xnlp_language" / "grammar"
    assert (word_analyzer_dir / "word_analyzer.py").exists(), "no word analyzer module"
    assert (word_analyzer_dir / "__init__.py").exists(), "no grammar package init"

    print(f"PASS: {len(records)} authoritative records; {len(lexicon)} lexical observations; "
          f"{len(noun)} prefix observations; {len(nc_classes)} noun classes; "
          f"{len(ag_entries)} agreement entries; {len(vm_entries)} verb entries; "
          f"{len(der_entries)} derivational entries; {len(schema_paths)} schema assets parsed")

if __name__ == "__main__": main()
