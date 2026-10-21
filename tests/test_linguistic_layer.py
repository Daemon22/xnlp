import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from xnlp_language.validation.linguistic_layer import (
    EVIDENCE_STATES,
    _build_claims,
    _partition_audit,
    apply_review_decision,
    validate_claims,
)


def test_evidence_state_is_independent_and_candidates_require_explicit_review():
    assert set(EVIDENCE_STATES) == {
        "OBSERVED",
        "CANDIDATE",
        "SOURCE_SUPPORTED",
        "CROSS_SOURCE_SUPPORTED",
        "VALIDATED",
        "ESTABLISHED",
        "REJECTED",
        "UNRESOLVED",
    }
    claim = {
        "claim_id": "claim-1",
        "evidence_state": "CANDIDATE",
        "evidence_record_ids": ["record-1"],
        "source_works": ["work-1"],
        "source_collections": ["collection-1"],
        "validated_by": [],
        "review_history": [],
    }
    with pytest.raises(ValueError, match="prior explicit VALIDATED"):
        apply_review_decision(
            claim, "ESTABLISHED", "reviewer-1", "protocol-1", {"record-1"}
        )
    assert claim["evidence_state"] == "CANDIDATE"

    apply_review_decision(
        claim, "VALIDATED", "reviewer-1", "protocol-1", {"record-1"}
    )
    assert claim["evidence_state"] == "VALIDATED"
    with pytest.raises(ValueError, match="strong-evidence basis"):
        apply_review_decision(
            claim, "ESTABLISHED", "reviewer-2", "protocol-1", {"record-1"}
        )
    apply_review_decision(
        claim,
        "ESTABLISHED",
        "reviewer-2",
        "protocol-1",
        {"record-1"},
        establishment_basis="independent expert review and documented counterexample checks",
    )
    validate_claims([claim], {"record-1"})
    assert claim["evidence_state"] == "ESTABLISHED"
    assert claim["validated_by"] == ["reviewer-1", "reviewer-2"]


def test_orphan_claims_are_rejected_and_rejected_claims_keep_provenance():
    orphan = {
        "claim_id": "orphan",
        "evidence_state": "CANDIDATE",
        "evidence_record_ids": [],
    }
    with pytest.raises(ValueError, match="Orphan claim"):
        validate_claims([orphan], {"record-1"})

    rejected = {
        "claim_id": "claim-rejected",
        "evidence_state": "CANDIDATE",
        "evidence_record_ids": ["record-1"],
        "source_works": ["work-1"],
        "source_collections": ["collection-1"],
        "validated_by": [],
        "review_history": [],
    }
    apply_review_decision(
        rejected, "REJECTED", "reviewer-1", "protocol-1", {"record-1"}
    )
    validate_claims([rejected], {"record-1"})
    assert rejected["evidence_state"] == "REJECTED"
    assert rejected["evidence_record_ids"] == ["record-1"]
    with pytest.raises(ValueError, match="cannot be promoted"):
        apply_review_decision(
            rejected, "VALIDATED", "reviewer-2", "protocol-1", {"record-1"}
        )


def test_split_audit_detects_record_work_source_and_near_duplicate_overlap():
    common_text = " ".join(f"distincttoken{index:04d}" for index in range(100))
    partitions = {
        "train": [{
            "record_id": "train-1",
            "source_title": "training-work",
            "source": "collection-a",
            "author": "writer-a",
            "text": common_text + " The final token is crimson.",
        }],
        "test": [{
            "record_id": "test-1",
            "source_title": "evaluation-work",
            "source": "collection-a",
            "author": "writer-a",
            "text": common_text + " The final token is scarlet.",
        }],
    }
    audit = _partition_audit(partitions)
    assert audit["record_overlaps"]["test_train"] == 0
    assert audit["source_work_overlaps"]["test_train"] == []
    assert audit["source_overlaps"]["test_train"] == ["collection-a"]
    assert audit["author_overlaps"]["test_train"] == ["writer-a"]
    assert audit["author_metadata_available"]["test_train"] is True
    assert audit["near_duplicate_pairs"]["test_train"]


def test_validation_schema_and_machine_readable_report_are_well_formed():
    from xnlp_language.build_bundle import ROOT

    for schema_name in (
        "linguistic_claim_schema.json",
        "lexical_analysis_task_schema.json",
        "semantic_frame_schema.json",
        "discourse_annotation_schema.json",
    ):
        schema_path = ROOT / "xnlp_language" / "schemas" / schema_name
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        assert schema["$schema"].endswith("draft-07/schema#")

    report_path = ROOT / "xnlp_language" / "generated" / "reports" / "linguistic_validation_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["verified"]["orphan_claims"] == 0
        assert report["verified"]["gold_validated_annotations"] == 0
        assert report["gold_evaluation"]["gold_set_populated"] is False


def test_semantic_and_discourse_review_tasks_keep_exact_source_provenance():
    from xnlp_language.build_bundle import ROOT
    from xnlp_language.build_bundle import _load_jsonl

    generated = ROOT / "xnlp_language" / "generated"
    corpus = {
        row["record_id"]: row.get("text", row.get("original_text"))
        for row in _load_jsonl(generated / "evidence" / "corpus_records.jsonl")
    }
    for task_name in ("semantic_review_queue.jsonl", "discourse_review_queue.jsonl"):
        tasks = _load_jsonl(generated / "annotation" / task_name)
        assert len(tasks) == 5719
        for task in tasks:
            record_id = task.get("source_record_id", task.get("record_id"))
            assert record_id in corpus
            assert task["source_text"] == corpus[record_id]
            assert task["proposed_analysis"] is None


def test_synthetic_only_claim_cannot_be_source_supported():
    claim = {
        "claim_id": "synthetic-claim",
        "evidence_state": "SOURCE_SUPPORTED",
        "evidence_record_ids": ["generated-1"],
        "provenance_summary": {"attested": 0, "synthetic": 1, "unknown": 0},
        "validated_by": [],
        "review_history": [],
    }
    with pytest.raises(ValueError, match="cannot source-support"):
        validate_claims(
            [claim],
            {"generated-1"},
            {
                "generated-1": {
                    "record_id": "generated-1",
                    "provenance_type": "generated",
                    "generator": "model-v1",
                }
            },
        )


def test_legacy_supported_claim_labels_and_provenance_are_traceable():
    from xnlp_language.build_bundle import GENERATED, ROOT, _load_jsonl
    from xnlp_language.validation.linguistic_layer import _provenance_summary

    source_registry = json.loads(
        (ROOT / GENERATED / "evidence" / "source_provenance.json").read_text(encoding="utf-8")
    )["sources"]
    corpus_by_id = {
        row["record_id"]: row
        for row in _load_jsonl(ROOT / GENERATED / "evidence" / "corpus_records.jsonl")
    }
    claims, summary = _build_claims(ROOT, corpus_by_id)
    validate_claims(claims, set(corpus_by_id), corpus_by_id)

    legacy = [claim for claim in claims if claim.get("legacy_confidence_label") == "SUPPORTED"]
    assert len(legacy) == 10
    assert all(claim["evidence_state"] == "SOURCE_SUPPORTED" for claim in legacy)
    assert all(
        claim["legacy_label_migration_note"].endswith(
            "#/definitions/legacy_label_migration"
        )
        for claim in legacy
    )
    for claim in claims:
        assert claim["provenance_summary"] == _provenance_summary(
            claim["evidence_record_ids"], corpus_by_id, source_registry
        )
    assert not summary["source_support_violations"]


def _write_test_queue(path: Path) -> bytes:
    rows = [
        {"task_id": f"task-{index}", "status": "pending", "untouched": f"value-{index}"}
        for index in range(3)
    ]
    original = b"".join(
        json.dumps(row, sort_keys=True).encode("utf-8") + b"\r\n" for row in rows
    )
    path.write_bytes(original)
    return original


def test_lexical_batch_updates_only_front_tasks_and_preserves_other_bytes():
    from xnlp_language.annotation.adjudicate_lexical import apply_batch

    with TemporaryDirectory(prefix="xnlp-lexical-test-") as temporary:
        queue = Path(temporary) / "queue.jsonl"
        original = _write_test_queue(queue)
        progress = apply_batch(
            queue,
            2,
            {
                "task-0": {
                    "status": "adjudicated",
                    "assigned_lemma": "reviewed-form",
                    "assigned_category": None,
                    "assigned_meaning": None,
                    "adjudicator": "rule-assisted-human",
                    "adjudication_note": "Test decision with explicit reviewer input.",
                },
                "task-1": {
                    "status": "disputed",
                    "assigned_lemma": None,
                    "assigned_category": None,
                    "assigned_meaning": None,
                    "adjudicator": "human",
                    "adjudication_note": "Needs specialist review.",
                },
            },
        )
        updated = queue.read_bytes()
        original_untouched = original.splitlines(keepends=True)[2]
        assert updated.splitlines(keepends=True)[2] == original_untouched
        assert progress["batch_size"] == 2
        assert progress["batch_status_counts"] == {"adjudicated": 1, "disputed": 1}
        assert progress["overall_status_counts"] == {
            "pending": 1,
            "adjudicated": 1,
            "disputed": 1,
        }


def test_lexical_batch_refuses_machine_only_without_flag():
    from xnlp_language.annotation.adjudicate_lexical import apply_batch

    with TemporaryDirectory(prefix="xnlp-lexical-test-") as temporary:
        queue = Path(temporary) / "queue.jsonl"
        original = _write_test_queue(queue)
        with pytest.raises(ValueError, match="pass --machine-assisted"):
            apply_batch(
                queue,
                1,
                {
                    "task-0": {
                        "status": "disputed",
                        "assigned_lemma": None,
                        "assigned_category": None,
                        "assigned_meaning": None,
                        "adjudicator": "machine-only",
                        "adjudication_note": "requires human isiXhosa review",
                    }
                },
            )
        assert queue.read_bytes() == original


def test_machine_flagged_disputes_have_no_adjudicator():
    from xnlp_language.annotation.adjudicate_lexical import apply_batch

    with TemporaryDirectory(prefix="xnlp-lexical-test-") as temporary:
        queue = Path(temporary) / "queue.jsonl"
        _write_test_queue(queue)
        apply_batch(queue, 1, default_disputed=True)
        task = json.loads(queue.read_bytes().splitlines()[0])
        assert task["status"] == "disputed"
        assert task["adjudicator"] is None
        assert task["flagged_by"] == "machine-only"


def test_lexical_bootstrap_threshold_audit_preservation_and_revert():
    from xnlp_language.annotation.adjudicate_lexical import bootstrap_queue, revert_batch

    with TemporaryDirectory(prefix="xnlp-bootstrap-test-") as temporary:
        folder = Path(temporary)
        queue = folder / "queue.jsonl"
        corpus = folder / "corpus.jsonl"
        registry = folder / "registry.json"
        tasks = [
            {"task_id": "auto", "surface_form": "Zanele", "status": "pending"},
            {"task_id": "one-work", "surface_form": "Nandi", "status": "pending"},
            {
                "task_id": "disputed",
                "surface_form": "Bawo",
                "status": "disputed",
                "flagged_by": "machine-only",
            },
            {"task_id": "untouched", "surface_form": "not-in-corpus", "status": "pending"},
        ]
        original_lines = [
            json.dumps(row, sort_keys=True).encode("utf-8") + b"\r\n"
            for row in tasks
        ]
        queue.write_bytes(b"".join(original_lines))
        source_registry = {
            "Book A": {"foundation_attested": True, "independence_group": "Book A"},
            "Book B": {"foundation_attested": True, "independence_group": "Book B"},
        }
        corpus_rows = [
            {
                "record_id": "a1", "source_title": "Book A",
                "provenance_type": "authoritative", "generator": "",
                "text": "Sibone Zanele namhlanje. Sibone Zanele kwakhona. Sibone Nandi namhlanje. Sibone Nandi kwakhona. Sibone Bawo namhlanje. Sibone Bawo kwakhona.",
            },
            {
                "record_id": "b1", "source_title": "Book B",
                "provenance_type": "authoritative", "generator": "",
                "text": "Sibone Zanele izolo. Sibone Zanele kwakhona. Sibone Bawo izolo. Sibone Bawo kwakhona.",
            },
        ]
        corpus.write_text(
            "".join(json.dumps(row) + "\n" for row in corpus_rows),
            encoding="utf-8",
        )
        registry.write_text(
            json.dumps({"sources": source_registry}), encoding="utf-8"
        )

        progress = bootstrap_queue(queue, corpus, registry)
        updated_lines = queue.read_bytes().splitlines(keepends=True)
        updated = [json.loads(line) for line in updated_lines]
        assert updated[0]["status"] == "adjudicated"
        assert updated[0]["assigned_category"] == "proper_noun"
        assert updated[0]["auto_evidence"]["rule_ids"] == [
            "AUTO_PROPER_NOUN_CAPITALIZED_MULTIWORK_V1"
        ]
        assert updated[0]["auto_evidence"]["supporting_record_ids"] == ["a1", "b1"]
        assert updated[0]["adjudication_history"][0]["action"] == "auto-adjudicated"
        assert updated[1]["status"] == "pending"
        assert updated[2]["status"] == "disputed"
        assert updated[2]["flagged_by"] == "machine-only"
        assert updated_lines[1] == original_lines[1]
        assert updated_lines[2] == original_lines[2]
        assert updated_lines[3] == original_lines[3]
        assert progress["auto_adjudicated_this_run"] == 1
        assert progress["rule_firing_histogram"] == {
            "AUTO_PROPER_NOUN_CAPITALIZED_MULTIWORK_V1": 1
        }
        assert progress["pending"] == 2
        assert progress["disputed"] == 1
        assert progress["tasks_requiring_human_attention"] == 4

        reverted = revert_batch(queue, 1, "reviewer-1")
        reverted_task = json.loads(queue.read_bytes().splitlines()[0])
        assert reverted_task["status"] == "pending"
        assert reverted_task["assigned_category"] is None
        assert reverted_task["adjudicator"] is None
        assert reverted_task["adjudication_history"][-1]["action"] == "human-reverted"
        assert reverted["reverted_tasks"] == 1
