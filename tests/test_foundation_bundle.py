import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from xnlp_language.build_bundle import ROOT, build_foundation_bundle


def _copy_bundle_inputs(destination: Path) -> None:
    generated_source = ROOT / "xnlp_language" / "generated"
    (destination / "xnlp_language").mkdir(parents=True, exist_ok=True)
    shutil.copytree(generated_source, destination / "xnlp_language" / "generated")

    files = (
        "data/processed/v2/corpus_manifest.json",
        "data/processed/v2/training_corpus_provenance.jsonl",
        "xnlp_language/build_foundation.py",
        "xnlp_language/validation/linguistic_layer.py",
        "xnlp_language/annotation/adjudicate_lexical.py",
        "xnlp_language/schemas/linguistic_claim_schema.json",
        "xnlp_language/schemas/lexical_analysis_task_schema.json",
        "xnlp_language/schemas/semantic_frame_schema.json",
        "xnlp_language/schemas/discourse_annotation_schema.json",
        "artifacts/xnlp_prose_pilot_work_isolated/training_report.json",
        "data/reports/mqhayi_checkpoint_verification.json",
        "data/reports/mqhayi_work_isolated_comparison.json",
        "artifacts/xnlp_prose_pilot/training_report.json",
        "artifacts/xnlp_prose_pilot_linebreaks/training_report.json",
        "artifacts/xnlp_prose_pilot_packed_context/training_report.json",
    )
    for relative in files:
        source = ROOT / relative
        if source.is_file():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    shutil.copytree(
        ROOT / "data/processed/v2/splits",
        destination / "data/processed/v2/splits",
    )
    shutil.copytree(
        ROOT / "artifacts/xnlp_prose_pilot_work_isolated/development_corpus",
        destination / "artifacts/xnlp_prose_pilot_work_isolated/development_corpus",
    )


def test_build_is_reproducible_and_valid():
    with TemporaryDirectory(prefix="xnlp-foundation-repro-") as temporary:
        destination = Path(temporary) / "generated"
        command = [
            sys.executable,
            "xnlp_language/build_foundation.py",
            "--output",
            str(destination),
        ]
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        first = (destination / "foundation_manifest.json").read_bytes()
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        second = (destination / "foundation_manifest.json").read_bytes()

    assert second == first
    subprocess.run(
        [sys.executable, "xnlp_language/validation/validate_foundation.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_bundle_connects_evidence_grammar_reviews_and_evaluations():
    with TemporaryDirectory(prefix="xnlp-bundle-tests-") as temporary:
        root = Path(temporary)
        _copy_bundle_inputs(root)
        output = root / "xnlp_language" / "generated" / "foundation_bundle.json"
        bundle = build_foundation_bundle(root=root, output_path=output)
        first_bytes = output.read_bytes()
        persisted = json.loads(first_bytes)
        build_foundation_bundle(root=root, output_path=output)
        second_bytes = output.read_bytes()
        artifact_hashes_match = all(
            (root / artifact["path"]).is_file()
            and hashlib.sha256((root / artifact["path"]).read_bytes()).hexdigest()
            == artifact["sha256"]
            for artifact in bundle["artifact_catalog"]
        )
        split_audit = json.loads(
            (root / "xnlp_language" / "generated" / "reports" / "split_integrity.json")
            .read_text(encoding="utf-8")
        )

    assert second_bytes == first_bytes
    assert persisted["bundle_id"] == "XNLP_ISIXHOSA_EVIDENCE_FOUNDATION_BUNDLE_V1"
    assert bundle["language_derivation"]["evidence_set_id"] == "DERIVATION_EVIDENCE_SET_V1"
    assert bundle["language_derivation"]["evidence_record_count"] == 5716
    assert bundle["language_derivation"]["linguistic_generalizations_promoted_to_supported"] == 0
    grammar_spec_path = (
        root / "xnlp_language" / "generated" / "grammar_derivation" / "grammar_specification.json"
    )
    if grammar_spec_path.is_file():
        assert bundle["formal_grammar_derivation"]["evidence_set_id"] == (
            "DERIVATION_EVIDENCE_SET_V1"
        )
        assert bundle["formal_grammar_derivation"]["evidence_record_count"] == 5716
        assert bundle["formal_grammar_derivation"]["supported_grammar_rule_count"] == 0
    assert bundle["source_corpus"]["records_preserved"] == 5839
    assert bundle["source_corpus"]["eligible_analysis_records"] == 5719
    assert bundle["lexicon"]["entries"] == 30017
    assert bundle["release_state"]["established_grammar_rule_count"] == 0
    assert bundle["grammar"]["evidence_link_audit"]["status"] == "PASS"
    assert bundle["grammar"]["evidence_link_audit"]["unresolved_record_ids"] == []
    assert bundle["annotation_work"]["semantic"]["all_analysis_fields_empty"] is True
    assert bundle["annotation_work"]["discourse"]["all_analysis_fields_empty"] is True
    assert len(bundle["grammar"]["domain_coverage"]) == 14
    assert bundle["split_integrity"]["historical_record_split"][
        "whole_work_isolation_verified"
    ] is False
    assert bundle["split_integrity"]["work_isolated_experiment"]["available"] is True
    validation = bundle["linguistic_validation"]
    assert validation["claims"]["total"] > 0
    assert validation["validation_report"]["legacy_supported_claims"] == 10
    assert validation["claims"]["domains"]["morphology"]["by_evidence_state"][
        "CANDIDATE"
    ] == 666
    assert validation["lexical_analysis"]["tasks"] == 30017
    assert validation["evaluation"]["gold_validated_annotations"] == 0
    assert validation["evaluation"]["gold_set_populated"] is False
    assert validation["validation_report"]["verified"]["orphan_claims"] == 0
    assert validation["status"] == "PASS_WITH_UNRESOLVED_REVIEW_WORK"
    assert split_audit["gold_evaluation"]["work_isolated_from_current_training"] is True
    assert split_audit["gold_evaluation"]["validated_gold_annotations"] == 0
    assert split_audit["work_isolated_split"]["author_isolated"] is None
    assert split_audit["historical_split"]["label"] == (
        "DIAGNOSTIC_ONLY_NOT_GENERALIZATION_EVIDENCE"
    )
    historical_work_overlaps = split_audit["historical_split"]["audit"][
        "source_work_overlaps"
    ]
    assert any(historical_work_overlaps.values())
    assert artifact_hashes_match
    assert any(
        artifact["path"] == "xnlp_language/annotation/adjudicate_lexical.py"
        for artifact in bundle["artifact_catalog"]
    )
