import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from xnlp_language.derive_language import derive_language


def test_derivation_spec_is_bounded_traceable_and_non_mutating():
    from xnlp_language.build_bundle import ROOT

    corpus_path = ROOT / "xnlp_language" / "generated" / "evidence" / "corpus_records.jsonl"
    corpus_hash_before = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    with TemporaryDirectory(prefix="xnlp-derivation-test-") as temporary:
        folder = Path(temporary)
        spec = derive_language(
            ROOT,
            folder / "derivation",
            folder / "language_derivation_report.md",
        )
        evidence = spec["DERIVATION_EVIDENCE_SET"]
        assert evidence["record_count"] == 5716
        assert evidence["correction_status"]["records_with_explicit_correction_metadata"] == 0
        assert len(evidence["correction_status"]["target_records_differing_from_preserved_original"]) == 3
        assert evidence["excluded_from_derivation"]["generated_candidate_file"]["record_count"] == 89
        assert spec["summary"]["linguistic_generalizations_promoted_to_supported"] == 0
        assert spec["summary"]["claims_validated_or_established"] == 0

        evidence_ids = set(evidence["record_ids"])
        assert all(
            record_id in evidence_ids
            for candidate in spec["candidate_structures"]
            for record_id in candidate["source_record_ids"]
        )
        assert all(
            observation["source_artifact_sha256"]
            and set(observation["source_record_ids"]) <= evidence_ids
            for observation in spec["source_linked_observations"]
        )
        assert (folder / "derivation" / "language_specification.json").is_file()
        assert "not an executable grammar" in (
            folder / "language_derivation_report.md"
        ).read_text(encoding="utf-8")

    assert hashlib.sha256(corpus_path.read_bytes()).hexdigest() == corpus_hash_before
