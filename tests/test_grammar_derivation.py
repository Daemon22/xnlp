import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from xnlp_language.derive_grammar import derive_grammar


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _output_hashes(folder: Path) -> dict[str, str]:
    return {
        path.relative_to(folder).as_posix(): _sha256(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def test_grammar_derivation_is_bounded_traceable_and_deterministic():
    from xnlp_language.build_bundle import ROOT

    corpus_path = ROOT / "xnlp_language" / "generated" / "evidence" / "corpus_records.jsonl"
    provenance_path = ROOT / "xnlp_language" / "generated" / "evidence" / "source_provenance.json"
    source_hashes_before = {
        corpus_path: _sha256(corpus_path),
        provenance_path: _sha256(provenance_path),
    }
    with TemporaryDirectory(prefix="xnlp-grammar-derivation-") as temporary:
        folder = Path(temporary)
        output = folder / "first"
        report = output / "grammar_derivation_report.md"
        specification = derive_grammar(ROOT, output, report)

        assert specification["evidence_set"]["id"] == "DERIVATION_EVIDENCE_SET_V1"
        assert specification["evidence_set"]["record_count"] == 5716
        assert specification["summary"]["derived_candidate_structure_count"] == 834
        assert specification["summary"]["legacy_structure_count"] == 681
        assert specification["summary"]["source_linked_observation_count"] == 153
        assert specification["summary"]["supported_grammar_rule_count"] == 0
        assert specification["summary"]["claims_validated_or_established"] == 0
        disposition_report = json.loads(
            (output / "candidate_dispositions.json").read_text(encoding="utf-8")
        )
        assert disposition_report["counts"]["PROMOTE_TO_SUPPORTED"] == 0
        assert disposition_report["counts"]["REJECT"] == 0
        assert all(rule["status"] != "SUPPORTED" for rule in specification["rules"])
        evidence_ids = set(specification["evidence_set"]["record_ids"])
        for rule in specification["rules"]:
            assert set(rule["evidence_record_ids"]) <= evidence_ids
            assert set(rule["out_of_boundary_record_ids"]).isdisjoint(evidence_ids)
            assert rule["confidence"]["assessment"] == "NOT_ASSESSED"
            assert rule["counterexamples"]["grammatical_counterexample_status"] == (
                "REQUIRES_HUMAN_ADJUDICATION"
            )
            assert "source_listed_counterexamples_as_recorded" in rule["counterexamples"]
        assert all(
            observation["status"] == "OBSERVATION"
            for observation in specification["observations"]
        )
        noun_class_inputs = [
            json.loads(line)
            for line in (
                ROOT / "xnlp_language" / "generated" / "grammar" / "noun_classes.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        class_one = next(item for item in noun_class_inputs if item["entry_id"] == "NC_001")
        class_one_rule = next(
            rule for rule in specification["rules"]
            if rule["rule_id"] == "LEGACY_NC_001"
        )
        assert {
            example["source_record_id"] for example in class_one["examples"]
        } <= set(class_one_rule["source_record_ids_as_recorded"])

        bundle_file = ROOT / "xnlp_language" / "generated" / "foundation_bundle.json"
        bundle = json.loads(bundle_file.read_text(encoding="utf-8"))
        assert bundle["bundle_id"] == specification["bundle_reference"]["bundle_id"]
        assert "a-" in json.dumps(specification["conflicts"], ensure_ascii=False)
        assert any(
            analysis["conflict_id"] == "CONFLICT_LINGUISTIC_001"
            for rule in specification["rules"]
            for analysis in rule["competing_analyses"]
        )
        counterexample_report = json.loads(
            (output / "counterexample_report.json").read_text(encoding="utf-8")
        )
        for search in counterexample_report["unique_surface_form_searches"]:
            assert set(search["matched_record_ids"]) <= evidence_ids
            assert search["matching_record_count"] == len(search["matched_record_ids"])
        search_ids = {
            search["search_id"]
            for search in counterexample_report["unique_surface_form_searches"]
        }
        assert all(
            cue["search_id"] in search_ids
            for rule in specification["rules"]
            for cue in rule["counterexamples"]["surface_cue_searches"]
            if "search_id" in cue
        )
        first_hashes = _output_hashes(output)

        second_output = folder / "second"
        derive_grammar(ROOT, second_output, second_output / "grammar_derivation_report.md")
        assert first_hashes == _output_hashes(second_output)

    assert all(_sha256(path) == digest for path, digest in source_hashes_before.items())


def test_legacy_supported_labels_are_not_promoted_to_grammar_rules():
    from xnlp_language.build_bundle import ROOT

    with TemporaryDirectory(prefix="xnlp-grammar-dispositions-") as temporary:
        folder = Path(temporary)
        derive_grammar(ROOT, folder / "output", folder / "report.md")
        specification = json.loads(
            (folder / "output" / "grammar_specification.json").read_text(encoding="utf-8")
        )
        legacy_supported = [
            rule for rule in specification["rules"]
            if rule["source_kind"] == "LEGACY_STRUCTURED_GRAMMAR_ENTRY"
            and rule["source_status_as_recorded"] == "SUPPORTED"
        ]
        assert len(legacy_supported) == 11
        assert all(rule["status"] != "SUPPORTED" for rule in legacy_supported)
        assert all(
            rule["disposition"] in {"RETAIN_AS_CANDIDATE", "MOVE_TO_UNKNOWN"}
            for rule in legacy_supported
        )
        counterexamples = json.loads(
            (folder / "output" / "counterexample_report.json").read_text(encoding="utf-8")
        )
        assert counterexamples["full_evidence_set_record_count_scanned"] == 5716
        assert all(
            item["grammatical_counterexample_status"] == "REQUIRES_HUMAN_ADJUDICATION"
            for item in counterexamples["rule_searches"]
        )
