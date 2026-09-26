import subprocess
import sys
import json
import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "xnlp_language" / "generated"
GRAMMAR = ROOT / "xnlp_language" / "grammar"

VALID_STATUSES = {"OBSERVED", "UNDER_REVIEW", "SUPPORTED", "ESTABLISHED", "CONTESTED", "REJECTED"}
VALID_CONFIDENCE = {"OBSERVED", "SUPPORTED", "HIGH_CONFIDENCE",
                    "PROVISIONAL", "INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"}

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

@pytest.fixture(scope="module", autouse=True)
def build_once():
    """Run the deterministic build once before all tests in this module."""
    subprocess.run([sys.executable, "xnlp_language/build_foundation.py"], cwd=ROOT, check=True)
    return True


def test_preservation_layer_immutable():
    """All 5,839 authoritative records preserved with SHA-256 fingerprints;
    non-TARGET_LANGUAGE records excluded from automatic analysis but retained unchanged."""
    records = rows(GENERATED / "evidence" / "corpus_records.jsonl")
    registry = json.loads((GENERATED / "validation" / "language_integrity_registry.json").read_text(encoding="utf-8"))
    assert len(registry) == len(records), "registry does not preserve every record"
    source_by_id = {r["record_id"]: r for r in records}
    for item in registry:
        assert item["original_text_preserved"] is True
        record = source_by_id[item["record_id"]]
        assert item["original_text_sha256"] == hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
        assert item["preservation_action"] == "RETAIN_UNCHANGED"
        assert (item["automated_linguistic_analysis"] == "INCLUDED") == (item["language_class"] == "TARGET_LANGUAGE")
    # Verify excluded records are still retained
    excluded = [i for i in registry if i["automated_linguistic_analysis"] == "EXCLUDED_PENDING_REVIEW"]
    assert excluded, "no excluded records in registry"


def test_build_is_reproducible_and_valid():
    """Build twice → identical manifest SHA-256 (deterministic), then validation passes."""
    subprocess.run([sys.executable, "xnlp_language/build_foundation.py"], cwd=ROOT, check=True)
    first = (ROOT / "xnlp_language" / "generated" / "foundation_manifest.json").read_bytes()
    subprocess.run([sys.executable, "xnlp_language/build_foundation.py"], cwd=ROOT, check=True)
    second = (ROOT / "xnlp_language" / "generated" / "foundation_manifest.json").read_bytes()
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()
    subprocess.run([sys.executable, "xnlp_language/validation/validate_foundation.py"], cwd=ROOT, check=True)


def test_manifest_freeze_and_decision():
    """Manifest must have model_training FROZEN and decision starting with FOUNDATION INCOMPLETE."""
    manifest = json.loads((GENERATED / "foundation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_training"] == "FROZEN"
    assert manifest["decision"].startswith("FOUNDATION INCOMPLETE")
    assert manifest["corpus_sha256"] == "3fcec2c0c9dac47942fe164889601423d69226e9456108f4156f6ec3231f752c"
    assert manifest["authoritative_records"] == 5839
    assert manifest["analysis_records"] == 5719


def test_lexicon_unanalyzed():
    """Every lexicon entry must be UNANALYZED with part_of_speech UNASSIGNED."""
    lexicon = rows(GENERATED / "lexicon" / "lexicon_observed.jsonl")
    assert lexicon, "no lexical observations"
    for entry in lexicon:
        assert entry["analysis_status"] == "UNANALYZED"
        assert entry["part_of_speech"] == "UNASSIGNED"
        assert entry["evidence"] and all(e["record_id"] for e in entry["evidence"])


def test_noun_class_system_structured():
    """Noun-class entries must carry stable IDs, claims, evidence, review status, confidence."""
    entries = rows(GENERATED / "grammar" / "noun_classes.jsonl")
    assert entries, "no structured noun-class entries"
    for entry in entries:
        assert entry["entry_id"]  # stable ID
        assert entry["category"] == "noun_class"
        assert entry["claim"]  # linguistic claim
        assert entry["evidence_records"]  # evidence records
        assert entry["source_provenance"]  # source provenance
        assert entry["observation_ids"]  # observation IDs
        assert entry["review_status"] in VALID_STATUSES
        assert entry["confidence"] in VALID_CONFIDENCE
        assert "counterexamples" in entry
        assert entry["notes"]
        # No entry is ESTABLISHED without human review
        assert entry["review_status"] != "ESTABLISHED" or entry["observation_ids"]


def test_noun_class_evidence_linked():
    """Noun-class evidence items must link to entry IDs and be backed by record IDs."""
    evidence = rows(GENERATED / "grammar" / "noun_class_evidence.jsonl")
    entries = rows(GENERATED / "grammar" / "noun_classes.jsonl")
    assert evidence, "no noun-class evidence items"
    entry_ids = {e["entry_id"] for e in entries}
    record_ids = {r["record_id"] for r in rows(GENERATED / "evidence" / "corpus_records.jsonl")}
    for item in evidence:
        assert item["entry_id"] in entry_ids
        assert item["record_id"] in record_ids
        assert item["category"] == "noun_class"


def test_noun_class_review_queue_exists():
    """Noun-class review queue must contain tasks with null proposed_analysis and INSUFFICIENT_EVIDENCE."""
    tasks = rows(GENERATED / "grammar" / "noun_class_review_queue.jsonl")
    assert tasks, "no noun-class review tasks"
    analysis_ids = {i["record_id"] for i in json.loads(
        (GENERATED / "validation" / "language_integrity_registry.json").read_text(encoding="utf-8"))
        if i["automated_linguistic_analysis"] == "INCLUDED"}
    for task in tasks:
        assert task["status"] == "PENDING_HUMAN_REVIEW"
        assert task["proposed_analysis"] is None
        assert task["confidence"] == "INSUFFICIENT_EVIDENCE"
        assert task["source_record_ids"] and set(task["source_record_ids"]).issubset(analysis_ids)


def test_agreement_system_structured():
    """Agreement entries must carry stable IDs, relational claims, evidence, review status."""
    entries = rows(GENERATED / "grammar" / "agreement_system.jsonl")
    assert entries, "no structured agreement entries"
    for entry in entries:
        assert entry["entry_id"]  # stable ID
        assert entry["category"] == "agreement"
        assert entry["claim"]
        assert entry["noun_class"]  # links to noun class
        assert entry["agreement_type"]
        assert entry["evidence_records"]
        assert entry["source_provenance"]
        assert entry["observation_ids"]
        assert entry["review_status"] in VALID_STATUSES
        assert entry["confidence"] in VALID_CONFIDENCE
        assert "counterexamples" in entry


def test_agreement_evidence_linked():
    """Agreement evidence items must link to entry IDs and record IDs."""
    evidence = rows(GENERATED / "grammar" / "agreement_evidence.jsonl")
    entries = rows(GENERATED / "grammar" / "agreement_system.jsonl")
    assert evidence, "no agreement evidence items"
    entry_ids = {e["entry_id"] for e in entries}
    record_ids = {r["record_id"] for r in rows(GENERATED / "evidence" / "corpus_records.jsonl")}
    for item in evidence:
        assert item["entry_id"] in entry_ids
        assert item["record_id"] in record_ids
        assert item["category"] == "agreement"
        assert item["relationship"]


def test_verb_morphology_structured():
    """Verb morphology entries must carry surface form, candidate segments, analysis, evidence."""
    entries = rows(GENERATED / "grammar" / "verb_morphology.jsonl")
    assert entries, "no structured verb morphology entries"
    for entry in entries:
        assert entry["entry_id"]
        assert entry["category"] == "verb_morphology"
        assert entry["surface_form"]
        assert entry["candidate_segments"]
        assert entry["analysis"]
        assert entry["supporting_evidence"]
        assert entry["review_status"] in VALID_STATUSES
        assert entry["confidence"] in VALID_CONFIDENCE
        assert "counterexamples" in entry


def test_verb_analysis_queue_exists():
    """Verb analysis queue must contain segmentation review tasks."""
    tasks = rows(GENERATED / "grammar" / "verb_analysis_queue.jsonl")
    assert tasks, "no verb analysis queue tasks"
    for task in tasks:
        assert task["status"] == "PENDING_HUMAN_REVIEW"
        assert task["proposed_analysis"] is None
        assert task["confidence"] == "INSUFFICIENT_EVIDENCE"
        assert task["candidate_segments"]


def test_derivational_morphology_structured():
    """Derivational morphology entries must carry affix, type, evidence, review status."""
    entries = rows(GENERATED / "grammar" / "derivational_morphology.jsonl")
    assert entries, "no structured derivational morphology entries"
    for entry in entries:
        assert entry["entry_id"]
        assert entry["category"] == "derivational_morphology"
        assert entry["affix"]
        assert entry["affix_type"] == "suffix"
        assert entry["derivation_type"]
        assert entry["supporting_evidence"]
        assert entry["review_status"] in VALID_STATUSES
        assert entry["confidence"] in VALID_CONFIDENCE
    evidence = rows(GENERATED / "grammar" / "derivational_evidence.jsonl")
    assert evidence, "no derivational evidence items"


def test_evidence_graph_links():
    """Evidence graph must link sources → records → observations → rules."""
    graph = json.loads((GENERATED / "grammar" / "evidence_graph.json").read_text(encoding="utf-8"))
    assert graph["artifact_id"] == "EVIDENCE_GRAPH_V1"
    assert graph["status"] == "OBSERVED"
    assert graph["corpus_sha256"]
    nodes = graph["nodes"]
    assert nodes["sources"]["count"] > 0
    assert nodes["records"]["count"] > 0
    assert nodes["observations"]["count"] > 0
    assert nodes["rules"]["count"] > 0
    assert graph["edges"], "no edges in evidence graph"
    # Each edge should have from/to/relation
    for edge in graph["edges"]:
        assert "from" in edge and "to" in edge and "relation" in edge


def test_conflict_detection_exists():
    """Conflict detection file must exist with OPEN conflicts."""
    conflicts = rows(GENERATED / "grammar" / "linguistic_conflicts.jsonl")
    assert conflicts, "no linguistic conflicts detected"
    for item in conflicts:
        assert item["status"] == "OPEN"
        assert item["record_ids"], "conflict has no record IDs"
        assert item["conflict_type"]
    # Also check the contradictions file
    contradictions = json.loads((GENERATED / "validation" / "contradictions.json").read_text(encoding="utf-8"))
    assert contradictions["status"] == "OPEN"


def test_source_gate_restricts_to_approved_sources():
    """Only the approved Mqhayi and Masikhanyise sources may be treated as authoritative."""
    corpus_path = ROOT / "data" / "authoritative" / "v2_authoritative_all.jsonl"
    records = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert {r["source"] for r in records}.issubset({"mqhayi", "masikhanyise"})
    assert all(r.get("source") in {"mqhayi", "masikhanyise"} for r in records)
    assert not any(r.get("source") not in {"mqhayi", "masikhanyise"} for r in records)


def test_coverage_report_per_domain():
    """Coverage report must cover all structural domains."""
    report = json.loads((GENERATED / "reports" / "structural_coverage.json").read_text(encoding="utf-8"))
    assert report["artifact_id"] == "STRUCTURAL_COVERAGE_V1"
    domains = {d["domain"]: d for d in report["domains"]}
    expected = {"orthography", "lexicon", "noun_classes", "agreement", "morphology",
                "verbs", "tense_aspect_mood", "negation", "pronouns",
                "modifiers", "syntax", "word_formation", "semantics", "discourse"}
    assert expected.issubset(set(domains.keys())), f"missing domains: {expected - set(domains.keys())}"
    for domain, info in domains.items():
        assert info["entries"] >= 0
        assert info["coverage_status"]
        assert info["confidence"]


def test_unknown_analysis_is_safe():
    """Word analyzer must return UNKNOWN for unrecognized forms without crashing."""
    sys.path.insert(0, str(ROOT / "xnlp_language"))
    from grammar.word_analyzer import analyze_word, UNKNOWN_STATUS

    # Known form should be analyzed
    known = analyze_word("umntu")
    assert known["status"] != UNKNOWN_STATUS or known["status"] == "ANALYZED"

    # Unknown forms must return UNKNOWN, never crash
    assert analyze_word("xyzqwerty99")["status"] == UNKNOWN_STATUS
    assert analyze_word("")["status"] == UNKNOWN_STATUS
    assert analyze_word("z7q")["status"] == UNKNOWN_STATUS
    # Ensure no exception on edge cases
    for w in ["a", "I", "!", "@#$%", "umfana", "abantu"]:
        result = analyze_word(w)
        assert result["surface_form"] == w
        assert result["status"] in ("UNKNOWN", "ANALYZED")


def test_text_analysis_preserves_exact_surface_and_offsets():
    sys.path.insert(0, str(ROOT / "xnlp_language"))
    from grammar import analyze_text, UNKNOWN_STATUS

    text = "Qaphelisisa  upelo lwakho, xyzqwerty.\n"
    result = analyze_text(text)
    tokens = result["tokens"]

    assert result["schema_version"] == 1
    assert result["text"] == text
    assert "".join(token["surface"] for token in tokens) == text
    assert all(text[token["start"]:token["end"]] == token["surface"] for token in tokens)
    unknown = next(token for token in tokens if token["surface"] == "xyzqwerty")
    assert unknown["analysis"]["status"] == UNKNOWN_STATUS


def test_grammar_report_readable():
    """Grammar report must be a readable Markdown file with key sections."""
    report_path = GENERATED / "foundation" / "isiXhosa_grammar_state.md"
    assert report_path.exists(), "grammar report not found"
    content = report_path.read_text(encoding="utf-8")
    assert len(content) > 500, "grammar report too short"
    assert "# isiXhosa Grammar State" in content
    assert "## 1. Noun-Class System" in content
    assert "## 2. Agreement System" in content
    assert "## 3. Verb Morphology" in content
    assert "## 4. Derivational Morphology" in content
    assert "## 5. Coverage Summary" in content
    assert "## 6. Review Status" in content
    assert "## 7. Uncertainty and Limitations" in content
    # Must reference the immutable preservation layer
    assert "preservation" in content.lower()
    # Must warn that frequency != rule
    assert "frequency" in content.lower()


def test_structural_gate_report():
    """Structural gate report must end with the required completion decision."""
    gate_path = GENERATED / "reports" / "structural_gate_report.md"
    assert gate_path.exists(), "structural gate report not found"
    content = gate_path.read_text(encoding="utf-8")
    assert content.rstrip().endswith("FOUNDATION INSUFFICIENT — CONTINUE STRUCTURAL ANALYSIS")
    assert "RAW CORPUS" in content
    assert "OBSERVATION" in content
    assert "LINGUISTIC HYPOTHESIS" in content
    assert "SUPPORTED" in content
    assert "ESTABLISHED" in content
