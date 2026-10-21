import pytest

from xnlp_language import comprehend, understand, validate_generation


def test_understanding_has_structured_tokens_and_frames():
    result = understand("Umntu uyafunda isiXhosa.")
    assert result["schema_version"] == 1
    assert result["tokens"]
    assert "clauses" in result
    assert "semantic_frames" in result
    assert result["comprehension_boundary"]["semantic_claims_are_abstaining"] is True


def test_comprehension_cites_context_or_abstains():
    result = comprehend("UMqhayi ngumbongi. Wabhala imibongo.", "Ngubani uMqhayi?")
    assert result["status"] == "SUPPORTED_EXTRACTION"
    assert result["support"] == ["UMqhayi ngumbongi."]
    assert result["claim_type"] == "EXPLICIT"
    assert result["evidence_spans"][0]["text"] == "UMqhayi ngumbongi."
    unknown = comprehend("Umntu uyafunda.", "Uhlala phi?")
    assert unknown["status"] == "ABSTAIN"
    assert unknown["answer"] is None
    assert unknown["inference"] is False


def test_generation_validation_is_not_a_false_grammar_certificate():
    report = validate_generation("Umntu", "Umntu ngumntu ngabantu.")
    assert report["extended"] is True
    assert report["status"] == "REVIEW_REQUIRED"
    assert "native-speaker" in report["limitation"]
