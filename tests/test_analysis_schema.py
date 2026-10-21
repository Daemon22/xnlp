from xnlp_language import understand
from xnlp_language.validation.analysis_schema import validate_analysis


def test_canonical_analysis_is_schema_valid():
    result = understand("Umntu uyafunda isiXhosa.")
    assert validate_analysis(result) == []
    assert result["original_text"] == "Umntu uyafunda isiXhosa."
    assert all("verification_status" in token for token in result["tokens"])


def test_schema_validator_rejects_missing_required_fields():
    errors = validate_analysis({"schema_version": 1, "language": "isiXhosa"})
    assert any("missing analysis field" in error for error in errors)
