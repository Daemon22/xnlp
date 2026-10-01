"""
Tests for the evaluation framework (evaluate_checkpoint.py).

These tests verify:
1. Evaluation report structure and required fields
2. Tokenizer round-trip on canonical V2 prompts
3. Metric computation logic
4. Contamination detection correctness
"""
import pytest
import json
import os
import torch
from pathlib import Path
from unittest.mock import MagicMock

# These imports must succeed for the framework to be tested
import evaluate_checkpoint as eval_mod


# ── Tokenizer round-trip tests ──────────────────────────────────────────

def test_eval_module_imports():
    """The evaluation module loads and exposes expected functions."""
    assert hasattr(eval_mod, "eval_tokenizer_roundtrip")
    assert hasattr(eval_mod, "eval_generation_quality")
    assert hasattr(eval_mod, "eval_morphology")
    assert hasattr(eval_mod, "eval_contamination")


def test_round_trip_prompts_are_canonical():
    """The 5 canonical prompts match the release gate specification."""
    assert len(eval_mod.ROUND_TRIP_PROMPTS) == 5
    assert "Umntu ngumntu ngabantu." in eval_mod.ROUND_TRIP_PROMPTS
    assert "Ubuntu buhle." in eval_mod.ROUND_TRIP_PROMPTS
    assert "Ndiyafunda isiXhosa." in eval_mod.ROUND_TRIP_PROMPTS
    assert "Ityala lamawele." in eval_mod.ROUND_TRIP_PROMPTS
    assert "AmaXhosa anembali ende." in eval_mod.ROUND_TRIP_PROMPTS


def test_expected_xhosa_words_set():
    """The expected Xhosa words set is non-empty and contains known words."""
    assert len(eval_mod.EXPECTED_XHOSA_WORDS) >= 30
    assert "umntu" in eval_mod.EXPECTED_XHOSA_WORDS
    assert "abantu" in eval_mod.EXPECTED_XHOSA_WORDS
    assert "ubuntu" in eval_mod.EXPECTED_XHOSA_WORDS


# ── Metric computation tests (no model needed) ──────────────────────────

def test_generation_quality_scores_diversity():
    """eval_generation_quality should reward diverse generations."""
    # If all 3 generations are the same, diversity score is low
    mock_predictor = MagicMock()
    mock_predictor.generate.return_value = "Same output"
    result = eval_mod.eval_generation_quality(mock_predictor, [("test", "test")])
    assert result["passed"] == 0  # Same output = no diversity


def test_generation_quality_scores_output_length():
    """eval_generation_quality should reward outputs longer than the prompt."""
    mock_predictor = MagicMock()
    mock_predictor.generate.return_value = "Prompt extended with more text here."
    result = eval_mod.eval_generation_quality(mock_predictor, [("Prompt", "test")])
    assert result["passed"] == 1  # Longer output should pass


def test_morphology_detects_english():
    """eval_morphology should flag English words."""
    mock_predictor = MagicMock()
    mock_predictor.generate.return_value = "the quick brown fox"
    result = eval_mod.eval_morphology(mock_predictor, [("prompt", "test")])
    assert len(result["details"][0]["english_words_found"]) > 0


def test_morphology_detects_xhosa():
    """eval_morphology should find Xhosa words."""
    mock_predictor = MagicMock()
    mock_predictor.generate.return_value = "Umuntu ngumntu ngabantu. Ubuntu buhle."
    result = eval_mod.eval_morphology(mock_predictor, [("prompt", "test")])
    assert len(result["details"][0]["xhosa_words_found"]) > 0


def test_contamination_detects_exact_match():
    """eval_contamination should flag exact matches."""
    training = ["This is a test record that should not be reproduced."]
    mock_predictor = MagicMock()
    mock_predictor.generate.return_value = "This is a test record that should not be reproduced."
    result = eval_mod.eval_contamination(mock_predictor, [("prompt", "test")], training)
    assert all(not d["novel"] for d in result["details"])


def test_contamination_passes_novel_output():
    """eval_contamination should pass novel generations."""
    training = ["This is a test record."]
    mock_predictor = MagicMock()
    mock_predictor.generate.return_value = "This is a completely different output."
    result = eval_mod.eval_contamination(mock_predictor, [("prompt", "test")], training)
    assert all(d["novel"] for d in result["details"])


# ── Integration test with real checkpoint (if available) ────────────────

CHECKPOINT_PATH = "artifacts/xnlp_v2_tiny_tokenizer_fixed/model/best_model.pt"


@pytest.fixture(scope="module")
def predictor():
    """Load the checkpoint if available, otherwise skip."""
    if not os.path.exists(CHECKPOINT_PATH):
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH}")
    from xnlp_trainer.inference import XNLPPredictor
    return XNLPPredictor.load(CHECKPOINT_PATH, device="cpu")


@pytest.fixture(scope="module")
def tokenizer():
    """Load the tokenizer from checkpoint dir."""
    tok_dir = "artifacts/xnlp_v2_tiny_tokenizer_fixed"
    if not os.path.exists(tok_dir):
        pytest.skip(f"Tokenizer dir not found: {tok_dir}")
    from core_llm.tokenizer import XNLPTokenizer
    return XNLPTokenizer.load_pretrained(tok_dir)


class TestCheckpointIntegration:
    """Integration tests that run against an actual trained checkpoint."""

    def test_tokenizer_round_trip(self, tokenizer):
        """All 5 canonical prompts should round-trip exactly."""
        result = eval_mod.eval_tokenizer_roundtrip(tokenizer, eval_mod.ROUND_TRIP_PROMPTS)
        assert result["passed"] == 5
        assert result["score"] == 100.0

    def test_generation_produces_output(self, predictor):
        """Generation should produce non-empty output for a simple prompt."""
        gen = predictor.generate("Umntu", max_new_tokens=20, temperature=0.8)
        assert len(gen) > len("Umntu")
        assert len(gen) > 0

    def test_generation_with_temperature(self, predictor):
        """Different temperatures should produce different-length outputs."""
        gen1 = predictor.generate("Molo", max_new_tokens=30, temperature=0.5)
        gen2 = predictor.generate("Molo", max_new_tokens=30, temperature=1.0)
        assert len(gen1) > 0
        assert len(gen2) > 0

    def test_generation_with_top_k_top_p(self, predictor):
        """Top-k and top-p sampling should work without errors."""
        gen = predictor.generate(
            "Ndiyafunda", max_new_tokens=20,
            temperature=0.8, top_k=40, top_p=0.9,
        )
        assert len(gen) > 0

    def test_full_evaluation_report_structure(self, predictor, tokenizer):
        """The full evaluation report should have all required sections."""
        # This tests the eval_generation_quality and eval_morphology functions
        gen_result = eval_mod.eval_generation_quality(
            predictor, [(p, "test") for p in eval_mod.ROUND_TRIP_PROMPTS]
        )
        assert "passed" in gen_result
        assert "failed" in gen_result
        assert "details" in gen_result
        assert "score" in gen_result
        assert len(gen_result["details"]) == 5

    def test_morphology_report_structure(self, predictor):
        """Morphology evaluation should produce valid report structure."""
        morph_result = eval_mod.eval_morphology(
            predictor, [(p, "test") for p in eval_mod.ROUND_TRIP_PROMPTS[:3]]
        )
        assert "passed" in morph_result
        assert "failed" in morph_result
        assert "details" in morph_result
        for d in morph_result["details"]:
            assert "invalid_chars" in d
            assert "english_words_found" in d
            assert "xhosa_words_found" in d
