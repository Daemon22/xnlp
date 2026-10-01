#!/usr/bin/env python
"""V2 checkpoint validation tests.

These tests verify the V2 tokenizer-fixed checkpoint meets the release
gate criteria:

  * The checkpoint dict passes ``validate_checkpoint()`` (format version 1,
    all 8 top-level fields, all 11 metadata fields, vocab consistency).
  * Checkpoint-only inference works — ``XNLPPredictor.load`` reconstructs
    model + tokenizer from the single ``.pt`` file with no external deps.
  * Whitespace round-trip is exact on all 5 release-gate prompts.
  * The release-gate report, if produced, reports ``generation_spacing: true``
    and ``checkpoint_only_inference: true``.

Tests that depend on a fully trained checkpoint are skipped when the
artifact is absent, so this suite is CI-safe even before training runs.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from core_llm.architecture import XNLPConfig, XNLPCoreLLM
from core_llm.tokenizer import XNLPTokenizer
from xnlp_trainer.config import (
    CHECKPOINT_FORMAT_VERSION,
    REQUIRED_CHECKPOINT_FIELDS,
    REQUIRED_METADATA_FIELDS,
    TrainingConfig,
    validate_checkpoint,
)
from xnlp_trainer.data import (
    tokenizer_to_state_dict,
    tokenizer_from_state_dict,
)
from xnlp_trainer.inference import XNLPPredictor

# V2 release-gate prompts (must match release_gate_v2_tokenizer_fixed.py)
V2_PROMPTS = [
    "Umntu ngumntu ngabantu.",
    "Ubuntu buhle.",
    "Ndiyafunda isiXhosa.",
    "Ityala lamawele.",
    "AmaXhosa anembali ende.",
]

V2_ARTIFACT_DIR = Path("artifacts/xnlp_v2_tiny_tokenizer_fixed")
V2_CHECKPOINT = V2_ARTIFACT_DIR / "model" / "best_model.pt"
V2_GATE_REPORT = Path("data/reports/v2_tokenizer_fixed_release_gate.json")
V2_BENCHMARK_REPORT = Path("data/reports/v2_tokenizer_fixed_benchmark.json")


# ── Synthetic checkpoint helper (fast, no training) ──────────────────────────


def _make_synthetic_checkpoint() -> dict:
    """Build a minimal but fully valid checkpoint dict matching V2 format.

    This mirrors the exact structure that ``XNLPTrainer._save_checkpoint_atomic``
    produces, so tests exercising ``validate_checkpoint`` and
    ``XNLPPredictor.load`` are realistic.
    """
    torch.manual_seed(42)
    # Build a tokenizer with a few merges so encode/decode round-trips.
    tokenizer = XNLPTokenizer(vocab_size=512, min_frequency=1)
    tokenizer.train(V2_PROMPTS, verbose=False)

    vocab_size = tokenizer.vocab_size_actual
    config = XNLPConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        intermediate_size=640,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        dropout_prob=0.1,
    )
    model = XNLPCoreLLM(config)

    cfg_dict = {
        "vocab_size": tokenizer.vocab_size_actual,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "max_position_embeddings": config.max_position_embeddings,
        "pad_token_id": config.pad_token_id,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "unk_token_id": config.unk_token_id,
        "rope_theta": config.rope_theta,
        "layer_norm_eps": config.layer_norm_eps,
        "dropout_prob": config.dropout_prob,
    }

    training_config = TrainingConfig(
        preset="tiny",
        vocab_size=tokenizer.vocab_size_actual,
        corpus_dir="corpus_v2",
    )

    metadata = {
        "epoch": 1,
        "global_step": 100,
        "best_val_loss": 4.5,
        "param_count": sum(p.numel() for p in model.parameters()),
        "vocab_size": tokenizer.vocab_size_actual,
        "seed": 42,
        "python_version": "3.12.0",
        "torch_version": torch.__version__,
        "device": "cpu",
        "preset": "tiny",
        "training_start_time": "2025-01-01T00:00:00+00:00",
    }

    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "config": cfg_dict,
        "model_state_dict": model.state_dict(),
        "training_config": training_config.as_dict(),
        "tokenizer_state": tokenizer_to_state_dict(tokenizer),
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "training_metadata": metadata,
    }


# ── Checkpoint format validation ──────────────────────────────────────────────


def test_synthetic_checkpoint_passes_strict_validation():
    """The synthetic checkpoint must satisfy every checkpoint-format rule."""
    ckpt = _make_synthetic_checkpoint()
    # Should not raise
    validate_checkpoint(ckpt)


def test_checkpoint_has_all_required_top_level_fields():
    ckpt = _make_synthetic_checkpoint()
    for field in REQUIRED_CHECKPOINT_FIELDS:
        assert field in ckpt, f"Missing top-level field: {field}"


def test_checkpoint_format_version_is_one():
    ckpt = _make_synthetic_checkpoint()
    assert ckpt["checkpoint_format_version"] == 1


def test_checkpoint_has_all_required_metadata_fields():
    ckpt = _make_synthetic_checkpoint()
    for field in REQUIRED_METADATA_FIELDS:
        assert field in ckpt["training_metadata"], f"Missing metadata field: {field}"


def test_checkpoint_rejects_incompatible_format_version():
    """A checkpoint with a wrong format version must raise."""
    ckpt = _make_synthetic_checkpoint()
    ckpt["checkpoint_format_version"] = 999
    with pytest.raises(ValueError, match="Unsupported checkpoint format version"):
        validate_checkpoint(ckpt)


def test_checkpoint_rejects_missing_top_level_field():
    """Removing any required field must trigger a validation error."""
    ckpt = _make_synthetic_checkpoint()
    del ckpt["model_state_dict"]
    with pytest.raises(ValueError, match="missing required field"):
        validate_checkpoint(ckpt)


def test_checkpoint_vocab_consistency():
    """Model config vocab_size must match tokenizer vocab_size."""
    ckpt = _make_synthetic_checkpoint()
    model_vocab = ckpt["config"]["vocab_size"]
    tok_vocab = len(ckpt["tokenizer_state"]["token2id"])
    assert model_vocab == tok_vocab


def test_checkpoint_rejects_vocab_mismatch():
    ckpt = _make_synthetic_checkpoint()
    ckpt["config"]["vocab_size"] = 999
    with pytest.raises(ValueError, match="Vocabulary size mismatch"):
        validate_checkpoint(ckpt)


# ── Checkpoint-only inference ────────────────────────────────────────────────


def test_checkpoint_only_inference_works():
    """XNLPPredictor.load must reconstruct model + tokenizer from one .pt file.

    This is the 'checkpoint-only inference' release-gate criterion: the
    checkpoint directory contains *only* best_model.pt — no external tokenizer
    files, no config files — and inference still works.
    """
    ckpt = _make_synthetic_checkpoint()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "best_model.pt")
        torch.save(ckpt, ckpt_path)
        # Only the .pt file should be in the directory
        assert os.listdir(tmpdir) == ["best_model.pt"]

        predictor = XNLPPredictor.load(ckpt_path, device="cpu")
        assert predictor.tokenizer.vocab_size_actual == len(ckpt["tokenizer_state"]["token2id"])
        assert predictor.metadata["epoch"] == 1


def test_round_trip_on_all_v2_prompts():
    """All 5 release-gate prompts must round-trip through encode/decode exactly."""
    ckpt = _make_synthetic_checkpoint()
    tokenizer = tokenizer_from_state_dict(ckpt["tokenizer_state"])
    for prompt in V2_PROMPTS:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        assert decoded == prompt, (
            f"Round-trip failed for '{prompt}': got '{decoded}'"
        )


def test_tokenizer_state_round_trips_through_checkpoint():
    """Tokenizer embedded in checkpoint must be byte-identical after reload."""
    ckpt = _make_synthetic_checkpoint()
    tok_state = ckpt["tokenizer_state"]
    restored = tokenizer_from_state_dict(tok_state)
    original = XNLPTokenizer.load_pretrained  # sanity reference
    # Reload via from_state_dict and compare key attributes
    assert restored.token2id == tok_state["token2id"]
    assert restored.merges == [tuple(m) for m in tok_state["merges"]]
    assert restored.merge_ranks == {
        tuple(m): i for i, m in enumerate(tok_state["merges"])
    }


# ── Real V2 checkpoint tests (skip if artifact absent) ────────────────────────


@pytest.fixture(scope="module")
def v2_checkpoint():
    """Return the path to the real V2 checkpoint, or skip if it doesn't exist."""
    if not V2_CHECKPOINT.exists():
        pytest.skip(f"V2 checkpoint not found at {V2_CHECKPOINT}")
    return V2_CHECKPOINT


def test_v2_checkpoint_exists(v2_checkpoint):
    """Smoke test: the V2 checkpoint file must exist and be non-empty."""
    assert v2_checkpoint.exists()
    assert v2_checkpoint.stat().st_size > 1_000_000


def test_v2_checkpoint_passes_validation(v2_checkpoint):
    """The real V2 checkpoint must pass strict validate_checkpoint()."""
    ckpt = torch.load(v2_checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint(ckpt)  # should not raise


def test_v2_checkpoint_metadata_complete(v2_checkpoint):
    """All required metadata fields must be present in the V2 checkpoint."""
    ckpt = torch.load(v2_checkpoint, map_location="cpu", weights_only=False)
    meta = ckpt["training_metadata"]
    for field in REQUIRED_METADATA_FIELDS:
        assert field in meta, f"V2 checkpoint missing metadata field: {field}"


def test_v2_checkpoint_format_version(v2_checkpoint):
    ckpt = torch.load(v2_checkpoint, map_location="cpu", weights_only=False)
    assert ckpt["checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION


def test_v2_checkpoint_only_inference(v2_checkpoint):
    """Load the real V2 checkpoint from an isolated directory with no other files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        isolated = os.path.join(tmpdir, "best_model.pt")
        shutil.copy2(str(v2_checkpoint), isolated)
        # Only the .pt file should be in the directory
        assert os.listdir(tmpdir) == ["best_model.pt"]
        predictor = XNLPPredictor.load(isolated, device="cpu")
        assert predictor.tokenizer.vocab_size_actual > 0


def test_v2_checkpoint_round_trip(v2_checkpoint):
    """All 5 release-gate prompts must round-trip through the real checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        isolated = os.path.join(tmpdir, "best_model.pt")
        shutil.copy2(str(v2_checkpoint), isolated)
        predictor = XNLPPredictor.load(isolated, device="cpu")
        for prompt in V2_PROMPTS:
            ids = predictor.tokenizer.encode(prompt, add_special_tokens=False)
            decoded = predictor.tokenizer.decode(ids)
            assert decoded == prompt, f"Round-trip failed for '{prompt}'"


def test_v2_release_gate_report_passes():
    """If the release-gate report exists, it must pass all gates."""
    if not V2_GATE_REPORT.exists():
        pytest.skip(f"Release gate report not found at {V2_GATE_REPORT}")
    report = json.loads(V2_GATE_REPORT.read_text(encoding="utf-8"))
    assert report["checkpoint_only_inference"] is True
    assert report["generation_spacing"] is True


def test_v2_benchmark_report_has_selection():
    """The V2 benchmark report must exist with a valid selection."""
    if not V2_BENCHMARK_REPORT.exists():
        pytest.skip(f"Benchmark report not found at {V2_BENCHMARK_REPORT}")
    report = json.loads(V2_BENCHMARK_REPORT.read_text(encoding="utf-8"))
    assert "selection" in report
    assert "candidates" in report
    selection = report["selection"]
    assert "artifact_dir" in selection
    assert "actual_vocabulary" in selection
    assert "target_vocabulary" in selection
    assert selection["actual_vocabulary"] == selection["target_vocabulary"]
    assert os.path.isdir(selection["artifact_dir"])
    # The selected candidate must have round-trip accuracy and whitespace preservation
    selected_vocab = selection["actual_vocabulary"]
    matching = [c for c in report["candidates"] if c["actual_vocabulary"] == selected_vocab]
    assert len(matching) == 1
    candidate = matching[0]
    assert candidate["corpus_metrics"]["round_trip_accuracy"] == 1.0
    assert candidate["corpus_metrics"]["whitespace_preserved"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
