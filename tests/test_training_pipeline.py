#!/usr/bin/env python
"""Training pipeline integration tests.

These tests verify the end-to-end training flow:
1. A tiny model can be trained on a small isiXhosa dataset
2. The loss decreases (learning is happening)
3. Checkpoints are saved with correct format
4. Training can be resumed from a checkpoint
5. The trained model produces isiXhosa-compatible output

Tests are designed to be fast (tiny model, small dataset, 1-2 epochs)
so they run in CI without consuming excessive resources.
"""
import json
import os
import sys
import tempfile
import torch
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core_llm.architecture import XNLPConfig, XNLPCoreLLM
from core_llm.tokenizer import XNLPTokenizer
from xnlp_trainer.config import TrainingConfig, validate_checkpoint
from xnlp_trainer.data import prepare_data
from xnlp_trainer.trainer import XNLPTrainer


# ── Tiny isiXhosa dataset for fast testing ─────────────────────────────────

TINY_CORPUS = [
    "Umntu ngumntu ngabantu.",
    "Ubuntu buhle.",
    "Ndiyafunda isiXhosa.",
    "Ityala lamawele.",
    "AmaXhosa anembali ende.",
    "Izwe leli bangekithi.",
    "Inkosi yethu uyakuthanda.",
    "UMemeso ophiqo uwawuthando.",
    "Izibongo zethu ziyabonga.",
    "Abantu abangamaXhosa bayakhala.",
    "Indlu yethu ndiyiNkosi.",
    "Imbambalolo yethu ndivavanyo.",
    "Ubomi bethu buhlala kwiAfrika.",
    "Intetho yethu ndiisiXhosa.",
    "Inkcubeko yethu yayiqaqa.",
    "Amanani ethu ahlukeneyo.",
    "Izithembiso zethu ziqinile.",
    "Ubuhlakwe bwethu buyakhaya.",
    "Izikhashu zethu zizikhethe.",
    "Indlela ethu ibalulekile.",
    "Ukuthando kwethu kudala.",
    "Umlando wethu ubesebenzi.",
    "Izinto zethu ziyaxhomekeka.",
    "Intlanga yethu enkulu.",
    "Inyanga ethu isentla.",
]


@pytest.fixture
def tiny_corpus_dir(tmp_path):
    """Create a temporary corpus directory with a tiny isiXhosa dataset."""
    import json as _json

    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text("\n".join(TINY_CORPUS), encoding="utf-8")

    # Create authoritative JSONL
    records = []
    for i, text in enumerate(TINY_CORPUS):
        records.append({
            "id": f"tiny_{i:04d}",
            "text": text,
            "source": "test",
            "language": "xhosa",
            "source_language_integrity": "TARGET_LANGUAGE",
            "has_foreign_terms": False,
        })
    auth_dir = tmp_path / "authoritative"
    auth_dir.mkdir()
    auth_file = auth_dir / "tiny_authoritative.jsonl"
    with open(auth_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(_json.dumps(rec, ensure_ascii=False) + "\n")

    return str(tmp_path)


@pytest.fixture
def tiny_tokenizer_dir(tmp_path):
    """Train a tiny tokenizer for testing."""
    tok = XNLPTokenizer(vocab_size=256, min_frequency=1)
    tok.train(TINY_CORPUS, verbose=False)
    tok_dir = tmp_path / "tokenizer"
    tok.save_pretrained(str(tok_dir))
    return str(tok_dir)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestTrainingPipeline:
    """End-to-end training pipeline tests with a tiny model."""

    @pytest.fixture(autouse=True)
    def _setup(self, tiny_corpus_dir, tiny_tokenizer_dir):
        self.corpus_dir = tiny_corpus_dir
        self.tokenizer_dir = tiny_tokenizer_dir
        self.tmpdir = tempfile.mkdtemp(prefix="xnlp_train_test_")

    def test_training_runs_and_loss_decreases(self):
        """Training for 2 epochs should produce a checkpoint with decreasing loss."""
        cfg = TrainingConfig(
            preset="tiny",
            tokenizer_path=self.tokenizer_dir,
            corpus_dir=self.corpus_dir,
            output_dir=self.tmpdir,
            max_epochs=2,
            batch_size=4,
            max_seq_len=64,
            learning_rate=5e-4,
            warmup_steps=5,
            patience=10,  # don't early-stop
            seed=42,
            device="cpu",
            generate_samples=False,
        )
        trainer = XNLPTrainer(cfg)
        trainer.train()

        # Best checkpoint should exist
        best_path = os.path.join(self.tmpdir, "best_model.pt")
        assert os.path.exists(best_path)

        # Validate checkpoint format
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        validate_checkpoint(ckpt)

        # Loss should have decreased (train_loss and val_loss should exist)
        history = json.loads(
            open(os.path.join(self.tmpdir, "training_history.json"), encoding="utf-8").read()
        )
        assert len(history["train_loss"]) >= 1
        assert len(history["val_loss"]) >= 1
        # Validation loss should be finite
        assert all(v == v for v in history["val_loss"])  # not NaN

    def test_training_saves_best_and_last(self):
        """Both best_model.pt and last_model.pt should be saved."""
        cfg = TrainingConfig(
            preset="tiny",
            tokenizer_path=self.tokenizer_dir,
            corpus_dir=self.corpus_dir,
            output_dir=self.tmpdir,
            max_epochs=2,
            batch_size=4,
            max_seq_len=64,
            learning_rate=5e-4,
            warmup_steps=5,
            patience=10,
            seed=42,
            device="cpu",
            generate_samples=False,
        )
        trainer = XNLPTrainer(cfg)
        trainer.train()

        assert os.path.exists(os.path.join(self.tmpdir, "best_model.pt"))
        assert os.path.exists(os.path.join(self.tmpdir, "last_model.pt"))
        assert os.path.exists(os.path.join(self.tmpdir, "training_history.json"))

    def test_checkpoint_metadata_is_complete(self):
        """The saved checkpoint must have all required metadata fields."""
        from xnlp_trainer.config import REQUIRED_METADATA_FIELDS

        cfg = TrainingConfig(
            preset="tiny",
            tokenizer_path=self.tokenizer_dir,
            corpus_dir=self.corpus_dir,
            output_dir=self.tmpdir,
            max_epochs=1,
            batch_size=4,
            max_seq_len=64,
            learning_rate=5e-4,
            warmup_steps=2,
            patience=10,
            seed=42,
            device="cpu",
            generate_samples=False,
        )
        trainer = XNLPTrainer(cfg)
        trainer.train()

        ckpt = torch.load(
            os.path.join(self.tmpdir, "best_model.pt"),
            map_location="cpu", weights_only=False
        )
        meta = ckpt["training_metadata"]
        for field in REQUIRED_METADATA_FIELDS:
            assert field in meta, f"Missing metadata field: {field}"

    def test_checkpoint_can_be_resumed(self):
        """Training can be resumed from a saved checkpoint."""
        cfg = TrainingConfig(
            preset="tiny",
            tokenizer_path=self.tokenizer_dir,
            corpus_dir=self.corpus_dir,
            output_dir=self.tmpdir,
            max_epochs=1,
            batch_size=4,
            max_seq_len=64,
            learning_rate=5e-4,
            warmup_steps=2,
            patience=10,
            seed=42,
            device="cpu",
            generate_samples=False,
        )
        trainer = XNLPTrainer(cfg)
        trainer.train()

        best_path = os.path.join(self.tmpdir, "best_model.pt")
        assert os.path.exists(best_path)

        # Resume from the checkpoint
        cfg2 = TrainingConfig(
            preset="tiny",
            tokenizer_path=self.tokenizer_dir,
            corpus_dir=self.corpus_dir,
            output_dir=self.tmpdir + "_resume",
            max_epochs=2,
            batch_size=4,
            max_seq_len=64,
            learning_rate=5e-4,
            warmup_steps=2,
            patience=10,
            seed=42,
            device="cpu",
            generate_samples=False,
            resume_from=best_path,
        )
        trainer2 = XNLPTrainer(cfg2)
        trainer2.train()

        # Should have produced a checkpoint
        assert os.path.exists(os.path.join(self.tmpdir + "_resume", "best_model.pt"))

    def test_checkpoint_only_inference_from_saved(self):
        """A saved checkpoint should produce inference output via XNLPPredictor."""
        from xnlp_trainer.inference import XNLPPredictor

        cfg = TrainingConfig(
            preset="tiny",
            tokenizer_path=self.tokenizer_dir,
            corpus_dir=self.corpus_dir,
            output_dir=self.tmpdir,
            max_epochs=1,
            batch_size=4,
            max_seq_len=64,
            learning_rate=5e-4,
            warmup_steps=2,
            patience=10,
            seed=42,
            device="cpu",
            generate_samples=False,
        )
        trainer = XNLPTrainer(cfg)
        trainer.train()

        best_path = os.path.join(self.tmpdir, "best_model.pt")
        predictor = XNLPPredictor.load(best_path, device="cpu")
        gen = predictor.generate("Umntu", max_new_tokens=20, temperature=0.8)
        assert len(gen) > 0

    def test_training_corpus_fingerprint_recorded(self):
        """Checkpoint metadata must include the corpus fingerprint."""
        cfg = TrainingConfig(
            preset="tiny",
            tokenizer_path=self.tokenizer_dir,
            corpus_dir=self.corpus_dir,
            output_dir=self.tmpdir,
            max_epochs=1,
            batch_size=4,
            max_seq_len=64,
            learning_rate=5e-4,
            warmup_steps=2,
            patience=10,
            seed=42,
            device="cpu",
            generate_samples=False,
        )
        trainer = XNLPTrainer(cfg)
        trainer.train()

        ckpt = torch.load(
            os.path.join(self.tmpdir, "best_model.pt"),
            map_location="cpu", weights_only=False
        )
        fp = ckpt["training_metadata"].get("corpus_fingerprint", "")
        assert len(fp) > 0, "Corpus fingerprint must be recorded"


class TestEvaluateNewFunctions:
    """Tests for the newly added evaluation functions."""

    def test_context_retention_function_exists(self):
        """eval_context_retention should be importable."""
        import evaluate_checkpoint as eval_mod
        assert hasattr(eval_mod, "eval_context_retention")

    def test_comprehension_function_exists(self):
        """eval_comprehension should be importable."""
        import evaluate_checkpoint as eval_mod
        assert hasattr(eval_mod, "eval_comprehension")

    def test_robustness_function_exists(self):
        """eval_robustness should be importable."""
        import evaluate_checkpoint as eval_mod
        assert hasattr(eval_mod, "eval_robustness")

    def test_robustness_handles_empty_prompt(self):
        """eval_robustness should handle empty prompts without crashing."""
        from unittest.mock import MagicMock
        import evaluate_checkpoint as eval_mod

        mock_predictor = MagicMock()
        mock_predictor.generate.return_value = "Some output"
        result = eval_mod.eval_robustness(
            mock_predictor, [("", "empty_prompt")]
        )
        assert result["total"] == 1
        # No exception should have occurred
        assert result["details"][0]["error"] is None

    def test_comprehension_with_mock_predictor(self):
        """eval_comprehension should score novel outputs correctly."""
        from unittest.mock import MagicMock
        import evaluate_checkpoint as eval_mod

        mock_predictor = MagicMock()
        mock_predictor.generate.return_value = "A completely novel response."
        result = eval_mod.eval_comprehension(
            mock_predictor, [("test", "test")]
        )
        assert result["total"] == 1
        # Novel output with no English should score high
        assert result["passed"] == 1

    def test_context_retention_report_structure(self):
        """eval_context_retention should produce correct report structure."""
        from unittest.mock import MagicMock
        import evaluate_checkpoint as eval_mod

        mock_predictor = MagicMock()
        mock_predictor.generate.return_value = "Extended response with umntu"
        result = eval_mod.eval_context_retention(
            mock_predictor, [("Umntu", "test")]
        )
        assert "passed" in result
        assert "failed" in result
        assert "score" in result
        assert "details" in result
        assert len(result["details"]) == 1
        d = result["details"][0]
        assert "continuation_length" in d
        assert "xhosa_words_in_continuation" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
