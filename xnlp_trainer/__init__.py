"""
XNLP isiXhosa Language Model Trainer
====================================
Professional independent training pipeline for building a fluent
traditional Xhosa language model from scratch.

Produces a **single self-contained checkpoint** file that bundles
model weights, model config, tokenizer state and training metadata.

Usage::

    # Full training from scratch
    python -m xnlp_trainer.run

    # Train a specific preset
    python -m xnlp_trainer.run --preset small --epochs 30

    # Resume from checkpoint
    python -m xnlp_trainer.run --resume outputs/last_model.pt --epochs 50

    # One-liner alias
    python -m xnlp_trainer.run

    # Programmatic
    from xnlp_trainer.config import TrainingConfig
    from xnlp_trainer.trainer import XNLPTrainer
    cfg = TrainingConfig(preset="small", max_epochs=30)
    XNLPTrainer(cfg).train()

    # Inference from a single checkpoint file
    from xnlp_trainer.inference import XNLPPredictor
    predictor = XNLPPredictor.load("outputs/best_model.pt")
    print(predictor.generate("Umntu ngumntu"))
"""

__version__ = "3.0.0"
__author__ = "XNLP Team"

from .config import (
    TrainingConfig,
    MODEL_PRESETS,
    CHECKPOINT_FORMAT_VERSION,
    REQUIRED_CHECKPOINT_FIELDS,
    REQUIRED_METADATA_FIELDS,
    validate_checkpoint,
)
from .data import (
    load_corpus,
    train_tokenizer,
    prepare_data,
    XhosaTextDataset,
    make_collate_fn,
    tokenizer_to_state_dict,
    tokenizer_from_state_dict,
    tokenize_texts,
    build_split_indices,
)
from .trainer import XNLPTrainer
from .evaluate import evaluate, compute_perplexity, compute_loss, generate_samples
from .inference import XNLPPredictor

__all__ = [
    "TrainingConfig",
    "MODEL_PRESETS",
    "CHECKPOINT_FORMAT_VERSION",
    "REQUIRED_CHECKPOINT_FIELDS",
    "REQUIRED_METADATA_FIELDS",
    "validate_checkpoint",
    "XNLPTrainer",
    "XNLPPredictor",
    "load_corpus",
    "train_tokenizer",
    "prepare_data",
    "XhosaTextDataset",
    "make_collate_fn",
    "tokenizer_to_state_dict",
    "tokenizer_from_state_dict",
    "tokenize_texts",
    "build_split_indices",
    "evaluate",
    "compute_perplexity",
    "compute_loss",
    "generate_samples",
]
