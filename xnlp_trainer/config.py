"""
XNLP Trainer – Configuration
=============================
Centralised, self-contained configuration for the XNLP training pipeline.

Every knob needed to train or resume a model lives here.  A ``TrainingConfig``
can be serialised to/from plain JSON so that a checkpoint directory is a
single source of truth for reproducibility.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import torch

# ── Checkpoint format ---------------------------------------------------------
# Bump this number whenever the checkpoint schema changes in an incompatible
# way.  The loader checks it and raises a clear error for unknown versions.
CHECKPOINT_FORMAT_VERSION = 1

# Fields that every valid checkpoint MUST contain.
REQUIRED_CHECKPOINT_FIELDS = (
    "checkpoint_format_version",
    "model_state_dict",
    "config",
    "training_config",
    "tokenizer_state",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "training_metadata",
)

# Fields required inside ``training_metadata``.
REQUIRED_METADATA_FIELDS = (
    "epoch",
    "global_step",
    "best_val_loss",
    "param_count",
    "vocab_size",
    "seed",
    "python_version",
    "torch_version",
    "device",
    "preset",
    "training_start_time",
)

# ── Model presets -------------------------------------------------------------
# Maps preset name → dict of overrides passed to XNLPConfig.
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny": dict(
        hidden_size=256, intermediate_size=640, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
    ),
    "small": dict(
        hidden_size=512, intermediate_size=1280, num_hidden_layers=8,
        num_attention_heads=8, num_key_value_heads=4, max_position_embeddings=1024,
    ),
    "medium": dict(
        hidden_size=768, intermediate_size=2048, num_hidden_layers=12,
        num_attention_heads=12, num_key_value_heads=6, max_position_embeddings=2048,
    ),
    "large": dict(
        hidden_size=1024, intermediate_size=2816, num_hidden_layers=16,
        num_attention_heads=16, num_key_value_heads=8, max_position_embeddings=2048,
    ),
    "xlarge": dict(
        hidden_size=1280, intermediate_size=3584, num_hidden_layers=20,
        num_attention_heads=20, num_key_value_heads=10, max_position_embeddings=2048,
    ),
}


@dataclass
class TrainingConfig:
    """All hyperparameters for a full independent XNLP training run."""

    # ── Paths ────────────────────────────────────────────────────────────────
    corpus_dir: str = "corpus"
    output_dir: str = "outputs"          # single checkpoint lives here
    checkpoint_name: str = "model.pt"    # the single self-contained file
    history_name: str = "training_history.json"

    # ── Model ────────────────────────────────────────────────────────────────
    preset: str = "tiny"                 # tiny / small / medium / large / xlarge
    vocab_size: int = 8000               # target vocab (tokenizer capped at this)
    tokenizer_min_freq: int = 2

    # ── Training ─────────────────────────────────────────────────────────────
    max_epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200
    grad_clip: float = 1.0
    dropout: float = 0.1
    seed: int = 42

    # ── Data splitting & batching ────────────────────────────────────────────
    max_seq_len: int = 256
    train_split: float = 0.8
    val_split: float = 0.15
    test_split: float = 0.05     # kept for completeness; evaluation uses val

    # ── Checkpointing & early stopping ───────────────────────────────────────
    save_every_n_epochs: int = 5
    save_best_only: bool = False         # if True only keep best + last
    early_stopping: bool = True
    patience: int = 5                   # epochs without improvement before stop
    min_delta: float = 1e-4

    # ── Generation (evaluated at end of training) ───────────────────────────
    generate_samples: bool = True
    sample_prompts: list = field(default_factory=lambda: [
        "Molo, ndiyabulela",
        "Umntu ngumntu ngabantu",
        "Umthetho wamaXhosa",
        "Izibongo zethu",
        "Ndiyavuya",
    ])
    max_new_tokens: int = 40
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    # ── Device ───────────────────────────────────────────────────────────────
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled: bool = field(default_factory=lambda: torch.cuda.is_available())
    num_workers: int = 0

    # ── Resume ───────────────────────────────────────────────────────────────
    resume_from: Optional[str] = None   # path to a checkpoint .pt file

    # ── Derived / runtime (not saved to JSON) ───────────────────────────────
    _tokenizer_state: Optional[Dict[str, Any]] = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------ helpers

    def validate(self) -> None:
        """Sanity-check configuration values."""
        assert self.preset in MODEL_PRESETS, f"Unknown preset '{self.preset}'. Choose from {list(MODEL_PRESETS)}"
        assert 0 < self.learning_rate <= 1.0, "learning_rate must be in (0, 1]"
        assert 1 <= self.batch_size, "batch_size must be >= 1"
        assert 1 <= self.max_epochs, "max_epochs must be >= 1"
        assert 0 < self.train_split + self.val_split + self.test_split <= 1.0 + 1e-6, \
            "train+val+test splits must sum to <= 1.0"

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict (excludes private/runtime fields)."""
        d = asdict(self)
        d.pop("_tokenizer_state", None)
        return d

    def to_json(self, path: str) -> None:
        """Save config to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.as_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def model_config_kwargs(self) -> Dict[str, Any]:
        """Build the keyword dict for ``XNLPConfig`` from preset + overrides."""
        cfg = dict(MODEL_PRESETS[self.preset])
        cfg.update({
            "vocab_size": self.vocab_size,
            "dropout_prob": self.dropout,
            "device": self.device,
        })
        return cfg


# ── Checkpoint validation ────────────────────────────────────────────────────
# Moved to a function so it can be used standalone (imported by tests, etc.)


def validate_checkpoint(ckpt: Dict[str, Any]) -> None:
    """
    Strictly validate a loaded checkpoint dict.

    Raises ``ValueError`` with an actionable message if any required field
    is missing or has an invalid structure.  Never silently accepts a
    partially valid checkpoint.
    """
    # Top-level keys
    missing = [f for f in REQUIRED_CHECKPOINT_FIELDS if f not in ckpt]
    if missing:
        raise ValueError(
            f"Checkpoint is missing required field(s): {missing}. "
            f"Required fields: {list(REQUIRED_CHECKPOINT_FIELDS)}"
        )

    # Format version
    version = ckpt["checkpoint_format_version"]
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format version: {version}. "
            f"This tool requires version {CHECKPOINT_FORMAT_VERSION}."
        )

    # config
    cfg = ckpt["config"]
    if not isinstance(cfg, dict) or "vocab_size" not in cfg:
        raise ValueError("Checkpoint 'config' must be a dict with 'vocab_size'.")

    # training_config
    tc = ckpt["training_config"]
    if not isinstance(tc, dict):
        raise ValueError("Checkpoint 'training_config' must be a dict.")

    # tokenizer_state
    tok = ckpt["tokenizer_state"]
    if not isinstance(tok, dict) or "token2id" not in tok or "merges" not in tok:
        raise ValueError(
            "Checkpoint 'tokenizer_state' must contain 'token2id' and 'merges'."
        )

    # optimizer_state_dict
    if not isinstance(ckpt["optimizer_state_dict"], dict):
        raise ValueError("Checkpoint 'optimizer_state_dict' must be a dict.")

    # scheduler_state_dict
    if not isinstance(ckpt["scheduler_state_dict"], dict):
        raise ValueError("Checkpoint 'scheduler_state_dict' must be a dict.")

    # training_metadata
    meta = ckpt["training_metadata"]
    if not isinstance(meta, dict):
        raise ValueError("Checkpoint 'training_metadata' must be a dict.")
    missing_meta = [f for f in REQUIRED_METADATA_FIELDS if f not in meta]
    if missing_meta:
        raise ValueError(
            f"Checkpoint 'training_metadata' is missing field(s): {missing_meta}. "
            f"Required: {list(REQUIRED_METADATA_FIELDS)}"
        )

    # Cross-check: vocab size in config vs tokenizer
    model_vocab = cfg["vocab_size"]
    tok_vocab = len(tok["token2id"])
    if model_vocab != tok_vocab:
        raise ValueError(
            f"Vocabulary size mismatch: model config says {model_vocab}, "
            f"but tokenizer has {tok_vocab} tokens."
        )

