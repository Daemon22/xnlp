#!/usr/bin/env python3
"""
Train XNLP v3 with clean corpus and clean tokenizer.
This should eliminate the digit contamination that caused bad generations.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\xnlp')

print("=" * 80)
print("TRAINING XNLP V3 - CLEAN CORPUS, CLEAN TOKENIZER")
print("=" * 80)
print()

from xnlp_trainer.config import TrainingConfig
from xnlp_trainer.trainer import XNLPTrainer
from pathlib import Path
import time

# Use v3 data and tokenizer
ARTIFACT_DIR = Path(r'D:\xnlp\artifacts\xnlp_v3_clean')
OUTPUT_DIR = ARTIFACT_DIR / "model"
TOKENIZER_DIR = Path(r'D:\xnlp\artifacts\xnlp_v3_clean_tokenizer')

# Data paths
TRAIN_FILE = Path(r'D:\xnlp\data\processed\v3\splits\train.txt')
VAL_FILE = Path(r'D:\xnlp\data\processed\v3\splits\val.txt')

print(f"Training data: {TRAIN_FILE}")
print(f"Validation data: {VAL_FILE}")
print(f"Tokenizer: {TOKENIZER_DIR}")
print(f"Output: {OUTPUT_DIR}")
print()

# Check files exist
assert TRAIN_FILE.exists(), f"Train file not found: {TRAIN_FILE}"
assert VAL_FILE.exists(), f"Val file not found: {VAL_FILE}"
assert (TOKENIZER_DIR / 'tokenizer_state.json').exists(), f"Tokenizer not found: {TOKENIZER_DIR}"

print("All files verified. Starting training...")
print()

# Training configuration
config = TrainingConfig(
    corpus_dir=str(TOKENIZER_DIR.parent),  # Directory containing tokenizer
    tokenizer_path=str(TOKENIZER_DIR / 'tokenizer_state.json'),
    output_dir=str(OUTPUT_DIR),
    checkpoint_name="best_model.pt",
    preset="tiny",
    vocab_size=4096,
    tokenizer_min_freq=2,
    max_epochs=30,
    batch_size=8,
    learning_rate=3e-4,
    weight_decay=0.1,
    warmup_steps=50,
    grad_clip=1.0,
    dropout=0.1,
    seed=42,
    max_seq_len=256,
    train_split=1.0,  # We provide our own train file
    val_split=0.0,    # We provide our own val file
    test_split=0.0,
    patience=5,
    early_stopping=True,
    save_every_n_epochs=5,
    sample_prompts=[
        "Molo, ndiyabulela.",
        "Ndiyakwazi ukuthetha isiXhosa.",
        "Ityala lamawele.",
        "Umntu ngumntu ngabantu.",
        "Ubuntu buhle.",
    ],
    max_new_tokens=50,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.1,
    device="cpu",
    amp_enabled=False,
)

# Save config
config_path = ARTIFACT_DIR / "training_config.json"
config_path.parent.mkdir(parents=True, exist_ok=True)
config.to_json(str(config_path))

print("Configuration:")
print(f"  Preset: {config.preset}")
print(f"  Vocab size: {config.vocab_size}")
print(f"  Max epochs: {config.max_epochs}")
print(f"  Batch size: {config.batch_size}")
print(f"  Learning rate: {config.learning_rate}")
print()

# Start training
started = time.time()
trainer = XNLPTrainer(config)
trainer.train()
elapsed = time.time() - started

print()
print(f"Training completed in {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")
print()

# Load and test the best model
import torch
from xnlp_trainer.inference import XNLPPredictor

predictor = XNLPPredictor.load(trainer.best_path, device="cpu")

print("=" * 80)
print("TESTING NEW MODEL")
print("=" * 80)
print()

# Test prompts
test_prompts = [
    "Molo. ",
    "Namhlanje, ",
    "Ndiyakwazi ukuthetha isiXhosa. ",
    "Ityala lamawele. ",
]

for prompt in test_prompts:
    output = predictor.generate(
        prompt,
        max_new_tokens=60,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True
    )
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")
    print()

print("=" * 80)
print("XNLP V3 TRAINING COMPLETE")
print(f"Best checkpoint: {trainer.best_path}")
print("=" * 80)
