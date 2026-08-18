"""
XNLP Full Model Training Pipeline
==================================
Trains a complete LLaMA-style transformer from scratch on isiXhosa text.
- Builds tokenizer from corpus
- Trains model weights via next-token prediction
- Saves trained model weights (.pt)
- NOT retrieval-based - full end-to-end weight training

Usage:
    python train_full_model.py

Requirements:
    pip install torch numpy tqdm
"""

import sys
import os
import json
import math
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from core_llm.architecture import XNLPCoreLLM, XNLPConfig
from core_llm.tokenizer import XNLPTokenizer, SpecialTokens


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    corpus_dir: str = "corpus"
    output_dir: str = "trained_model"
    vocab_size: int = 8000
    hidden_size: int = 256
    intermediate_size: int = 640
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 4
    max_seq_len: int = 256
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_epochs: int = 50
    grad_clip: float = 1.0
    dropout: float = 0.1
    save_every: int = 5
    eval_every: int = 10
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Corpus Loading ───────────────────────────────────────────────────────────

def load_corpus(corpus_dir: str) -> List[str]:
    """Load all .txt files from corpus directory."""
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    all_sentences = []
    for txt_file in sorted(corpus_path.glob("*.txt")):
        print(f"  Loading: {txt_file.name}")
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
        sentences = [s.strip() for s in text.split("\n") if s.strip()]
        all_sentences.extend(sentences)

    print(f"  Total sentences loaded: {len(all_sentences)}")
    return all_sentences


# ─── Tokenizer Training ──────────────────────────────────────────────────────

def train_tokenizer(sentences: List[str], vocab_size: int = 8000) -> XNLPTokenizer:
    """Train BPE tokenizer from corpus sentences."""
    print("\n[1/4] Training BPE tokenizer...")
    tokenizer = XNLPTokenizer(vocab_size=vocab_size, min_frequency=2)
    tokenizer.train(sentences, verbose=True)
    return tokenizer


# ─── Dataset ──────────────────────────────────────────────────────────────────

class XhosaTextDataset(Dataset):
    """Tokenized text dataset for language model training."""

    def __init__(self, sentences: List[str], tokenizer: XNLPTokenizer, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []

        for sent in sentences:
            ids = tokenizer.encode(sent, add_special_tokens=True)
            if len(ids) < 4:
                continue
            self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        ids = torch.tensor(ids, dtype=torch.long)
        labels = ids.clone()
        return {"input_ids": ids, "labels": labels}


def collate_fn(batch, pad_token_id=0, max_len=256):
    """Pad batch to same length."""
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]

    padded_inputs = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, (inp, lab) in enumerate(zip(input_ids, labels)):
        length = min(len(inp), max_len)
        padded_inputs[i, :length] = inp[:length]
        padded_labels[i, :length] = lab[:length]

    return {"input_ids": padded_inputs, "labels": padded_labels}


# ─── Learning Rate Schedule ───────────────────────────────────────────────────

def get_lr(step, warmup_steps, max_steps, base_lr):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ─── Training Loop ────────────────────────────────────────────────────────────

def train(config: TrainConfig):
    """Full model training loop."""
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    os.makedirs(config.output_dir, exist_ok=True)

    # ── Load Corpus ──
    print("=" * 60)
    print("XNLP Full Model Training Pipeline")
    print("=" * 60)
    print(f"\nDevice: {config.device}")
    print(f"\n[0/4] Loading corpus from '{config.corpus_dir}/'...")
    sentences = load_corpus(config.corpus_dir)
    if len(sentences) == 0:
        print("ERROR: No sentences found in corpus!")
        return

    # ── Train Tokenizer ──
    tokenizer = train_tokenizer(sentences, config.vocab_size)
    tokenizer.save_pretrained(os.path.join(config.output_dir, "tokenizer"))

    vocab_actual = tokenizer.vocab_size_actual
    print(f"  Final vocab size: {vocab_actual}")

    # ── Build Dataset ──
    print("\n[2/4] Building dataset...")
    dataset = XhosaTextDataset(sentences, tokenizer, config.max_seq_len)
    print(f"  Dataset size: {len(dataset)} examples")

    pad_id = tokenizer.pad_token_id
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id, config.max_seq_len),
        num_workers=0,
    )

    # ── Build Model ──
    print("\n[3/4] Building model...")
    model_config = XNLPConfig(
        vocab_size=vocab_actual,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_heads,
        num_key_value_heads=config.num_kv_heads,
        max_position_embeddings=config.max_seq_len,
        dropout_prob=config.dropout,
        device=config.device,
    )
    model = XNLPCoreLLM(model_config)
    model.to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,}")
    print(f"  Trainable:        {trainable_params:,}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    max_steps = config.max_epochs * len(loader)

    # ── Training ──
    print(f"\n[4/4] Training for {config.max_epochs} epochs ({max_steps} steps)...")
    print("=" * 60)

    global_step = 0
    best_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(config.device)
            labels = batch["labels"].to(config.device)

            # Update learning rate
            lr = get_lr(global_step, config.warmup_steps, max_steps, config.learning_rate)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % 50 == 0:
                avg_loss = epoch_loss / max(epoch_steps, 1)
                print(
                    f"  Epoch {epoch+1}/{config.max_epochs} | "
                    f"Step {global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg: {avg_loss:.4f} | "
                    f"LR: {lr:.2e}"
                )

        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t0
        history["train_loss"].append(avg_epoch_loss)
        history["lr"].append(lr)

        print(f"\nEpoch {epoch+1}/{config.max_epochs} complete "
              f"| Loss: {avg_epoch_loss:.4f} | Time: {elapsed:.1f}s")

        # ── Save checkpoint ──
        if (epoch + 1) % config.save_every == 0 or avg_epoch_loss < best_loss:
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                tag = "best"
            else:
                tag = f"epoch{epoch+1}"

            save_path = os.path.join(config.output_dir, f"model_{tag}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_epoch_loss,
                "config": {
                    "vocab_size": vocab_actual,
                    "hidden_size": config.hidden_size,
                    "intermediate_size": config.intermediate_size,
                    "num_layers": config.num_layers,
                    "num_heads": config.num_heads,
                    "num_kv_heads": config.num_kv_heads,
                    "max_seq_len": config.max_seq_len,
                },
            }, save_path)
            print(f"  Saved checkpoint: {save_path}")

    # ── Save final model ──
    final_path = os.path.join(config.output_dir, "best_model.pt")
    torch.save({
        "epoch": config.max_epochs,
        "model_state_dict": model.state_dict(),
        "loss": best_loss,
        "config": {
            "vocab_size": vocab_actual,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "num_kv_heads": config.num_kv_heads,
            "max_seq_len": config.max_seq_len,
        },
    }, final_path)

    # Save training history
    history_path = os.path.join(config.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Best loss:     {best_loss:.4f}")
    print(f"  Model saved:   {final_path}")
    print(f"  Tokenizer:     {config.output_dir}/tokenizer/")
    print(f"  History:       {history_path}")
    print(f"  Total params:  {total_params:,}")
    print("=" * 60)

    return model, tokenizer


# ─── Generation Test ──────────────────────────────────────────────────────────

def test_generation(model, tokenizer, config):
    """Test text generation with the trained model."""
    model.eval()
    print("\n--- Text Generation Test ---")

    prompts = [
        "Umntu ngumntu",
        "Imbongi yethu",
        "Ityala lamawele",
        "UMqhayi waseNtab'ozuko",
        "Izibongo zamaXhosa",
    ]

    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([ids], dtype=torch.long, device=config.device)

        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=30,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        generated = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        print(f"\n  Prompt:  {prompt}")
        print(f"  Output:  {generated}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = TrainConfig()
    model, tokenizer = train(config)
    if model is not None:
        test_generation(model, tokenizer, config)
