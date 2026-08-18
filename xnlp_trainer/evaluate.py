"""
XNLP Trainer – Evaluation
==========================
Validation loss, perplexity and sample generation utilities.

These functions are deliberately lightweight so they can be called from
the training loop or standalone.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

__all__ = [
    "compute_loss",
    "compute_perplexity",
    "evaluate",
    "generate_samples",
]


@torch.no_grad()
def compute_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    device: str,
) -> float:
    """Forward pass and return scalar cross-entropy loss."""
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    outputs = model(input_ids=input_ids, labels=labels)
    return outputs["loss"].item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    device: str,
) -> float:
    """Compute average loss over a dataloader (no gradient)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        total_loss += outputs["loss"].item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def compute_perplexity(loss: float) -> float:
    """Convert cross-entropy loss to perplexity."""
    return math.exp(min(loss, 20.0))  # clamp to avoid overflow


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    tokenizer,
    prompts: list,
    device: str,
    max_new_tokens: int = 40,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> list:
    """Generate text for each prompt and return list of (prompt, text) tuples."""
    model.eval()
    results: list = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            results.append((prompt, ""))
            continue
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            inp,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        results.append((prompt, text))
    model.train()
    return results
