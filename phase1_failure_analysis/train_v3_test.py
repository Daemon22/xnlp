#!/usr/bin/env python3
"""
Quick test: train for just 1 epoch to verify the pipeline works.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\xnlp')

import torch
import json
from pathlib import Path

# Quick test with small data
corpus_path = Path(r'D:\xnlp\corpus_v2\v2_training_corpus_clean_final.txt')
with open(corpus_path, 'r', encoding='utf-8') as f:
    sentences = [line.strip() for line in f if line.strip()][:1000]  # Only 1000 sentences

print(f"Quick test: {len(sentences)} sentences")

# Load tokenizer
from xnlp_trainer.data import tokenizer_from_state_dict
with open(r'D:\xnlp\artifacts\xnlp_v3_clean_tokenizer\tokenizer_state.json', 'r', encoding='utf-8') as f:
    tokenizer_state = json.load(f)
tokenizer = tokenizer_from_state_dict(tokenizer_state)

# Tokenize
from xnlp_trainer.data import tokenize_texts
tokenized = tokenize_texts(sentences, tokenizer, 256)

# Split
import random
random.seed(42)
random.shuffle(tokenized)
train_data = tokenized[:800]
val_data = tokenized[800:]

# Datasets
from xnlp_trainer.data import XhosaTextDataset, make_collate_fn
pad_id = tokenizer.token2id[tokenizer.special_tokens.pad_token]
train_ds = XhosaTextDataset(train_data, 256)
val_ds = XhosaTextDataset(val_data, 256)

collate = make_collate_fn(pad_id, 256)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate, num_workers=0)

print(f"Data: {len(train_loader)} train batches, {len(val_loader)} val batches")

# Model
from core_llm.architecture import XNLPCoreLLM, XNLPConfig
from xnlp_trainer.config import MODEL_PRESETS

config = XNLPConfig(vocab_size=tokenizer.vocab_size, **MODEL_PRESETS['tiny'])
model = XNLPCoreLLM(config).to('cpu')

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

print("Training for 1 epoch (quick test)...")
model.train()
for batch_idx, batch in enumerate(train_loader):
    batch = {k: v.to('cpu') for k, v in batch.items()}
    optimizer.zero_grad()
    out = model(**batch)
    out['loss'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if batch_idx % 10 == 0:
        print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {out['loss'].item():.4f}")

print("Quick test complete!")
print("Full training script should work.")
