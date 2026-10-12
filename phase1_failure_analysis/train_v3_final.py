#!/usr/bin/env python3
"""
Final training script for XNLP v3 with clean corpus and clean tokenizer.
This is a focused, minimal training script.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\xnlp')

print("XNLP V3 TRAINING - CLEAN DATA, CLEAN TOKENIZER")
print("=" * 80)

import torch
import json
import time
from pathlib import Path

# Load clean corpus
corpus_path = Path(r'D:\xnlp\corpus_v2\v2_training_corpus_clean_final.txt')
with open(corpus_path, 'r', encoding='utf-8') as f:
    sentences = [line.strip() for line in f if line.strip()]

print(f"Corpus: {len(sentences)} clean sentences")

# Load clean tokenizer
from xnlp_trainer.data import tokenizer_from_state_dict
with open(r'D:\xnlp\artifacts\xnlp_v3_clean_tokenizer\tokenizer_state.json', 'r', encoding='utf-8') as f:
    tokenizer_state = json.load(f)
tokenizer = tokenizer_from_state_dict(tokenizer_state)
print(f"Tokenizer: vocab={tokenizer.vocab_size}, merges={tokenizer.num_merges}")

# Tokenize
from xnlp_trainer.data import tokenize_texts
max_seq_len = 256
tokenized = tokenize_texts(sentences, tokenizer, max_seq_len)
print(f"Tokenized: {len(tokenized)} sequences")

# Split
import random
random.seed(42)
random.shuffle(tokenized)
n = len(tokenized)
train_data = tokenized[:int(n * 0.8)]
val_data = tokenized[int(n * 0.8):int(n * 0.9)]
print(f"Train: {len(train_data)}, Val: {len(val_data)}")

# Create datasets
from xnlp_trainer.data import XhosaTextDataset, make_collate_fn
max_seq_len = 256
pad_id = tokenizer.token2id[tokenizer.special_tokens.pad_token]

train_ds = XhosaTextDataset(train_data, max_seq_len)
val_ds = XhosaTextDataset(val_data, max_seq_len)

collate = make_collate_fn(pad_id, max_seq_len)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate, num_workers=0)

# Model
from core_llm.architecture import XNLPCoreLLM, XNLPConfig
from xnlp_trainer.config import MODEL_PRESETS

config = XNLPConfig(vocab_size=tokenizer.vocab_size, **MODEL_PRESETS['tiny'])
model = XNLPCoreLLM(config).to('cpu')
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

# Optimizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
scheduler = LambdaLR(optimizer, lambda s: min(s / 50, 1.0))

# Training
model.train()
best_val = float('inf')
start = time.time()

for epoch in range(30):
    model.train()
    train_loss = 0.0
    
    for batch in train_loader:
        batch = {k: v.to('cpu') for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(**batch)
        out['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        train_loss += out['loss'].item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to('cpu') for k, v in batch.items()}
            out = model(**batch)
            val_loss += out['loss'].item()
    
    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    
    elapsed = time.time() - start
    print(f"Epoch {epoch+1:2d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | Time: {elapsed:.0f}s")
    
    if avg_val < best_val:
        best_val = avg_val
        # Save best
        ckpt = {
            "checkpoint_format_version": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
            "training_config": {"preset": "tiny", "lr": 3e-4, "batch_size": 8},
            "tokenizer_state": tokenizer_state,
            "training_metadata": {
                "epoch": epoch + 1,
                "global_step": epoch * len(train_loader),
                "best_val_loss": best_val,
                "param_count": sum(p.numel() for p in model.parameters()),
                "vocab_size": tokenizer.vocab_size,
                "seed": 42,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "torch_version": torch.__version__,
                "device": "cpu",
                "preset": "tiny",
                "training_start_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            }
        }
        Path(r'D:\xnlp\artifacts\xnlp_v3_clean\model').mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, r'D:\xnlp\artifacts\xnlp_v3_clean\model\best_model.pt')
        print("  -> Saved best model")
    else:
        print("  -> No improvement")
    
    # Early stopping
    if epoch > 5 and avg_val >= best_val * 1.001:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Save final
ckpt = {
    "checkpoint_format_version": 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "config": {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
    "training_config": {"preset": "tiny", "lr": 3e-4, "batch_size": 8},
    "tokenizer_state": tokenizer_state,
    "training_metadata": {
        "epoch": epoch + 1,
        "global_step": epoch * len(train_loader),
        "best_val_loss": best_val,
        "final_val_loss": avg_val,
        "param_count": sum(p.numel() for p in model.parameters()),
        "vocab_size": tokenizer.vocab_size,
        "seed": 42,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch_version": torch.__version__,
        "device": "cpu",
        "preset": "tiny",
        "training_start_time": time.strftime('%Y-%m-%d %H:%M:%S'),
    }
}
torch.save(ckpt, r'D:\xnlp\artifacts\xnlp_v3_clean\model\last_model.pt')

print(f"\nTraining complete in {time.time() - start:.0f}s")
print(f"Best val loss: {best_val:.4f}")
