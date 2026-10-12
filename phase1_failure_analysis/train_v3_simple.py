#!/usr/bin/env python3
"""
Simplified training script that uses pre-cleaned data and pre-trained tokenizer.
This avoids the complexity of the full trainer and lets us train directly.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\xnlp')

print("=" * 80)
print("TRAINING XNLP V3 - SIMPLIFIED APPROACH")
print("=" * 80)
print()

import torch
from pathlib import Path
import time
import json
import os

# Paths
CLEAN_CORPUS = Path(r'D:\xnlp\corpus_v2\v2_training_corpus_clean_final.txt')
TOKENIZER_STATE = Path(r'D:\xnlp\artifacts\xnlp_v3_clean_tokenizer\tokenizer_state.json')
OUTPUT_DIR = Path(r'D:\xnlp\artifacts\xnlp_v3_clean\model')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Clean corpus: {CLEAN_CORPUS}")
print(f"Tokenizer: {TOKENIZER_STATE}")
print(f"Output: {OUTPUT_DIR}")
print()

# Load sentences
with open(CLEAN_CORPUS, 'r', encoding='utf-8') as f:
    sentences = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(sentences)} clean sentences")
print()

# Load tokenizer
from xnlp_trainer.data import tokenizer_from_state_dict
with open(TOKENIZER_STATE, 'r', encoding='utf-8') as f:
    tokenizer_state = json.load(f)
tokenizer = tokenizer_from_state_dict(tokenizer_state)
print(f"Tokenizer loaded: vocab={tokenizer.vocab_size}, merges={tokenizer.num_merges}")
print()

# Tokenize all sentences
from xnlp_trainer.data import tokenize_texts, XhosaTextDataset, make_collate_fn

tokenized = tokenize_texts(sentences, tokenizer)
print(f"Tokenized {len(tokenized)} sentences")
print()

# Split into train/val
import random
random.seed(42)
random.shuffle(tokenized)

n = len(tokenized)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

train_data = tokenized[:train_end]
val_data = tokenized[train_end:val_end]

print(f"Train: {len(train_data)} sentences")
print(f"Val: {len(val_data)} sentences")
print()

# Create datasets
max_seq_len = 256
pad_token_id = tokenizer.token2id[tokenizer.special_tokens.pad_token]

train_dataset = XhosaTextDataset(train_data, max_seq_len)
val_dataset = XhosaTextDataset(val_data, max_seq_len)

# Create data loaders
batch_size = 8
collate_fn = make_collate_fn(pad_token_id, max_seq_len)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    collate_fn=collate_fn, num_workers=0
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    collate_fn=collate_fn, num_workers=0
)

print(f"Data loaders created")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print()

# Create model
from core_llm.architecture import XNLPCoreLLM, XNLPConfig
from xnlp_trainer.config import MODEL_PRESETS

model_config = XNLPConfig(
    vocab_size=tokenizer.vocab_size,
    **MODEL_PRESETS['tiny']
)
model = XNLPCoreLLM(model_config)
model.to('cpu')

print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
print()

# Training setup
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

device = 'cpu'
model.to(device)

optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

# Learning rate schedule
def lr_lambda(current_step):
    if current_step < 50:  # warmup
        return float(current_step) / 50
    return 1.0

scheduler = LambdaLR(optimizer, lr_lambda)

# Training loop
model.train()
global_step = 0
best_val_loss = float('inf')
max_epochs = 30
patience = 5
early_stop_counter = 0

print("Starting training loop...")
print()

for epoch in range(max_epochs):
    epoch_start = time.time()
    model.train()
    
    total_train_loss = 0.0
    train_steps = 0
    
    for batch_idx, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs['loss']
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_train_loss += loss.item()
        train_steps += 1
        global_step += 1
        
        if (batch_idx + 1) % 100 == 0:
            avg_train_loss = total_train_loss / train_steps
            print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, "
                  f"Train Loss: {avg_train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
    
    # Validation
    model.eval()
    total_val_loss = 0.0
    val_steps = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs['loss']
            total_val_loss += loss.item()
            val_steps += 1
    
    avg_train_loss = total_train_loss / train_steps if train_steps > 0 else 0
    avg_val_loss = total_val_loss / val_steps if val_steps > 0 else 0
    
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1}/{max_epochs} | "
          f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
          f"Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e}")
    
    # Checkpointing
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        
        # Save best checkpoint
        checkpoint = {
            "checkpoint_format_version": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": {
                "vocab_size": model_config.vocab_size,
                "hidden_size": model_config.hidden_size,
                "intermediate_size": model_config.intermediate_size,
                "num_hidden_layers": model_config.num_hidden_layers,
                "num_attention_heads": model_config.num_attention_heads,
                "num_key_value_heads": model_config.num_key_value_heads,
                "max_position_embeddings": model_config.max_position_embeddings,
            },
            "training_config": {
                "preset": "tiny",
                "batch_size": batch_size,
                "learning_rate": 3e-4,
                "max_epochs": max_epochs,
                "seed": 42,
            },
            "tokenizer_state": tokenizer_state,
            "training_metadata": {
                "epoch": epoch + 1,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
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
        
        best_path = OUTPUT_DIR / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"  New best model saved to {best_path}")
    else:
        early_stop_counter += 1
        print(f"  No improvement for {early_stop_counter}/{patience} epochs")
    
    # Save last checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch + 1 == max_epochs:
        last_path = OUTPUT_DIR / f"checkpoint_epoch{epoch+1}.pt"
        checkpoint = {
            "checkpoint_format_version": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": model_config.__dict__,
            "training_config": {
                "preset": "tiny",
                "batch_size": batch_size,
                "learning_rate": 3e-4,
                "max_epochs": max_epochs,
            },
            "tokenizer_state": tokenizer_state,
            "training_metadata": {
                "epoch": epoch + 1,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "current_val_loss": avg_val_loss,
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
        torch.save(checkpoint, last_path)
        print(f"  Checkpoint saved to {last_path}")
    
    # Early stopping
    if early_stop_counter >= patience and epoch + 1 < max_epochs:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break
    
    print()

# Save final model
final_path = OUTPUT_DIR / "last_model.pt"
checkpoint = {
    "checkpoint_format_version": 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "config": model_config.__dict__,
    "training_config": {
        "preset": "tiny",
        "batch_size": batch_size,
        "learning_rate": 3e-4,
        "max_epochs": max_epochs,
    },
    "tokenizer_state": tokenizer_state,
    "training_metadata": {
        "epoch": max_epochs,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "final_val_loss": avg_val_loss,
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
torch.save(checkpoint, final_path)

print("=" * 80)
print(f"TRAINING COMPLETE")
print(f"Best model: {best_path}")
print(f"Final model: {final_path}")
print(f"Best validation loss: {best_val_loss:.4f}")
print("=" * 80)
