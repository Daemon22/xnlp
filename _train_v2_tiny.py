#!/usr/bin/env python
"""
Phase V2-16: V2 Model Scaling Experiment (Tiny)
Trains the V2 tiny model (4.66M params, 2048-vocab) on the V2 corpus.
"""
import sys, io, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True)
import torch
torch.set_num_threads(os.cpu_count() or 4)

from core_llm.tokenizer import XNLPTokenizer
from core_llm.architecture import XNLPConfig, XNLPCoreLLM
from xnlp_trainer.config import TrainingConfig, MODEL_PRESETS, CHECKPOINT_FORMAT_VERSION
from xnlp_trainer.data import tokenize_texts, build_split_indices, XhosaTextDataset, make_collate_fn
from xnlp_trainer.evaluate import evaluate, compute_perplexity, generate_samples
from torch.utils.data import DataLoader

# ─── Load V2 corpus ───────────────────────────────────────────────────────
v2_corpus = 'data/processed/v2/training_corpus.txt'
with open(v2_corpus, 'r', encoding='utf-8') as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]
print(f'V2 corpus: {len(lines)} unique lines')

# Load V2 tokenizer
v2_tokenizer = XNLPTokenizer.load_pretrained('tokenizer_v2')
print(f'V2 tokenizer: {v2_tokenizer.vocab_size_actual} tokens')

MAX_SEQ_LEN = 256
BATCH_SIZE = 16
SEED = 42

# ─── Split corpus ──────────────────────────────────────────────────────────
train_idx, val_idx, test_idx = build_split_indices(len(lines), 0.8, 0.1, SEED)
train_sents = [lines[i] for i in train_idx]
val_sents = [lines[i] for i in val_idx]
print(f'Train: {len(train_sents)}, Val: {len(val_sents)}')

# ─── Tokenize ──────────────────────────────────────────────────────────────
print('Tokenizing...')
train_ids = tokenize_texts(train_sents, v2_tokenizer, MAX_SEQ_LEN)
val_ids = tokenize_texts(val_sents, v2_tokenizer, MAX_SEQ_LEN)

train_ds = XhosaTextDataset(train_ids, MAX_SEQ_LEN)
val_ds = XhosaTextDataset(val_ids, MAX_SEQ_LEN)
collate = make_collate_fn(v2_tokenizer.pad_token_id, MAX_SEQ_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0)
print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')

# ─── Build tiny model ─────────────────────────────────────────────────────
pc = MODEL_PRESETS['tiny']
model_cfg = XNLPConfig(
    vocab_size=v2_tokenizer.vocab_size_actual,
    hidden_size=pc['hidden_size'], intermediate_size=pc['intermediate_size'],
    num_hidden_layers=pc['num_hidden_layers'],
    num_attention_heads=pc['num_attention_heads'],
    num_key_value_heads=pc['num_key_value_heads'],
    max_position_embeddings=pc['max_position_embeddings'],
    dropout_prob=0.1, device='cpu',
)
model = XNLPCoreLLM(model_cfg)
param_count = sum(p.numel() for p in model.parameters())
print(f'Tiny model: {param_count:,} params, vocab={v2_tokenizer.vocab_size_actual}')

# ─── Optimizer & Scheduler ─────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
MAX_EPOCHS = 30
PATIENCE = 5
WARMUP = 50
total_steps = MAX_EPOCHS * len(train_loader)
print(f'Total steps: {total_steps}, warmup: {WARMUP}')

import math
def lr_lambda(step):
    if step < WARMUP:
        return step / max(WARMUP, 1)
    progress = (step - WARMUP) / max(total_steps - WARMUP, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# ─── Training loop ─────────────────────────────────────────────────────────
device = 'cpu'
best_val_loss = float('inf')
global_step = 0
history = {'train_loss': [], 'val_loss': [], 'lr': [], 'epoch': []}
early_stop_counter = 0

print(f'\nStarting training: {len(train_loader)} batches/epoch, {MAX_EPOCHS} max epochs')
start_time = time.time()

for epoch in range(MAX_EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    t0 = time.time()
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs['loss']
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        n_batches += 1
        global_step += 1
        
        if global_step % 100 == 0:
            avg = epoch_loss / n_batches
            lr = scheduler.get_last_lr()[0]
            print(f'  Ep {epoch+1}/{MAX_EPOCHS} step {global_step} loss {loss.item():.4f} avg {avg:.4f} lr {lr:.2e}')
    
    avg_loss = epoch_loss / n_batches
    epoch_time = time.time() - t0
    history['train_loss'].append(avg_loss)
    history['lr'].append(scheduler.get_last_lr()[0])
    
    # Validation
    val_loss = evaluate(model, val_loader, device)
    ppl = compute_perplexity(val_loss)
    history['val_loss'].append(val_loss)
    history['epoch'].append(epoch + 1)
    
    print(f'\n[epoch {epoch+1}/{MAX_EPOCHS}] train_loss={avg_loss:.4f} val_loss={val_loss:.4f} ppl={ppl:.1f} time={epoch_time:.1f}s')
    
    # Save best
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        early_stop_counter = 0
        
        output_dir = 'outputs_v2_tiny'
        os.makedirs(output_dir, exist_ok=True)
        ckpt = {
            'checkpoint_format_version': CHECKPOINT_FORMAT_VERSION,
            'model_state_dict': model.state_dict(),
            'config': {
                'vocab_size': v2_tokenizer.vocab_size_actual,
                'hidden_size': model_cfg.hidden_size,
                'intermediate_size': model_cfg.intermediate_size,
                'num_hidden_layers': model_cfg.num_hidden_layers,
                'num_attention_heads': model_cfg.num_attention_heads,
                'num_key_value_heads': model_cfg.num_key_value_heads,
                'max_position_embeddings': model_cfg.max_position_embeddings,
                'pad_token_id': model_cfg.pad_token_id,
                'bos_token_id': model_cfg.bos_token_id,
                'eos_token_id': model_cfg.eos_token_id,
                'unk_token_id': model_cfg.unk_token_id,
                'rope_theta': model_cfg.rope_theta,
                'layer_norm_eps': model_cfg.layer_norm_eps,
                'dropout_prob': model_cfg.dropout_prob,
            },
            'training_config': TrainingConfig(
                preset='tiny', vocab_size=8000, max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
                learning_rate=3e-4, warmup_steps=WARMUP, max_seq_len=MAX_SEQ_LEN,
            ).as_dict(),
            'tokenizer_state': __import__('xnlp_trainer.data', fromlist=['tokenizer_to_state_dict']).tokenizer_to_state_dict(v2_tokenizer),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'training_metadata': {
                'epoch': epoch + 1, 'global_step': global_step,
                'best_val_loss': best_val_loss, 'param_count': param_count,
                'vocab_size': v2_tokenizer.vocab_size_actual, 'seed': SEED,
                'python_version': '3.14.5', 'torch_version': torch.__version__,
                'device': device, 'preset': 'tiny', 'max_seq_len': MAX_SEQ_LEN,
                'batch_size': BATCH_SIZE, 'learning_rate': 3e-4, 'weight_decay': 0.1,
                'warmup_steps': WARMUP, 'max_steps': total_steps,
                'steps_per_epoch': len(train_loader),
                'training_start_time': '2026-08-18T12:00:00+00:00',
                'training_end_time': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                'rng_state': {'torch_cpu': torch.get_rng_state()},
            },
        }
        torch.save(ckpt, os.path.join(output_dir, 'best_model.pt'))
        torch.save(ckpt, os.path.join(output_dir, 'last_model.pt'))
        # Save history
        with open(os.path.join(output_dir, 'training_history.json'), 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        print(f'  -> Saved best_model.pt (val_loss={best_val_loss:.4f})')
    else:
        early_stop_counter += 1
        print(f'  -> No improvement for {early_stop_counter}/{PATIENCE}')
    
    if early_stop_counter >= PATIENCE:
        print(f'\nEarly stopping at epoch {epoch+1}')
        break

total_time = time.time() - start_time
print(f'\n{"="*60}')
print(f'V2 Tiny Training Complete')
print(f'  Best val_loss: {best_val_loss:.4f}')
print(f'  Perplexity: {compute_perplexity(best_val_loss):.1f}')
print(f'  Best epoch: {best_epoch}')
print(f'  Total time: {total_time:.1f}s')

# Generate samples
print(f'\nSample outputs:')
prompts = ['Molo, ndiyabulela', 'Umntu ngumntu ngabantu', 'Umthetho wamaXhosa', 'Izibongo zethu', 'Ndiyavuya']
samples = generate_samples(model, v2_tokenizer, prompts, device, max_new_tokens=50, temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.1)
for p, t in samples:
    print(f'  "{p}" -> "{t}"')

# Save results
os.makedirs('data/reports', exist_ok=True)
results = {
    'tiny_v2': {
        'preset': 'tiny', 'param_count': param_count,
        'vocab_size': v2_tokenizer.vocab_size_actual,
        'best_val_loss': round(best_val_loss, 4),
        'perplexity': round(compute_perplexity(best_val_loss), 1),
        'best_epoch': best_epoch, 'training_time_seconds': round(total_time, 1),
        'max_epochs': MAX_EPOCHS, 'patience': PATIENCE,
        'batch_size': BATCH_SIZE, 'max_seq_len': MAX_SEQ_LEN,
        'learning_rate': 3e-4, 'warmup_steps': WARMUP,
        'train_examples': len(train_ds), 'val_examples': len(val_ds),
        'train_batches': len(train_loader), 'global_step': global_step,
        'best_model_path': 'outputs_v2_tiny/best_model.pt',
        'training_history': history,
        'samples': {p: t for p, t in samples},
    }
}
with open('data/reports/v2_tiny_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f'\nSaved: data/reports/v2_tiny_results.json')
