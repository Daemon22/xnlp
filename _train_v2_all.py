#!/usr/bin/env python
"""
Phase V2-16: V2 Model Scaling Experiment

Trains XNLP models (Tiny, Small) on the V2 corpus using the pre-trained
V2 tokenizer (2,048 vocab). Optimized for CPU: max_seq_len=64 (99.4% of sentences fit),
batch_size=8 to minimize per-epoch time.
"""
import sys, io, os, json, time, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True)
import torch
torch.set_num_threads(4)

from core_llm.tokenizer import XNLPTokenizer
from core_llm.architecture import XNLPConfig, XNLPCoreLLM
from xnlp_trainer.config import TrainingConfig, MODEL_PRESETS, CHECKPOINT_FORMAT_VERSION
from xnlp_trainer.data import tokenize_texts, build_split_indices, XhosaTextDataset, make_collate_fn, tokenizer_to_state_dict
from xnlp_trainer.evaluate import evaluate, compute_perplexity, generate_samples
from torch.utils.data import DataLoader

V2_CORPUS = 'data/processed/v2/training_corpus.txt'
PROMPTS = ['Molo, ndiyabulela', 'Umntu ngumntu ngabantu', 'Umthetho wamaXhosa', 'Izibongo zethu', 'Ndiyavuya', 'Nkosi Sikelel', 'UMqhayi', 'Ubuntu']

# ─── Load corpus and tokenizer ──────────────────────────────────────────────
with open(V2_CORPUS, 'r', encoding='utf-8') as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]
print(f'V2 corpus: {len(lines)} unique lines')

v2_tokenizer = XNLPTokenizer.load_pretrained('tokenizer_v2')
print(f'V2 tokenizer: {v2_tokenizer.vocab_size_actual} tokens')

MAX_SEQ_LEN = 64  # 99.4% of sentences fit; max is 70 tokens
BATCH_SIZE = 8
SEED = 42

train_idx, val_idx, test_idx = build_split_indices(len(lines), 0.8, 0.1, SEED)
train_sents = [lines[i] for i in train_idx]
val_sents = [lines[i] for i in val_idx]
test_sents = [lines[i] for i in test_idx]
print(f'Train: {len(train_sents)}, Val: {len(val_sents)}, Test: {len(test_sents)}')

print('Tokenizing...')
t0 = time.time()
train_ids = tokenize_texts(train_sents, v2_tokenizer, MAX_SEQ_LEN)
val_ids = tokenize_texts(val_sents, v2_tokenizer, MAX_SEQ_LEN)
test_ids = tokenize_texts(test_sents, v2_tokenizer, MAX_SEQ_LEN)
print(f'Tokenization done in {time.time()-t0:.1f}s')

train_ds = XhosaTextDataset(train_ids, MAX_SEQ_LEN)
val_ds = XhosaTextDataset(val_ids, MAX_SEQ_LEN)
collate = make_collate_fn(v2_tokenizer.pad_token_id, MAX_SEQ_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0)
print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')

# ─── Training configurations ────────────────────────────────────────────────
MODELS = [
    {
        'name': 'tiny_v2', 'preset': 'tiny',
        'max_epochs': 20, 'lr': 3e-4, 'patience': 5,
        'out_dir': 'outputs_v2_tiny',
    },
    {
        'name': 'small_v2', 'preset': 'small',
        'max_epochs': 12, 'lr': 2e-4, 'patience': 3,
        'out_dir': 'outputs_v2_small',
    },
]

for m in MODELS:
    pc = MODEL_PRESETS[m['preset']]
    vc = dict(pc)
    vc['vocab_size'] = v2_tokenizer.vocab_size_actual
    vc['dropout_prob'] = 0.1
    vc['device'] = 'cpu'
    mcfg = XNLPConfig(**vc)
    mc = XNLPCoreLLM(mcfg)
    pc_count = sum(p.numel() for p in mc.parameters())
    est_steps = len(train_sents) // BATCH_SIZE
    print(f'  {m["name"]}: {pc_count:,} params, ~{est_steps} steps/epoch, {pc["hidden_size"]}h/{pc["num_hidden_layers"]}L')


def train_model(model_cfg, tokenizer, train_loader, val_loader, config_dict):
    """Train a single model and return results."""
    name = config_dict['name']
    preset = config_dict['preset']
    max_epochs = config_dict['max_epochs']
    lr = config_dict['lr']
    patience = config_dict['patience']
    out_dir = config_dict['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    model = XNLPCoreLLM(model_cfg)
    param_count = sum(p.numel() for p in model.parameters())
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = max_epochs * len(train_loader)
    warmup = 50
    
    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    best_val_loss = float('inf')
    best_epoch = 0
    global_step = 0
    early_stop_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'epoch': []}
    
    print(f'\n[training] {name}: {param_count:,} params, {len(train_loader)} batches/epoch, {max_epochs} max epochs')
    start_time = time.time()
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        
        for batch in train_loader:
            input_ids = batch['input_ids']
            labels = batch['labels']
            
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
                cur_lr = scheduler.get_last_lr()[0]
                print(f'  Ep {epoch+1}/{max_epochs} step {global_step} loss {loss.item():.4f} avg {avg:.4f} lr {cur_lr:.2e} ({time.time()-start_time:.0f}s)')
        
        avg_loss = epoch_loss / n_batches
        epoch_time = time.time() - t0
        
        # Validation
        val_loss = evaluate(model, val_loader, 'cpu')
        ppl = compute_perplexity(val_loss)
        history['train_loss'].append(round(avg_loss, 4))
        history['val_loss'].append(round(val_loss, 4))
        history['lr'].append(round(scheduler.get_last_lr()[0], 8))
        history['epoch'].append(epoch + 1)
        
        print(f'[epoch {epoch+1}/{max_epochs}] train={avg_loss:.4f} val={val_loss:.4f} ppl={ppl:.1f} time={epoch_time:.1f}s ({time.time()-start_time:.0f}s total)')
        
        # Save best
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            early_stop_counter = 0
            
            ckpt = {
                'checkpoint_format_version': CHECKPOINT_FORMAT_VERSION,
                'model_state_dict': model.state_dict(),
                'config': {
                    'vocab_size': model_cfg.vocab_size,
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
                    preset=preset, vocab_size=8000, max_epochs=max_epochs, batch_size=BATCH_SIZE,
                    learning_rate=lr, warmup_steps=warmup, max_seq_len=MAX_SEQ_LEN,
                ).as_dict(),
                'tokenizer_state': tokenizer_to_state_dict(tokenizer),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'training_metadata': {
                    'epoch': epoch + 1, 'global_step': global_step,
                    'best_val_loss': best_val_loss, 'param_count': param_count,
                    'vocab_size': tokenizer.vocab_size_actual, 'seed': SEED,
                    'python_version': '3.14.5', 'torch_version': torch.__version__,
                    'device': 'cpu', 'preset': preset, 'max_seq_len': MAX_SEQ_LEN,
                    'batch_size': BATCH_SIZE, 'learning_rate': lr, 'weight_decay': 0.1,
                    'warmup_steps': warmup, 'max_steps': total_steps,
                    'steps_per_epoch': len(train_loader),
                    'training_start_time': time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime(start_time)),
                    'training_end_time': time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()),
                    'rng_state': {'torch_cpu': torch.get_rng_state()},
                },
            }
            torch.save(ckpt, os.path.join(out_dir, 'best_model.pt'))
            print(f'  -> Saved best_model.pt (val_loss={best_val_loss:.4f})')
        else:
            early_stop_counter += 1
            print(f'  -> No improvement ({early_stop_counter}/{patience})')
        
        ckpt_last = dict(ckpt)
        torch.save(ckpt_last, os.path.join(out_dir, 'last_model.pt'))
        with open(os.path.join(out_dir, 'training_history.json'), 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        
        if early_stop_counter >= patience:
            print(f'[early-stop] Stopping at epoch {epoch+1}')
            break
    
    total_time = time.time() - start_time
    print(f'\n[{name}] Done: val_loss={best_val_loss:.4f}, ppl={compute_perplexity(best_val_loss):.1f}, epoch={best_epoch}, time={total_time:.1f}s')
    
    model.load_state_dict(torch.load(os.path.join(out_dir, 'best_model.pt'), map_location='cpu', weights_only=False)['model_state_dict'])
    model.eval()
    samples = generate_samples(model, tokenizer, PROMPTS, 'cpu', max_new_tokens=50, temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.1)
    
    return {
        'preset': preset, 'param_count': param_count,
        'vocab_size': tokenizer.vocab_size_actual,
        'best_val_loss': round(best_val_loss, 4),
        'perplexity': round(compute_perplexity(best_val_loss), 1),
        'best_epoch': best_epoch, 'training_time_seconds': round(total_time, 1),
        'max_epochs': max_epochs, 'patience': patience,
        'batch_size': BATCH_SIZE, 'max_seq_len': MAX_SEQ_LEN,
        'learning_rate': lr, 'warmup_steps': warmup,
        'train_examples': len(train_ds), 'val_examples': len(val_ds),
        'train_batches': len(train_loader), 'global_step': global_step,
        'best_model_path': os.path.join(out_dir, 'best_model.pt'),
        'training_history': history,
        'samples': {p: t for p, t in samples},
    }


results = {}

for model_cfg_dict in MODELS:
    pc = MODEL_PRESETS[model_cfg_dict['preset']]
    model_cfg = XNLPConfig(
        vocab_size=v2_tokenizer.vocab_size_actual,
        hidden_size=pc['hidden_size'], intermediate_size=pc['intermediate_size'],
        num_hidden_layers=pc['num_hidden_layers'],
        num_attention_heads=pc['num_attention_heads'],
        num_key_value_heads=pc['num_key_value_heads'],
        max_position_embeddings=pc['max_position_embeddings'],
        dropout_prob=0.1, device='cpu',
    )
    r = train_model(model_cfg, v2_tokenizer, train_loader, val_loader, model_cfg_dict)
    results[model_cfg_dict['name']] = r
    
    with open('data/reports/v2_model_scaling_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

# ─── Summary ────────────────────────────────────────────────────────────────
print(f'\n{"="*70}')
print('V2 Model Scaling Summary')
print(f'{"="*70}')
print(f'{"Model":>12} {"Params":>12} {"Val Loss":>10} {"PPL":>8} {"Epochs":>8} {"Time":>8}')
print('-' * 60)
for name in ['tiny_v2', 'small_v2']:
    if name in results:
        r = results[name]
        print(f'{name:>12} {r["param_count"]:>12,} {r["best_val_loss"]:>10.4f} {r["perplexity"]:>8.1f} {r["best_epoch"]:>8} {r["training_time_seconds"]:>7.0f}s')
print(f'{"V1 tiny":>12} {"4,321,280":>12} {"4.7701":>10} {"117.9":>8} {"14":>8} {"~24min":>8}')
