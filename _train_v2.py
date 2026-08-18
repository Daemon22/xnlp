#!/usr/bin/env python
"""
Phase V2-16: Model Scaling Experiment
Trains XNLP models at Tiny -> Small -> Medium scales on the V2 corpus
using the V2 tokenizer (2,048 vocab).

All training is fully independent — no retrieval, no external APIs, no web.
The model learns language through its own parameters.
"""
import sys, io, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True)

import torch
from core_llm.tokenizer import XNLPTokenizer
from core_llm.architecture import XNLPConfig, XNLPCoreLLM
from xnlp_trainer.config import TrainingConfig, MODEL_PRESETS
from xnlp_trainer.data import prepare_data, tokenizer_to_state_dict, tokenizer_from_state_dict
from xnlp_trainer.trainer import XNLPTrainer
from xnlp_trainer.evaluate import evaluate, compute_perplexity, generate_samples

# ─── Setup V2 corpus directory ─────────────────────────────────────────────

v2_corpus_dir = 'corpus_v2'
os.makedirs(v2_corpus_dir, exist_ok=True)

# Copy V2 training corpus to the corpus directory
src_corpus = 'data/processed/v2/training_corpus.txt'
dst_corpus = os.path.join(v2_corpus_dir, 'v2_training_corpus.txt')
with open(src_corpus, 'r', encoding='utf-8') as f:
    content = f.read()
with open(dst_corpus, 'w', encoding='utf-8') as f:
    f.write(content)
print(f'V2 corpus: {len(content):,} chars, {len(content.splitlines())} lines')

# ─── Load V2 tokenizer ──────────────────────────────────────────────────────

tok_dir = 'tokenizer_v2'
v2_tokenizer = XNLPTokenizer.load_pretrained(tok_dir)
print(f'V2 tokenizer loaded: {v2_tokenizer.vocab_size_actual} tokens, {v2_tokenizer.num_merges} merges')

# ─── Training configurations ──────────────────────────────────────────────

training_configs = {
    'tiny_v2': {
        'preset': 'tiny',
        'max_epochs': 30,
        'batch_size': 8,
        'learning_rate': 3e-4,
        'warmup_steps': 50,
        'max_seq_len': 256,
        'patience': 5,
        'output_dir': 'outputs_v2_tiny',
    },
    'small_v2': {
        'preset': 'small',
        'max_epochs': 30,
        'batch_size': 8,
        'learning_rate': 3e-4,
        'warmup_steps': 50,
        'max_seq_len': 256,
        'patience': 5,
        'output_dir': 'outputs_v2_small',
    },
    'medium_v2': {
        'preset': 'medium',
        'max_epochs': 15,
        'batch_size': 8,
        'learning_rate': 2e-4,
        'warmup_steps': 50,
        'max_seq_len': 256,
        'patience': 3,
        'output_dir': 'outputs_v2_medium',
    },
}

# ─── Train models ───────────────────────────────────────────────────────────

results = {}

for model_name, cfg_dict in training_configs.items():
    print(f'\n{"="*70}')
    print(f'V2-16: Training {model_name.upper()}')
    print(f'{"="*70}')
    
    # Create training config
    config = TrainingConfig(
        corpus_dir=v2_corpus_dir,
        output_dir=cfg_dict['output_dir'],
        preset=cfg_dict['preset'],
        vocab_size=8000,  # Will be overridden by actual tokenizer vocab
        tokenizer_min_freq=2,
        max_epochs=cfg_dict['max_epochs'],
        batch_size=cfg_dict['batch_size'],
        learning_rate=cfg_dict['learning_rate'],
        warmup_steps=cfg_dict['warmup_steps'],
        max_seq_len=cfg_dict['max_seq_len'],
        patience=cfg_dict['patience'],
        early_stopping=True,
        seed=42,
        sample_prompts=[
            'Molo, ndiyabulela',
            'Umntu ngumntu ngabantu',
            'Umthetho wamaXhosa',
            'Izibongo zethu',
            'Ndiyavuya',
            'Nkosi Sikelel',
            'UMqhayi',
            'Ubuntu',
        ],
        max_new_tokens=50,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    
    # Show model config
    preset_config = MODEL_PRESETS[cfg_dict['preset']]
    print(f'  Model: {cfg_dict["preset"]}')
    print(f'  Hidden: {preset_config["hidden_size"]}')
    print(f'  Layers: {preset_config["num_hidden_layers"]}')
    print(f'  Heads: {preset_config["num_attention_heads"]}')
    
    # Calculate expected parameter count
    vc = dict(preset_config)
    vc['vocab_size'] = v2_tokenizer.vocab_size_actual
    vc['dropout_prob'] = 0.1
    vc['device'] = 'cpu'
    model_cfg = XNLPConfig(**vc)
    model = XNLPCoreLLM(model_cfg)
    param_count = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {param_count:,}')
    print(f'  Vocab: {v2_tokenizer.vocab_size_actual}')
    print(f'  Epochs: {cfg_dict["max_epochs"]} (early stopping, patience={cfg_dict["patience"]})')
    print(f'  Batch size: {cfg_dict["batch_size"]}')
    print(f'  LR: {cfg_dict["learning_rate"]}')
    print(f'  Max seq len: {cfg_dict["max_seq_len"]}')
    
    # Train
    start = time.time()
    trainer = XNLPTrainer(config)
    trainer.train()
    elapsed = time.time() - start
    
    # Load best model and evaluate
    best_ckpt = torch.load(trainer.best_path, map_location='cpu', weights_only=False)
    best_val_loss = best_ckpt['training_metadata']['best_val_loss']
    best_epoch = best_ckpt['training_metadata']['epoch']
    perplexity = compute_perplexity(best_val_loss)
    
    print(f'\n  Results:')
    print(f'  Best val loss: {best_val_loss:.4f}')
    print(f'  Perplexity: {perplexity:.1f}')
    print(f'  Best epoch: {best_epoch}')
    print(f'  Training time: {elapsed:.1f}s')
    
    # Generate samples
    print(f'  Sample outputs:')
    samples = generate_samples(
        trainer.model, v2_tokenizer, config.sample_prompts, 'cpu',
        max_new_tokens=50, temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.1,
    )
    for prompt, text in samples:
        print(f'    "{prompt}" -> "{text}"')
    
    results[model_name] = {
        'preset': cfg_dict['preset'],
        'param_count': param_count,
        'vocab_size': v2_tokenizer.vocab_size_actual,
        'best_val_loss': best_val_loss,
        'perplexity': round(perplexity, 1),
        'best_epoch': best_epoch,
        'training_time_seconds': round(elapsed, 1),
        'outputs_dir': cfg_dict['output_dir'],
        'best_model_path': trainer.best_path,
        'samples': {p: t for p, t in samples},
    }

# ─── Comparison summary ─────────────────────────────────────────────────────

print(f'\n{"="*70}')
print('V2 Model Scaling Summary')
print(f'{"="*70}')
print(f'{"Model":>10} {"Params":>12} {"Val Loss":>10} {"PPL":>8} {"Epochs":>8} {"Time":>8}')
print('-' * 60)
for name, r in results.items():
    print(f'{name:>10} {r["param_count"]:>12,} {r["best_val_loss"]:>10.4f} {r["perplexity"]:>8.1f} {r["best_epoch"]:>8} {r["training_time_seconds"]:>7.0f}s')

# V1 comparison
v1_meta = ckpt['training_metadata'] if False else None  # We already have V1 baseline
print(f'{"V1 tiny":>10} {"4,321,280":>12} {"4.7701":>10} {"117.9":>8} {"14":>8} {"~24min":>8}')

# Save results
results_path = 'data/reports/v2_model_scaling_results.json'
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f'\nSaved: {results_path}')

print(f'\n{"="*70}')
print('V2-16: Model Scaling Complete')
print(f'{"="*70}')
