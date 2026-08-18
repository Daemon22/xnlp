#!/usr/bin/env python
"""
Phase V2-8/9: V2 Tokenizer Benchmark (full scale)
Trains V2 tokenizer on full V2 corpus, compares with V1 tokenizer.
"""
import sys, io, json, os, re, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True)
import torch
from core_llm.tokenizer import XNLPTokenizer

# ─── Load V2 corpus ─────────────────────────────────────────────────────────

corpus_path = 'data/processed/v2/training_corpus.txt'
with open(corpus_path, 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f if line.strip()]

corpus_text = '\n'.join(texts)
print(f'V2 corpus: {len(texts)} lines, {len(corpus_text):,} chars, {len(corpus_text.split()):,} words')

# ─── Load V1 tokenizer ──────────────────────────────────────────────────────
v1_ckpt_path = 'v1_baseline/best_model.pt'
ckpt = torch.load(v1_ckpt_path, map_location='cpu', weights_only=False)
v1_state = ckpt['tokenizer_state']
v1_vocab_size = len(v1_state['token2id'])
v1_merges_list = v1_state.get('merges', [])
if v1_merges_list and isinstance(v1_merges_list[0], list):
    v1_merges_list = [tuple(m) for m in v1_merges_list]

v1_tok = XNLPTokenizer(vocab_size=v1_vocab_size, min_frequency=2)
v1_tok.token2id = v1_state['token2id']
v1_tok.id2token = v1_state['id2token']
v1_tok.merges = v1_merges_list
v1_tok.merge_ranks = {pair: idx for idx, pair in enumerate(v1_tok.merges)}
v1_tok.num_merges = len(v1_tok.merges)

# Encode V2 corpus with V1 tokenizer
print(f'\nV1 tokenizer: {v1_vocab_size} tokens, {len(v1_tok.merges)} merges')
start = time.time()
v1_ids = []
for text in texts:
    ids = v1_tok.encode(text, add_special_tokens=False)
    v1_ids.extend(ids)
v1_time = time.time() - start
v1_tokens = len(v1_ids)
v1_compression = len(corpus_text) / v1_tokens if v1_tokens > 0 else 0
print(f'  Encoded {len(corpus_text):,} chars -> {v1_tokens:,} tokens (compression: {v1_compression:.2f}, time: {v1_time:.1f}s)')

# ─── Train V2 tokenizer on full corpus ──────────────────────────────────────
# Use vocab_size=2048 (will converge before that)
# Train on full corpus for best coverage
print(f'\n--- Training V2 tokenizer on FULL corpus (vocab_size=2048) ---')
start = time.time()
v2_tok = XNLPTokenizer(vocab_size=2048, min_frequency=2)
v2_tok.train(texts, verbose=False)
elapsed = time.time() - start

v2_vocab = len(v2_tok.token2id)
v2_merges = v2_tok.num_merges
print(f'  Vocab: {v2_vocab}, Merges: {v2_merges}, Training time: {elapsed:.1f}s')

# Encode V2 corpus with V2 tokenizer
start = time.time()
v2_ids = []
for text in texts:
    ids = v2_tok.encode(text, add_special_tokens=False)
    v2_ids.extend(ids)
v2_time = time.time() - start
v2_tokens = len(v2_ids)
v2_compression = len(corpus_text) / v2_tokens if v2_tokens > 0 else 0
print(f'  Encoded {len(corpus_text):,} chars -> {v2_tokens:,} tokens (compression: {v2_compression:.2f}, time: {v2_time:.1f}s)')

# ─── Also benchmark on V1 corpus for fair comparison ────────────────────────
v1_corpus_path = 'v1_baseline/training_corpus.txt'
with open(v1_corpus_path, 'r', encoding='utf-8') as f:
    v1_texts = [line.strip() for line in f if line.strip()]

# V1 tokenizer on V1 corpus
v1_ids_v1 = []
for text in v1_texts:
    ids = v1_tok.encode(text, add_special_tokens=False)
    v1_ids_v1.extend(ids)
v1_v1_tokens = len(v1_ids_v1)
v1_v1_compression = sum(len(t) for t in v1_texts) / v1_v1_tokens if v1_v1_tokens > 0 else 0

# V2 tokenizer on V1 corpus
v2_ids_v1 = []
for text in v1_texts:
    ids = v2_tok.encode(text, add_special_tokens=False)
    v2_ids_v1.extend(ids)
v2_v1_tokens = len(v2_ids_v1)
v2_v1_compression = sum(len(t) for t in v1_texts) / v2_v1_tokens if v2_v1_tokens > 0 else 0

print(f'\n--- Cross-evaluation ---')
print(f'V1 tok on V1 corpus: {v1_v1_tokens:,} tokens, compression={v1_v1_compression:.2f}')
print(f'V2 tok on V1 corpus: {v2_v1_tokens:,} tokens, compression={v2_v1_compression:.2f}')
print(f'V1 tok on V2 corpus: {v1_tokens:,} tokens, compression={v1_compression:.2f}')
print(f'V2 tok on V2 corpus: {v2_tokens:,} tokens, compression={v2_compression:.2f}')

# ─── Also test smaller vocab sizes for the V2 tokenizer ──────────────────────
# Train on sample for speed
sample = texts[:1200]
print(f'\n--- V2 tokenizer at different vocab sizes (trained on 1200-line sample) ---')
print(f'{"Vocab":>8} {"Actual":>8} {"Merges":>8} {"Tokens":>10} {"Comp":>8} {"Delta%":>7}')
print('-' * 55)

results_table = []

for vs in [256, 512, 1024, 2048, 4096, 8192]:
    start = time.time()
    tok = XNLPTokenizer(vocab_size=vs, min_frequency=2)
    tok.train(sample, verbose=False)
    elapsed = time.time() - start
    
    # Encode full corpus
    all_ids = []
    for text in texts:
        ids = tok.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
    n_total = len(all_ids)
    comp = len(corpus_text) / n_total if n_total > 0 else 0
    
    delta_pct = (comp - v1_compression) / v1_compression * 100 if v1_compression > 0 else 0
    print(f'{vs:>8} {len(tok.token2id):>8} {tok.num_merges:>8} {n_total:>10,} {comp:>8.2f} {delta_pct:>+7.1f}%')
    
    results_table.append({
        'vocab_size_target': vs,
        'vocab_size_actual': len(tok.token2id),
        'num_merges': tok.num_merges,
        'total_tokens': n_total,
        'compression_ratio': round(comp, 2),
        'delta_pct_vs_v1': round(delta_pct, 1),
        'training_time_seconds': round(elapsed, 2),
    })

# ─── Save V2 tokenizer ────────────────────────────────────────────────────────
tok_dir = 'tokenizer_v2'
v2_tok.save_pretrained(tok_dir)
print(f'\nV2 tokenizer saved to {tok_dir}/')

# ─── Save full results ───────────────────────────────────────────────────────
results = {
    'benchmark_type': 'V2 Tokenizer vs V1 Tokenizer',
    'v2_corpus': {
        'total_records': len(texts),
        'total_chars': len(corpus_text),
        'total_words': len(corpus_text.split()),
    },
    'v1_baseline_tokenizer': {
        'vocab_size': v1_vocab_size,
        'num_merges': len(v1_tok.merges),
        'on_v2_corpus': {
            'total_tokens': v1_tokens,
            'compression_ratio': round(v1_compression, 2),
            'encoding_time_seconds': round(v1_time, 2),
        },
        'on_v1_corpus': {
            'total_tokens': v1_v1_tokens,
            'compression_ratio': round(v1_v1_compression, 2),
        },
    },
    'v2_full_tokenizer': {
        'vocab_size_actual': v2_vocab,
        'num_merges': v2_merges,
        'training_time_seconds': round(elapsed, 2),
        'training_sample': 'full corpus ({0} lines)'.format(len(texts)),
        'on_v2_corpus': {
            'total_tokens': v2_tokens,
            'compression_ratio': round(v2_compression, 2),
            'encoding_time_seconds': round(v2_time, 2),
        },
        'on_v1_corpus': {
            'total_tokens': v2_v1_tokens,
            'compression_ratio': round(v2_v1_compression, 2),
        },
    },
    'vocab_size_table': results_table,
    'comparison_summary': {
        'v1_comp_on_v2': round(v1_compression, 2),
        'v2_comp_on_v2': round(v2_compression, 2),
        'improvement_pct': round((v2_compression - v1_compression) / v1_compression * 100, 1),
        'v1_vocab': v1_vocab_size,
        'v2_vocab': v2_vocab,
        'vocab_expansion': v2_vocab / v1_vocab_size,
    },
    'recommendation': {
        'vocab_size': v2_vocab,
        'rationale': f'V2 tokenizer trained on {len(texts)} lines of expanded corpus, converged at {v2_vocab} tokens. Provides {round(v2_compression, 2)} chars/token compression (vs V1: {round(v1_compression, 2)}), a {(v2_compression - v1_compression) / v1_compression * 100:.1f}% improvement.',
    },
}

os.makedirs('data/reports', exist_ok=True)
with open('data/reports/tokenizer_benchmark_v2.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f'\n{"="*60}')
print('V2 Tokenizer Summary')
print(f'{"="*60}')
print(f'V1 tokenizer: {v1_vocab_size} vocab, compression={v1_compression:.2f} on V2 corpus')
print(f'V2 tokenizer: {v2_vocab} vocab, compression={v2_compression:.2f} on V2 corpus')
print(f'Improvement: {(v2_compression - v1_compression) / v1_compression * 100:.1f}%')
print(f'Saved: data/reports/tokenizer_benchmark_v2.json')
print(f'Saved: {tok_dir}/ (tokenizer)')
