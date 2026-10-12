#!/usr/bin/env python3
"""
Prepare clean data splits for training.
Copy the clean corpus to the data pipeline and create splits.
"""
import sys
import os
import json
import random
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("PREPARING CLEAN DATA SPLITS")
print("=" * 80)
print()

# Load clean corpus
clean_corpus_path = r'D:\xnlp\corpus_v2\v2_training_corpus_clean_final.txt'
with open(clean_corpus_path, 'r', encoding='utf-8') as f:
    sentences = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(sentences)} clean sentences")
print()

# Create v3 directory
v3_dir = r'D:\xnlp\data\processed\v3'
v3_splits_dir = os.path.join(v3_dir, 'splits')
os.makedirs(v3_splits_dir, exist_ok=True)

# Save full clean corpus
full_corpus_path = os.path.join(v3_dir, 'training_corpus.txt')
with open(full_corpus_path, 'w', encoding='utf-8') as f:
    for s in sentences:
        f.write(s + '\n')

# Create provenance tracking
provenance = []
for i, s in enumerate(sentences):
    provenance.append({
        'line_number': i + 1,
        'source': 'v2_training_corpus_clean_final.txt',
        'processing': 'OCR_error_removal',
        'text_hash': hash(s) % 1000000
    })

provenance_path = os.path.join(v3_dir, 'training_corpus_provenance.jsonl')
with open(provenance_path, 'w', encoding='utf-8') as f:
    for p in provenance:
        f.write(json.dumps(p, ensure_ascii=False) + '\n')

# Create train/val/test splits (80/10/10)
random.seed(42)  # Reproducible
random.shuffle(sentences)

n = len(sentences)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

train_sentences = sentences[:train_end]
val_sentences = sentences[train_end:val_end]
test_sentences = sentences[val_end:]

print(f"Split sizes:")
print(f"  Train: {len(train_sentences)} ({len(train_sentences)/n*100:.1f}%)")
print(f"  Val:   {len(val_sentences)} ({len(val_sentences)/n*100:.1f}%)")
print(f"  Test:  {len(test_sentences)} ({len(test_sentences)/n*100:.1f}%)")
print()

# Save splits
train_path = os.path.join(v3_splits_dir, 'train.txt')
val_path = os.path.join(v3_splits_dir, 'val.txt')
test_path = os.path.join(v3_splits_dir, 'test.txt')

with open(train_path, 'w', encoding='utf-8') as f:
    for s in train_sentences:
        f.write(s + '\n')

with open(val_path, 'w', encoding='utf-8') as f:
    for s in val_sentences:
        f.write(s + '\n')

with open(test_path, 'w', encoding='utf-8') as f:
    for s in test_sentences:
        f.write(s + '\n')

# Save provenance for splits
for split_name, split_sentences in [('train', train_sentences), ('val', val_sentences), ('test', test_sentences)]:
    split_provenance = []
    for s in split_sentences:
        split_provenance.append({
            'source': 'v3_clean_corpus',
            'split': split_name,
            'text_hash': hash(s) % 1000000
        })
    with open(os.path.join(v3_splits_dir, f'{split_name}_provenance.jsonl'), 'w', encoding='utf-8') as f:
        for p in split_provenance:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

# Save split summary
split_summary = {
    'total_sentences': len(sentences),
    'train_size': len(train_sentences),
    'val_size': len(val_sentences),
    'test_size': len(test_sentences),
    'split_ratio': [0.8, 0.1, 0.1],
    'random_seed': 42,
    'source': 'v2_training_corpus_clean_final.txt',
    'processing': 'OCR_error_removal_and_deduplication'
}

with open(os.path.join(v3_splits_dir, 'split_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(split_summary, f, ensure_ascii=False, indent=2)

print("Data splits created:")
print(f"  {v3_dir}")
print(f"  {v3_splits_dir}")
print()

# Copy clean tokenizer to v3 artifacts
import shutil
tokenizer_src = r'D:\xnlp\artifacts\xnlp_v3_clean_tokenizer'
tokenizer_dst = r'D:\xnlp\artifacts\xnlp_v3_clean_tokenizer_data'
shutil.copytree(tokenizer_src, tokenizer_dst)

print("Tokenizer copied to:")
print(f"  {tokenizer_dst}")
print()

print("=" * 80)
print("CLEAN DATA SPLITS PREPARED")
print("=" * 80)
