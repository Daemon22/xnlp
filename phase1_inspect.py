#!/usr/bin/env python3
"""PHASE 1: Inspect current system to diagnose language competence blockage."""
import json
from pathlib import Path

print("="*70)
print("XNLP LANGUAGE COMPETENCE - PHASE 1: SYSTEM INSPECTION")
print("="*70)

# 1. Repository state
print("\n1. REPOSITORY STATE")
print("-" * 40)

import subprocess
result = subprocess.run(
    ["git", "log", "--oneline", "-5"],
    cwd="D:/xnlp",
    capture_output=True,
    text=True
)
print("Recent commits:")
print(result.stdout)

# 2. Corpus inspection
print("\n2. CORPUS INSPECTION")
print("-" * 40)

corpus_path = Path("D:/xnlp/data/authoritative/v2_authoritative_all.jsonl")
with open(corpus_path, 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f if line.strip()]

print(f"Total records: {len(records)}")

# Sample records
print("\nSample records:")
for i, rec in enumerate(records[:10]):
    text = rec.get('text', '')[:100]
    source = rec.get('source', 'unknown')
    print(f"  {i+1}. [{source}] {text}...")

# Text length analysis
lengths = [len(rec.get('text', '')) for rec in records]
print(f"\nText length statistics:")
print(f"  Min: {min(lengths)}")
print(f"  Max: {max(lengths)}")
print(f"  Avg: {sum(lengths)/len(lengths):.1f}")

# Check for potential issues
print("\nPotential issues in corpus:")

# Count short texts
short = sum(1 for l in lengths if l < 10)
print(f"  Very short (<10 chars): {short}")

# Count with numbers/abbreviations
import re
num_pattern = re.compile(r'\d+')
abbr_pattern = re.compile(r'[A-Z]{2,}')

with_numbers = sum(1 for rec in records if num_pattern.search(rec.get('text', '')))
with_abbr = sum(1 for rec in records if abbr_pattern.search(rec.get('text', '')))
print(f"  With numbers: {with_numbers}")
print(f"  With abbreviations: {with_abbr}")

# 3. Training data inspection
print("\n3. TRAINING DATA")
print("-" * 40)

# Check how training data is constructed
from xnlp_trainer.data import prepare_data
print("Training data preparation function exists")

# Check actual training files
corpus_v2_path = Path("D:/xnlp/corpus_v2")
if corpus_v2_path.exists():
    files = list(corpus_v2_path.glob("*.txt"))
    print(f"Corpus v2 files: {len(files)}")
    for f in files[:5]:
        size = f.stat().st_size
        print(f"  {f.name}: {size} bytes")

# 4. Tokenizer inspection
print("\n4. TOKENIZER INSPECTION")
print("-" * 40)

from core_llm.tokenizer import XNLPTokenizer

tokenizer_path = Path("D:/xnlp/artifacts/xnlp_v2_tiny_tokenizer_fixed")
tokenizer = XNLPTokenizer.load(tokenizer_path)
print(f"Tokenizer loaded: vocab={tokenizer.vocab_size}, merges={tokenizer.num_merges}")

# Test tokenization of isiXhosa words
test_words = [
    "Umntu",
    "ngumntu",
    "ngabantu",
    "Inkosi",
    "yethu",
    "AmaXhosa",
    "anembali",
    "ende",
    "Ndiyafunda",
    "isiXhosa"
]

print("\nTokenization examples:")
for word in test_words:
    tokens = tokenizer.encode(word, add_special_tokens=False)
    decoded = tokenizer.decode(tokens)
    match = "✓" if decoded == word else "✗"
    print(f"  '{word}' -> {tokens} -> '{decoded}' {match}")

# 5. Checkpoint inspection
print("\n5. CHECKPOINT INSPECTION")
print("-" * 40)

import torch
checkpoint_path = Path("D:/xnlp/artifacts/xnlp_v2_tiny_tokenizer_fixed/model/best_model.pt")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print(f"Checkpoint keys: {list(checkpoint.keys())[:10]}...")
print(f"Model type: {type(checkpoint.get('model_state_dict', {}))}")
print(f"Config: {checkpoint.get('config', {}).get('vocab_size', 'N/A')}")

# 6. Training config
print("\n6. TRAINING CONFIGURATION")
print("-" * 40)

config_path = Path("D:/xnlp/artifacts/xnlp_v2_tiny_tokenizer_fixed/training_config.json")
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

for key, value in config.items():
    print(f"  {key}: {value}")

print("\n" + "="*70)
print("PHASE 1 INSPECTION COMPLETE")
print("="*70)
