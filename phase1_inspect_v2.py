#!/usr/bin/env python3
"""PHASE 1: System inspection - Version 2."""
import json
from pathlib import Path

print("="*70)
print("XNLP LANGUAGE COMPETENCE - PHASE 1: SYSTEM INSPECTION")
print("="*70)

# 1. Corpus inspection - look for problems
print("\n1. CORPUS CONTENT ANALYSIS")
print("-" * 40)

corpus_path = Path("D:/xnlp/data/authoritative/v2_authoritative_all.jsonl")
with open(corpus_path, 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f if line.strip()]

print(f"Total records: {len(records)}")

# Check for English contamination
import re
english_words = ['Col', 'Hon', 'Denys', 'Reitz', 'Jimmy', 'Gray', 'Maya', 'Koboka',
                 'Chalmers', 'James', 'Brown', 'Lovedale', 'Mary', 'Geddes',
                 'W.L.', 'Hail', 'Sithwalandwe', 'The']

english_contamination = 0
for rec in records:
    text = rec.get('text', '')
    for word in english_words:
        if word in text:
            english_contamination += 1
            break

print(f"Records with English names/abbreviations: {english_contamination}")

# Sample the problematic records
print("\nProblematic records (with English):")
count = 0
for rec in records:
    text = rec.get('text', '')
    if any(word in text for word in english_words):
        print(f"  [{rec.get('source')}] {text[:120]}...")
        count += 1
        if count >= 10:
            break

# 2. Check training corpus
print("\n2. TRAINING CORPUS")
print("-" * 40)

corpus_v2_path = Path("D:/xnlp/corpus_v2/v2_training_corpus.txt")
if corpus_v2_path.exists():
    with open(corpus_v2_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"Training sentences: {len(lines)}")
    print(f"Sample sentences:")
    for i, line in enumerate(lines[:10]):
        print(f"  {i+1}. {line.strip()[:100]}...")

# 3. Check tokenizer
print("\n3. TOKENIZER")
print("-" * 40)

from core_llm.tokenizer import XNLPTokenizer

tokenizer_path = Path("D:/xnlp/artifacts/xnlp_v2_tiny_tokenizer_fixed")
# Create tokenizer from files
tokenizer = XNLPTokenizer(vocab_size=4096)
vocab_path = tokenizer_path / "vocab.json"
merges_path = tokenizer_path / "merges.json"

if vocab_path.exists() and merges_path.exists():
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    with open(merges_path, 'r', encoding='utf-8') as f:
        merges = [json.loads(line) for line in f if line.strip()]
    
    tokenizer.token2id = vocab
    tokenizer.id2token = {v: k for k, v in vocab.items()}
    tokenizer.merges = merges
    tokenizer.num_merges = len(merges)
    tokenizer.vocab_size = len(vocab)
    
    print(f"Tokenizer: vocab={tokenizer.vocab_size}, merges={tokenizer.num_merges}")
    
    # Test tokenization
    test_words = ["Umntu", "Inkosi", "AmaXhosa", "Ngubani", "Kodwa"]
    print("\nTokenization tests:")
    for word in test_words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        decoded = tokenizer.decode(tokens)
        match = "✓" if decoded == word else "✗"
        print(f"  '{word}' -> {tokens} -> '{decoded}' {match}")

# 4. Check checkpoint metadata
print("\n4. CHECKPOINT METADATA")
print("-" * 40)

import torch
checkpoint_path = Path("D:/xnlp/artifacts/xnlp_v2_tiny_tokenizer_fixed/model/best_model.pt")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print(f"Checkpoint format version: {checkpoint.get('checkpoint_format_version', 'N/A')}")
print(f"Model params: {sum(p.numel() for p in checkpoint.get('model_state_dict', {}).values()):,}")
print(f"Tokenizer vocab size: {checkpoint.get('tokenizer_state', {}).get('vocab_size', 'N/A')}")
print(f"Training metadata: {checkpoint.get('training_metadata', {}).get('epoch', 'N/A')} epochs")

# 5. Trace the "P. M. D." artifact
print("\n5. TRACING 'P. M. D.' ARTIFACT")
print("-" * 40)

# Search corpus for "P. M. D." or similar patterns
pattern = re.compile(r'[A-Z]\.\s*[A-Z]\.\s*[A-Z]')
found = []
for rec in records:
    text = rec.get('text', '')
    matches = pattern.findall(text)
    if matches:
        found.append((rec.get('source'), text[:150]))

if found:
    print(f"Found {len(found)} records with abbreviation patterns:")
    for source, text in found[:10]:
        print(f"  [{source}] {text}...")
else:
    print("No abbreviation patterns found in corpus")

# Check if tokenizer splits these
if 'tokenizer' in locals():
    test_abbr = ["P. M. D.", "Col. The Hon.", "W.L."]
    print("\nTokenizer handling of abbreviations:")
    for abbr in test_abbr:
        tokens = tokenizer.encode(abbr, add_special_tokens=False)
        decoded = tokenizer.decode(tokens)
        print(f"  '{abbr}' -> {tokens} -> '{decoded}'")

print("\n" + "="*70)
print("PHASE 1: INITIAL INSPECTION COMPLETE")
print("="*70)
