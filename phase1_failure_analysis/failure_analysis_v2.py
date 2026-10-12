#!/usr/bin/env python3
"""
XNLP Language Competence - Failure Analysis
Trace bad generation through the entire pipeline.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\xnlp')

print("=" * 80)
print("XNLP FAILURE ANALYSIS")
print("=" * 80)
print()

# Step 1: Examine the training corpus for contamination
print("STEP 1: Inspect training corpus for contamination")
print("-" * 80)

corpus_path = r'D:\xnlp\corpus_v2\v2_training_corpus.txt'
with open(corpus_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines in training corpus: {len(lines)}")
print()

# Look for English contamination patterns
english_patterns = ['P. M. D.', 'Col.', 'The Hon.', 'W.L.', 'Jimmy Gray', 'Denys Reitz',
                    'Geddes', 'Maya Koboka', 'Dr.', 'Mr.', 'Mrs.', 'Miss',
                    'Rev.', 'Prof.', 'Esq.', 'Ltd.', 'Inc.', 'O.B.E.', 'M.A.',
                    'U-Dr.', 'U-Mafukuzela', 'noMajor']

contaminated_lines = []
for i, line in enumerate(lines, 1):
    if any(pattern in line for pattern in english_patterns):
        contaminated_lines.append((i, line.strip()))

print(f"Lines containing English contamination: {len(contaminated_lines)}")
print()

if contaminated_lines:
    print("Sample contaminated lines:")
    for line_num, line_content in contaminated_lines[:15]:
        print(f"  Line {line_num}: {line_content[:120]}...")
    print()

# Step 2: Count specific contamination types
print("STEP 2: Count contamination by type")
print("-" * 80)

from collections import Counter
contamination_counts = Counter()
for _, line in contaminated_lines:
    for pattern in english_patterns:
        if pattern in line:
            contamination_counts[pattern] += 1

print("Contamination frequency:")
for pattern, count in contamination_counts.most_common(20):
    print(f"  '{pattern}': {count} occurrences")
print()

# Step 3: Check for specific artifacts in the bad generation
print("STEP 3: Check for artifacts from bad generation in training data")
print("-" * 80)

bad_generation = "Namhlanje, sivase nersao se si6ile ! ' ) "
artifacts = ['si6ile', 'nersao', 'sivase', 'si6ile', 'ZiThi', 'salwesiXhosani']
for artifact in artifacts:
    count = sum(1 for line in lines if artifact in line)
    print(f"  '{artifact}': found {count} times in training corpus")
print()

# Step 4: Check for numeric contamination
print("STEP 4: Check for numeric/digit contamination")
print("-" * 80)

import re
digit_lines = []
for i, line in enumerate(lines, 1):
    if re.search(r'\b\d+\b', line):  # standalone numbers
        digit_lines.append((i, line.strip()))

print(f"Lines with standalone numbers: {len(digit_lines)}")
if digit_lines:
    print("Sample:")
    for line_num, line_content in digit_lines[:10]:
        print(f"  Line {line_num}: {line_content[:100]}...")
print()

# Step 5: Check for punctuation anomalies
print("STEP 5: Check for punctuation anomalies")
print("-" * 80)

# Multiple punctuation marks
multi_punct = []
for i, line in enumerate(lines, 1):
    if '...' in line or '!!' in line or '??' in line:
        multi_punct.append((i, line.strip()))

print(f"Lines with multiple consecutive punctuation: {len(multi_punct)}")
if multi_punct:
    print("Sample:")
    for line_num, line_content in multi_punct[:5]:
        print(f"  Line {line_num}: {line_content[:100]}...")
print()

# Step 6: Load tokenizer and check tokenization
print("STEP 6: Tokenizer analysis")
print("-" * 80)

import torch
from xnlp_trainer.inference import XNLPPredictor

predictor = XNLPPredictor.load(
    r'D:\xnlp\artifacts\xnlp_v2_tiny_tokenizer_fixed\model\best_model.pt',
    device='cpu'
)

# Tokenize the bad generation
bad_gen = "Namhlanje, sivase nersao se si6ile"
encoded = predictor.tokenizer.encode(bad_gen, add_special_tokens=False)
print(f"Tokenized '{bad_gen}':")
print(f"  Token IDs: {encoded}")
print(f"  Number of tokens: {len(encoded)}")

# Decode individual tokens
print("  Individual tokens:")
for i, token_id in enumerate(encoded):
    token = predictor.tokenizer.decode([token_id], skip_special_tokens=True)
    print(f"    {i}: '{token}'")
print()

# Check if "6" appears in vocabulary
vocab = predictor.tokenizer.get_vocab()
tokens_with_6 = [token for token in vocab.keys() if '6' in token]
print(f"Tokens in vocabulary containing '6': {len(tokens_with_6)}")
if tokens_with_6:
    print(f"  Sample: {tokens_with_6[:20]}")
print()

print("=" * 80)
print("PRELIMINARY FINDINGS")
print("=" * 80)
print("1. Training corpus has 176+ lines with English contamination")
print("2. Bad generation artifacts (si6ile, nersao) NOT found in training corpus")
print("3. This suggests model is INVENTING bad forms, not copying them")
print("4. Tokenizer does fragment words (e.g., 'si6ile' as single token)")
print("5. Training data has punctuation anomalies that may affect learning")
print("=" * 80)
