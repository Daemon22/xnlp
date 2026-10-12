#!/usr/bin/env python3
"""
XNLP Language Competence - Failure Analysis
Trace bad generation through the entire pipeline.

Bad generation example:
"Namhlanje, sivase nersao se si6ile ! ’ ” " ZiThi le nto ni na le nto ma saama sathu ke se salwesiXhosani le ?"

We need to determine where these errors come from.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

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
                    'Rev.', 'Prof.', 'Esq.', 'Ltd.', 'Inc.']

contaminated_lines = []
for i, line in enumerate(lines, 1):
    if any(pattern in line for pattern in english_patterns):
        contaminated_lines.append((i, line.strip()))

print(f"Lines containing English contamination: {len(contaminated_lines)}")
print()

if contaminated_lines:
    print("Sample contaminated lines:")
    for line_num, line_content in contaminated_lines[:10]:
        print(f"  Line {line_num}: {line_content[:100]}...")
    print()

# Step 2: Check tokenization of the bad generation
print("STEP 2: Tokenize the bad generation to see fragmentation")
print("-" * 80)

from core_llm.tokenizer import XNLPTokenizer
import os

# Find the tokenizer state
checkpoint_path = r'D:\xnlp\artifacts\xnlp_v2_tiny_tokenizer_fixed\model\best_model.pt'
import torch
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
tokenizer_state = ckpt['tokenizer_state']

from xnlp_trainer.data import tokenizer_from_state_dict
tokenizer = tokenizer_from_state_dict(tokenizer_state)

bad_generation = "Namhlanje, sivase nersao se si6ile ! ’ ” \" ZiThi le nto ni na le nto ma saama sathu ke se salwesiXhosani le ?"
print(f"Bad generation: {bad_generation}")
print()

# Tokenize
encoded = tokenizer.encode(bad_generation, add_special_tokens=False)
print(f"Token IDs: {encoded}")
print(f"Number of tokens: {len(encoded)}")
print()

# Decode each token
tokens = []
for token_id in encoded:
    token = tokenizer.decode([token_id], skip_special_tokens=True)
    tokens.append(token)

print("Individual tokens:")
for i, token in enumerate(tokens):
    print(f"  {i}: '{token}'")
print()

# Step 3: Check if these artifacts exist in the training data
print("STEP 3: Check for specific artifacts in training data")
print("-" * 80)

artifacts = ['si6ile', 'nersao', 'se si6ile', 'ZiThi', 'le nto ni na', 'salwesiXhosani']
for artifact in artifacts:
    count = sum(1 for line in lines if artifact in line)
    print(f"  '{artifact}': found {count} times in training corpus")
print()

# Step 4: Check tokenizer vocabulary for unusual tokens
print("STEP 4: Check tokenizer vocabulary")
print("-" * 80)
vocab = tokenizer.get_vocab()
print(f"Vocabulary size: {len(vocab)}")
print()

# Look for problematic tokens
problematic = []
for token, idx in vocab.items():
    if any(c.isdigit() for c in token) and len(token) > 1:
        problematic.append(token)
    if '6' in token or 'P.M.D' in token or 'P. M. D.' in token:
        problematic.append(token)

print(f"Tokens containing digits: {len(problematic)}")
if problematic:
    print(f"Sample: {problematic[:20]}")
print()

print("=" * 80)
print("ANALYSIS COMPLETE - See detailed report")
print("=" * 80)
