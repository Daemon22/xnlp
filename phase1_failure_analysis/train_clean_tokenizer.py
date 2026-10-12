#!/usr/bin/env python3
"""
Train a new tokenizer on the clean corpus (without OCR digit errors).
This will prevent the model from learning that digits are valid in isiXhosa words.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\xnlp')

print("=" * 80)
print("TRAINING CLEAN TOKENIZER")
print("=" * 80)
print()

from xnlp_trainer.data import train_tokenizer

# Load clean corpus
clean_corpus_path = r'D:\xnlp\corpus_v2\v2_training_corpus_clean_final.txt'
with open(clean_corpus_path, 'r', encoding='utf-8') as f:
    sentences = [line.strip() for line in f if line.strip()]

print(f"Training on {len(sentences)} clean sentences")
print()

# Train tokenizer with same params as before but on clean data
tokenizer = train_tokenizer(
    sentences,
    vocab_size=4096,  # Same as before
    min_frequency=2,
    verbose=True
)

print()
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Number of merges: {tokenizer.num_merges}")
print()

# Check for digits in vocabulary
import re
digit_tokens = [token for token in tokenizer.get_vocab().keys() if re.search(r'\d', token)]
print(f"Tokens containing digits: {len(digit_tokens)}")
if digit_tokens:
    print(f"  Sample: {digit_tokens[:20]}")
else:
    print("  SUCCESS: No digits in vocabulary!")
print()

# Test tokenization of problematic strings
bad_strings = ["si6ile", "nersao", "namhlanje", "sivase"]
print("Tokenization test:")
for s in bad_strings:
    encoded = tokenizer.encode(s, add_special_tokens=False)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    print(f"  '{s}' -> {encoded} -> '{decoded}'")
print()

# Save tokenizer
output_dir = r'D:\xnlp\artifacts\xnlp_v3_clean_tokenizer'
import os
os.makedirs(output_dir, exist_ok=True)

from xnlp_trainer.data import tokenizer_to_state_dict
import json

tokenizer_state = tokenizer_to_state_dict(tokenizer)
with open(os.path.join(output_dir, 'tokenizer_state.json'), 'w', encoding='utf-8') as f:
    json.dump(tokenizer_state, f, ensure_ascii=False, indent=2)

print(f"Tokenizer saved to: {output_dir}")
print()

# Save vocab for inspection
with open(os.path.join(output_dir, 'vocab.txt'), 'w', encoding='utf-8') as f:
    for token, idx in sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]):
        f.write(f"{idx}\t{token}\n")

print("=" * 80)
print("CLEAN TOKENIZER TRAINED")
print("=" * 80)
