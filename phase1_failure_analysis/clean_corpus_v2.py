#!/usr/bin/env python3
"""
XNLP - Create Clean Training Corpus
More careful cleaning that preserves valid isiXhosa while removing contamination.
"""
import sys
import re
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("CREATING CLEAN XNLP CORPUS v2")
print("=" * 80)
print()

# Input corpus
input_path = r'D:\xnlp\corpus_v2\v2_training_corpus.txt'
output_path = r'D:\xnlp\corpus_v2\v2_training_corpus_clean_v2.txt'

with open(input_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Input: {len(lines)} lines")
print()

# Cleaning rules - more careful this time
# The CRITICAL issue is standalone numbers being treated as tokens
# We need to remove:
# 1. Lines with English titles/abbreviations
# 2. Lines with standalone numbers (the main cause of "6ile" type errors)
# 3. Keep lines with normal isiXhosa text even if they have ellipsis

cleaned_lines = []
removed_count = 0
removed_reasons = {}

# English titles and names - remove these completely
english_titles = [
    r'Col\.', r'Rev\.', r'Dr\.', r'Mr\.', r'Mrs\.', r'Miss\.',
    r'Prof\.', r'Esq\.', r'The Hon\.', r'Major\.', r'Lt\.',
    r'Capt\.', r'Gen\.', r'Sir\.', r'Lady\.', r'Hon\.',
    r'Denys Reitz', r'Jimmy Gray', r'Geddes', r'Maya Koboka',
    r'James Chalmers', r'Mary Brown', r'W\.L\.', r'W\. W\.',
    r'Stewart', r'R\. GODFREY', r'Willie Geddes',
    r'M\.A\.', r'B\.Sc\.', r'O\.B\.E\.'
]

# Compile patterns
english_patterns = [re.compile(p, re.IGNORECASE) for p in english_titles]

# CRITICAL: Remove lines with standalone numbers
# This is the root cause of generating "si6ile", "nersao", etc.
# The tokenizer learns that digits can be part of words
digit_pattern = re.compile(r'\b\d+\b')

# Remove lines with excessive consecutive punctuation (4+ of same punct)
excessive_punct = re.compile(r'[.!?]{4,}')

# Clean each line
for i, line in enumerate(lines, 1):
    original_line = line.strip()
    
    if not original_line:
        removed_count += 1
        removed_reasons['empty'] = removed_reasons.get('empty', 0) + 1
        continue
    
    # Check for English titles - REMOVE THESE
    has_english = False
    for pattern in english_patterns:
        if pattern.search(original_line):
            has_english = True
            break
    
    if has_english:
        removed_count += 1
        removed_reasons['english_titles'] = removed_reasons.get('english_titles', 0) + 1
        continue
    
    # Check for standalone numbers - CRITICAL: REMOVE THESE
    # This is what causes the model to generate "6ile" and similar artifacts
    if digit_pattern.search(original_line):
        removed_count += 1
        removed_reasons['standalone_numbers'] = removed_reasons.get('standalone_numbers', 0) + 1
        continue
    
    # Check for excessive punctuation
    if excessive_punct.search(original_line):
        removed_count += 1
        removed_reasons['excessive_punctuation'] = removed_reasons.get('excessive_punctuation', 0) + 1
        continue
    
    # Check for lines that are too short
    if len(original_line.split()) < 3:
        removed_count += 1
        removed_reasons['too_short'] = removed_reasons.get('too_short', 0) + 1
        continue
    
    cleaned_lines.append(original_line)

# Write clean corpus
with open(output_path, 'w', encoding='utf-8') as f:
    for line in cleaned_lines:
        f.write(line + '\n')

print(f"Output: {len(cleaned_lines)} lines")
print(f"Removed: {removed_count} lines")
print()
print("Removal reasons:")
for reason, count in sorted(removed_reasons.items(), key=lambda x: x[1], reverse=True):
    print(f"  {reason}: {count}")
print()

print(f"Retention rate: {len(cleaned_lines)/len(lines)*100:.1f}%")
print()

# Sample of cleaned lines
print("Sample cleaned lines:")
for i, line in enumerate(cleaned_lines[:10], 1):
    print(f"  {i}: {line[:100]}...")
print()

# Verify cleanliness
with open(output_path, 'r', encoding='utf-8') as f:
    output_lines = f.readlines()

numbers_in_clean = sum(1 for line in output_lines if digit_pattern.search(line))
print(f"Lines with standalone numbers in clean corpus: {numbers_in_clean}")

english_in_clean = 0
for line in output_lines:
    for pattern in english_patterns:
        if pattern.search(line):
            english_in_clean += 1
            break

print(f"Lines with English titles in clean corpus: {english_in_clean}")

# Count lines with ellipsis (these are OK)
ellipsis_count = sum(1 for line in output_lines if '...' in line)
print(f"Lines with ellipsis (...): {ellipsis_count}")
print()

print("=" * 80)
print("CLEAN CORPUS CREATED")
print(f"Input:  {input_path}")
print(f"Output: {output_path}")
print("=" * 80)
