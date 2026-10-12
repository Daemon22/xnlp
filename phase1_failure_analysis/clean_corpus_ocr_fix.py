#!/usr/bin/env python3
"""
XNLP - Clean Corpus with OCR Error Removal
The training corpus has MASSIVE OCR contamination where digits replace letters.
This is the root cause of generating "si6ile", "nersao", etc.

We need to remove all lines containing digits that are being used as letters.
"""
import sys
import re
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("CREATING CLEAN XNLP CORPUS - OCR ERROR REMOVAL")
print("=" * 80)
print()

# Input and output
input_path = r'D:\xnlp\corpus_v2\v2_training_corpus.txt'
output_path = r'D:\xnlp\corpus_v2\v2_training_corpus_clean_final.txt'

with open(input_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Input: {len(lines)} lines")
print()

# Cleaning: Remove lines containing ANY digits
# The OCR errors use digits (especially '6') as letter substitutes
# This is the root cause of all bad generation artifacts
digit_pattern = re.compile(r'\d')

cleaned_lines = []
removed_with_digits = 0
removed_other = 0

# English titles
english_patterns = [
    re.compile(r'Col\.', re.I), re.compile(r'Rev\.', re.I), re.compile(r'Dr\.', re.I),
    re.compile(r'Mr\.', re.I), re.compile(r'Mrs\.', re.I), re.compile(r'Miss\.', re.I),
    re.compile(r'Denys Reitz', re.I), re.compile(r'Jimmy Gray', re.I),
    re.compile(r'Geddes', re.I), re.compile(r'Maya Koboka', re.I),
]

for i, line in enumerate(lines, 1):
    original_line = line.strip()
    
    if not original_line:
        continue
    
    # Remove lines with digits (OCR errors)
    if digit_pattern.search(original_line):
        removed_with_digits += 1
        continue
    
    # Remove lines with English titles
    has_english = False
    for pattern in english_patterns:
        if pattern.search(original_line):
            has_english = True
            break
    
    if has_english:
        removed_other += 1
        continue
    
    # Keep the line
    cleaned_lines.append(original_line)

# Write clean corpus
with open(output_path, 'w', encoding='utf-8') as f:
    for line in cleaned_lines:
        f.write(line + '\n')

print(f"Output: {len(cleaned_lines)} lines")
print(f"Removed with digits (OCR errors): {removed_with_digits}")
print(f"Removed with English titles: {removed_other}")
print(f"Total removed: {removed_with_digits + removed_other}")
print()
print(f"Retention rate: {len(cleaned_lines)/len(lines)*100:.1f}%")
print()

# Verify
with open(output_path, 'r', encoding='utf-8') as f:
    output_lines = f.readlines()

digits_in_clean = sum(1 for line in output_lines if digit_pattern.search(line))
print(f"Lines with digits in clean corpus: {digits_in_clean}")

# Sample
print("\nSample cleaned lines:")
for i, line in enumerate(cleaned_lines[:15], 1):
    print(f"  {i}: {line[:100]}...")
print()

print("=" * 80)
print("CLEAN CORPUS WITH OCR ERRORS REMOVED")
print(f"Input:  {input_path}")
print(f"Output: {output_path}")
print("=" * 80)
