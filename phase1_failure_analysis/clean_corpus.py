#!/usr/bin/env python3
"""
XNLP - Create Clean Training Corpus
Remove contamination that causes the model to generate bad forms like "si6ile"
"""
import sys
import re
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("CREATING CLEAN XNLP CORPUS")
print("=" * 80)
print()

# Input corpus
input_path = r'D:\xnlp\corpus_v2\v2_training_corpus.txt'
output_path = r'D:\xnlp\corpus_v2\v2_training_corpus_clean.txt'

with open(input_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Input: {len(lines)} lines")
print()

# Cleaning rules
# 1. Remove lines that are mostly numbers or start with numbers
# 2. Remove English abbreviations and titles
# 3. Remove excessive punctuation
# 4. Keep only lines with substantial isiXhosa content

cleaned_lines = []
removed_count = 0
removed_reasons = {}

# Patterns to identify contamination
english_titles = [
    r'\bCol\.', r'\bRev\.', r'\bDr\.', r'\bMr\.', r'\bMrs\.', r'\bMiss\.',
    r'\bProf\.', r'\bEsq\.', r'\bThe Hon\.', r'\bMajor\.', r'\bLt\.',
    r'\bCapt\.', r'\bGen\.', r'\bSir\.', r'\bLady\.', r'\bHon\.',
    r'Denys Reitz', r'Jimmy Gray', r'Geddes', r'Maya Koboka',
    r'James Chalmers', r'Mary Brown', r'W\.L\.', r'W\. W\.',
    r'Stewart', r'R\. GODFREY', r'Willie Geddes',
    r'M\.A\.', r'B\.Sc\.', r'O\.B\.E\.'
]

# Compile patterns
english_patterns = [re.compile(p, re.IGNORECASE) for p in english_titles]

# Digit pattern - standalone numbers or numbers with letters
# This is the CRITICAL fix - remove lines where numbers appear as tokens
digit_pattern = re.compile(r'\b\d+\b')

# Excessive punctuation
punct_pattern = re.compile(r'[{}]'.format(re.escape('... !!! ??? ......')))

# Clean each line
for i, line in enumerate(lines, 1):
    original_line = line.strip()
    
    if not original_line:
        removed_count += 1
        removed_reasons['empty'] = removed_reasons.get('empty', 0) + 1
        continue
    
    # Check for English titles
    has_english = False
    for pattern in english_patterns:
        if pattern.search(original_line):
            has_english = True
            break
    
    if has_english:
        removed_count += 1
        removed_reasons['english_titles'] = removed_reasons.get('english_titles', 0) + 1
        continue
    
    # Check for standalone numbers
    if digit_pattern.search(original_line):
        # But allow some numbers if they're part of valid isiXhosa context
        # For now, remove all lines with standalone numbers
        removed_count += 1
        removed_reasons['standalone_numbers'] = removed_reasons.get('standalone_numbers', 0) + 1
        continue
    
    # Check for excessive punctuation
    if punct_pattern.search(original_line):
        removed_count += 1
        removed_reasons['excessive_punctuation'] = removed_reasons.get('excessive_punctuation', 0) + 1
        continue
    
    # Check for lines that are too short (likely fragments)
    if len(original_line.split()) < 3:
        removed_count += 1
        removed_reasons['too_short'] = removed_reasons.get('too_short', 0) + 1
        continue
    
    # Check for lines that are mostly punctuation
    punct_only = re.sub(r'[\w\s]', '', original_line)
    if len(punct_only) > len(original_line) * 0.5:
        removed_count += 1
        removed_reasons['mostly_punctuation'] = removed_reasons.get('mostly_punctuation', 0) + 1
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
for i, line in enumerate(cleaned_lines[:5], 1):
    print(f"  {i}: {line[:100]}...")
print()

# Verify no numbers remain
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
print()

print("=" * 80)
print("CLEAN CORPUS CREATED")
print(f"Input:  {input_path}")
print(f"Output: {output_path}")
print("=" * 80)
