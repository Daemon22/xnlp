#!/usr/bin/env python
"""Examine extracted Mqhayi text structure for cleaning and extraction."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True)
import os, re

source_dir = 'corpus_sources'

for fname in sorted(os.listdir(source_dir)):
    if not fname.endswith('.txt') or os.path.getsize(os.path.join(source_dir, fname)) == 0:
        continue
    fpath = os.path.join(source_dir, fname)
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    print(f'\n{"="*60}')
    print(f'{fname}: {len(content)} chars, {len(lines)} lines')
    print(f'{"="*60}')
    
    # Show first 15 non-empty lines
    non_empty = [l.strip() for l in lines if l.strip()]
    print(f'Non-empty lines: {len(non_empty)}')
    
    print(f'\nFirst 8 content lines:')
    for i, line in enumerate(non_empty[:8]):
        print(f'  {i+1:3d}| {line[:150]}')
    
    # Identify page number patterns
    page_nums = [l for l in non_empty if re.match(r'^\d+$', l) or re.match(r'^[ivx]+\.?$', l.lower())]
    print(f'\nPage number lines: {len(page_nums)} (sample: {page_nums[:5]})')
    
    # Identify header patterns (short, repeated)
    from collections import Counter
    short_lines = Counter(l for l in non_empty if len(l) < 40)
    print(f'\nMost common short lines:')
    for text, count in short_lines.most_common(8):
        print(f'  [{count:3d}] "{text}"')
    
    # Count lines with traditional orthography (6 for bh, etc.)
    old_ortho = sum(1 for l in non_empty if re.search(r'\b\w*6\w*', l) or re.search(r'\b\w*5\w*', l))
    print(f'\nLines with old orthography: {old_ortho}')
    
    # Count lines in isiXhosa (has Xhosa-specific patterns)
    xhosa_indicators = re.findall(r'(ndiya|ndiy|uku|kwi|ngok|emva|ngokuba|kodwa|njeng|xa\b)', 
                                   ' '.join(non_empty).lower())
    print(f'Xhosa indicator matches: {len(xhosa_indicators)}')
    
    # Show a sample from the middle
    mid = len(non_empty) // 2
    print(f'\nMiddle section:')
    for i in range(max(0, mid-3), min(len(non_empty), mid+3)):
        print(f'  {i+1:3d}| {non_empty[i][:150]}')
