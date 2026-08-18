#!/usr/bin/env python
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True)

for fname in ['corpus/mqhayi_complete.txt', 'corpus/masikhanyise_complete.txt']:
    with open(fname, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    meta_start = None
    for i, line in enumerate(lines):
        if 'Ukufunda ngolwimi lwethu' in line:
            meta_start = i
            break
    print(f'{fname}:')
    print(f'  Total lines: {len(lines)}')
    if meta_start:
        print(f'  Meta lines: {len(lines) - meta_start}')
        print(f'  Actual content lines: {meta_start}')
        print(f'  Last 3 content lines: {lines[meta_start-3:meta_start]}')
    else:
        print(f'  No meta section found')
    print()

# Check content categories
print('=== Corpus content categories ===')
for fname in ['corpus/mqhayi_complete.txt', 'corpus/masikhanyise_complete.txt']:
    with open(fname, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f'{fname}:')
    print('  Ityala Lamawele:', 'tyala' in content.lower())
    print('  U-Don Jadu:', 'Don Jadu' in content)
    print('  Izibongo:', 'izibong' in content.lower())
    print('  Nkosi Sikelel:', 'Nkosi Sikelel' in content)
    print('  Isiganeko:', 'siganek' in content.lower())
    print('  Abantu besizwe:', 'Abantu besizwe' in content)
    print('  Traditional orth (6):', '6' in content)
    print()
