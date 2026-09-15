import re

content = open('xnlp_language/build_foundation.py', encoding='utf-8').read()
count = content.count('if review_status in ("UNDER_REVIEW", "OBSERVED"):')
print(f'Occurrences of conditions: {count}')
for i, pos in enumerate([m.start() for m in re.finditer(re.escape('review_status in ("UNDER_REVIEW", "OBSERVED")'), content)]):
    start = max(0, pos - 80)
    end = min(len(content), pos + 200)
    print(f'--- match {i} at pos {pos} ---')
    print(repr(content[start:end]))
    print()
# Also check the build manifest decision
for pos in [m.start() for m in re.finditer(r'FOUNDATION INCOMPLETE', content)]:
    start = max(0, pos - 30)
    end = min(len(content), pos + 60)
    print(f'--- FOUNDATION INCOMPLETE at pos {pos} ---')
    print(repr(content[start:end]))
    print()
for pos in [m.start() for m in re.finditer(r'FOUNDATION INSUFFICIENT', content)]:
    start = max(0, pos - 30)
    end = min(len(content), pos + 60)
    print(f'--- FOUNDATION INSUFFICIENT at pos {pos} ---')
    print(repr(content[start:end]))
    print()
