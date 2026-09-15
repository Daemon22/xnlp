#!/usr/bin/env python3
"""Quick corpus analysis - just read small files."""
import json
import os

# Read the smaller masikhanyise file (37 records, clean modern text)
PATH = os.path.join(os.path.dirname(__file__), "data", "authoritative", "mqhayi_masikhanyise.jsonl")
with open(PATH, "r", encoding="utf-8") as f:
    records = [json.loads(l) for l in f if l.strip()]

print(f"Loaded {len(records)} clean masikhanyise records")
print()

for r in records:
    text = r.get("text", "").replace("\n", " ")
    print(f"[{r['source']}|{r.get('orthography','')}|{r.get('source_type','')}] {r['record_id']}")
    print(f"  {text}")
    print()
