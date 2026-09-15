#!/usr/bin/env python3
"""Analyze the v2 corpus to understand preservation layer and observations."""
import json
import os
import hashlib

BASE = os.path.dirname(__file__)

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# Load v2 authoritative
v2_path = os.path.join(BASE, "data", "authoritative", "v2_authoritative_all.jsonl")
v2_records = load_jsonl(v2_path)
print(f"v2_authoritative_all.jsonl: {len(v2_records)} records")
print(f"  Unique texts: {len(set(r['text'] for r in v2_records))}")
print(f"  Fingerprints: {len(set(fingerprint(r['text']) for r in v2_records))}")

# Categorize
source_breakdown = {}
lang_class = {}
ortho = {}
for r in v2_records:
    s = r.get("source", "unknown")
    source_breakdown[s] = source_breakdown.get(s, 0) + 1
    lc = r.get("language_class", "unknown")
    lang_class[lc] = lang_class.get(lc, 0) + 1
    o = r.get("orthography", "unknown")
    ortho[o] = ortho.get(o, 0) + 1

print(f"\nSource breakdown: {json.dumps(source_breakdown, indent=2)}")
print(f"\nLanguage class: {json.dumps(lang_class, indent=2)}")
print(f"\nOrthography: {json.dumps(ortho, indent=2)}")

# Check for existing observation/preservation files
print("\n=== Looking for existing observation/preservation files ===")
for root, dirs, files in os.walk(os.path.join(BASE, "data")):
    for f in files:
        if "observ" in f.lower() or "preserv" in f.lower() or "review" in f.lower() or "annot" in f.lower():
            fpath = os.path.join(root, f)
            print(f"  {fpath} ({os.path.getsize(fpath)} bytes)")

# Check xnlp_language/data
lang_data = os.path.join(BASE, "xnlp_language", "data")
if os.path.exists(lang_data):
    print(f"\n=== xnlp_language/data contents ===")
    for root, dirs, files in os.walk(lang_data):
        for f in files:
            print(f"  {os.path.join(root, f)} ({os.path.getsize(os.path.join(root, f))} bytes)")
else:
    print(f"\nxnlp_language/data does not exist")

# Check xnlp_language for any data directory
lang_dir = os.path.join(BASE, "xnlp_language")
print(f"\n=== xnlp_language directory tree ===")
for root, dirs, files in os.walk(lang_dir):
    rel = os.path.relpath(root, lang_dir)
    if rel != ".":
        print(f"  {rel}/")
    for f in files:
        print(f"    {f} ({os.path.getsize(os.path.join(root, f))} bytes)")

# Show some example record texts from different sources
print("\n=== Sample texts from each source ===")
seen_sources = {}
for r in v2_records:
    s = r.get("source", "unknown")
    if s not in seen_sources:
        seen_sources[s] = True
        text = r.get("text", "").replace("\n", " ")
        print(f"\n  [{s}] ortho={r.get('orthography','')} | lang_class={r.get('language_class','')}")
        print(f"  source_type: {r.get('source_type','')}")
        print(f"  source_title: {r.get('source_title','')}")
        print(f"  record_id: {r['record_id']}")
        print(f"  text: {text[:200]}")
