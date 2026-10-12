#!/usr/bin/env python3
"""
Independent verification of the V3 clean corpus, data splits, and tokenizer.

This is an AUDIT script: it does NOT trust the reported characteristics and
re-derives every fact independently.  It prints a structured report so that
we can decide whether to proceed to training.

Checks
------
CORPUS
  - valid UTF-8 / decodable
  - exact line count, empty records
  - duplicate analysis (intra-corpus)
  - digit-in-word contamination (the OCR pathology: digits inside alnum runs)
  - English-title contamination (Col./Rev./Dr./Mr./Mrs./Miss)
  - sentence-boundary validity (ends with terminal punctuation)
SPLITS (train/val/test)
  - sizes, reproducibility vs split_summary.json
  - cross-split leakage (exact-sentence duplicates across train/val/test)
  - no internal duplicates within a split
TOKENIZER
  - vocab internal consistency (bijective token2id<->id2token, contiguous IDs)
  - 0 digit-bearing tokens (the V2 pathology had 146)
  - round-trip encode->decode on representative isiXhosa
  - [UNK] rate across the corpus
  - no unexpected corruption introduced
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\xnlp')

import json
import re
from pathlib import Path
from collections import Counter

from core_llm.tokenizer import XNLPTokenizer, SpecialTokens
from xnlp_trainer.data import tokenizer_from_state_dict

CORPUS = Path(r'D:\xnlp\corpus_v2/v2_training_corpus_clean_final.txt')
SPLITS = Path(r'D:\xnlp/data/processed/v3/splits')
TOKENIZER_STATE = Path(r'D:\xnlp/artifacts/xnlp_v3_clean_tokenizer/tokenizer_state.json')
SUMMARY = SPLITS / 'split_summary.json'

SEP = "=" * 70


def ok(msg): print(f"  [PASS] {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")


# ---------------------------------------------------------------------------
# 1. CORPUS
# ---------------------------------------------------------------------------
print(SEP)
print("CORPUS VERIFICATION")
print(SEP)

raw_bytes = CORPUS.read_bytes()
# UTF-8 validity
try:
    raw_text = raw_bytes.decode('utf-8')
    ok(f"Valid UTF-8 ({len(raw_bytes)} bytes)")
except UnicodeDecodeError as e:
    fail(f"Invalid UTF-8: {e}")
    raw_text = raw_text if 'raw_text' in dir() else raw_bytes.decode('utf-8', errors='replace')

lines = raw_text.splitlines()
nonempty = [l for l in lines if l.strip()]
empty = len(lines) - len(nonempty)
print(f"  Total lines        : {len(lines)}")
print(f"  Non-empty lines    : {len(nonempty)}")
print(f"  Empty lines        : {empty}")
if empty == 0:
    ok("No empty records")
else:
    fail(f"{empty} empty records")

# Duplicates within corpus
dup = len(nonempty) - len(set(nonempty))
if dup == 0:
    ok("No exact duplicate lines in corpus")
else:
    warn(f"{dup} duplicate lines in corpus")

# Digit-in-word contamination.
# The OCR pathology substitutes digits (esp '6') for letters INSIDE words.
# We flag any run of [letter][letter/digit]*[digit][letter/digit]*[letter]...
# i.e. a word that mixes letters and digits.
digit_in_word = re.compile(r'\b\w*\d\w*\b', re.UNICODE)
digit_lines = []
for i, l in enumerate(nonempty):
    if digit_in_word.search(l):
        digit_lines.append((i, l))
if not digit_lines:
    ok("Zero digit-in-word contamination (no alnum-mixed tokens)")
else:
    fail(f"{len(digit_lines)} lines contain digit-in-word contamination")
    for idx, l in digit_lines[:5]:
        print(f"      line {idx}: {l}")

# English-title contamination (the specific pathology targeted by cleaning).
# These are abbreviations used as standalone English titles, NOT proper names.
english_titles = re.compile(r'\b(Col\.|Rev\.|Dr\.|Mr\.|Mrs\.|Miss)\b', re.IGNORECASE)
title_hits = [(i, l) for i, l in enumerate(nonempty) if english_titles.search(l)]
if not title_hits:
    ok("Zero English-title contamination (Col./Rev./Dr./Mr./Mrs./Miss absent)")
else:
    warn(f"{len(title_hits)} lines still contain English titles")
    for idx, l in title_hits[:5]:
        print(f"      line {idx}: {l}")

# Sentence-boundary validity: a sentence should end with . ! ? or Xhosa
# punctuation.  We check the fraction of lines lacking terminal punctuation.
terminals = re.compile(r'[.!?…]+$|"|"\s*$|…|\]$')  # accept common terminals
no_terminal = [l for l in nonempty if not re.search(r'[.!?…]\s*$', l)]
if not no_terminal:
    ok("All lines end with terminal punctuation (. ! ? …)")
else:
    warn(f"{len(no_terminal)} lines lack terminal punctuation")
    for l in no_terminal[:5]:
        print(f"      {l}")

# Note on historical proper names (informational, not a failure):
english_wordish = re.compile(r'\b(Hail|General|Smuts|The late|Provincial|William|Gcule|Matthews|Edwin|Ncwana|Soga|Keodirelang|Zachariah)\b')
name_lines = [(i, l) for i, l in enumerate(nonempty) if english_wordish.search(l)]
print(f"  [INFO] {len(name_lines)} lines contain historical proper names / English")
print("         words within isiXhosa historical text (legitimate, not contamination).")
print("         Examples:")
for idx, l in name_lines[:5]:
    print(f"      line {idx}: {l}")


# ---------------------------------------------------------------------------
# 2. DATA SPLITS
# ---------------------------------------------------------------------------
print()
print(SEP)
print("DATA SPLIT VERIFICATION")
print(SEP)

expected = json.loads(SUMMARY.read_text(encoding='utf-8'))
for split in ('train', 'val', 'test'):
    p = SPLITS / f'{split}.txt'
    slines = [l for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
    exp = expected[f'{split}_size']
    match = "OK" if len(slines) == exp else "MISMATCH"
    print(f"  {split:5s}: {len(slines)} lines (expected {exp})  [{match}]")
    if len(slines) != exp:
        fail(f"{split} size mismatch")

# Cross-split leakage: exact sentences appearing in more than one split
all_splits = {}
for split in ('train', 'val', 'test'):
    p = SPLITS / f'{split}.txt'
    all_splits[split] = [l for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]

train_set, val_set, test_set = (set(all_splits[s]) for s in ('train', 'val', 'test'))
leak_val = len(train_set & val_set)
leak_test = len(train_set & test_set)
leak_vt = len(val_set & test_set)
print(f"  train∩val leakage : {leak_val}")
print(f"  train∩test leakage: {leak_test}")
print(f"  val∩test leakage  : {leak_vt}")
if leak_val == 0 and leak_test == 0 and leak_vt == 0:
    ok("No cross-split leakage (train/val/test disjoint)")
else:
    fail("Cross-split leakage detected!")

# Internal duplicates within each split
for split, s in all_splits.items():
    d = len(s) - len(set(s))
    if d == 0:
        ok(f"{split}: no internal duplicates")
    else:
        warn(f"{split}: {d} internal duplicates")

# Check splits partition the full corpus (union == corpus, no extras)
corpus_set = set(nonempty)
split_union = train_set | val_set | test_set
if split_union == corpus_set:
    ok("Splits exactly partition the clean corpus (no loss, no extras)")
else:
    only_corpus = corpus_set - split_union
    only_split = split_union - corpus_set
    warn(f"{len(only_corpus)} corpus lines NOT in any split; {len(only_split)} split lines NOT in corpus")


# ---------------------------------------------------------------------------
# 3. TOKENIZER
# ---------------------------------------------------------------------------
print()
print(SEP)
print("TOKENIZER VERIFICATION")
print(SEP)

tok_state = json.loads(TOKENIZER_STATE.read_text(encoding='utf-8'))
tok: XNLPTokenizer = tokenizer_from_state_dict(tok_state)

t2i = tok.token2id
i2t = tok.id2token
print(f"  vocab_size (target)   : {tok.vocab_size}")
print(f"  actual vocab size     : {len(t2i)}")
print(f"  num_merges            : {tok.num_merges}")
print(f"  training_data_size    : {tok.training_data_size}")

# Bijectivity
if dict(i2t.values()) and len(set(i2t.keys())) == len(i2t) and len(set(t2i.values())) == len(t2i):
    pass  # sanity
# Check token2id / id2token are exact inverses
inverse_ok = all(i2t.get(v) == k for k, v in t2i.items()) and all(t2i.get(v) == k for k, v in i2t.items())
if inverse_ok:
    ok("token2id / id2token are exact inverses (no duplicate IDs)")
else:
    fail("token2id / id2token are NOT exact inverses")

# Contiguous IDs 0..N-1
ids = sorted(i2t.keys())
expected_ids = list(range(len(ids)))
if ids == expected_ids:
    ok(f"IDs are contiguous 0..{max(ids)}")
else:
    gaps = set(expected_ids) - set(ids)
    dups = [k for k, v in Counter(ids).items() if v > 1]
    fail(f"IDs non-contiguous. missing={sorted(gaps)[:10]}, dup_count={len(dups)}")

# Vocab size agreement
if len(t2i) == tok.vocab_size:
    ok(f"Actual vocab ({len(t2i)}) == target ({tok.vocab_size})")
else:
    warn(f"Actual vocab ({len(t2i)}) != target ({tok.vocab_size})")

# Digit-bearing tokens (the V2 pathology had 146)
digit_tokens = [(t, i) for t, i in t2i.items() if any(c.isdigit() for c in t)]
# Exclude special tokens (they contain no digits anyway).
if not digit_tokens:
    ok("Zero digit-bearing tokens in vocabulary (vs 146 in V2)")
else:
    warn(f"{len(digit_tokens)} digit-bearing tokens remain")
    for t, i in digit_tokens[:10]:
        print(f"      '{t}' (id {i})")

# [UNK] rate across the clean corpus.
# Count how many *subword* tokens (when encoding the corpus) fall on the [UNK] id.
unk_id = tok.unk_token_id
unk_total = 0
enc_total = 0
sample_for_unk = nonempty[:3000]  # sample for speed
for s in sample_for_unk:
    ids = tok.encode(s, add_special_tokens=True)
    enc_total += len(ids)
    unk_total += sum(1 for i in ids if i == unk_id)
unk_rate = unk_total / enc_total if enc_total else 0
print(f"  [UNK] rate over {len(sample_for_unk)} corpus lines: {unk_total}/{enc_total} = {unk_rate:.4f}")
if unk_rate < 0.005:
    ok(f"[UNK] rate is negligible ({unk_rate:.4%})")
else:
    warn(f"[UNK] rate elevated ({unk_rate:.4%}) — investigate coverage")

# Round-trip tests on representative isiXhosa
print("  Round-trip (text -> ids -> decode):")
tests = [
    "Molo, ndiyabulela",
    "Umntu ngumntu ngabantu.",
    "AmaXhosa anebali endle.",
    "Namhlanje sivase ngoMvulo.",
    "Abafundi bayithanda isiXhosa nofundo lwabo.",
    "Umthetho wamaXhosa wuxhomekeke kwigqibo.",
    "Ndiyafunda isiXhosa kwi-University.",
    "Ityala lamawele leli layiqala kwimibuzo.",
    "Kodwa ke, sihlala sihlangabezana nentombazane.",
    "Ingqokelela yethu inxhoja xa sicima intombazane.",
    "1 2 3",                 # legitimate numerals
    "Buku lamawele elilandelayo.",
]
all_roundtrip_ok = True
for t in tests:
    ids = tok.encode(t, add_special_tokens=False)
    decoded = tok.decode(ids, skip_special_tokens=True)
    status = "OK" if decoded == t else "DIFF"
    if decoded != t:
        all_roundtrip_ok = False
    print(f"    [{status}] {repr(t)} -> ids={ids[:8]}{'...' if len(ids)>8 else ''} -> {repr(decoded)}")
if all_roundtrip_ok:
    ok("All round-trips exact (no corruption introduced by tokenizer)")
else:
    warn("Some round-trips differ (see above)")

# Verify the documented V2 pathology: "6" inside words should NOT tokenize
# as a coherent digit-substituted token in V3.
pathology_checks = ["si6ile", "Aka6a", "ku6a", "6antu"]
print("  Pathology-token checks (these should decompose, not merge):")
for w in pathology_checks:
    ids = tok.encode(w, add_special_tokens=False)
    decoded = tok.decode(ids, skip_special_tokens=True)
    print(f"    {w!r:12s} -> ids {ids} -> {decoded!r}")

# Sanity: a real isiXhosa word vs a digit-word
real_word = tok.encode("namhlanje", add_special_tokens=False)
print(f"  Real word 'namhlanje' -> {real_word} (single token if in vocab)")

print()
print(SEP)
print("VERIFICATION COMPLETE")
print(SEP)
