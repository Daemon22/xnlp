#!/usr/bin/env python
"""Comprehensive evaluation ladder for XNLP checkpoints.

Produces machine-readable JSON reports and prints a human-readable summary.

Usage::
    python evaluate_checkpoint.py [checkpoint_path]
    python evaluate_checkpoint.py artifacts/xnlp_v2_tiny_tokenizer_fixed/model/best_model.pt
"""
from __future__ import annotations
import json
import os
import sys
import time
import re
import torch
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from xnlp_trainer.inference import XNLPPredictor
from core_llm.tokenizer import XNLPTokenizer


# ─── Evaluation prompt sets ────────────────────────────────────────────────

# 5 canonical tokenizer round-trip prompts (used by release_gate)
ROUND_TRIP_PROMPTS = [
    "Umntu ngumntu ngabantu.",
    "Ubuntu buhle.",
    "Ndiyafunda isiXhosa.",
    "Ityala lamawele.",
    "AmaXhosa anembali ende.",
]

# Generation evaluation prompts for isiXhosa capability
GEN_EVAL_PROMPTS = [
    # Basic sentence completion
    ("Umntu ngumntu ngabantu.", "sentence_completion"),
    ("Ubuntu buhle.", "sentence_completion"),
    ("Ndiyafunda isiXhosa.", "sentence_completion"),
    ("Ityala lamawele.", "sentence_completion"),
    ("AmaXhosa anembali ende.", "sentence_completion"),
    # Open-ended generation
    ("Inkosi yethu", "open_ended"),
    ("Emva kwemini", "open_ended"),
    ("UMemeso ophiqo", "open_ended"),
    # Instruction-like prompts (the model is NOT instruction-tuned yet,
    # so we measure how it handles them as next-token prediction)
    ("Bhala malunga", "instruction"),
    ("Qaphela ngokuba", "instruction"),
    ("Xhumana ke", "conversation"),
    ("Molo", "conversation"),
]

# Xhosa orthography: valid characters in the Xhosa alphabet
# Includes click consonants (c, q, x) which are part of the Xhosa orthography
XHOSA_VALID_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "'-.,!? "
)
# Xhosa also uses digraphs for clicks: ch, qh, xh, etc.
# These are represented as two characters, so we don't need special handling.

# Xhosa noun class prefixes (for morphology evaluation)
NOUN_CLASS_PREFIXES = [
    ("um", "umuntu"), ("ab", "abantu"), ("i", "intlwini"), ("izi", "izibopho"),
    ("ama", "amahhala"), ("in", "indlela"), ("ii", "iimvo"), ("u", "umsebenzi"),
    ("e", "endle"), ("a", "abantu"), ("un", "umntu"),
]

# Expected Xhosa words (morphology check)
EXPECTED_XHOSA_WORDS = [
    "umntu", "abantu", "izwe", "intetho", "umthetho", "incwadi", "umkhulu",
    "umama", "ubaba", "isikolo", "indlela", "ukuthando", "ubomi", "ubuntu",
    "amandla", "izicoco", "izilwandle", "izikole", "izikhala", "isenzo",
    "isichazi", "isiduko", "isimboni", "isifundo", "ithoni", "ithoni",
    "imibongo", "izibongo", "inkosi", "iziqalingisi",
]


def eval_generation_quality(predictor, prompts):
    """Evaluate generation quality: diversity, length, coherence."""
    results = {"total": 0, "passed": 0, "failed": 0, "details": [], "score": 0.0}
    
    for prompt_text, category in prompts:
        generations = []
        for i in range(3):
            gen = predictor.generate(
                prompt_text, max_new_tokens=50, temperature=0.8,
                top_k=40, top_p=0.9, repetition_penalty=1.1,
            )
            generations.append(gen)
        
        unique = len(set(generations))
        min_len = min(len(g) for g in generations)
        avg_len = sum(len(g) for g in generations) / len(generations)
        
        # Check for excessive repetition (repetition penalty should help)
        has_repetition = any(
            len(g) < 10 or g.count(g.split()[-1] if g.split() else "") > 2
            for g in generations
        )
        
        entry = {
            "prompt": prompt_text,
            "category": category,
            "unique_generations": unique,
            "min_output_length": min_len,
            "avg_output_length": round(avg_len, 1),
            "sample_outputs": [g[:80] for g in generations],
        }
        results["details"].append(entry)
        results["total"] += 1
        
        score = 0
        if unique >= 2:
            score += 3
        if min_len > len(prompt_text):
            score += 2
        if not has_repetition:
            score += 5
        if avg_len > 20:
            score += 5
        
        if score >= 10:
            results["passed"] += 1
        else:
            results["failed"] += 1
        entry["score"] = score
    
    results["score"] = round(results["passed"] / max(results["total"], 1) * 100, 1)
    return results


def eval_morphology(predictor, prompts):
    """Evaluate morphological correctness: noun class prefixes, orthography."""
    results = {"total": 0, "passed": 0, "failed": 0, "details": [], "score": 0.0}
    
    for prompt_text, _ in prompts:
        gen = predictor.generate(
            prompt_text, max_new_tokens=40, temperature=0.7,
            top_k=40, top_p=0.9, repetition_penalty=1.1,
        )
        
        # Check for non-Xhosa characters
        invalid_chars = set()
        for ch in gen:
            if ch not in XHOSA_VALID_CHARS:
                invalid_chars.add(ch)
        
        # Check for English words
        gen_lower = gen.lower()
        english_found = []
        for word in ["the", "and", "of", "to", "in", "is", "that"]:
            if word in gen_lower.split():
                english_found.append(word)
        
        # Check for Xhosa words
        xhosa_found = [w for w in EXPECTED_XHOSA_WORDS if w in gen_lower]
        
        entry = {
            "prompt": prompt_text,
            "generation": gen[:100],
            "invalid_chars": sorted(invalid_chars),
            "english_words_found": english_found,
            "xhosa_words_found": xhosa_found,
        }
        results["details"].append(entry)
        results["total"] += 1
        
        score = 0
        if not invalid_chars:
            score += 4
        if not english_found:
            score += 3
        if len(xhosa_found) >= 1:
            score += 3
        if len(gen) > len(prompt_text):
            score += 5
        
        if score >= 10:
            results["passed"] += 1
        else:
            results["failed"] += 1
        entry["score"] = score
    
    results["score"] = round(results["passed"] / max(results["total"], 1) * 100, 1)
    return results


def eval_tokenizer_roundtrip(tokenizer, prompts):
    """Verify tokenizer round-trip: encode → decode is lossless."""
    results = {"total": 0, "passed": 0, "details": []}
    
    for prompt in prompts:
        encoded = tokenizer.encode(prompt, add_special_tokens=True)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        match = prompt.strip() == decoded.strip()
        
        results["details"].append({
            "prompt": prompt,
            "tokens": len(encoded),
            "decoded": decoded[:100],
            "round_trip_match": match,
        })
        results["total"] += 1
        if match:
            results["passed"] += 1
    
    results["score"] = round(results["passed"] / max(results["total"], 1) * 100, 1)
    return results


def eval_contamination(predictor, prompts, training_records):
    """Check that generated text is not verbatim from training data."""
    results = {"total": 0, "passed": 0, "details": []}
    
    for prompt_text, _ in prompts:
        gen = predictor.generate(
            prompt_text, max_new_tokens=50, temperature=0.8,
            top_k=40, top_p=0.9, repetition_penalty=1.1,
        )
        
        # Check exact match against training records
        exact = any(gen.strip() == rec.strip() for rec in training_records)
        # Check long substring match
        substring = any(
            len(gen.strip()) > 20 and gen.strip() in rec.strip()
            for rec in training_records
        )
        
        results["details"].append({
            "prompt": prompt_text,
            "generation": gen[:80],
            "exact_match": exact,
            "substring_match": substring,
            "novel": not exact and not substring,
        })
        results["total"] += 1
        if not exact and not substring:
            results["passed"] += 1
    
    results["score"] = round(results["passed"] / max(results["total"], 1) * 100, 1)
    return results


def main():
    # Find the checkpoint
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else \
        "artifacts/xnlp_v2_tiny_tokenizer_fixed/model/best_model.pt"
    
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)
    
    print(f"=== Loading checkpoint: {ckpt_path} ===")
    predictor = XNLPPredictor.load(ckpt_path, device="cpu")
    
    # Load training corpus for contamination check
    training_records = []
    corpus_files = [
        "data/processed/v2/training_corpus.txt",
        "data/processed/v2/splits/train.txt",
    ]
    for f in corpus_files:
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    s = line.strip()
                    if s:
                        training_records.append(s)
    print(f"Loaded {len(training_records)} training records for contamination check")
    
    report = {
        "checkpoint": ckpt_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tests": {},
    }
    
    # 1. Tokenizer round-trip
    print("\n=== Test 1: Tokenizer Round-Trip ===")
    tok_report = eval_tokenizer_roundtrip(predictor.tokenizer, ROUND_TRIP_PROMPTS)
    print(f"  Passed: {tok_report['passed']}/{tok_report['total']} ({tok_report['score']}%)")
    for d in tok_report["details"]:
        status = "PASS" if d["round_trip_match"] else "FAIL"
        print(f"  {status}: {d['prompt']} → {d['decoded'][:60]}")
    report["tests"]["tokenizer_roundtrip"] = tok_report
    
    # 2. Generation quality
    print("\n=== Test 2: Generation Quality ===")
    gen_report = eval_generation_quality(predictor, GEN_EVAL_PROMPTS)
    print(f"  Passed: {gen_report['passed']}/{gen_report['total']} ({gen_report['score']}%)")
    for d in gen_report["details"]:
        print(f"  {d['prompt'][:30]:30s} → {d['sample_outputs'][0][:60]}")
    report["tests"]["generation_quality"] = gen_report
    
    # 3. Morphology
    print("\n=== Test 3: Morphology & Orthography ===")
    morph_report = eval_morphology(predictor, GEN_EVAL_PROMPTS[:6])
    print(f"  Passed: {morph_report['passed']}/{morph_report['total']} ({morph_report['score']}%)")
    for d in morph_report["details"]:
        print(f"  {d['prompt'][:30]:30s} → {d['generation'][:60]}")
    report["tests"]["morphology"] = morph_report
    
    # 4. Contamination
    print("\n=== Test 4: Contamination Detection ===")
    contam_report = eval_contamination(predictor, GEN_EVAL_PROMPTS[:5], training_records)
    print(f"  Passed: {contam_report['passed']}/{contam_report['total']} ({contam_report['score']}%)")
    for d in contam_report["details"]:
        status = "PASS" if d["novel"] else "FAIL"
        print(f"  {status}: {d['generation'][:60]}")
    report["tests"]["contamination"] = contam_report
    
    # Summary
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    overall = 0
    for name, result in report["tests"].items():
        score = result.get("score", 0)
        overall += score
        passed = result.get("passed", 0)
        total = result.get("total", 0)
        print(f"  {name:30s}: {passed}/{total}  ({score:.0f}%)")
    overall = overall / max(len(report["tests"]), 1)
    print(f"\n  Overall: {overall:.0f}%")
    
    # Save report
    report["overall_score"] = round(overall, 1)
    report_path = f"data/reports/evaluation_{time.strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("data/reports", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved to: {report_path}")


if __name__ == "__main__":
    main()
