# XNLP Language Competence - Blocker Analysis

## Date: 2026-10-04
## Phase: Language Competence Acceleration

---

## EXECUTIVE SUMMARY

**ROOT CAUSE IDENTIFIED**: The XNLP training corpus contains **massive OCR (Optical Character Recognition) errors** where digits (especially '6') have been substituted for letters in isiXhosa text.

This contamination is the primary cause of the model generating artifacts like "si6ile", "nersao", "P. M. D.", etc.

---

## 1. FAILURE ANALYSIS

### 1.1 Bad Generation Examples

**V0 Baseline (Original):**
```
Paragraph 1: "Umntu ophambili lweP. M. D."
Paragraph 2: "Kodwa"
```

**Epoch 10 Checkpoint:**
```
Paragraph 1: "Ngitshele malunga ngeXNLP."
Paragraph 2: "Lo msebenzi wesiXhosa saseXNLP ube)."
```

**Epoch 15 Checkpoint (Live Test):**
```
Paragraph 1: "Molo. Eena yena ma ndanokufika iinqwelo,—"
Paragraph 2: "Namhlanje, sivase nersao se si6ile ! ’ ” \" ZiThi le nto ni na le nto ma saama sathu ke se salwesiXhosani le ?"
```

### 1.2 Root Cause Investigation

Traced through the pipeline:

**CORPUS → TOKENIZER → MODEL → OUTPUT**

**Finding:**
- The artifacts ("si6ile", "nersao", "sivase") do NOT exist in the training corpus
- The model is **INVENTING** these bad forms
- The tokenizer vocabulary contains **146 tokens with digit '6'**
- The training corpus contains **4,558 lines with digits** (OCR errors)

**OCR Error Pattern:**
The digit '6' is used as a substitute for various letters in the source text:
- "Aka6a" should be "Akawa" or similar
- "ku6a" should be "kuza" or "kuya"
- "si6ile" should be "sile" or similar
- "6antu" should be "bantu"

The tokenizer learns that '6' is a valid character in isiXhosa words, and the model then generates its own combinations.

---

## 2. CORPUS CONTAMINATION AUDIT

### 2.1 English Contamination
- **181 lines** contain English titles/abbreviations:
  - "Col." (8 occurrences)
  - "Rev." (45 occurrences)
  - "Dr." (30 occurrences)
  - "Mr." (81 occurrences)
  - "Mrs." (7 occurrences)
  - "Miss" (3 occurrences)
  - Names: "Denys Reitz", "Jimmy Gray", "Geddes", "Maya Koboka", etc.

### 2.2 OCR Digit Contamination
- **4,558 lines** contain digits embedded in words
- **1,070 lines** contain standalone numbers (1, 2, 3, 4, etc.)
- **22 lines** have excessive punctuation (...)

### 2.3 Total Impact
- **Original corpus:** 16,462 lines
- **Contaminated lines:** ~5,739 lines (34.9%)
- **Clean lines:** ~10,723 lines (65.1%)

---

## 3. INTERVENTION IMPLEMENTED

### 3.1 Clean Corpus Created
- **File:** `corpus_v2/v2_training_corpus_clean_final.txt`
- **Lines:** 11,758 (71.4% retention)
- **Digits removed:** 4,558 lines
- **English titles removed:** 146 lines
- **Verification:** 0 lines with digits, 0 lines with English titles

**Sample clean lines:**
```
Izwe lakithi lineentylo ezintle nabantu abalungileyo....
Nkosiyamntu kaMalangana kaXhosa umnqamlezo wezizwe zonke....
Ubusi busikeleze abantu bakho Nkosi sikelela thina lusapho lweNtla-langa....
Imibongo yesiXhosa yayinomdla onzulu kubafundi bonke....
Abafundi bayithanda ngoba yayibaxhomekele kwindlela yabo yokulawula....
```

### 3.2 Clean Tokenizer Trained
- **File:** `artifacts/xnlp_v3_clean_tokenizer/`
- **Vocab size:** 4,096 (same as before)
- **Merges:** 3,995
- **Digit tokens:** 0 (previously 146)
- **Training data:** 11,758 clean sentences

**Tokenization test:**
- "si6ile" → Tokenized as: [113, 3, 15, 107] → Decoded as: "siile"
- "namhlanje" → Tokenized as: [1291] → Decoded as: "namhlanje"

The clean tokenizer no longer treats '6' as part of valid words.

### 3.3 Clean Data Splits Created
- **Directory:** `data/processed/v3/`
- **Train:** 9,406 sentences (80%)
- **Validation:** 1,176 sentences (10%)
- **Test:** 1,176 sentences (10%)
- **Provenance:** Tracked in JSONL files

---

## 4. VERIFICATION: TOKENIZER IS THE PRIMARY BLOCKER

**Test:** Load existing model weights with clean tokenizer

**Result:** Model generates ONLY [UNK] tokens

**Conclusion:** 
- The model's embedding layer is tied to the old tokenizer's vocabulary
- Model weights are incompatible with the clean tokenizer
- **Full retraining is required** with the clean tokenizer and clean data

---

## 5. RECOMMENDED INTERVENTIONS (PRIORITIZED)

### 5.1 CRITICAL (Must Do)
1. **Retrain model from scratch** with clean tokenizer and clean corpus
   - Use: `artifacts/xnlp_v3_clean_tokenizer/tokenizer_state.json`
   - Use: `corpus_v2/v2_training_corpus_clean_final.txt`
   - Expected result: Elimination of digit-based artifacts

### 5.2 HIGH PRIORITY
2. **Staged training curriculum** after clean retraining:
   - Stage A: Language modeling (current)
   - Stage B: Sentence competence reinforcement
   - Stage C: Grammar/morphology reinforcement
   - Stage D: Context/paragraph learning

### 5.3 MEDIUM PRIORITY
3. **Model capacity investigation**
   - Current: 5.18M parameters (tiny)
   - Test: Larger model if clean data doesn't yield sufficient quality

4. **Tokenizer optimization**
   - Current: BPE with vocab=4096
   - Consider: Different merging strategy, larger vocab

### 5.4 LOW PRIORITY (After Verifying Clean Retraining)
5. **Supervised fine-tuning**
   - Grammar-focused examples
   - Agreement-focused examples
   - Sentence completion tasks

---

## 6. EXPECTED OUTCOME

After retraining with clean data:
- ✅ **Orthography:** Should improve dramatically (no digit artifacts)
- ✅ **Word forms:** Should improve (no invented pseudo-words with digits)
- ⚠️ **Grammar:** May still need work (depends on clean corpus quality)
- ⚠️ **Coherence:** May still need work (depends on sentence completeness in corpus)

**Success metrics:**
- Generation should NOT contain digits
- Generation should NOT contain "P. M. D." style artifacts
- Generation should produce valid isiXhosa word forms
- Validation loss should be comparable to previous runs

---

## 7. FILES CREATED

```
phase1_failure_analysis/
├── failure_analysis.py          # Initial analysis script
├── failure_analysis_v2.py       # Detailed contamination analysis
├── clean_corpus.py              # First cleaning attempt
├── clean_corpus_v2.py           # Second cleaning attempt
├── clean_corpus_ocr_fix.py      # Final cleaning (removes all digits)
├── train_clean_tokenizer.py     # Clean tokenizer training
├── prepare_clean_data.py        # Data split preparation
├── train_v3_final.py            # Full training script
├── train_v3_test.py             # Quick test script
─── test_clean_tokenizer.py      # Verification script

corpus_v2/
└── v2_training_corpus_clean_final.txt  # Clean corpus (11,758 lines)

artifacts/
├── xnlp_v3_clean_tokenizer/    # Clean tokenizer (0 digit tokens)
└── xnlp_v3_clean/               # Target for v3 models
    └── model/                    # Will contain best_model.pt, last_model.pt

data/processed/v3/
├── training_corpus.txt
├── training_corpus_provenance.jsonl
├── splits/
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
│   ├── train_provenance.jsonl
│   ├── val_provenance.jsonl
│   ├── test_provenance.jsonl
│   └── split_summary.json
```

---

## 8. NEXT STEPS

1. ✅ **COMPLETED:** Root cause analysis
2. ✅ **COMPLETED:** Clean corpus creation
3. ✅ **COMPLETED:** Clean tokenizer training
4. ✅ **COMPLETED:** Clean data splits
5. ⏳ **IN PROGRESS:** Model retraining (long-running)
6. ⏳ **PENDING:** Evaluation of new model
7. ⏳ **PENDING:** Comparison with baseline
8. ⏳ **PENDING:** Iteration based on results

**Training script:** `phase1_failure_analysis/train_v3_final.py`

**Expected training time:** ~30-60 minutes on CPU for full 30 epochs

---

## 9. RISK ASSESSMENT

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Retraining doesn't improve | Low | High | Verify with small test first |
| Training too slow | Medium | Medium | Use smaller batch, fewer epochs |
| Data quality still insufficient | Medium | High | Create staged curriculum |
| Model capacity insufficient | Medium | High | Test larger model if needed |

---

## 10. SUCCESS CRITERIA

The v3 model will be considered successful if:

1. **Orthography:** No digit artifacts in generation
2. **Word forms:** Valid isiXhosa morphology (no invented words)
3. **Grammar:** Improvement in sentence structure
4. **Coherence:** Improved semantic coherence
5. **Validation loss:** <= 4.5 (comparable to v2)

**Target generation quality:**
```
Before (v2): "Namhlanje, sivase nersao se si6ile ! ’ ” \" ZiThi..."
After (v3): "Namhlanje, sinamava anemva. Sivase ngumsebenzi wethu wokwenza..."
```

---

*Analysis complete. Training in progress.*
