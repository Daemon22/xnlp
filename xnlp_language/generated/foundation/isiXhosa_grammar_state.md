# isiXhosa Grammar State — XNLP Sovereign Foundation

**Foundation Release**: XNLP_ISIXHOSA_FOUNDATION_V1
**Corpus SHA-256**: 3fcec2c0c9dac479…
**Generation Mode**: deterministic_from_corpus_hash

## Preservation Layer (Immutable)

- **5839** authoritative records preserved exactly.
- **5719** TARGET_LANGUAGE records eligible for automated observation.
- **120** mixed/foreign/uncertain records retained but excluded from automatic analysis.
- Original surface text is immutable; normalization policy is identity-only.

## 1. Noun-Class System

The noun-class inventory is derived from corpus prefix co-occurrence evidence and
hand-curated foundation entries. Each class carries evidence record IDs, observation IDs,
and review status. Classes are NOT automatically assumed from prefix distributions alone.

| Class | Singular Prefix | Plural Prefix | Status | Confidence | Evidence Records |
|-------|----------------|---------------|--------|------------|-----------------|
| 1 | um- | aba- | SUPPORTED | HIGH_CONFIDENCE | 3 |
| 2 | aba- | aba- | SUPPORTED | HIGH_CONFIDENCE | 3 |
| 3 | um- | imi- | SUPPORTED | HIGH_CONFIDENCE | 3 |
| 5 | i- | ama- | SUPPORTED | HIGH_CONFIDENCE | 3 |
| 6 | ama- | ama- | SUPPORTED | HIGH_CONFIDENCE | 3 |
| 7 | isi- | izi- | SUPPORTED | HIGH_CONFIDENCE | 3 |
| 9 | u- | o- | SUPPORTED | HIGH_CONFIDENCE | 1 |

Total noun classes: **7**

## 2. Agreement System

Agreement is modelled as a relational system: noun class → agreement feature → concord →
construction → corpus evidence. Each entry links to specific corpus records.

| Agreement ID | Type | Noun Class | Status | Evidence Records |
|-------------|------|------------|--------|-----------------|
| AGR_POSS_NC9 | possessive_concord | NC_009 | SUPPORTED | 2 |
| AGR_SUBJ_NC1 | subject_concord | NC_001 | SUPPORTED | 2 |
| AGR_SUBJ_NC2 | subject_concord | NC_002 | SUPPORTED | 2 |

Total agreement entries: **3**

## 3. Verb Morphology

Verb forms are analysed as candidate segmentations. Each entry records surface form,
candidate segments, analysis, supporting evidence, and counterexamples. Segmentations are
hypotheses requiring human morphological review — never automatic rules.

| Entry ID | Surface Form | Segments | Status | Evidence |
|----------|-------------|----------|--------|----------|
| VMORPH_0001 | ababephila | a/babephila | UNDER_REVIEW | 5 |
| VMORPH_0002 | abafazi | a/bafazi | UNDER_REVIEW | 7 |
| VMORPH_0003 | abafo | a/bafo | UNDER_REVIEW | 10 |
| VMORPH_0004 | abafundi | a/bafundi | UNDER_REVIEW | 20 |
| VMORPH_0005 | abafundisi | a/bafundisi | UNDER_REVIEW | 18 |
| VMORPH_0006 | abakhulu | a/bakhulu | UNDER_REVIEW | 7 |
| VMORPH_0007 | abamhlophe | a/bamhlophe | UNDER_REVIEW | 6 |
| VMORPH_0008 | abaninzi | a/baninzi | UNDER_REVIEW | 6 |
| VMORPH_0009 | abantsundu | a/bantsundu | UNDER_REVIEW | 20 |
| VMORPH_0010 | abantu | a/bantu | UNDER_REVIEW | 20 |
| VMORPH_0011 | abantwana | a/bantw/ana | UNDER_REVIEW | 20 |
| VMORPH_0012 | abanye | a/banye | UNDER_REVIEW | 20 |
| VMORPH_0013 | abaya | a/baya | UNDER_REVIEW | 12 |
| VMORPH_0014 | abefundisi | a/befundisi | UNDER_REVIEW | 7 |
| VMORPH_0015 | abuye | a/buye | UNDER_REVIEW | 12 |
| VMORPH_0016 | abuze | a/buze | UNDER_REVIEW | 5 |
| VMORPH_0017 | adala | a/dala | UNDER_REVIEW | 6 |
| VMORPH_0018 | afana | a/f/ana | UNDER_REVIEW | 5 |
| VMORPH_0019 | afazi | a/fazi | UNDER_REVIEW | 20 |
| VMORPH_0020 | afike | a/fike | UNDER_REVIEW | 9 |

Total verb morphology entries: **666**

## 4. Derivational Morphology

Derivational patterns (passive, causative, applicative, benefactive) are extracted from
repeated lexical families. Each entry links to corpus examples and observation IDs.

| Entry ID | Affix | Type | Status | Evidence |
|----------|-------|------|--------|----------|
| DERMORPH_0001 | ana | benefactive | UNDER_REVIEW | 20 |
| DERMORPH_0002 | isa | causative | UNDER_REVIEW | 20 |
| DERMORPH_0003 | elwa | passive | UNDER_REVIEW | 20 |
| DERMORPH_0004 | iswa | passive | UNDER_REVIEW | 20 |

Total derivational entries: **4**

## 5. Coverage Summary

- **agreement**: 3 entries, status: SUPPORTED, confidence: HIGH_CONFIDENCE
- **discourse**: 0 entries, status: INSUFFICIENT_EVIDENCE, confidence: INSUFFICIENT_EVIDENCE
- **lexicon**: 30017 entries, status: OBSERVED, confidence: HIGH_CONFIDENCE
- **modifiers**: 2 entries, status: OBSERVED, confidence: OBSERVED
- **morphology**: 7 entries, status: SUPPORTED, confidence: HIGH_CONFIDENCE
- **negation**: 2 entries, status: OBSERVED, confidence: HIGH_CONFIDENCE
- **noun_classes**: 7 entries, status: SUPPORTED, confidence: HIGH_CONFIDENCE
- **orthography**: 1 entries, status: OBSERVED, confidence: HIGH_CONFIDENCE
- **pronouns**: 6 entries, status: OBSERVED, confidence: SUPPORTED
- **semantics**: 0 entries, status: INSUFFICIENT_EVIDENCE, confidence: INSUFFICIENT_EVIDENCE
- **syntax**: 74 entries, status: OBSERVED, confidence: OBSERVED
- **tense_aspect_mood**: 7 entries, status: OBSERVED, confidence: HIGH_CONFIDENCE
- **verbs**: 666 entries, status: UNDER_REVIEW, confidence: SUPPORTED
- **word_formation**: 4 entries, status: UNDER_REVIEW, confidence: SUPPORTED

## 6. Review Status

All automated analyses are marked as OBSERVED, UNDER_REVIEW, or SUPPORTED.
No entry is ESTABLISHED without human peer review.
- Model training: FROZEN
- Decision: FOUNDATION INCOMPLETE — CONTINUE STRUCTURAL ANALYSIS

## 7. Uncertainty and Limitations

- Frequency counts in the corpus are OBSERVATIONS, never rules.
- Prefix distributions suggest but do not prove noun-class membership.
- Verb segmentations are candidate analyses, not established morphology.
- The foundation remains intentionally incomplete.
- See `generated/reports/structural_gate_report.md` for the structural gate decision.
