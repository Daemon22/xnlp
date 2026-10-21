# XNLP Language Derivation Report

- Specification: `xnlp_language/generated/derivation/language_specification.json` (version 0.1.0).
- Status: exploratory, reversible, and not an executable grammar.

## A. Exact DERIVATION_EVIDENCE_SET

**ID:** `DERIVATION_EVIDENCE_SET_V1`; **records:** 5716; **independence groups represented:** 18.

A record from generated/evidence/corpus_records.jsonl is included iff its source_title exists in source_provenance.json with foundation_attested=true, provenance_class=attested, provenance_type=authoritative, generator is empty, validation_status=authoritative, language_class=TARGET_LANGUAGE, and text equals original_text. This is an evidence boundary, not a new corpus or correction.

This set identifies source-attested, target-language rows whose stored text equals `original_text`. It does **not** mean the transcription was verified against an edition: there are 0 records with explicit correction/transcription review metadata. Three otherwise target-language rows differ from `original_text` by punctuation and are held out pending review. The corpus was not modified.

### Source works and coverage

| Source title | Canonical records | Included records | Independence group | Attested |
|---|---:|---:|---|---|
| Abantu besizwe | 2 | 1 | Abantu besizwe | yes |
| General Mqhayi | 61 | 60 | General Mqhayi | yes |
| Imibongo | 1 | 1 | Imibongo | yes |
| Imihobe Nemibongo | 433 | 409 | Imihobe Nemibongo | yes |
| Imihobe nemibongo | 3 | 2 | Imihobe Nemibongo | yes |
| Inzuzo | 559 | 539 | Inzuzo | yes |
| Ityala Lamawele | 1770 | 1748 | Ityala Lamawele | yes |
| Izibongo zoogxa | 1 | 1 | Izibongo zoogxa | yes |
| Iziganeko zesizwe | 2 | 1 | Iziganeko zesizwe | yes |
| Izwe | 2 | 2 | Izwe | yes |
| Masikhanyise | 38 | 38 | Masikhanyise | yes |
| Masikhanyise isiXhosa Home Language textbook series | 127 | 122 | Masikhanyise isiXhosa Home Language textbook series | yes |
| Mqhayi - Various works (original corpus) | 86 | 77 | mqhayi_complete.txt | yes |
| U John Knox Bokwe (Biography) | 662 | 648 | U John Knox Bokwe (Biography) | yes |
| U-Don Jadu | 1184 | 1172 | U-Don Jadu | yes |
| UMqhayi waseNtabozuko | 800 | 792 | UMqhayi waseNtabozuko | yes |
| Umhlekazi u Hintsa | 71 | 70 | Umhlekazi u Hintsa | yes |
| Unknown Mqhayi | 36 | 32 | mqhayi_complete.txt | yes |
| Usapho | 1 | 1 | Usapho | yes |

Included records by independence group:

- Abantu besizwe: 1
- General Mqhayi: 60
- Imibongo: 1
- Imihobe Nemibongo: 412
- Inzuzo: 540
- Ityala Lamawele: 1755
- Izibongo zoogxa: 1
- Iziganeko zesizwe: 2
- Izwe: 2
- Masikhanyise: 38
- Masikhanyise isiXhosa Home Language textbook series: 122
- U John Knox Bokwe (Biography): 729
- U-Don Jadu: 1173
- U-Samson: 3
- UMqhayi waseNtabozuko: 793
- Umhlekazi u Hintsa: 70
- Usapho: 1
- mqhayi_complete.txt: 13

Canonical corpus exclusions: {"language_class_FOREIGN_LANGUAGE": 5, "language_class_MIXED_LANGUAGE": 24, "language_class_UNCERTAIN": 91, "target_text_differs_from_preserved_original": 3}. An additional 89 generated candidate records are excluded.

### Provenance and correction boundary

- Registry source-title entries: 19; independence groups: 17.
- Unregistered canonical corpus records: 0.
- Owner/source attestation is not transcription-fidelity verification.
- Derivations describe stored text only; ambiguity or OCR/transcription-looking forms are not repaired.

## B. Observed language inventory

These are observations about the evidence text. They do not by themselves establish linguistic categories.

- Alphabetic Unicode code points observed: 52 case-sensitive symbols / 26 casefolded symbols.
- Symbol occurrence inventory: `A` (1512), `a` (84080), `B` (3013), `b` (12686), `C` (230), `c` (2914), `D` (496), `d` (12508), `E` (1074), `e` (55298), `F` (186), `f` (4425), `G` (640), `g` (15303), `H` (581), `h` (25028), `I` (2368), `i` (52913), `J` (418), `j` (2913), `K` (2726), `k` (35837), `L` (1264), `l` (31711), `M` (2305), `m` (19592), `N` (3839), `n` (54034), `O` (538), `o` (37619), `P` (396), `p` (5023), `Q` (251), `q` (3477), `R` (202), `r` (1344), `S` (986), `s` (15474), `T` (769), `t` (17424), `U` (2573), `u` (39896), `V` (170), `v` (3020), `W` (855), `w` (19801), `X` (572), `x` (2486), `Y` (390), `y` (15890), `Z` (759), `z` (15576).
- Regex-defined tokens: 98363 occurrences / 30017 casefolded types. Regex is a measurement convention, not proven wordhood.
- Whitespace-delimited units: 99075.
- Punctuation: `,` (11216), `.` (6841), `-` (3600), `’` (2783), `;` (1951), `!` (1209), `—` (926), `“` (880), `”` (749), `?` (440), `:` (437), `'` (404), `)` (399), `(` (396), `*` (130), `‘` (104), `&` (68), `/` (44), `\` (43), `•` (35), `"` (27), `_` (16), `{` (10), `[` (9), `»` (6).
- Within-token letter bigrams: `ku` (13585), `ng` (13299), `an` (11168), `wa` (10363), `in` (9545), `la` (8800), `th` (8646), `nd` (8386), `el` (8340), `en` (8272), `le` (7667), `zi` (7415), `na` (7348), `ka` (6498), `we` (6493), `ba` (6313), `si` (5953), `am` (5902), `ma` (5823), `ak` (5804).
- Within-token letter trigrams: `nga` (4665), `ndi` (3462), `uku` (3398), `thi` (3048), `kwa` (2926), `nge` (2777), `ama` (2733), `ngo` (2690), `oku` (2448), `ele` (2392), `ang` (2322), `ela` (2321), `ath` (2307), `the` (2204), `kub` (2197), `ith` (2171), `ana` (2029), `uba` (2009), `ala` (1963), `ing` (1900).
- Case-variant token types: 2640; types represented in at least two independent groups: 5535.
- These n-grams are `OBSERVED_PATTERN`, not syllables, phonemes, morphemes, or phonotactic rules.

### Frequent surface forms (not lemmas, POS, or meanings)

| Casefolded form | Occurrences | Records | Independent groups |
|---|---:|---:|---:|
| `a` | 1688 | 1054 | 10 |
| `ke` | 1000 | 848 | 11 |
| `ukuba` | 786 | 676 | 8 |
| `e` | 636 | 536 | 10 |
| `le` | 594 | 549 | 13 |
| `nto` | 571 | 509 | 10 |
| `nje` | 496 | 404 | 9 |
| `lo` | 470 | 408 | 9 |
| `na` | 461 | 398 | 11 |
| `ku` | 437 | 379 | 9 |
| `u` | 411 | 342 | 9 |
| `kuba` | 385 | 357 | 7 |
| `into` | 380 | 342 | 9 |
| `uku` | 378 | 346 | 9 |
| `kwa` | 351 | 319 | 8 |
| `i` | 345 | 280 | 9 |
| `o` | 341 | 292 | 8 |
| `loo` | 291 | 275 | 7 |
| `xa` | 291 | 267 | 10 |
| `ma` | 286 | 255 | 9 |

## C. Supported structures

No linguistic generalization is promoted to **SUPPORTED** in this phase. The directly supported facts are corpus/statistical facts in the specification (record inclusion, character counts, tokenization convention, and observed source-linked strings); they are not grammar rules.

## D. Candidate structures

There are 834 existing source-linked claim records, including 716 morphology, 78 syntax/modifier/negation, 13 agreement/concord, and 26 functional candidates. Existing `SOURCE_SUPPORTED` legacy labels are retained as provenance metadata but remain candidates for language derivation; they are not human-validated conclusions.

The machine specification preserves all 153 source-linked observation records with evidence IDs, domains, raw patterns, and the distinction between observed count and candidate interpretation.

No morpheme, stem, lexical category, agreement rule, word order, or semantic meaning is asserted by this report. The 29 prior auto-evidence lexical category decisions remain queue decisions only and are not treated as linguistic facts.

## E. Unknown / insufficiently evidenced

- Phonemes, syllables, phonotactic constraints, and sound-to-spelling rules: `EXTERNAL_KNOWLEDGE_REQUIRED` or direct phonetic evidence required.
- Morpheme boundaries, roots, affixes, inflection, derivation, and category assignments: candidates only.
- Sentence segmentation beyond punctuation-delimited text; syntactic constituent structure; agreement interpretation; negation/interrogation/subordination function.
- Word sense, predicate semantics, discourse references, and pragmatic/focus interpretations.
- Transcription fidelity: unverified for all rows; stored `text == original_text` only establishes internal identity.

## F. Counterexamples and conflicts

No systematic positive-vs-negative search was performed. Existing claim artifacts mostly contain empty counterexample lists; those are not evidence of no exceptions.
- `CONFLICT_LINGUISTIC_001` (prefix_ambiguity): a-; competing labels: subject_concord_class_6, noun_class_prefix_class_6, negation_marker; 46 in-boundary records across 9 groups. Status remains OPEN.

Competing explanations and unresolved records must remain open; the full claims, record IDs, source titles, and work-group counts are in the specification.

## G. Proposed language representation

The versioned machine-readable representation separates ORTHOGRAPHY, PHONOLOGICAL/PHONOTACTIC observations, MORPHOLOGY, LEXICON, SYNTACTIC_STRUCTURE, AGREEMENT_CONCORD, SEMANTIC_FUNCTIONAL, DISCOURSE_CONSTRUCTION, and UNKNOWN_UNRESOLVED. Every candidate retains record references and provenance. The representation is data, not executable grammar.

## H. Provenance map

The source registry and independence groups are embedded in `DERIVATION_EVIDENCE_SET`; each lexical count and claim/observation stores source record references. Mqhayi compilation records use the shared registry independence group unless an explicit internal-heading attribution is present. No generated candidate is accepted as attested.

## I. Test results

- test_command: python -m pytest tests/test_language_derivation.py tests/test_linguistic_layer.py tests/test_foundation_bundle.py xnlp_language/tests/test_foundation.py -q
- before_derivation: 34 passed in 1611.06s
- after_derivation: 35 passed in 236.95s

## J. Gaps requiring more evidence

- Record-level transcription review, with corrected text/version provenance rather than silent edits.
- Human-annotated lexical, morphological, syntactic, semantic, and discourse examples.
- Independent counterexample and competing-analysis review across genuinely independent works.
- Sound/phonetic evidence if phonological statements are required.
- Resolved review of the three punctuation-different text/original pairs and the open prefix ambiguity.

## K. Proposed sequence: LANGUAGE DERIVATION → GRAMMAR DERIVATION

1. Freeze the evidence-set manifest and record-level correction trail.
2. Add adjudicated, source-linked examples plus explicit negative/counterexample cases.
3. Derive candidate relationships and constructions with independent-work replication.
4. Have qualified reviewers assess competing analyses and unknowns.
5. Promote only reviewed generalizations to SUPPORTED in a versioned specification.
6. Implement grammar only after those derivations and tests are approved; this phase implements none.

## Invariants

No corpus row is changed; no provenance is broadened; no claim is validated; no machine inference is described as human review; and all derived outputs can be regenerated or removed.
