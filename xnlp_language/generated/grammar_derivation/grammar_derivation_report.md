# XNLP Formal Grammar Derivation Report

- Specification: `grammar_specification.json` (version 0.1.0).
- Scope: evidence-bounded, reviewable, non-executable grammar audit.
- No corpus records, provenance labels, or existing observations were changed.

## A. Existing candidate inventory

- Derived candidate structures audited: 834.
- Legacy structured grammar entries audited: 681.
- Source-linked observations preserved: 153.
- Total disposition items: 1515.
- Existing legacy `SUPPORTED` labels are recorded as-is but are not treated as promoted linguistic rules.

### Candidate dispositions

| Disposition | Count |
|---|---:|
| MOVE_TO_UNKNOWN | 2 |
| PROMOTE_TO_SUPPORTED | 0 |
| REJECT | 0 |
| RETAIN_AS_CANDIDATE | 1513 |

No grammatical generalization met the promotion gate. Items with an explicit data-governance scope are kept UNKNOWN/out-of-grammar; linguistic analyses remain candidates pending adjudication.

## B. Candidate dispositions and C. promoted supported rules

- Promoted supported grammar rules: **0**.
- The specification contains no executable rule and no rule with status SUPPORTED.
- Candidates remain candidates when interpretation, full counterexample search, or author independence is unresolved.

## D. Rules retained as candidates and E. rejected rules

- Retained candidate analyses: 1513.
- Rejected analyses: 0.
- No source-backed candidate was rejected solely from absence of evidence; rejection requires a contradictory evidential basis.

## F. Unknown grammatical areas

- ORTHOGRAPHIC_TOKEN_STRUCTURE: Which recurring whitespace or punctuation patterns correspond to language-defined word and boundary units? **REQUIRES HUMAN ADJUDICATION**.
- MORPHOLOGY: Which recurring surface sequences are independently validated morphemes, and what distributional/function evidence supports each segmentation? **REQUIRES HUMAN ADJUDICATION**.
- LEXICAL_CATEGORIES: Which forms exhibit reproducible distributional behavior supporting lexical-category labels? **REQUIRES HUMAN ADJUDICATION**.
- AGREEMENT_CONCORD: Do recurring forms show agreement relationships across independently annotated contexts? **REQUIRES HUMAN ADJUDICATION**.
- NOUN_CLASS_LIKE_SYSTEMS: Do the observed prefix and pairing candidates form a supported system, with what exceptions and agreement behavior? **REQUIRES HUMAN ADJUDICATION**.
- VERB_STRUCTURE: Which candidate segments, alternations, and clause roles are supported by attested positive and negative examples? **REQUIRES HUMAN ADJUDICATION**.
- PHRASE_STRUCTURE: Which recurring category combinations form evidence-supported phrases? **REQUIRES HUMAN ADJUDICATION**.
- CLAUSE_STRUCTURE: What constituent orders and argument relationships are present, and in which constructions? **REQUIRES HUMAN ADJUDICATION**.
- SENTENCE_CONSTRUCTION: Which complete constructions recur with independently adjudicated form and function? **REQUIRES HUMAN ADJUDICATION**.
- MORPHOLOGY_AND_CLAUSE_STRUCTURE: How should the attested surface form a- be analyzed in its conflicting contexts? **REQUIRES HUMAN ADJUDICATION**.
- PROVENANCE: Do stored records faithfully represent their source editions? **REQUIRES ADDITIONAL EVIDENCE**.
- PHONOLOGICAL_PHONOTACTIC: What phonological or phonotactic generalizations, if any, follow from direct evidence? **EXTERNAL_KNOWLEDGE_REQUIRED**.
- PROVENANCE: Are the cited works independent by author as well as by source title? **REQUIRES ADDITIONAL EVIDENCE**.
- ALL_GRAMMAR_LAYERS: Which corpus-wide contexts are positive, negative, exceptional, or competing evidence for each candidate? **REQUIRES HUMAN ADJUDICATION**.

## G–M. Morphology, agreement, nominal, verbal, phrase, clause, and construction structures

No grammatical rule was promoted in these layers. Exact counts of linked observations and candidate items are in `grammar_coverage.json`. Existing labels, interpretations, and examples remain source metadata—not independent validation.

## N. Counterexamples

- The frozen evidence set was scanned in full (5716 records) for exact token sequences explicitly present in linked source-observation examples.
- This is exhaustive surface-form retrieval for those cues, not a grammatical counterexample search. The corpus does not provide exhaustive positive/negative grammatical labels; all candidate rules require human counterexample adjudication.
- Existing listed counterexamples are preserved when their record IDs are inside the evidence set; no absence of counterexamples is inferred from empty legacy lists.

## O. Competing analyses

The open `a-` conflict is preserved with the competing analyses subject concord, noun-prefix, and negation marker; it remains unresolved. Full conflict records are included in the specification.

## P. Independence evidence

- Rules with cited evidence from different work groups: 1330.
- Rules with evidence limited to one work group: 180.
- Rules with one cited record: 5.
- Author independence is UNKNOWN because verified author metadata is not recorded; source-title plurality is not counted as author independence.

## Q. Provenance coverage

- Evidence set: `DERIVATION_EVIDENCE_SET_V1`, 5716 records.
- Source author metadata recorded on cited records: 0.
- Out-of-boundary candidate citations excluded from evidence: 11.
- Every in-boundary rule reference resolves to a source record; authorship and transcription fidelity were not inferred.

## R. Grammar coverage

| Layer | Candidate items | Observations | Supported rules | Coverage |
|---|---:|---:|---:|---|
| ORTHOGRAPHIC_TOKEN_STRUCTURE | 0 | 384 | 0 | OBSERVATIONS_AND_CANDIDATES_ONLY |
| MORPHOLOGY | 1386 | 18 | 0 | OBSERVATIONS_AND_CANDIDATES_ONLY |
| LEXICAL_CATEGORIES | 26 | 26 | 0 | OBSERVATIONS_AND_CANDIDATES_ONLY |
| AGREEMENT_CONCORD | 16 | 10 | 0 | OBSERVATIONS_AND_CANDIDATES_ONLY |
| NOUN_CLASS_LIKE_SYSTEMS | 7 | 0 | 0 | OBSERVATIONS_AND_CANDIDATES_ONLY |
| VERB_STRUCTURE | 1382 | 18 | 0 | OBSERVATIONS_AND_CANDIDATES_ONLY |
| PHRASE_STRUCTURE | 78 | 82 | 0 | OBSERVATIONS_AND_CANDIDATES_ONLY |
| CLAUSE_STRUCTURE | 78 | 82 | 0 | OBSERVATIONS_AND_CANDIDATES_ONLY |
| SENTENCE_CONSTRUCTION | 78 | 82 | 0 | OBSERVATIONS_AND_CANDIDATES_ONLY |

## S. Human-adjudication requirements

- Review proposed interpretations and lexical/morphological segments against the linked, attested record contexts.
- Annotate positive, negative, competing, and counterexample contexts across the full evidence set.
- Record validated source work and author relationships; do not infer them from title strings.
- Review the unresolved `a-` conflict and transcription fidelity.

## T. Test results

- Command: `python -m pytest tests/test_language_derivation.py tests/test_grammar_derivation.py tests/test_linguistic_layer.py tests/test_foundation_bundle.py xnlp_language/tests/test_foundation.py -q`.
- Result: 37 passed in 313.55s.
- Tests assert evidence boundaries, trace resolution, no silent candidate promotion, corpus non-mutation, and deterministic regeneration.

## U. Exact artifacts changed

- `xnlp_language/derive_grammar.py` — evidence-bounded grammar audit generator.
- `xnlp_language/build_bundle.py` — direct bundle reference to the grammar audit.
- `tests/test_grammar_derivation.py` — grammar audit validation tests.
- `tests/test_foundation_bundle.py` — bundle catalog assertions.
- `xnlp_language/README.md` — audit regeneration and evidence limits.

## V. Exact artifacts created

- `xnlp_language/generated/grammar_derivation/grammar_specification.json`
- `xnlp_language/generated/grammar_derivation/candidate_dispositions.json`
- `xnlp_language/generated/grammar_derivation/counterexample_report.json`
- `xnlp_language/generated/grammar_derivation/unresolved_grammar_questions.json`
- `xnlp_language/generated/grammar_derivation/grammar_coverage.json`
- `xnlp_language/generated/grammar_derivation/independence_report.json`
- `xnlp_language/generated/grammar_derivation/provenance_map.json`
- `xnlp_language/generated/grammar_derivation/grammar_derivation_report.md`

## W. Next phase required

Formal grammar validation with human-adjudicated examples, negative evidence, and independently reviewed provenance is required before executable grammar implementation.

## Invariants

No grammar engine, parser, generator, tagger, grammar checker, training run, or model integration was created. No unsupported language fact was made authoritative.
