# XNLP Linguistic Validation Report

## What do we actually know?
- The preserved corpus contains source-linked isiXhosa text records.
- Lexical artifacts establish surface-form occurrence, not lemma, part of speech, or sense.
- Structured morphology includes proposed analyses with source evidence; most remain candidates.

## What is strongly supported?
- The existence of cited source forms and contexts.
- 10 structured noun-class/agreement claims carry legacy SUPPORTED labels and direct source links; these are source-supported, not project-validated.

## What remains candidate?
- Grammar observations, noun-class and agreement interpretations, and verb segmentations not explicitly human-validated.
- Lexical queue adjudication progress: 29963 pending, 29 adjudicated, 25 disputed.
- 30017 lexical tasks still need some human attention; auto-evidence rows have only a category assignment, with lemma and meaning left unassigned.
- Auto-evidence lexical decisions are reversible and retain their rule IDs and supporting record IDs in an audit trail; rule histogram: {'AUTO_PROPER_NOUN_CAPITALIZED_MULTIWORK_V1': 29}.

## Lexical adjudication progress
- adjudicated: 29
- disputed: 25
- pending: 29963
- Source-supported provenance violations: 0.
- Distinct source works: 19.
- Work attribution counts: {"by_basis": {"internal_heading": 106, "not_applicable": 5717, "unattributed_compilation_record": 16}, "by_work": {"Imihobe Nemibongo": 1, "Inzuzo": 1, "Ityala Lamawele": 7, "Iziganeko zesizwe": 1, "U John Knox Bokwe (Biography)": 91, "U-Don Jadu": 1, "U-Samson": 3, "UMqhayi waseNtabozuko": 1}, "compilation_records_by_source_title": {"Mqhayi - Various works (original corpus)": {"internal_heading": 77, "unattributed_compilation_record": 9}, "Unknown Mqhayi": {"internal_heading": 29, "unattributed_compilation_record": 7}}}.
- Derived record provenance counts: {"attested": 5839}.
- Unregistered source records: 0.
- Auto-evidence decisions are reversible and retain their evidence/rule audit trail.
- Auto-evidence rule histogram: {"AUTO_PROPER_NOUN_CAPITALIZED_MULTIWORK_V1": 29}.
- Tasks still needing any human attention: 30017; unassigned dimensions: {"assigned_category": 29988, "assigned_lemma": 30017, "assigned_meaning": 30017}.

## What remains unresolved?
- Lexical lemmas, categories, semantic senses, and contextual meanings pending review.
- Discourse functions, references, and relations pending review.
- No claim currently meets the project's explicit VALIDATED or ESTABLISHED standard.
- Unresolved linguistic claims: 0
- Unresolved lexical analysis tasks: 30017
- Unresolved semantic annotations: 5719
- Unresolved discourse annotations: 5719

## Claim evidence-state totals
- CANDIDATE: 823
- CROSS_SOURCE_SUPPORTED: 0
- ESTABLISHED: 0
- OBSERVED: 1
- REJECTED: 0
- SOURCE_SUPPORTED: 10
- UNRESOLVED: 0
- VALIDATED: 0

## Useful evidence and sparse domains
- Observed lexicon and corpus-backed noun-class/agreement/morphology candidates.
- Source-linked syntax, TAM, negation, function-word, modifier, and word-formation observations.
- Sparse: Validated semantic and discourse annotations: none yet.
- Sparse: Validated syntax and generation/comprehension gold examples: none yet.

## Is the gold evaluation set genuinely work-isolated?
- 539 unannotated candidate records from `Inzuzo` are separate from the designated work-isolated training partition: yes.
- Validated gold annotations: 0. This is a pending candidate pool, not a populated gold set.
- Do not evaluate checkpoints trained on historical or other data containing the held-out work.
- Author metadata is unavailable; the source collection is not isolated across the work-isolated experiment.
- Historical record splits overlap source works and are diagnostic only.
- The work-isolated partition audit found zero record, exact-text, and near-duplicate overlaps across train/validation/test.

## What is still blocking professional-grade training?
- Human-reviewed and independently adjudicated linguistic claims.
- Validated lexical, semantic, discourse, morphology, and syntax annotations.
- An appropriately diverse, work-isolated gold set with validated labels.
- Author/source generalization and professional-quality generation evaluation.

No claim of professional isiXhosa competence is made.
