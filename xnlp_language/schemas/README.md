# XNLP Language Foundation Schemas

This directory contains JSON schemas for the XNLP IsiXhosa Language Foundation. Each schema defines the structure for machine-readable linguistic data.

## Schema Files

### Core Linguistic Schemas
- `grammar_rules_schema.json` - Grammatical rules with evidence links
- `noun_classes_schema.json` - Noun class system
- `agreement_system_schema.json` - Agreement/concord patterns
- `morphology_schema.json` - Morphological analysis
- `verb_system_schema.json` - Verb system representation
- `tense_aspect_mood_schema.json` - Tense, aspect, mood, and polarity
- `orthography_schema.json` - Orthographic conventions
- `phonology_schema.json` - Phonological system

### Lexical and Semantic Schemas
- `lexicon_schema.json` - Lexical database entries
- `lexical_analysis_task_schema.json` - Evidence-linked lexical review tasks with unassigned lemma/category fields
- `semantic_relations_schema.json` - Semantic relationships
- `semantic_frame_schema.json` - Evidence-linked predicate/argument meaning annotations and review decisions
- `function_words_schema.json` - Function words (pronouns, demonstratives, etc.)

### Structural and Discourse Schemas
- `syntax_schema.json` - Syntactic structures
- `word_formation_schema.json` - Word formation patterns
- `discourse_schema.json` - Discourse-level structures
- `discourse_annotation_schema.json` - Evidence-linked discourse components and relations
- `linguistic_claim_schema.json` - Traceable linguistic claims with explicit evidence states

### Meta-Schemas
- `linguistic_evidence_schema.json` - Evidence linking rules to corpus
- `contradiction_detection_schema.json` - Contradiction tracking
- `coverage_matrix_schema.json` - Coverage and confidence tracking

## Schema Principles

All schemas follow these principles:

1. **Evidence-First**: Every linguistic assertion must link to corpus evidence
2. **Separate Evidence and Confidence**: Evidence state (OBSERVED, CANDIDATE, SOURCE_SUPPORTED, CROSS_SOURCE_SUPPORTED, VALIDATED, ESTABLISHED, REJECTED, UNRESOLVED) is independent of confidence
3. **Historical Distinction**: Entries track historical vs modern forms
4. **Machine-Readable**: Structured JSON for computational processing
5. **Traceability**: All rules traceable to source corpus records

Semantic and discourse schemas define annotation formats, not grammatical claims.
Generated semantic and discourse queues carry source text and provenance but leave
the proposed analysis empty until qualified human review. Span offsets are zero-based,
end-exclusive Python string offsets into the cited, unchanged `source_text`.
Machine suggestions remain candidates; explicit reviewer identity, protocol, and
evidence are required for validation. A rejected analysis remains traceable rather
than being deleted. `ESTABLISHED` requires a prior explicit `VALIDATED` decision.

The lexical adjudication utility, `xnlp_language.annotation.adjudicate_lexical`,
applies imported JSONL decisions only to the requested front-of-queue batch.
Machine-only rows require `--machine-assisted`; its conservative disputed-batch
mode does not assign a lemma, category, or meaning.

## Usage

### Validate Data Against Schemas
```python
import jsonschema

# Load schema
with open('grammar_rules_schema.json', 'r') as f:
    schema = json.load(f)

# Validate data
jsonschema.validate(instance=data, schema=schema)
```

### Generate Data Files
Data files should be named according to their schema:
- `grammar_rules.jsonl` - One JSON object per line
- `noun_classes.json` - Single JSON array
- `morphology.jsonl` - One analysis per line

## Schema Versioning

Current version: 1.0
- Initial release for XNLP_ISIXHOSA_FOUNDATION_V1

## Adding New Schemas

When adding new schemas:
1. Follow the existing naming convention: `*_schema.json`
2. Include required fields: `*_id`, `confidence`, evidence fields
3. Update this README
4. Ensure schema is valid JSON Schema Draft 7