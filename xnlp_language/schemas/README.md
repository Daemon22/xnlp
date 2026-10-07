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
- `semantic_relations_schema.json` - Semantic relationships
- `semantic_sense_schema.json` - Evidence-linked lexical senses
- `semantic_construction_schema.json` - Source-attested agreement constructions
- `function_words_schema.json` - Function words (pronouns, demonstratives, etc.)

### Structural and Discourse Schemas
- `syntax_schema.json` - Syntactic structures
- `word_formation_schema.json` - Word formation patterns
- `discourse_schema.json` - Discourse-level structures

### Meta-Schemas
- `linguistic_evidence_schema.json` - Evidence linking rules to corpus
- `contradiction_detection_schema.json` - Contradiction tracking
- `coverage_matrix_schema.json` - Coverage and confidence tracking

## Schema Principles

All schemas follow these principles:

1. **Evidence-First**: Every linguistic assertion must link to corpus evidence
2. **Confidence Levels**: All entries must include confidence levels (OBSERVED, HIGH_CONFIDENCE, SUPPORTED, PROVISIONAL, INSUFFICIENT_EVIDENCE, CONFLICTING_EVIDENCE)
3. **Historical Distinction**: Entries track historical vs modern forms
4. **Machine-Readable**: Structured JSON for computational processing
5. **Traceability**: All rules traceable to source corpus records

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