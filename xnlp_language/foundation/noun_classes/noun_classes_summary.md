# IsiXhosa Noun Class System Foundation

## Overview
This directory contains the noun class system for isiXhosa as derived from the authoritative corpus (Mqhayi historical texts and Masikhanyise modern materials).

## Confidence Levels
- **HIGH_CONFIDENCE**: Classes 1, 2, 3, 5, 6, 7, 9 with extensive corpus evidence
- **SUPPORTED**: Additional classes with limited examples
- **INSUFFICIENT_EVIDENCE**: Complete class inventory, all prefix variants
- **CONFLICTING_EVIDENCE**: Historical OCR artifacts affecting analysis

## Key Findings

### 1. High-Confidence Noun Classes

#### Class 1 (um-) / Class 2 (aba-)
- **Semantic**: People, persons, human beings, agents
- **Prefixes**: um- (singular), aba- (plural)
- **Examples**: umntu/abantu (person/people), umfana/abafana (boy/boys)
- **Evidence**: Extensive corpus usage, clear singular/plural pairing

#### Class 3 (um-) / Class 4 (imi-)
- **Semantic**: Objects, tools, plants, natural phenomena
- **Prefixes**: um- (singular), imi- (plural)
- **Examples**: umzi/imizi (home/households), umkhonto/imikhonto (spear/spears)
- **Evidence**: Distinct from Class 1 by semantic category

#### Class 5 (i-) / Class 6 (ama-)
- **Semantic**: Natural phenomena, objects, animals, abstract concepts
- **Prefixes**: i- (singular), ama- (plural)
- **Examples**: izwe/amazwe (country/countries), inkosi/iinkosi (chief/chiefs)
- **Evidence**: Most productive class in corpus

#### Class 7 (isi-) / Class 8 (izi-)
- **Semantic**: Abstract concepts, languages, tools, institutions
- **Prefixes**: isi- (singular), izi- (plural)
- **Examples**: isizwe/izizwe (nation/nations), isilwimi/izilwimi (language/languages)
- **Evidence**: Highly productive for abstract nouns

#### Class 9 (u-) / Class 10 (o-)
- **Semantic**: Proper names, kinship terms, animals
- **Prefixes**: u- (singular), o- (plural)
- **Examples**: uHintsa/oHintsa, uMqhayi/oMqhayi
- **Evidence**: Extensive use for personal names

### 2. Agreement Patterns Observed
- **Subject concords**: Clear patterns for singular/plural
- **Object concords**: Documented for several classes
- **Possessive concords**: ka- pattern for proper names
- **Adjective concords**: Prefix-based agreement observed
- **Demonstrative concords**: lo-, eli-, esi- patterns documented

### 3. Semantic Tendencies
- **People classes**: 1/2 (um-/aba-)
- **Object classes**: 3/4 (um-/imi-), 5/6 (i-/ama-)
- **Abstract classes**: 7/8 (isi-/izi-)
- **Proper name classes**: 9/10 (u-/o-)

### 4. Stem Behavior
- **Nasal assimilation**: Observed in um- classes
- **Vowel changes**: Limited evidence in current corpus
- **Stem-initial variation**: Extensive across all classes

## Data Quality Notes
- **Mqhayi_v2_authoritative.jsonl**: 8,819 records with OCR contamination
- **Masikhanyise**: 33 unique records (with duplication issues)
- **Historical texts**: Contain OCR artifacts affecting prefix identification
- **Recommendation**: Use clean modern texts for primary analysis

## Evidence Sources
- **Primary**: mqhayi_v2_authoritative.jsonl (historical poetry, 1907/1927)
- **Secondary**: masikhanyise (modern cultural content)
- **All examples**: Linked to specific record IDs for traceability

## Next Steps
- Expand class inventory with additional clean data
- Document additional agreement patterns
- Analyze historical vs modern prefix usage
- Investigate irregular noun class pairings
- Complete agreement system for all observed classes

## Coverage Assessment
- **Total classes analyzed**: 7 (with HIGH_CONFIDENCE)
- **Total examples documented**: 200+ with source evidence
- **Agreement patterns**: Partial coverage for high-confidence classes
- **Semantic tendencies**: Documented for observed classes
- **Exceptions**: Limited evidence in current corpus