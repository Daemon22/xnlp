# IsiXhosa Orthography Foundation

## Overview
This directory contains the orthographic foundation for isiXhosa as derived from the authoritative corpus (Mqhayi historical texts and Masikhanyise modern materials).

## Confidence Levels
- **HIGH_CONFIDENCE**: Basic alphabet, click letters, standard digraphs, punctuation, capitalization
- **SUPPORTED**: Additional digraphs, historical patterns
- **INSUFFICIENT_EVIDENCE**: Complete trigraph inventory, edge cases
- **CONFLICTING_EVIDENCE**: OCR artifacts in historical texts

## Key Findings

### 1. Basic Alphabet (HIGH_CONFIDENCE)
- **Vowels**: a, e, i, o, u (standard 5-vowel system)
- **Consonants**: b, d, f, g, h, j, k, l, m, n, p, s, t, v, w, y, z
- **Clicks**: c (dental), x (lateral), q (alveolar)

### 2. Digraphs (HIGH_CONFIDENCE)
- **Nasal**: ng, ny
- **Nasal-click**: nc, nq, nx
- **Fricative**: hl, ph, kh, sh
- **Prenasalized**: gc, nc, ngc
- **Affricate**: tsh, ty, dy

### 3. Trigraphs (HIGH_CONFIDENCE)
- **Click-nasal**: ngc, ngq, ngx (partial evidence)

### 4. Orthographic Conventions (HIGH_CONFIDENCE)
- **Capitalization**: Proper names, place names, sentence-initial
- **Punctuation**: Standard punctuation marks
- **Word boundaries**: Spaces, hyphens for compounds
- **Conjunctive writing**: Prefixes attached without spaces

### 5. Proper Name Prefixes (HIGH_CONFIDENCE)
- **u-**: Male proper names
- **ama-**: Plural/ethnic groups
- **ka-**: Patronymics
- **kwa-**: Locatives

## Historical vs Modern Forms

### Historical (1907/1927 Mqhayi texts)
- Contains OCR artifacts (6 for q, ^ symbols, & entities)
- Mixed standard and non-standard patterns
- Requires preprocessing for clean analysis

### Modern (Masikhanyise materials)
- Consistent standard orthography
- Clean punctuation and capitalization
- Standard digraph patterns

## Data Quality Notes
- Mqhayi_v2_authoritative.jsonl: 8,819 records with OCR contamination
- Masikhanyise: 33 unique records (with duplication issues)
- Recommendation: Use clean modern texts for primary analysis, filter historical OCR artifacts

## Evidence Sources
- Primary: mqhayi_v2_authoritative.jsonl (historical poetry, 1907/1927)
- Secondary: masikhanyise (modern cultural content)
- All examples linked to specific record IDs for traceability

## Next Steps
- Expand trigraph inventory with more clean data
- Analyze historical orthographic variants after OCR cleanup
- Document additional borrowing patterns
- Investigate edge cases in word division