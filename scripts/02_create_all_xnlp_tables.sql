-- XNLP Migration: All Tables Creation
-- Batch execution for performance

-- ============================================================================
-- SOURCE AND CORPUS TABLES
-- ============================================================================

-- Source documents
CREATE TABLE IF NOT EXISTS xnlp.source_documents (
    source_document_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES xnlp.sources(source_id),
    document_name TEXT NOT NULL,
    document_type TEXT CHECK (document_type IN ('chapter', 'section', 'file', 'page', 'other')),
    position INTEGER,
    total_positions INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_source_docs_source ON xnlp.source_documents(source_id);

-- Corpus records
CREATE TABLE IF NOT EXISTS xnlp.corpus_records (
    record_id TEXT PRIMARY KEY,
    source_document_id TEXT REFERENCES xnlp.source_documents(source_document_id),
    exact_text TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    position INTEGER NOT NULL,
    language_classification TEXT NOT NULL CHECK (language_classification IN ('xh', 'en', 'mixed', 'foreign', 'uncertain')),
    authority_level TEXT NOT NULL CHECK (authority_level IN ('authoritative', 'review', 'generated')),
    historical_or_modern TEXT CHECK (historical_or_modern IN ('historical', 'modern', 'unknown')),
    generated_flag BOOLEAN NOT NULL DEFAULT FALSE,
    english_overlay_flag BOOLEAN NOT NULL DEFAULT FALSE,
    ingestion_batch TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(source_document_id, position)
);
CREATE INDEX IF NOT EXISTS idx_corpus_records_sha256 ON xnlp.corpus_records(sha256);
CREATE INDEX IF NOT EXISTS idx_corpus_records_language ON xnlp.corpus_records(language_classification);
CREATE INDEX IF NOT EXISTS idx_corpus_records_authority ON xnlp.corpus_records(authority_level);
CREATE INDEX IF NOT EXISTS idx_corpus_records_source_doc ON xnlp.corpus_records(source_document_id);

-- Corpus record versions
CREATE TABLE IF NOT EXISTS xnlp.corpus_record_versions (
    version_id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL REFERENCES xnlp.corpus_records(record_id),
    exact_text TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    version_number INTEGER NOT NULL,
    change_description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_corpus_record_versions_record ON xnlp.corpus_record_versions(record_id);

-- Provenance links
CREATE TABLE IF NOT EXISTS xnlp.provenance_links (
    provenance_id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL REFERENCES xnlp.corpus_records(record_id),
    source_id TEXT REFERENCES xnlp.sources(source_id),
    source_document_id TEXT REFERENCES xnlp.source_documents(source_document_id),
    original_position INTEGER,
    fingerprint TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_provenance_record ON xnlp.provenance_links(record_id);

-- Language classifications
CREATE TABLE IF NOT EXISTS xnlp.language_classifications (
    classification_id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL REFERENCES xnlp.corpus_records(record_id),
    classification TEXT NOT NULL,
    confidence FLOAT,
    classifier TEXT CHECK (classifier IN ('automatic', 'manual', 'heal')),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_language_classifications_record ON xnlp.language_classifications(record_id);

-- ============================================================================
-- OBSERVATIONS AND LEXICAL ITEMS
-- ============================================================================

CREATE TABLE IF NOT EXISTS xnlp.observations (
    observation_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    observation_type TEXT NOT NULL,
    description TEXT NOT NULL,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    source_records TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_observations_category ON xnlp.observations(category);
CREATE INDEX IF NOT EXISTS idx_observations_type ON xnlp.observations(observation_type);

CREATE TABLE IF NOT EXISTS xnlp.lexical_items (
    lexical_item_id TEXT PRIMARY KEY,
    surface_form TEXT NOT NULL,
    normalized_form TEXT,
    language TEXT NOT NULL CHECK (language IN ('xh', 'en', 'loan', 'other')),
    item_type TEXT NOT NULL CHECK (item_type IN ('word', 'stem', 'root', 'affix', 'clitic', 'other')),
    frequency INTEGER DEFAULT 0,
    part_of_speech TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_lexical_items_form ON xnlp.lexical_items(surface_form);
CREATE INDEX IF NOT EXISTS idx_lexical_items_language ON xnlp.lexical_items(language);

-- ============================================================================
-- MORPHEMES AND MORPHOLOGICAL PATTERNS
-- ============================================================================

CREATE TABLE IF NOT EXISTS xnlp.morphemes (
    morpheme_id TEXT PRIMARY KEY,
    surface_forms TEXT[] NOT NULL,
    function TEXT NOT NULL,
    category TEXT NOT NULL CHECK (category IN ('prefix', 'suffix', 'infix', 'root', 'clitic', 'other')),
    distribution TEXT,
    allomorphs JSONB DEFAULT '{}',
    examples TEXT[] DEFAULT '{}',
    evidence TEXT[] DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'OBSERVED' CHECK (status IN ('OBSERVED', 'SUPPORTED', 'UNDER_REVIEW', 'ESTABLISHED', 'CONTESTED', 'REJECTED')),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_morphemes_category ON xnlp.morphemes(category);
CREATE INDEX IF NOT EXISTS idx_morphemes_status ON xnlp.morphemes(status);

CREATE TABLE IF NOT EXISTS xnlp.morphological_patterns (
    pattern_id TEXT PRIMARY KEY,
    pattern_name TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT NOT NULL CHECK (category IN ('derivational', 'inflectional', 'compounding', 'other')),
    pattern_string TEXT,
    productivity TEXT NOT NULL CHECK (productivity IN ('PRODUCTIVE', 'PROBABLY_PRODUCTIVE', 'LEXICALIZED', 'UNCERTAIN')),
    examples TEXT[] DEFAULT '{}',
    evidence TEXT[] DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'OBSERVED' CHECK (status IN ('OBSERVED', 'SUPPORTED', 'UNDER_REVIEW', 'ESTABLISHED', 'CONTESTED', 'REJECTED')),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- NOUN CLASSES AND AGREEMENT
-- ============================================================================

CREATE TABLE IF NOT EXISTS xnlp.noun_classes (
    class_id TEXT PRIMARY KEY,
    class_number INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    description TEXT,
    singular_prefix TEXT,
    plural_prefix TEXT,
    prefix_variants JSONB DEFAULT '{}',
    semantic_tendency TEXT,
    corpus_evidence_count INTEGER DEFAULT 0,
    agreement_evidence_count INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'DOCUMENTED' CHECK (status IN ('DOCUMENTED', 'CORPUS-ATTESTED', 'ESTABLISHED')),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_noun_classes_number ON xnlp.noun_classes(class_number);
CREATE INDEX IF NOT EXISTS idx_noun_classes_status ON xnlp.noun_classes(status);

CREATE TABLE IF NOT EXISTS xnlp.noun_class_forms (
    form_id TEXT PRIMARY KEY,
    surface_form TEXT NOT NULL,
    augment_preprefix TEXT,
    class_prefix TEXT NOT NULL,
   
