-- XNLP Migration Batch 01: Source and Corpus Tables
CREATE TABLE IF NOT EXISTS xnlp.sources (
    source_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    authority_level TEXT NOT NULL CHECK (authority_level IN ('authoritative', 'supporting', 'external', 'uncertain')),
    source_type TEXT NOT NULL CHECK (source_type IN ('book', 'corpus', 'hand_curated', 'generated', 'other')),
    language TEXT NOT NULL CHECK (language IN ('xh', 'en', 'mixed', 'other')),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sources_authority ON xnlp.sources(authority_level);
CREATE INDEX IF NOT EXISTS idx_sources_type ON xnlp.sources(source_type);
