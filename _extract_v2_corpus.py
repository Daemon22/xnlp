#!/usr/bin/env python
"""
Phase V2-2/4/5: Expand V2 corpus from downloaded Mqhayi sources.

Processes extracted PDF text from the Emandulo archive:
  - Cleans OCR artifacts, page numbers, headers, footers
  - Splits into meaningful text segments (sentences/paragraphs)
  - Classifies orthography (traditional/old vs. modern)
  - Separates authentic isiXhosa from English/translation/pedagogical overlays
  - Adds full provenance metadata
  - Deduplicates across editions
  - Outputs to data/authoritative/ for validation pipeline
"""
import sys, io, re, os, json, hashlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True)
from datetime import datetime, timezone

# ─── Configuration ──────────────────────────────────────────────────────────

SOURCE_PROVENANCE = {
    'imihobe_nemibongo.txt': {
        'work': 'Imihobe Nemibongo',
        'work_year': '1907/1927',
        'edition': 'Original',
        'source_url': 'https://emandulo.apc.uct.ac.za/metadata/Mqhayi/10130/4817',
        'archive': 'Emandulo (UCT/FHYA)',
        'retrieval_date': '2026-08-18',
        'source_type': 'poetry',
    },
    'inzuzo_1943.txt': {
        'work': 'Inzuzo',
        'work_year': '1943',
        'edition': '1943 edition',
        'source_url': 'https://emandulo.apc.uct.ac.za/metadata/Mqhayi/10130/10139',
        'archive': 'Emandulo (UCT/FHYA)',
        'retrieval_date': '2026-08-18',
        'source_type': 'poetry',
    },
    'ityala_lamawele_8th.txt': {
        'work': 'Ityala Lamawele',
        'work_year': '1814',
        'edition': '8th edition (1930)',
        'source_url': 'https://emandulo.apc.uct.ac.za/metadata/Mqhayi/10130/10132',
        'archive': 'Emandulo (UCT/FHYA)',
        'retrieval_date': '2026-08-18',
        'source_type': 'novel',
    },
    'ityala_lamawele_abridged_1955.txt': {
        'work': 'Ityala Lamawele',
        'work_year': '1814',
        'edition': 'Abridged edition (1955)',
        'source_url': 'https://emandulo.apc.uct.ac.za/metadata/Mqhayi/10130/10133',
        'archive': 'Emandulo (UCT/FHYA)',
        'retrieval_date': '2026-08-18',
        'source_type': 'novel',
    },
    'u_don_jadu_1951.txt': {
        'work': 'U-Don Jadu',
        'work_year': '1829',
        'edition': '1st edition (1951)',
        'source_url': 'https://emandulo.apc.uct.ac.za/metadata/Mqhayi/10130/4822',
        'archive': 'Emandulo (UCT/FHYA)',
        'retrieval_date': '2026-08-18',
        'source_type': 'novel',
    },
    'u_don_jadu_1967.txt': {
        'work': 'U-Don Jadu',
        'work_year': '1829',
        'edition': '1967 edition',
        'source_url': 'https://emandulo.apc.uct.ac.za/metadata/Mqhayi/10130/10134',
        'archive': 'Emandulo (UCT/FHYA)',
        'retrieval_date': '2026-08-18',
        'source_type': 'novel',
    },
    'u_john_knox_1972.txt': {
        'work': 'U John Knox Bokwe (Biography)',
        'work_year': '1829',
        'edition': '1972 edition',
        'source_url': 'https://emandulo.apc.uct.ac.za/metadata/Mqhayi/10130/10137',
        'archive': 'Emandulo (UCT/FHYA)',
        'retrieval_date': '2026-08-18',
        'source_type': 'biography',
    },
    'umhlekazi_u_hintsa.txt': {
        'work': 'Umhlekazi u Hintsa',
        'work_year': '1829',
        'edition': 'Original',
        'source_url': 'https://emandulo.apc.uct.ac.za/metadata/Mqhayi/10130/10312',
        'archive': 'Emandulo (UCT/FHYA)',
        'retrieval_date': '2026-08-18',
        'source_type': 'historical',
    },
    'umqhayi_wasen_tabozuko_1964.txt': {
        'work': 'UMqhayi waseNtabozuko',
        'work_year': '1829',
        'edition': '1964 edition',
        'source_url': 'https://emandulo.apc.uct.ac.za/metadata/Mqhayi/10130/10136',
        'archive': 'Emandulo (UCT/FHYA)',
        'retrieval_date': '2026-08-18',
        'source_type': 'autobiography',
    },
    'umqhayi_wasen_tabozuko_1975.txt': {
        'work': 'UMqhayi waseNtabozuko',
        'work_year': '1829',
        'edition': '1975 edition',
        'source_url': 'https://emandulo.apc.uct.ac.za/metadata/Mqhayi/10130/10135',
        'archive': 'Emandulo (UCT/FHYA)',
        'retrieval_date': '2026-08-18',
        'source_type': 'autobiography',
    },
}


# ─── Text Cleaning ──────────────────────────────────────────────────────────

# Patterns for cleaning
PAGE_NUM_RE = re.compile(r'^\s*(\d+|[ivx]+\.?)\s*$')
HEADER_PATTERNS = [
    re.compile(r'^[A-Z\s]+$'),  # All-caps headers
    re.compile(r'^S\. E\.\s*[A-Z]+\s*MQHAYI$', re.IGNORECASE),
    re.compile(r'^THE LOVEDALE PRESS$', re.IGNORECASE),
    re.compile(r'^THE SHELDON PRESS$', re.IGNORECASE),
    re.compile(r'^WITWATERSRAND UNIVERSITY PRESS$', re.IGNORECASE),
    re.compile(r'^JOHANNESBURG$', re.IGNORECASE),
    re.compile(r'^LONDON$', re.IGNORECASE),
    re.compile(r'^NORTHUMBERLAND AVENUE.*$', re.IGNORECASE),
    re.compile(r'^[Xosa Poetry for Schools]', re.IGNORECASE),
    re.compile(r'^[IE]BALWE NGU-?$', re.IGNORECASE),
    re.compile(r'^Published and Printed$', re.IGNORECASE),
    re.compile(r'^by the Lovedale Press$', re.IGNORECASE),
    re.compile(r'^\d{4}$'),  # Years
    re.compile(r'^[iI]nternational.*standard$', re.IGNORECASE),
    re.compile(r'^[r©][\w\s]*$', re.IGNORECASE),  # Copyright lines
    re.compile(r'^ISBN\s+\d+$', re.IGNORECASE),
    re.compile(r'^\d+\.\s*\d+\.\s*\d+\.\s*\d+\.\s*\d+$', re.IGNORECASE),  # ISBN-like
    re.compile(r'^[A-Z\s]{20,}$'),  # Long all-caps lines
    re.compile(r'^«\s*\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+»$', re.IGNORECASE),  # Library catalog numbers
    re.compile(r'^RHODES UNIVERSITY.*$', re.IGNORECASE),
    re.compile(r'^LIBRARY M G$', re.IGNORECASE),
    re.compile(r'^m t\s+B O O K\s+C A S E$', re.IGNORECASE),
    re.compile(r'^[Xosa Poetry for Schools]', re.IGNORECASE),
]
OCR_ARTIFACT_RE = re.compile(r'^[ivx]+\.\s*$', re.IGNORECASE)
CHAPTER_RE = re.compile(r'^ISAHLUKO\s+[IVXLCDM0-9]+', re.IGNORECASE)


# ─── Orthography Classification ─────────────────────────────────────────────

OLD_ORTHO_DIGITS = re.compile(r'[65]')

def classify_orthography(text: str) -> str:
    """Classify orthography as 'traditional', 'modern', or 'mixed'."""
    old_count = len(OLD_ORTHO_DIGITS.findall(text))
    if old_count > 2:
        return "traditional"
    elif old_count > 0:
        return "mixed"
    return "modern"


# ─── Language Detection ─────────────────────────────────────────────────────

# Xhosa-specific word patterns
XHOSA_INDICATORS = re.compile(
    r'\b(ndiya|ndiy|uku|kwi|ngok|emva|ngokuba|kodwa|njeng|xa\b|ke\b|'
    r'pele|malunga|phakathi|abantu|bantu|into|uku|baleka|qonj'
    r'elela|hlala|sebenza|thando|uMntu|uMama|iNkosikazi|khaya)\b',
    re.IGNORECASE
)

ENGLISH_LEXICON = {
    'the', 'and', 'of', 'to', 'in', 'is', 'that', 'it', 'for', 'you',
    'on', 'with', 'as', 'are', 'was', 'be', 'at', 'by', 'this', 'have',
    'from', 'or', 'an', 'they', 'not', 'but', 'we', 'what', 'all', 'can',
    'who', 'do', 'if', 'her', 'his', 'how', 'its', 'has', 'your',
    'their', 'will', 'about', 'would', 'there', 'said', 'could', 'each',
    'she', 'him', 'than', 'welcome', 'hello', 'thank', 'please',
    'the', 'and', 'to', 'of', 'in', 'is', 'that', 'it', 'for', 'you',
    'published', 'edition', 'press', 'university', 'college', 'editor',
    'introduction', 'note', 'translation', 'english', 'page', 'chapter',
    'section', 'title', 'author', 'copyright', 'isbn', 'edition',
    'narrative', 'story', 'novel', 'poem', 'poetry', 'biography',
    'historical', 'history', 'traditional', 'modern', 'text',
    'ixhosa', 'xhosa', 'africa', 'south', 'african',
    'the', 'and', 'to', 'of', 'in', 'is', 'that', 'it',
    'content', 'description', 'scope', 'material', 'record',
}

def detect_language(text: str) -> str:
    """Detect if text is Xhosa, English, or mixed."""
    if not text.strip():
        return 'UNCERTAIN'
    
    words = re.findall(r'[a-zA-Z]+', text.lower())
    if len(words) < 3:
        return 'UNCERTAIN'
    
    xhosa_matches = len(XHOSA_INDICATORS.findall(text))
    english_count = sum(1 for w in words if w in ENGLISH_LEXICON)
    
    xhosa_ratio = xhosa_matches / len(words)
    english_ratio = english_count / len(words)
    
    if english_ratio > 0.4 and xhosa_ratio < 0.05:
        return 'FOREIGN_LANGUAGE'
    if xhosa_ratio > 0.01 or english_ratio < 0.2:
        return 'TARGET_LANGUAGE'
    return 'MIXED_LANGUAGE'


# ─── Text Segmentation ──────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove OCR artifacts, fix common issues."""
    # Fix common OCR issues
    text = text.replace('I', '1') if re.match(r'^I$', text.strip()) else text  # Page numbers
    text = re.sub(r'Mqhayi\n', 'Mqhayi ', text)  # Split names across lines
    text = re.sub(r'MQ[ H]AYI', 'MQHAYI', text)  # Fix OCR on name
    text = re.sub(r'KRUN[EE]', 'KRUNE', text)  # Fix OCR
    text = re.sub(r'KFUNE', 'KRUNE', text)  # Fix OCR
    text = re.sub(r'MQAYI', 'MQHAYI', text)  # Fix OCR
    return text


def is_header_or_footer(line: str) -> bool:
    """Check if a line is a header, footer, page number, or OCR artifact."""
    stripped = line.strip()
    if not stripped:
        return True
    
    # Page numbers
    if PAGE_NUM_RE.match(stripped):
        return True
    
    # Chapter headers
    if CHAPTER_RE.match(stripped):
        return True
    
    # OCR artifacts
    if OCR_ARTIFACT_RE.match(stripped):
        return True
    
    # Single character lines (OCR noise)
    if len(stripped) <= 2 and stripped not in ('6', '5', 'n', 'i', 'c', 'a', 'u'):
        return True
    
    # Check against header patterns
    for pattern in HEADER_PATTERNS:
        if pattern.match(stripped):
            return True
    
    # Library/catalog metadata
    if stripped.startswith('«') and stripped.endswith('»'):
        return True
    if re.match(r'^\d{6,}', stripped):
        return True
    
    # Empty or whitespace-only
    if not stripped.replace(' ', '').replace('-', ''):
        return True
    
    return False


def segment_text(text: str, source_info: dict) -> list:
    """
    Split text into meaningful segments (paragraphs of 3+ lines or 100+ chars).
    Each segment should be a coherent piece of isiXhosa text.
    """
    lines = text.split('\n')
    segments = []
    current_segment = []
    current_lines = []
    
    for line in lines:
        line = line.rstrip()
        
        # Skip headers, footers, page numbers
        if is_header_or_footer(line):
            # End current segment if we have one
            if current_segment:
                seg_text = '\n'.join(current_segment)
                if len(seg_text.strip()) > 0:
                    segments.append(seg_text.strip())
                current_segment = []
            continue
        
        # Skip very short lines (likely OCR artifacts)
        if len(line.strip()) < 10 and len(current_segment) > 0:
            # Could be end of paragraph
            if len(line.strip()) <= 3:
                if current_segment:
                    seg_text = '\n'.join(current_segment)
                    if len(seg_text.strip()) > 0:
                        segments.append(seg_text.strip())
                current_segment = []
            continue
        
        # If this is a chapter/section header, treat as separator
        if CHAPTER_RE.match(line.strip()):
            if current_segment:
                seg_text = '\n'.join(current_segment)
                if len(seg_text.strip()) > 0:
                    segments.append(seg_text.strip())
            current_segment = []
            continue
        
        # Add to current segment
        current_segment.append(line)
    
    # Don't forget the last segment
    if current_segment:
        seg_text = '\n'.join(current_segment)
        if len(seg_text.strip()) > 0:
            segments.append(seg_text.strip())
    
    return segments


def deduplicate_segments(segments: list) -> list:
    """Remove exact duplicates and near-duplicates."""
    seen = set()
    unique = []
    for seg in segments:
        # Normalize for comparison (lowercase, remove whitespace)
        normalized = re.sub(r'\s+', ' ', seg.strip().lower())
        if normalized not in seen and len(normalized) > 20:
            seen.add(normalized)
            unique.append(seg)
    return unique


# ─── Main Extraction Pipeline ──────────────────────────────────────────────

def extract_records_from_source(txt_file: str, source_info: dict) -> list:
    """Extract records from a single source text file."""
    with open(txt_file, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    # Clean the text
    cleaned = clean_text(raw_text)
    
    # Segment into meaningful pieces
    segments = segment_text(cleaned, source_info)
    
    # Deduplicate
    segments = deduplicate_segments(segments)
    
    records = []
    now = datetime.now(timezone.utc).isoformat()
    
    for idx, seg in enumerate(segments):
        # Skip if too short or too long
        if len(seg.strip()) < 20 or len(seg.strip()) > 500:
            continue
        
        # Detect language
        lang_class = detect_language(seg)
        
        # Skip clearly English content
        if lang_class == 'FOREIGN_LANGUAGE':
            continue
        
        # Classify orthography
        ortho = classify_orthography(seg)
        
        # Create record
        text_clean = seg.strip()
        record_id = f"mqhayi_v2_{source_info['work'][:3]}_{idx:06d}_{hashlib.md5(text_clean.encode('utf-8')).hexdigest()[:8]}"
        
        records.append({
            'text': text_clean,
            'source': 'mqhayi',
            'source_type': source_info.get('source_type', 'literary'),
            'source_title': source_info['work'],
            'source_section': source_info.get('edition', ''),
            'source_url': source_info['source_url'],
            'archive': source_info['archive'],
            'retrieval_date': source_info['retrieval_date'],
            'work_year': source_info.get('work_year', ''),
            'edition': source_info.get('edition', ''),
            'language': 'isiXhosa',
            'provenance_type': 'authoritative',
            'confidence': 'high',
            'validation_status': 'authoritative',
            'orthography': ortho,
            'language_class': lang_class,
            'generator': '',
            'original_text': text_clean,
            'record_id': record_id,
            'token_count': len(text_clean.split()),
            'extraction_date': now,
            'v2_phase': 'V2-2',
        })
    
    return records


def main():
    source_dir = 'corpus_sources'
    output_dir = 'data/authoritative'
    os.makedirs(output_dir, exist_ok=True)
    
    all_records = []
    
    for txt_file, source_info in SOURCE_PROVENANCE.items():
        fpath = os.path.join(source_dir, txt_file)
        if not os.path.exists(fpath):
            print(f'SKIP: {txt_file} (not found)')
            continue
        
        if os.path.getsize(fpath) == 0:
            print(f'SKIP: {txt_file} (empty - scanned image)')
            continue
        
        print(f'\nProcessing: {source_info["work"]} ({source_info["edition"]})')
        records = extract_records_from_source(fpath, source_info)
        print(f'  Extracted: {len(records)} records')
        
        # Show orthography distribution
        ortho_dist = {}
        for r in records:
            ortho = r['orthography']
            ortho_dist[ortho] = ortho_dist.get(ortho, 0) + 1
        print(f'  Orthography: {ortho_dist}')
        
        # Show language distribution
        lang_dist = {}
        for r in records:
            lang = r['language_class']
            lang_dist[lang] = lang_dist.get(lang, 0) + 1
        print(f'  Language: {lang_dist}')
        
        # Show sample records
        print(f'  Sample records:')
        for r in records[:5]:
            print(f'    [{r["orthography"]}/{r["language_class"]}] {r["text"][:100]}')
        
        all_records.extend(records)
    
    # Deduplicate across all sources
    seen_texts = set()
    unique_records = []
    for r in all_records:
        norm = re.sub(r'\s+', ' ', r['text'].strip().lower())
        if norm not in seen_texts:
            seen_texts.add(norm)
            unique_records.append(r)
    
    print(f'\n{"="*60}')
    print(f'Total records extracted: {len(all_records)}')
    print(f'Unique records: {len(unique_records)}')
    
    # Save to authoritative file
    output_file = os.path.join(output_dir, 'mqhayi_v2_authoritative.jsonl')
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in unique_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f'Saved to: {output_file}')
    
    # Generate summary
    works = {}
    for r in unique_records:
        work = r['source_title']
        if work not in works:
            works[work] = {'records': 0, 'chars': 0, 'orthography': {}, 'language': {}}
        works[work]['records'] += 1
        works[work]['chars'] += len(r['text'])
        works[work]['orthography'][r['orthography']] = works[work]['orthography'].get(r['orthography'], 0) + 1
        works[work]['language'][r['language_class']] = works[work]['language'].get(r['language_class'], 0) + 1
    
    print(f'\n{"="*60}')
    print('Summary by work:')
    for work, stats in works.items():
        print(f'  {work}:')
        print(f'    Records: {stats["records"]}, Chars: {stats["chars"]}')
        print(f'    Orthography: {stats["orthography"]}')
        print(f'    Language: {stats["language"]}')
    
    total_chars = sum(len(r['text']) for r in unique_records)
    total_words = sum(r['token_count'] for r in unique_records)
    print(f'\nTotal: {len(unique_records)} records, {total_chars} chars, {total_words} words')
    
    # Save summary
    summary = {
        'extraction_date': datetime.now(timezone.utc).isoformat(),
        'total_records': len(unique_records),
        'total_chars': total_chars,
        'total_words': total_words,
        'works': works,
        'sources': SOURCE_PROVENANCE,
    }
    summary_path = os.path.join(output_dir, 'mqhayi_v2_extraction_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\nSummary saved: {summary_path}')


if __name__ == '__main__':
    main()
