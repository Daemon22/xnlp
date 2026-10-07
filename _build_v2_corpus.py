#!/usr/bin/env python
"""Phase V2-3/4/5: Review and clean non-Xhosa records, classify sources by tier."""
import sys, io, json, os, re, hashlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True)

# Load V2 corpus
v2_file = 'data/authoritative/v2_authoritative_all.jsonl'
def has_traceable_publication(record):
    """Require a direct publication citation before corpus records enter the clean set."""
    if record.get("provenance_type") != "authoritative" or record.get("validation_status") != "authoritative":
        return False
    if record.get("generator"):
        return False
    if not record.get("source_title") or not (record.get("edition") or record.get("source_section")):
        return False
    if record.get("original_text") != record.get("text"):
        return False
    source_url = record.get("source_url", "")
    if record.get("source") == "mqhayi":
        return source_url.startswith("https://emandulo.apc.uct.ac.za/metadata/Mqhayi/")
    if record.get("source") == "masikhanyise":
        approved_domains = (
            "shop.snapplify.com", "ebooks.unisaenterprise.ac.za",
            "dcebooks.co.za", "amazon.co.za", "pearson.com",
        )
        has_publication_url = any(domain in source_url for domain in approved_domains)
        has_isbn = bool(record.get("isbn") or record.get("book_isbn"))
        return has_publication_url and has_isbn
    return False

with open(v2_file, "r", encoding="utf-8") as f:
    all_records = [json.loads(line) for line in f if line.strip()]
records = [record for record in all_records if has_traceable_publication(record)]
excluded_untraceable = len(all_records) - len(records)
print(f"Excluded {excluded_untraceable} records without traceable publication metadata")

# Check FOREIGN_LANGUAGE and MIXED_LANGUAGE records
print("=== FOREIGN_LANGUAGE records ===")
for rec in records:
    if rec.get('language_class') == 'FOREIGN_LANGUAGE':
        print(f"  [{rec['source']}] {rec['text'][:150]}")

print(f"\n=== MIXED_LANGUAGE records ===")
for rec in records:
    if rec.get('language_class') == 'MIXED_LANGUAGE':
        print(f"  [{rec['source']}] {rec['text'][:150]}")

# Check UNCERTAIN records (sample)
uncertain = [r for r in records if r.get('language_class') == 'UNCERTAIN']
print(f"\n=== UNCERTAIN records (showing first 20 of {len(uncertain)}) ===")
for rec in uncertain[:20]:
    print(f"  [{rec['source']}] {rec['text'][:120]}")

# Remove FOREIGN_LANGUAGE records (they should not be in training corpus)
# Keep UNCERTAIN and MIXED_LANGUAGE but flag them for review
# Actually, let's be conservative: only keep TARGET_LANGUAGE records
# The UNCERTAIN ones might still be Xhosa (just hard to detect)
clean_records = [r for r in records if r.get('language_class') != 'FOREIGN_LANGUAGE']
print(f"\nAfter removing FOREIGN_LANGUAGE: {len(clean_records)} records")

# Phase V2-3: Source tier classification
print("\n=== Phase V2-3: Source Tier Classification ===")
tier_summary = {
    'Tier 1': [],
    'Tier 2': [],
    'Tier 3': [],
    'Tier 4': [],
    'Tier 5': [],
}

# All Mqhayi sources are Tier 1 (authoritative literary author)
mqhayi_works = set()
for r in clean_records:
    if r.get('source') == 'mqhayi':
        mqhayi_works.add(r.get('source_title', 'Unknown'))

tier_summary['Tier 1'].append({
    'source': 'S.E.K. Mqhayi',
    'works': sorted(mqhayi_works),
    'archive': 'Emandulo (UCT Five Hundred Year Archive)',
    'rationale': 'Authoritative literary works by recognized Xhosa literary master',
    'license': 'CC BY-NC-ND 4.0 (from Emandulo archive)',
})

# All Masikhanyise sources are Tier 1 (authoritative educational publisher)
masik_works = set()
for r in clean_records:
    if r.get('source') == 'masikhanyise':
        masik_works.add(r.get('source_title', 'Unknown'))

tier_summary['Tier 1'].append({
    'source': 'Masikhanyise Textbook Series',
    'works': sorted(masik_works),
    'publisher': 'Maskew Miller Learning / Pearson South Africa',
    'rationale': 'CAPs-aligned educational textbooks, authentic isiXhosa content',
    'license': 'Commercial (educational use)',
})

# Phase V2-4: Authentic text separation
print("\n=== Phase V2-4: Authentic Text Separation ===")
authentic = [r for r in clean_records if r.get('language_class') == 'TARGET_LANGUAGE']
non_authentic = [r for r in clean_records if r.get('language_class') != 'TARGET_LANGUAGE']
print(f"Authentic isiXhosa: {len(authentic)} records")
print(f"Non-authentic (UNCERTAIN/MIXED): {len(non_authentic)} records")
print(f"  UNCERTAIN: {len([r for r in non_authentic if r['language_class']=='UNCERTAIN'])}")
print(f"  MIXED_LANGUAGE: {len([r for r in non_authentic if r['language_class']=='MIXED_LANGUAGE'])}")

# Phase V2-5: Orthography classification
print("\n=== Phase V2-5: Orthography Classification ===")
ortho_dist = {}
for r in clean_records:
    ortho = r.get('orthography', 'unknown')
    ortho_dist[ortho] = ortho_dist.get(ortho, 0) + 1
for ortho, count in sorted(ortho_dist.items()):
    print(f"  {ortho}: {count} records")

# Create the clean training corpus
# Only TARGET_LANGUAGE records, deduplicated
final_records = authentic

# Deduplicate
seen = set()
unique_final = []
for r in final_records:
    norm = re.sub(r'\s+', ' ', r['text'].strip().lower())
    if norm not in seen and len(norm) > 10:
        seen.add(norm)
        unique_final.append(r)

print(f"\n=== Final V2 Training Corpus ===")
print(f"Total records: {len(records)}")
print(f"After removing FOREIGN_LANGUAGE: {len(clean_records)}")
print(f"AUTHENTIC (TARGET_LANGUAGE): {len(authentic)}")
print(f"After final dedup: {len(unique_final)}")

# Save training corpus text file
os.makedirs('data/processed/v2', exist_ok=True)
corpus_path = 'data/processed/v2/training_corpus.txt'
with open(corpus_path, 'w', encoding='utf-8') as f:
    for r in unique_final:
        f.write(r['text'] + '\n')

# Save provenance
prov_path = 'data/processed/v2/training_corpus_provenance.jsonl'
with open(prov_path, 'w', encoding='utf-8') as f:
    for r in unique_final:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

# Generate stats
total_chars = sum(len(r['text']) for r in unique_final)
total_words = sum(r.get('token_count', len(r['text'].split())) for r in unique_final)
unique_words = len(set(w.lower() for r in unique_final for w in r['text'].split()))

# Count by work
by_work = {}
for r in unique_final:
    work = r.get('source_title', 'Unknown')
    if work not in by_work:
        by_work[work] = {'records': 0, 'chars': 0, 'words': 0}
    by_work[work]['records'] += 1
    by_work[work]['chars'] += len(r['text'])
    by_work[work]['words'] += r.get('token_count', len(r['text'].split()))

manifest = {
    'name': 'XNLP V2 Training Corpus',
    'phase': 'V2-2/3/4/5',
    'total_records': len(unique_final),
    'excluded_untraceable_records': excluded_untraceable,
    'total_chars': total_chars,
    'total_words': total_words,
    'unique_words': unique_words,
    'by_work': by_work,
    'by_source': {
        'mqhayi': sum(1 for r in unique_final if r.get('source') == 'mqhayi'),
        'masikhanyise': sum(1 for r in unique_final if r.get('source') == 'masikhanyise'),
    },
    'by_orthography': ortho_dist,
    'source_tiers': tier_summary,
    'authentic_texts': len(authentic),
    'non_authentic': len(non_authentic),
    'comparison_with_v1': {
        'v1_records': 233,
        'v2_records': len(unique_final),
        'expansion_factor': len(unique_final) / 233,
        'v1_words': 1473,
        'v2_words': total_words,
        'word_expansion_factor': total_words / 1473,
    },
}

manifest_path = 'data/processed/v2/corpus_manifest.json'
with open(manifest_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f'\nTraining corpus saved: {corpus_path}')
print(f'Provenance saved: {prov_path}')
print(f'Manifest saved: {manifest_path}')

print(f'\n=== V2 Corpus Stats ===')
print(f'Records: {len(unique_final)} (V1: 233, {len(unique_final)/233:.1f}x expansion)')
print(f'Characters: {total_chars:,} (V1: 12,223)')
print(f'Words: {total_words:,} (V1: 1,473, {total_words/1473:.1f}x expansion)')
print(f'Unique words: {unique_words} (V1: 770)')
print(f'Dataset: {sum(1 for r in unique_final if r.get("source")=="mqhayi")} Mqhayi + {sum(1 for r in unique_final if r.get("source")=="masikhanyise")} Masikhanyise')

print(f'\n=== By Work ===')
for work, stats in sorted(by_work.items(), key=lambda x: x[1]['records'], reverse=True):
    print(f'  {work}: {stats["records"]} records, {stats["chars"]:,} chars')
