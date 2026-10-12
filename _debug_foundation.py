"""Debug script to trace foundation review queue filtering."""
import collections
import re
import json
from pathlib import Path
from xnlp_language.build_foundation import (
    _read_hand_curated_foundation, read_records, DEFAULT_CORPUS
)

WORD = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)

records = read_records(DEFAULT_CORPUS)
print(f"Records: {len(records)}")

token_occurrences = collections.defaultdict(list)
for record in records:
    for match in WORD.finditer(record["text"]):
        form = match.group(0).lower()
        token_occurrences[form].append({
            "record_id": record["record_id"],
            "surface_form": form,
        })

lexical_entries = []
for index, (form, occurrences) in enumerate(sorted(token_occurrences.items()), 1):
    lexical_entries.append({
        "lexeme_id": f"LEX_OBS_{index:06d}",
        "lemma": form,
        "frequency": len(occurrences),
    })

hand_curated = _read_hand_curated_foundation()
curated_lemmas = {e.get("lemma", "").lower() for e in hand_curated.get("lexicon", [])}
print(f"Curated lemmas: {len(curated_lemmas)}")
print(f"Curated lemmas: {sorted(curated_lemmas)}")

unfiltered_candidates = [e for e in lexical_entries if len(e["lemma"]) >= 3]
print(f"Unfiltered candidates: {len(unfiltered_candidates)}")

review_candidates = [
    e for e in unfiltered_candidates
    if e["lemma"].lower() not in curated_lemmas
]
print(f"Filtered candidates: {len(review_candidates)}")

top250_unfiltered = sorted(unfiltered_candidates, key=lambda x: (-x["frequency"], x["lemma"]))[:250]
top250_filtered = sorted(review_candidates, key=lambda x: (-x["frequency"], x["lemma"]))[:250]

unfiltered_lemmas = set(e["lemma"].lower() for e in top250_unfiltered)
filtered_lemmas = set(e["lemma"].lower() for e in top250_filtered)

curated_in_unfiltered = curated_lemmas & unfiltered_lemmas
curated_in_filtered = curated_lemmas & filtered_lemmas

print(f"\nCurated in unfiltered top 250: {len(curated_in_unfiltered)}")
print(f"Curated in filtered top 250: {len(curated_in_filtered)}")

# Calculate reduction
lex_tasks_unfiltered = 250 * 3  # LEXICAL_LEMMA, MORPHOLOGY_SEGMENTATION, NOUN_CLASS_OR_AGREEMENT
lex_tasks_filtered = len(top250_filtered) * 3
print(f"\nLexical review tasks (unfiltered): {lex_tasks_unfiltered}")
print(f"Lexical review tasks (filtered): {lex_tasks_filtered}")
print(f"Reduction: {lex_tasks_unfiltered - lex_tasks_filtered}")

# Check what the actual review queue looks like
queue_path = Path("xnlp_language/generated/annotation/linguistic_review_queue.jsonl")
with open(queue_path, encoding="utf-8") as f:
    queue = [json.loads(l) for l in f if l.strip()]

lex_tasks_in_queue = [t for t in queue if t.get("task_type") in ("LEXICAL_LEMMA", "MORPHOLOGY_SEGMENTATION", "NOUN_CLASS_OR_AGREEMENT")]
print(f"\nLexical-type tasks in current queue: {len(lex_tasks_in_queue)}")

# Check which curated lemmas are still in the queue
surface_forms = set(t.get("surface_form", "") for t in lex_tasks_in_queue)
curated_in_queue = curated_lemmas & surface_forms
print(f"Curated lemmas in current queue: {len(curated_in_queue)}")
print(f"Curated lemmas in queue: {sorted(curated_in_queue)}")
