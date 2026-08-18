"""Check generated records in training corpus."""
import sys, json
sys.stdout.reconfigure(encoding="utf-8")

with open("data/processed/training_corpus_provenance.jsonl", "r", encoding="utf-8") as f:
    provs = [json.loads(l) for l in f]

gen = [p for p in provs if p["source"] == "generated"]
print(f"=== Generated records in training corpus ===")
print(f"Count: {len(gen)}")
for p in gen[:20]:
    text = p["text"][:100]
    print(f"  {text}")
if len(gen) > 20:
    print(f"  ... and {len(gen) - 20} more")
