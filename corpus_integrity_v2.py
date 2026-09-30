#!/usr/bin/env python
"""Audit the exact V2 corpus and its provenance without modifying it."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

CORPUS = Path("data/processed/v2/training_corpus.txt")
PROVENANCE = Path("data/processed/v2/training_corpus_provenance.jsonl")
OUTPUT = Path("data/reports/v2_corpus_integrity.json")
ENGLISH = {
    "the", "and", "of", "to", "in", "is", "that", "it", "for", "you", "on", "with", "as", "are", "was", "be", "at", "by", "this", "have", "from", "or", "an", "they", "not", "but", "we", "what", "all", "can", "who", "do", "if", "her", "his", "how", "its", "may", "has", "your", "their", "will", "about", "would", "there", "said", "could", "each",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    records = [json.loads(line) for line in PROVENANCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    corpus_lines = [line.strip() for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    words = [word for line in corpus_lines for word in line.split()]
    generated = [record for record in records if record.get("generator")]
    english_lines = [line for line in corpus_lines if set(re.findall(r"[A-Za-z]+", line.lower())) & ENGLISH]
    report = {
        "corpus": str(CORPUS),
        "corpus_sha256": sha256(CORPUS),
        "provenance": str(PROVENANCE),
        "records_in_corpus": len(corpus_lines),
        "records_in_provenance": len(records),
        "words": len(words),
        "unique_words": len(set(words)),
        "duplicate_records": len(corpus_lines) - len(set(corpus_lines)),
        "duplicate_rate": round((len(corpus_lines) - len(set(corpus_lines))) / len(corpus_lines), 6),
        "generated_records": len(generated),
        "english_like_records": len(english_lines),
        "source_distribution": dict(Counter(record.get("source", "") for record in records)),
        "source_type_distribution": dict(Counter(record.get("source_type", "") for record in records)),
        "provenance_type_distribution": dict(Counter(record.get("provenance_type", "") for record in records)),
        "validation_status_distribution": dict(Counter(record.get("validation_status", "") for record in records)),
        "language_class_distribution": dict(Counter(record.get("language_class", "") for record in records)),
        "morphology_source_counts": {
            "mqhayi": sum(record.get("source") == "mqhayi" for record in records),
            "masikhanyise": sum(record.get("source") == "masikhanyise" for record in records),
            "authoritative": sum(record.get("provenance_type") == "authoritative" for record in records),
            "validated": sum(record.get("validation_status") == "validated" for record in records),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
