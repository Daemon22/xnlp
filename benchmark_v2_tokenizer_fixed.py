#!/usr/bin/env python
"""Benchmark fresh whitespace-preserving tokenizers for the V2 corpus."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from core_llm.tokenizer import XNLPTokenizer

SAMPLES = [
    "Umntu ngumntu ngabantu.",
    "Ubuntu buhle.",
    "Ndiyafunda isiXhosa.",
    "Ityala lamawele.",
    "AmaXhosa anembali ende.",
    "ukuguquguquka nokwakha ubudlelwane boluntu.",
]


def load_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metrics(tokenizer: XNLPTokenizer, lines: list[str]) -> dict:
    token_count = 0
    word_count = 0
    character_count = 0
    fragments = 0
    unique_words = set()
    covered_words = 0
    for line in lines:
        ids = tokenizer.encode(line, add_special_tokens=False)
        token_count += len(ids)
        character_count += len(line)
        for word in line.split():
            word_count += 1
            unique_words.add(word)
            word_ids = tokenizer.encode(word, add_special_tokens=False)
            fragments += max(len(word_ids) - 1, 0)
            if tokenizer.unk_token_id not in word_ids:
                covered_words += 1
    round_trip = all(
        tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)) == text
        for text in SAMPLES
    )
    return {
        "records": len(lines),
        "words": word_count,
        "tokens": token_count,
        "tokens_per_word": round(token_count / word_count, 6),
        "characters_per_token": round(character_count / token_count, 6),
        "unique_words": len(unique_words),
        "vocabulary_coverage": round(covered_words / len(unique_words), 6),
        "fragmentation_rate": round(fragments / word_count, 6),
        "round_trip_accuracy": 1.0 if round_trip else 0.0,
        "whitespace_preserved": round_trip,
        "sample_decodes": {
            text: tokenizer.decode(tokenizer.encode(text, add_special_tokens=False))
            for text in SAMPLES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/processed/v2/splits/train.txt"))
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/v2/training_corpus.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/v2_tokenizer_fixed_benchmark.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/xnlp_v2_tiny_tokenizer_fixed"))
    args = parser.parse_args()

    train_lines = load_lines(args.train)
    corpus_lines = load_lines(args.corpus)
    results = []
    for target_vocab in (512, 1024, 1565, 2048, 4096):
        started = time.time()
        tokenizer = XNLPTokenizer(vocab_size=target_vocab, min_frequency=2)
        tokenizer.train(train_lines, verbose=True)
        result = {
            "target_vocabulary": target_vocab,
            "actual_vocabulary": tokenizer.vocab_size_actual,
            "merges": tokenizer.num_merges,
            "training_seconds": round(time.time() - started, 3),
            "train_metrics": metrics(tokenizer, train_lines),
            "corpus_metrics": metrics(tokenizer, corpus_lines),
        }
        results.append(result)
        print(json.dumps({"target": target_vocab, "actual": tokenizer.vocab_size_actual, "tokens_per_word": result["corpus_metrics"]["tokens_per_word"], "round_trip": result["corpus_metrics"]["whitespace_preserved"]}))

    eligible = [item for item in results if item["corpus_metrics"]["whitespace_preserved"]]
    selected = min(eligible, key=lambda item: (item["corpus_metrics"]["tokens_per_word"], item["actual_vocabulary"]))
    tokenizer = XNLPTokenizer(vocab_size=selected["target_vocabulary"], min_frequency=2)
    tokenizer.train(train_lines, verbose=False)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(args.artifact_dir))

    report = {
        "artifact_identity": "XNLP_V2_TINY_TOKENIZER_FIXED",
        "train_source": str(args.train),
        "corpus_source": str(args.corpus),
        "candidates": results,
        "selection": {
            "target_vocabulary": selected["target_vocabulary"],
            "actual_vocabulary": selected["actual_vocabulary"],
            "criterion": "lowest full-corpus tokens_per_word among exact round-trip candidates",
            "artifact_dir": str(args.artifact_dir),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["selection"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
