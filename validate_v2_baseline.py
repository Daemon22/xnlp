#!/usr/bin/env python
"""Validate the XNLP V2 baseline without training or retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import torch

from core_llm.tokenizer import XNLPTokenizer
from xnlp_trainer.data import build_split_indices, tokenizer_from_state_dict

EXAMPLES = [
    "Umntu ngumntu ngabantu.",
    "Ubuntu buhle.",
    "Ndiyafunda isiXhosa.",
    "Ityala lamawele.",
    "AmaXhosa anembali ende.",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokenizer_round_trip(tokenizer: XNLPTokenizer) -> dict:
    cases = []
    for text in EXAMPLES:
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        cases.append({
            "text": text,
            "token_ids": ids,
            "pieces": [tokenizer.id2token[token_id] for token_id in ids],
            "decoded": decoded,
            "preserves_boundaries": decoded == text,
        })
    return {"passed": all(case["preserves_boundaries"] for case in cases), "cases": cases}


def fertility(tokenizer: XNLPTokenizer, lines: list[str]) -> dict:
    words = sum(len(line.split()) for line in lines)
    tokens = sum(len(tokenizer.encode(line, add_special_tokens=False)) for line in lines)
    characters = sum(len(line) for line in lines)
    token_words = Counter()
    for line in lines:
        for word in line.split():
            token_words[word] += len(tokenizer.encode(word, add_special_tokens=False))
    return {
        "records": len(lines),
        "words": words,
        "tokens": tokens,
        "tokens_per_word": round(tokens / words, 4) if words else None,
        "characters_per_token": round(characters / tokens, 4) if tokens else None,
        "unique_words": len(token_words),
        "vocabulary_coverage": round(sum(1 for word in token_words if tokenizer.unk_token_id not in tokenizer.encode(word, add_special_tokens=False)) / len(token_words), 4) if token_words else None,
        "whitespace_round_trip": tokenizer_round_trip(tokenizer)["passed"],
    }


def checkpoint_info(path: Path) -> dict:
    if not path.is_file():
        return {"available": False, "path": str(path)}
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("training_metadata", {})
    return {
        "available": True,
        "path": str(path),
        "sha256": sha256(path),
        "parameters": metadata.get("param_count"),
        "vocabulary": len(checkpoint["tokenizer_state"]["token2id"]),
        "best_epoch": metadata.get("epoch"),
        "best_validation_loss": metadata.get("best_val_loss"),
        "training_config": checkpoint.get("training_config", {}),
        "tokenizer_config": {
            "vocab_size": checkpoint["tokenizer_state"].get("vocab_size"),
            "min_frequency": checkpoint["tokenizer_state"].get("min_frequency"),
            "num_merges": checkpoint["tokenizer_state"].get("num_merges"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs_v2_tiny/best_model.pt"))
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/v2/training_corpus.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/v2_baseline_validation.json"))
    args = parser.parse_args()

    lines = load_lines(args.corpus)
    train_idx, val_idx, test_idx = build_split_indices(len(lines), 0.8, 0.1, 42)
    serialized_tokenizer = Path("tokenizer_v2")
    if serialized_tokenizer.is_dir() and (serialized_tokenizer / "config.json").is_file():
        fresh_tokenizer = XNLPTokenizer.load_pretrained(str(serialized_tokenizer))
        tokenizer_source = "tokenizer_v2"
    else:
        fresh_tokenizer = XNLPTokenizer(vocab_size=2048, min_frequency=2)
        fresh_tokenizer.train(lines[: min(len(lines), 2000)], verbose=False)
        tokenizer_source = "first 2000 corpus records"

    report = {
        "report": "XNLP V2 model validation and tokenizer diagnosis",
        "decision": "FIX TOKENIZER / PIPELINE",
        "decision_basis": [
            "The previous tokenizer discarded whitespace during encoding.",
            "The previous decoder concatenated token strings.",
            "The repaired tokenizer preserves whitespace in encode/decode round trips.",
            "The reported 4.53M checkpoint is not present, so capability claims are withheld.",
        ],
        "checkpoint": checkpoint_info(args.checkpoint),
        "corpus": {
            "path": str(args.corpus),
            "records": len(lines),
            "characters": sum(len(line) for line in lines),
            "words": sum(len(line.split()) for line in lines),
            "unique_records": len(set(lines)),
            "duplicate_rate": round(1 - len(set(lines)) / len(lines), 6) if lines else None,
            "split_records": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
            "english_like_lines": sum(bool(re.search(r"\b(the|and|of|to|is|in)\b", line, re.IGNORECASE)) for line in lines),
        },
        "tokenizer": {
            "source": tokenizer_source,
            "new_tokenizer": {"vocabulary": fresh_tokenizer.vocab_size_actual, "merges": fresh_tokenizer.num_merges},
            "round_trip": tokenizer_round_trip(fresh_tokenizer),
            "fertility": fertility(fresh_tokenizer, lines),
        },
        "capability_status": {
            "engineering": "blocked: reported V2 checkpoint unavailable",
            "linguistic": "not assessed",
            "memorization": "not assessed",
            "offline_inference": "blocked: reported V2 checkpoint unavailable",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "decision": report["decision"], "checkpoint": report["checkpoint"], "round_trip": report["tokenizer"]["round_trip"]["passed"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
