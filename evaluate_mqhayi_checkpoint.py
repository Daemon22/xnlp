#!/usr/bin/env python3
"""Evaluate a checkpoint on Mqhayi test records absent from its train text."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch

from xnlp_trainer.inference import XNLPPredictor

ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_PROVENANCE = ROOT / "data" / "processed" / "v2" / "splits" / "test_provenance.jsonl"
DEFAULT_BASELINE_TRAINING_TEXT = ROOT / "v1_baseline" / "training_corpus.txt"
DEFAULT_CHECKPOINT = ROOT / "v1_baseline" / "best_model.pt"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_mqhayi_record_ids(
    test_provenance: Path,
    training_texts: Sequence[Path],
    sample_size: int,
) -> list[str]:
    """Select a deterministic sample absent from every supplied training text."""
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    seen_training_texts = set()
    for path in training_texts:
        if not path.is_file():
            raise FileNotFoundError(f"Training text not found: {path}")
        seen_training_texts.update(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    eligible = [
        record
        for record in _read_jsonl(test_provenance)
        if record.get("source") == "mqhayi"
        and record.get("language_class") == "TARGET_LANGUAGE"
        and record.get("text", "").strip()
        and record["text"].strip() not in seen_training_texts
    ]
    if not eligible:
        raise ValueError("No non-overlapping Mqhayi test records are available")
    count = min(sample_size, len(eligible))
    indices = [index * len(eligible) // count for index in range(count)]
    return [eligible[index]["record_id"] for index in indices]


def evaluate_mqhayi_checkpoint(
    checkpoint: Path,
    sample_size: int = 128,
    test_provenance: Path = DEFAULT_TEST_PROVENANCE,
    baseline_training_text: Path = DEFAULT_BASELINE_TRAINING_TEXT,
    record_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Report held-out source-text/tokenizer metrics without judging grammar."""
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    test_records = _read_jsonl(test_provenance)
    training_texts = {
        line.strip()
        for line in baseline_training_text.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    mqhayi_records = [
        record
        for record in test_records
        if record.get("source") == "mqhayi"
        and record.get("language_class") == "TARGET_LANGUAGE"
        and record.get("text", "").strip()
    ]
    overlapping = [
        record for record in mqhayi_records
        if record["text"].strip() in training_texts
    ]
    eligible = [
        record for record in mqhayi_records
        if record["text"].strip() not in training_texts
    ]
    if not eligible:
        raise ValueError("No non-overlapping Mqhayi test records are available")

    if record_ids is None:
        count = min(sample_size, len(eligible))
        indices = [index * len(eligible) // count for index in range(count)]
        sample = [eligible[index] for index in indices]
        selection_method = "evenly_spaced_non_overlapping_records"
    else:
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("record_ids must not contain duplicates")
        eligible_by_id = {record["record_id"]: record for record in eligible}
        missing = [record_id for record_id in record_ids if record_id not in eligible_by_id]
        if missing:
            raise ValueError(
                "Requested Mqhayi record IDs are unavailable or overlap training text: "
                + ", ".join(missing[:5])
            )
        sample = [eligible_by_id[record_id] for record_id in record_ids]
        selection_method = "explicit_record_ids"

    predictor = XNLPPredictor.load(str(checkpoint), device="cpu")
    tokenizer = predictor.tokenizer
    max_length = predictor.model.config.max_position_embeddings
    encoded: list[list[int]] = []
    round_trip_failures = []
    whitespace_only_failures = []
    for record in sample:
        text = record["text"]
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
        )
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        if decoded != text.strip():
            round_trip_failures.append(record["record_id"])
            if "".join(decoded.split()) == "".join(text.split()):
                whitespace_only_failures.append(record["record_id"])
        encoded.append(token_ids)

    model = predictor.model.eval()
    batch_size = 8
    total_nll = 0.0
    total_targets = 0
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            batch = encoded[start:start + batch_size]
            width = max(map(len, batch))
            input_ids = torch.full(
                (len(batch), width), tokenizer.pad_token_id, dtype=torch.long
            )
            labels = torch.full((len(batch), width), -100, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
            for row_index, token_ids in enumerate(batch):
                length = len(token_ids)
                values = torch.tensor(token_ids, dtype=torch.long)
                input_ids[row_index, :length] = values
                labels[row_index, :length] = values
                attention_mask[row_index, :length] = 1

            output = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            targets = int(attention_mask[:, 1:].sum().item())
            total_nll += float(output["loss"].item()) * targets
            total_targets += targets

    if total_targets == 0:
        raise ValueError("Selected Mqhayi records contain no next-token targets")
    loss = total_nll / total_targets
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_epoch": predictor.metadata.get("epoch"),
        "test_provenance": str(test_provenance),
        "test_provenance_sha256": hashlib.sha256(test_provenance.read_bytes()).hexdigest(),
        "selection": {
            "source": "mqhayi",
            "language_class": "TARGET_LANGUAGE",
            "test_records": len(mqhayi_records),
            "exact_training_text_overlaps_excluded": len(overlapping),
            "eligible_non_overlapping_records": len(eligible),
            "sample_size": len(sample),
            "selection_method": selection_method,
            "sample_record_ids": [record["record_id"] for record in sample],
        },
        "tokenizer_round_trip": {
            "passed": len(sample) - len(round_trip_failures),
            "total": len(sample),
            "failed_record_ids": round_trip_failures,
            "whitespace_only_failure_count": len(whitespace_only_failures),
            "other_surface_loss_count": len(round_trip_failures) - len(whitespace_only_failures),
            "whitespace_only_failed_record_ids": whitespace_only_failures,
        },
        "next_token_evaluation": {
            "target_tokens": total_targets,
            "cross_entropy_loss": loss,
            "perplexity": math.exp(loss) if loss < 709 else math.inf,
        },
        "interpretation": (
            "Measures exact-text overlap exclusion, tokenizer fidelity, and held-out "
            "next-token fit only. The checkpoint tokenizer may lose whitespace and other "
            "surface characters. It does not verify semantics, grammatical correctness, "
            "or professional generation quality."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--test-provenance", type=Path, default=DEFAULT_TEST_PROVENANCE)
    parser.add_argument(
        "--baseline-training-text",
        type=Path,
        default=DEFAULT_BASELINE_TRAINING_TEXT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "reports" / "mqhayi_checkpoint_verification.json",
        help="Path to save the machine-readable report.",
    )
    args = parser.parse_args()
    report = evaluate_mqhayi_checkpoint(
        args.checkpoint,
        sample_size=args.sample_size,
        test_provenance=args.test_provenance,
        baseline_training_text=args.baseline_training_text,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Saved verification report: {args.output}")
    print(rendered, end="")


if __name__ == "__main__":
    main()
