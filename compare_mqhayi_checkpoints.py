#!/usr/bin/env python3
"""Compare XNLP checkpoints on one shared, non-overlapping Mqhayi sample."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluate_mqhayi_checkpoint import (
    DEFAULT_TEST_PROVENANCE,
    evaluate_mqhayi_checkpoint,
    select_mqhayi_record_ids,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_BASELINE_CHECKPOINT = ROOT / "v1_baseline" / "best_model.pt"
DEFAULT_BASELINE_TRAINING_TEXT = ROOT / "v1_baseline" / "training_corpus.txt"
DEFAULT_PILOT_CHECKPOINT = ROOT / "artifacts" / "xnlp_prose_pilot" / "model" / "best_model.pt"
DEFAULT_PILOT_TRAINING_TEXT = (
    ROOT / "artifacts" / "xnlp_prose_pilot" / "development_corpus" / "development.txt"
)
DEFAULT_OUTPUT = ROOT / "data" / "reports" / "mqhayi_pilot_comparison.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_checkpoints(
    baseline_checkpoint: Path = DEFAULT_BASELINE_CHECKPOINT,
    baseline_training_text: Path = DEFAULT_BASELINE_TRAINING_TEXT,
    pilot_checkpoint: Path = DEFAULT_PILOT_CHECKPOINT,
    pilot_training_text: Path = DEFAULT_PILOT_TRAINING_TEXT,
    test_provenance: Path = DEFAULT_TEST_PROVENANCE,
    sample_size: int = 128,
) -> dict[str, Any]:
    """Evaluate both models on identical records outside both training sets."""
    record_ids = select_mqhayi_record_ids(
        test_provenance,
        (baseline_training_text, pilot_training_text),
        sample_size,
    )
    evaluations = {
        "baseline": evaluate_mqhayi_checkpoint(
            baseline_checkpoint,
            sample_size=len(record_ids),
            test_provenance=test_provenance,
            baseline_training_text=baseline_training_text,
            record_ids=record_ids,
        ),
        "pilot": evaluate_mqhayi_checkpoint(
            pilot_checkpoint,
            sample_size=len(record_ids),
            test_provenance=test_provenance,
            baseline_training_text=pilot_training_text,
            record_ids=record_ids,
        ),
    }
    if (
        evaluations["baseline"]["selection"]["sample_record_ids"]
        != evaluations["pilot"]["selection"]["sample_record_ids"]
    ):
        raise RuntimeError("Checkpoint evaluations did not use the same Mqhayi records")

    return {
        "experiment": "XNLP_MQHAYI_PAIRED_CHECKPOINT_COMPARISON_V1",
        "test_provenance_sha256": _sha256(test_provenance),
        "shared_sample_record_ids": record_ids,
        "training_text_sha256": {
            "baseline": _sha256(baseline_training_text),
            "pilot": _sha256(pilot_training_text),
        },
        "evaluations": evaluations,
        "interpretation": (
            "Both checkpoints were evaluated on the same deterministic Mqhayi records, "
            "excluding exact text overlaps with either training corpus. The models use "
            "different tokenizers, so token-level loss and perplexity are not strictly "
            "comparable. These metrics do not assess grammar, meaning, or prose quality."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument("--baseline-training-text", type=Path, default=DEFAULT_BASELINE_TRAINING_TEXT)
    parser.add_argument("--pilot-checkpoint", type=Path, default=DEFAULT_PILOT_CHECKPOINT)
    parser.add_argument("--pilot-training-text", type=Path, default=DEFAULT_PILOT_TRAINING_TEXT)
    parser.add_argument("--test-provenance", type=Path, default=DEFAULT_TEST_PROVENANCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = compare_checkpoints(
        baseline_checkpoint=args.baseline_checkpoint,
        baseline_training_text=args.baseline_training_text,
        pilot_checkpoint=args.pilot_checkpoint,
        pilot_training_text=args.pilot_training_text,
        test_provenance=args.test_provenance,
        sample_size=args.sample_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved paired checkpoint comparison: {args.output}")
    for name, evaluation in report["evaluations"].items():
        print(
            f"{name}: loss={evaluation['next_token_evaluation']['cross_entropy_loss']:.4f}, "
            f"perplexity={evaluation['next_token_evaluation']['perplexity']:.2f}, "
            f"round-trip={evaluation['tokenizer_round_trip']['passed']}/"
            f"{evaluation['tokenizer_round_trip']['total']}"
        )


if __name__ == "__main__":
    main()
