#!/usr/bin/env python3
"""Train and evaluate a reproducible isiXhosa prose-model pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from evaluate_mqhayi_checkpoint import evaluate_mqhayi_checkpoint
from xnlp_trainer.config import TrainingConfig
from xnlp_trainer.evaluate import compute_perplexity
from xnlp_trainer.inference import XNLPPredictor
from xnlp_trainer.trainer import XNLPTrainer

ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "data" / "processed" / "v2" / "splits"
ARTIFACT_DIR = ROOT / "artifacts" / "xnlp_prose_pilot"
DEV_FILES = (SPLIT_DIR / "train.txt", SPLIT_DIR / "val.txt")
OUTER_TEST = SPLIT_DIR / "test_provenance.jsonl"
SOURCE_PROVENANCE = ROOT / "data" / "processed" / "v2" / "training_corpus_provenance.jsonl"
WORK_HELDOUT_VALIDATION = "U John Knox Bokwe (Biography)"
WORK_HELDOUT_TEST = "Inzuzo"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_development_corpus(corpus_dir: Path) -> dict[str, Any]:
    """Stage only train/validation partitions; never read the outer test text."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    staged = corpus_dir / "development.txt"
    content = []
    for source_path in DEV_FILES:
        if not source_path.is_file():
            raise FileNotFoundError(f"Development split not found: {source_path}")
        text = source_path.read_text(encoding="utf-8")
        content.append(text.rstrip("\r\n"))
    staged.write_text("\n".join(content) + "\n", encoding="utf-8")
    return {
        "staged_corpus": str(staged),
        "staged_corpus_sha256": sha256(staged),
        "development_inputs": [
            {"path": str(path), "sha256": sha256(path)}
            for path in DEV_FILES
        ],
        "outer_test_provenance": str(OUTER_TEST),
        "outer_test_provenance_sha256": sha256(OUTER_TEST),
        "split_policy": (
            "Trainer creates deterministic train/validation/test subsets only from "
            "the original train and validation partitions. The original test split "
            "is held out for final Mqhayi evaluation."
        ),
    }


def prepare_work_isolated_corpus(
    corpus_dir: Path,
    source_provenance: Path = SOURCE_PROVENANCE,
    validation_work: str = WORK_HELDOUT_VALIDATION,
    test_work: str = WORK_HELDOUT_TEST,
) -> dict[str, Any]:
    """Stage provenance records with entire works reserved for validation/test."""
    records = [
        json.loads(line)
        for line in source_provenance.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eligible = [
        record for record in records
        if record.get("language_class") == "TARGET_LANGUAGE"
        and record.get("validation_status") == "authoritative"
        and record.get("text", "").strip()
    ]
    if validation_work == test_work:
        raise ValueError("Validation and test works must be different")
    available_works = {record.get("source_title") for record in eligible}
    missing_works = {validation_work, test_work} - available_works
    if missing_works:
        raise ValueError(
            "Requested holdout works not found in eligible provenance: "
            + ", ".join(sorted(missing_works))
        )

    split_records = {
        "train": [
            record for record in eligible
            if record.get("source_title") not in {validation_work, test_work}
        ],
        "val": [
            record for record in eligible
            if record.get("source_title") == validation_work
        ],
        "test": [
            record for record in eligible
            if record.get("source_title") == test_work
        ],
    }
    if any(not values for values in split_records.values()):
        raise ValueError("Work-isolated split produced an empty partition")

    title_sets = {
        name: {record["source_title"] for record in values}
        for name, values in split_records.items()
    }
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = title_sets[first] & title_sets[second]
        if overlap:
            raise RuntimeError(
                f"Work leakage between {first} and {second}: {sorted(overlap)}"
            )

    text_sets = {
        name: {record["text"].strip() for record in values}
        for name, values in split_records.items()
    }
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = text_sets[first] & text_sets[second]
        if overlap:
            raise ValueError(
                f"Exact text leakage between {first} and {second}: {len(overlap)} records"
            )

    corpus_dir.mkdir(parents=True, exist_ok=True)
    split_summary = {}
    for name, values in split_records.items():
        split_dir = corpus_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)
        text_path = split_dir / f"{name}.txt"
        text_path.write_text(
            "".join(record["text"].rstrip("\r\n") + "\n" for record in values),
            encoding="utf-8",
        )
        provenance_path = split_dir / f"{name}_provenance.jsonl"
        provenance_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in values),
            encoding="utf-8",
        )
        split_summary[name] = {
            "records": len(values),
            "works": sorted(title_sets[name]),
            "text_path": str(text_path),
            "text_sha256": sha256(text_path),
            "provenance_path": str(provenance_path),
            "provenance_sha256": sha256(provenance_path),
        }

    test_provenance = corpus_dir / "test_provenance.jsonl"
    test_provenance.write_text(
        (corpus_dir / "test" / "test_provenance.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return {
        "source_provenance": str(source_provenance),
        "source_provenance_sha256": sha256(source_provenance),
        "split_policy": "Whole-source-title holdout; no source work crosses splits.",
        "validation_work": validation_work,
        "test_work": test_work,
        "splits": split_summary,
        "test_provenance": str(test_provenance),
        "test_provenance_sha256": sha256(test_provenance),
        "eligible_records": len(eligible),
        "excluded_records": len(records) - len(eligible),
    }


def train_pilot(
    epochs: int = 5,
    preset: str = "tiny",
    sample_size: int = 128,
    artifact_dir: Path = ARTIFACT_DIR,
    work_isolated: bool = False,
) -> dict[str, Any]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")

    torch.set_num_threads(min(torch.get_num_threads(), 2))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = artifact_dir / "development_corpus"
    if work_isolated:
        data_provenance = prepare_work_isolated_corpus(corpus_dir)
        outer_test_provenance = Path(data_provenance["test_provenance"])
        pilot_training_text = Path(data_provenance["splits"]["train"]["text_path"])
    else:
        data_provenance = prepare_development_corpus(corpus_dir)
        outer_test_provenance = OUTER_TEST
        pilot_training_text = Path(data_provenance["staged_corpus"])
    model_dir = artifact_dir / "model"

    config = TrainingConfig(
        corpus_dir=str(corpus_dir),
        output_dir=str(model_dir),
        checkpoint_name="model.pt",
        preset=preset,
        vocab_size=2048,
        tokenizer_min_freq=2,
        max_epochs=epochs,
        batch_size=8,
        learning_rate=3e-4,
        weight_decay=0.1,
        warmup_steps=100,
        grad_clip=1.0,
        dropout=0.1,
        seed=42,
        max_seq_len=256,
        train_split=0.8,
        val_split=0.15,
        test_split=0.05,
        preserve_line_breaks=True,
        predefined_splits=work_isolated,
        patience=3,
        early_stopping=True,
        save_every_n_epochs=1,
        sample_prompts=[
            "Molo, ndiyavuya.",
            "Ubuntu bubonakala xa",
            "Bhala isincoko esimalunga",
        ],
        max_new_tokens=48,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.1,
        device="cpu",
        amp_enabled=False,
    )
    config_path = artifact_dir / "training_config.json"
    config.to_json(str(config_path))

    started = time.time()
    trainer = XNLPTrainer(config)
    trainer.train()
    elapsed = time.time() - started

    if trainer.best_path is None or not Path(trainer.best_path).is_file():
        raise RuntimeError("Training finished without producing a best checkpoint")

    checkpoint = torch.load(trainer.best_path, map_location="cpu", weights_only=False)
    predictor = XNLPPredictor.load(trainer.best_path, device="cpu")
    prompts = config.sample_prompts
    generated = {
        prompt: predictor.generate(
            prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
        )
        for prompt in prompts
    }
    mqhayi = evaluate_mqhayi_checkpoint(
        Path(trainer.best_path),
        sample_size=sample_size,
        test_provenance=outer_test_provenance,
        baseline_training_text=pilot_training_text,
    )
    report = {
        "experiment": "XNLP_ISIXHOSA_PROSE_PILOT_V1",
        "training_status": "EXPERIMENTAL",
        "training_data": data_provenance,
        "checkpoint": str(trainer.best_path),
        "checkpoint_sha256": sha256(Path(trainer.best_path)),
        "training_seconds": round(elapsed, 3),
        "preset": preset,
        "parameters": checkpoint["training_metadata"]["param_count"],
        "vocabulary": checkpoint["training_metadata"]["vocab_size"],
        "best_epoch": checkpoint["training_metadata"]["epoch"],
        "best_validation_loss": checkpoint["training_metadata"]["best_val_loss"],
        "validation_perplexity": compute_perplexity(
            checkpoint["training_metadata"]["best_val_loss"]
        ),
        "internal_test_loss": (
            trainer.history["test_loss"][-1] if trainer.history["test_loss"] else None
        ),
        "tokenizer_round_trip": {
            prompt: predictor.tokenizer.decode(
                predictor.tokenizer.encode(prompt, add_special_tokens=False)
            ) == prompt
            for prompt in prompts
        },
        "sample_generations": generated,
        "mqhayi_holdout_evaluation": mqhayi,
        "release_note": (
            "A small-corpus pilot only. Corpus next-token fit, tokenizer fidelity, and "
            "sample outputs do not establish grammatical correctness or professional "
            "quality; native-speaker evaluation and broader licensed data remain required."
        ),
    }
    report_path = artifact_dir / "training_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved pilot report: {report_path}")
    print(json.dumps({
        "checkpoint": report["checkpoint"],
        "training_seconds": report["training_seconds"],
        "preset": report["preset"],
        "parameters": report["parameters"],
        "best_epoch": report["best_epoch"],
        "best_validation_loss": report["best_validation_loss"],
        "validation_perplexity": report["validation_perplexity"],
        "tokenizer_round_trip": report["tokenizer_round_trip"],
        "mqhayi_holdout": {
            "selection": report["mqhayi_holdout_evaluation"]["selection"],
            "tokenizer_round_trip": report["mqhayi_holdout_evaluation"]["tokenizer_round_trip"],
            "next_token_evaluation": report["mqhayi_holdout_evaluation"]["next_token_evaluation"],
        },
        "sample_generations": generated,
    }, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--preset", choices=("tiny", "small"), default="tiny")
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument(
        "--work-isolated",
        action="store_true",
        help="Hold out complete source works for validation and Mqhayi testing.",
    )
    args = parser.parse_args()
    train_pilot(
        args.epochs, args.preset, args.sample_size, args.artifact_dir,
        work_isolated=args.work_isolated,
    )


if __name__ == "__main__":
    main()
