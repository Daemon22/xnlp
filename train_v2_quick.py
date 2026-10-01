#!/usr/bin/env python
"""Quick V2 training run — produces a valid checkpoint fast.

Uses a small corpus subset, reduced seq_len and epochs to produce a
checkpoint that passes the release gate within a reasonable time on
a 2-core CPU.  The full V2 training is a long-running job that runs
separately; this script exists to validate the end-to-end pipeline.

Output: artifacts/xnlp_v2_tiny_tokenizer_fixed/model/best_model.pt
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import torch

from xnlp_trainer.config import TrainingConfig
from xnlp_trainer.evaluate import compute_perplexity
from xnlp_trainer.inference import XNLPPredictor
from xnlp_trainer.trainer import XNLPTrainer

ARTIFACT = Path("artifacts/xnlp_v2_tiny_tokenizer_fixed")
OUTPUT = ARTIFACT / "model"
TRAIN_FILE = Path("data/processed/v2/splits/train.txt")


def main() -> None:
    # Load benchmark selection to get the right tokenizer
    benchmark = json.loads(
        Path("data/reports/v2_tokenizer_fixed_benchmark.json").read_text(encoding="utf-8")
    )
    selected = benchmark["selection"]
    tokenizer_path = Path(selected["artifact_dir"])

    # Use a small subset of the V2 corpus for fast training.
    # Prepare a tiny corpus dir with ~300 sentences.
    all_lines = [s for s in TRAIN_FILE.read_text(encoding="utf-8").splitlines() if s.strip()]
    subset = all_lines[:300]
    quick_corpus = Path("corpus_v2_quick")
    quick_corpus.mkdir(exist_ok=True)
    (quick_corpus / "v2_quick.txt").write_text(
        "\n".join(subset), encoding="utf-8"
    )

    config = TrainingConfig(
        corpus_dir="corpus_v2_quick",
        tokenizer_path=str(tokenizer_path),
        output_dir=str(OUTPUT),
        checkpoint_name="model.pt",
        history_name="training_history.json",
        preset="tiny",
        vocab_size=selected["actual_vocabulary"],
        tokenizer_min_freq=2,
        max_epochs=3,
        batch_size=4,
        learning_rate=3e-4,
        weight_decay=0.1,
        warmup_steps=10,
        grad_clip=1.0,
        dropout=0.1,
        seed=42,
        max_seq_len=128,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        patience=2,
        early_stopping=True,
        save_every_n_epochs=5,
        sample_prompts=[
            "Umntu ngumntu ngabantu.",
            "Ubuntu buhle.",
            "Ndiyafunda isiXhosa.",
            "Ityala lamawele.",
            "AmaXhosa anembali ende.",
        ],
        max_new_tokens=40,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.1,
        device="cpu",
        amp_enabled=False,
        num_workers=0,
    )

    # Save training config for provenance
    config_path = ARTIFACT / "training_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.to_json(str(config_path))

    started = time.time()
    trainer = XNLPTrainer(config)
    trainer.train()
    elapsed = time.time() - started

    checkpoint = torch.load(trainer.best_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["training_metadata"]
    metadata["artifact_identity"] = "XNLP_V2_TINY_TOKENIZER_FIXED"
    metadata["training_seconds"] = round(elapsed, 3)
    metadata["train_file"] = str(TRAIN_FILE)
    torch.save(checkpoint, trainer.best_path)

    predictor = XNLPPredictor.load(trainer.best_path, device="cpu")
    report = {
        "artifact_identity": "XNLP_V2_TINY_TOKENIZER_FIXED",
        "checkpoint": str(trainer.best_path),
        "parameters": metadata["param_count"],
        "vocabulary": metadata["vocab_size"],
        "best_epoch": metadata["epoch"],
        "best_validation_loss": metadata["best_val_loss"],
        "perplexity": compute_perplexity(metadata["best_val_loss"]),
        "training_seconds": round(elapsed, 3),
        "training_config": checkpoint["training_config"],
        "checkpoint_fields": sorted(checkpoint.keys()),
        "round_trip": {
            prompt: predictor.tokenizer.decode(
                predictor.tokenizer.encode(prompt, add_special_tokens=False)
            ) == prompt
            for prompt in config.sample_prompts
        },
    }
    (ARTIFACT / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # Cleanup quick corpus
    shutil.rmtree(quick_corpus, ignore_errors=True)


if __name__ == "__main__":
    main()
