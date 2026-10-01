#!/usr/bin/env python
"""Durable V2 training launcher.

Creates an isolated run directory, redirects all stdout/stderr to a
persistent log file, configures the full-corpus tiny-model training run,
and hands off to ``XNLPTrainer``.  Supports ``--resume`` to continue from
a previously saved checkpoint.

Usage::

    python start_v2_training.py                     # fresh run
    python start_v2_training.py --resume PATH.pt    # resume
    python start_v2_training.py --epochs 20         # override epochs

Output layout::

    runs/v2_run_<timestamp>/
        run.json               # run metadata
        training_config.json   # exact TrainingConfig used
        train.log              # full stdout/stderr log
    artifacts/xnlp_v2_tiny_tokenizer_fixed/
        tokenizer/             # (loaded from benchmark)
        model/
            best_model.pt      # best checkpoint (model selection)
            last_model.pt      # latest checkpoint (resumable)
            training_history.json
            training_report.json
        training_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path

# Ensure UTF-8 output on all platforms
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from xnlp_trainer.config import TrainingConfig
from xnlp_trainer.trainer import XNLPTrainer
from xnlp_trainer.data import corpus_fingerprint

ARTIFACT_DIR = Path("artifacts/xnlp_v2_tiny_tokenizer_fixed")
TOKENIZER_DIR = ARTIFACT_DIR  # tokenizer files are saved directly here
MODEL_DIR = ARTIFACT_DIR / "model"
BENCHMARK_REPORT = Path("data/reports/v2_tokenizer_fixed_benchmark.json")


def make_run_dir() -> Path:
    """Create a timestamped run directory under runs/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"runs/v2_run_{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class Tee:
    """Redirect stdout/stderr to both console and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    parser = argparse.ArgumentParser(description="Durable V2 full-corpus training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Max epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--max-seq-len", type=int, default=64,
                        help="Max sequence length (default: 64, captures all V2 corpus sentences)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (default: 5)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Warmup steps (default: 100)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping")
    args = parser.parse_args()

    # ── Load benchmark selection for tokenizer ─────────────────────────
    benchmark = json.loads(
        BENCHMARK_REPORT.read_text(encoding="utf-8")
    )
    selected = benchmark["selection"]
    vocab_size = selected["actual_vocabulary"]

    # ── Create run directory ─────────────────────────────────────────────
    run_dir = make_run_dir()
    log_path = run_dir / "train.log"

    # ── Write PID for durability / monitoring ────────────────────────────
    pid_path = run_dir / "run.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    # ── Tee stdout/stderr to log file ────────────────────────────────────
    log_file = open(log_path, "w", encoding="utf-8")
    # When launched via Start-Process -WindowStyle Hidden, sys.stdout may
    # be a limited console handle.  We still tee to the original stdout
    # if it is writable; otherwise we fall back to file-only logging.
    try:
        _orig_stdout = sys.stdout
        _orig_stderr = sys.stderr
        _orig_stdout.write("")  # probe
    except Exception:
        _orig_stdout = None
        _orig_stderr = None

    if _orig_stdout is not None:
        tee = Tee(_orig_stdout, log_file)
        sys.stdout = tee
        sys.stderr = Tee(_orig_stderr, log_file)
    else:
        sys.stdout = log_file
        sys.stderr = log_file

    print(f"=== V2 Full-Corpus Training Run ===")
    print(f"Run directory: {run_dir}")
    print(f"Log file:      {log_path}")
    print(f"Started:       {datetime.now(timezone.utc).isoformat()}")
    print(f"Resuming from: {args.resume or 'scratch'}")
    print()

    # ── Save run metadata ────────────────────────────────────────────────
    run_meta = {
        "run_id": run_dir.name,
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "python_version": sys.version,
        "torch_version": torch_version_safe(),
        "corpus_dir": "corpus_v2",
        "corpus_fingerprint": corpus_fingerprint("corpus_v2"),
        "tokenizer_path": str(TOKENIZER_DIR),
        "tokenizer_vocab_size": vocab_size,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "early_stopping": not args.no_early_stop,
        "patience": args.patience,
        "learning_rate": args.lr,
        "warmup_steps": args.warmup_steps,
        "dropout": args.dropout,
        "seed": args.seed,
        "max_seq_len_note": "64 captures all V2 corpus sentences (avg 18 tokens, max 62)",
        "resume_from": args.resume,
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── Build training config ───────────────────────────────────────────
    config = TrainingConfig(
        corpus_dir="corpus_v2",
        tokenizer_path=str(TOKENIZER_DIR),
        output_dir=str(MODEL_DIR),
        checkpoint_name="model.pt",
        history_name="training_history.json",
        preset="tiny",
        vocab_size=vocab_size,
        tokenizer_min_freq=2,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.1,
        warmup_steps=args.warmup_steps,
        grad_clip=1.0,
        dropout=args.dropout,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        patience=args.patience,
        early_stopping=not args.no_early_stop,
        save_every_n_epochs=5,
        generate_samples=True,
        sample_prompts=[
            "Umntu ngumntu ngabantu.",
            "Ubuntu buhle.",
            "Ndiyafunda isiXhosa.",
            "Ityala lamawele.",
            "AmaXhosa anembali ende.",
        ],
        max_new_tokens=60,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.1,
        device="cpu",
        amp_enabled=False,
        num_workers=0,
        resume_from=args.resume,
    )

    # Save training config
    config.to_json(str(run_dir / "training_config.json"))
    config.to_json(str(ARTIFACT_DIR / "training_config.json"))

    # ── Train ────────────────────────────────────────────────────────────
    started = time.time()
    try:
        trainer = XNLPTrainer(config)
        trainer.train()
    finally:
        elapsed = time.time() - started
        log_file.flush()

    # ── Write run summary ────────────────────────────────────────────────
    summary = {
        "run_id": run_dir.name,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "best_model_path": str(MODEL_DIR / "best_model.pt"),
        "last_model_path": str(MODEL_DIR / "last_model.pt"),
        "history_path": str(MODEL_DIR / "training_history.json"),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log_file.close()


def torch_version_safe() -> str:
    try:
        import torch
        return torch.__version__
    except ImportError:
        return "unknown"


if __name__ == "__main__":
    main()
