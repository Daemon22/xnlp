#!/usr/bin/env python
"""Resume a V2 training run from its latest checkpoint.

Usage::

    python resume_v2_training.py                           # auto-detect latest last_model.pt
    python resume_v2_training.py --checkpoint PATH.pt      # specific checkpoint
    python resume_v2_training.py --epochs 10               # override epochs

The script re-uses the TrainingConfig saved alongside the checkpoint,
so the resume is faithful to the original run configuration.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import shutil
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from xnlp_trainer.config import TrainingConfig

MODEL_DIR = Path("artifacts/xnlp_v2_tiny_tokenizer_fixed/model")


def find_latest_checkpoint() -> str:
    """Find the most recent resumable checkpoint."""
    candidates = [
        MODEL_DIR / "last_model.pt",
        MODEL_DIR / "best_model.pt",
    ]
    for c in candidates:
        if c.exists():
            size = c.stat().st_size
            # Check it's a real checkpoint (>1MB)
            if size > 1_000_000:
                return str(c)
    raise FileNotFoundError(
        f"No training checkpoint found in {MODEL_DIR}/. "
        "Run start_v2_training.py first."
    )


def main():
    parser = argparse.ArgumentParser(description="Resume V2 training from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (default: auto-detect last_model.pt)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Max additional epochs (default: 30, capped by original max)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Override early-stop patience")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    args = parser.parse_args()

    # Find checkpoint
    ckpt_path = args.checkpoint or find_latest_checkpoint()
    print(f"Resuming from: {ckpt_path}")

    # Load the training config that was saved alongside this checkpoint
    artifact_dir = Path("artifacts/xnlp_v2_tiny_tokenizer_fixed")
    config_json = artifact_dir / "training_config.json"
    if config_json.exists():
        cfg = TrainingConfig.from_json(str(config_json))
        print(f"Loaded config from: {config_json}")
    else:
        # Fallback: construct from checkpoint metadata
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        meta = ckpt["training_metadata"]
        tc = ckpt["training_config"]
        cfg = TrainingConfig(**tc)
        print(f"Loaded config from checkpoint metadata")

    # Override settings
    cfg.resume_from = ckpt_path
    if args.epochs:
        # Add to existing epoch count
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        prev_epoch = ckpt["training_metadata"].get("epoch", 0)
        cfg.max_epochs = prev_epoch + args.epochs
        print(f"Previous epoch: {prev_epoch}, new max_epochs: {cfg.max_epochs}")
    if args.patience:
        cfg.patience = args.patience
    if args.lr:
        cfg.learning_rate = args.lr

    # Create a new run directory for this resume
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"runs/v2_resume_{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_dir}")
    print(f"Output dir:    {cfg.output_dir}")
    print(f"  Epochs:      {cfg.max_epochs}")
    print(f"  Patience:    {cfg.patience}")
    print(f"  LR:          {cfg.learning_rate}")

    # Save resume config
    cfg.to_json(str(run_dir / "training_config.json"))

    # Launch training (reuse start_v2_training's config but without re-creating run dir)
    # We call XNLPTrainer directly
    from xnlp_trainer.trainer import XNLPTrainer
    from xnlp_trainer.data import corpus_fingerprint

    # Set up logging
    log_path = run_dir / "train.log"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file

    # Write PID
    (run_dir / "run.pid").write_text(str(os.getpid()), encoding="utf-8")

    print(f"=== V2 Training Resume ===")
    print(f"  PID: {os.getpid()}")
    print(f"  Resume: {ckpt_path}")
    print(f"  Max epochs: {cfg.max_epochs}")
    print(f"  Corpus fingerprint: {corpus_fingerprint(cfg.corpus_dir)}")
    print()

    trainer = XNLPTrainer(cfg)
    trainer.train()

    log_file.close()
    print(f"Resume training complete. Log: {log_path}")


if __name__ == "__main__":
    main()
