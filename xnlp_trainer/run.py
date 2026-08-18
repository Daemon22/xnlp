"""
XNLP Trainer - CLI Entry Point
==============================

Train or resume an XNLP Core LLM from the command line::

    # Full training from scratch (tiny preset, CPU)
    python -m xnlp_trainer.run

    # Larger model, more epochs
    python -m xnlp_trainer.run --preset small --epochs 30

    # Resume from a checkpoint
    python -m xnlp_trainer.run --resume outputs/last_model.pt --epochs 50

    # Custom output directory and generation settings
    python -m xnlp_trainer.run --output-dir my_run --temperature 0.7

    # Just generate text from a trained single-file checkpoint
    python -m xnlp_trainer.run --generate outputs/best_model.pt --prompt "Molo"
"""

from __future__ import annotations

import argparse
import sys
import os

# Ensure UTF-8 output on all platforms (avoids Windows cp1252 crashes
# when printing Unicode characters from Xhosa text or generation output).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Make project root importable when running as ``python -m xnlp_trainer.run``
# from within the xnlp directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from xnlp_trainer.config import TrainingConfig, MODEL_PRESETS
from xnlp_trainer.trainer import XNLPTrainer
from xnlp_trainer.inference import XNLPPredictor


# -- Argument parser ---------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="xnlp-trainer",
        description="Professional independent training pipeline for the XNLP "
                    "Core LLM (isiXhosa). Produces a single self-contained "
                    "model checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python -m xnlp_trainer.run                      # train tiny from scratch
  python -m xnlp_trainer.run --preset small       # train small model
  python -m xnlp_trainer.run --resume ckpt.pt     # resume training
  python -m xnlp_trainer.run --generate ckpt.pt   # text generation only
""",
    )

    # -- Modes --
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--generate", metavar="CHECKPOINT",
                      help="Load CHECKPOINT and generate text (skip training).")

    # -- Paths --
    p.add_argument("--corpus", default="corpus",
                   help="Directory containing .txt corpus files (default: corpus)")
    p.add_argument("--output-dir", default="outputs",
                   help="Directory for checkpoints (default: outputs)")

    # -- Model --
    p.add_argument("--preset", default="tiny", choices=list(MODEL_PRESETS.keys()),
                   help="Model size preset (default: tiny)")
    p.add_argument("--vocab-size", type=int, default=8000,
                   help="Target BPE vocabulary size (default: 8000)")
    p.add_argument("--min-freq", type=int, default=2,
                   help="Minimum token frequency for BPE merges (default: 2)")

    # -- Training hyperparameters --
    p.add_argument("--epochs", type=int, default=50,
                   help="Maximum number of training epochs (default: 50)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Batch size (default: 8)")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Peak learning rate (default: 3e-4)")
    p.add_argument("--weight-decay", type=float, default=0.1,
                   help="Weight decay (default: 0.1)")
    p.add_argument("--warmup-steps", type=int, default=200,
                   help="Linear warm-up steps (default: 200)")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max gradient norm (default: 1.0)")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout probability (default: 0.1)")

    # -- Data / sequence --
    p.add_argument("--max-seq-len", type=int, default=256,
                   help="Maximum sequence length in tokens (default: 256)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")

    # -- Checkpointing --
    p.add_argument("--resume", metavar="PATH",
                   help="Resume training from checkpoint PATH.")
    p.add_argument("--save-every", type=int, default=5,
                   help="Save periodic checkpoint every N epochs (default: 5)")
    p.add_argument("--patience", type=int, default=5,
                   help="Early-stopping patience in epochs (default: 5)")

    # -- Generation --
    p.add_argument("--no-early-stop", action="store_true",
                   help="Disable early stopping")
    p.add_argument("--prompt", default="Molo, ndiyabulela",
                   help="Prompt for generation mode (default: 'Molo, ndiyabulela')")
    p.add_argument("--max-new-tokens", type=int, default=50,
                   help="Max new tokens to generate (default: 50)")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature (default: 0.8)")
    p.add_argument("--top-k", type=int, default=40,
                   help="Top-k sampling (default: 40)")
    p.add_argument("--top-p", type=float, default=0.9,
                   help="Top-p nucleus sampling (default: 0.9)")
    p.add_argument("--repetition-penalty", type=float, default=1.1,
                   help="Repetition penalty (default: 1.1)")

    return p


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """Translate CLI args into a TrainingConfig."""
    return TrainingConfig(
        corpus_dir=args.corpus,
        output_dir=args.output_dir,
        preset=args.preset,
        vocab_size=args.vocab_size,
        tokenizer_min_freq=args.min_freq,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        save_every_n_epochs=args.save_every,
        patience=args.patience,
        early_stopping=not args.no_early_stop,
        resume_from=args.resume,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )


# -- Modes --------------------------------------------------------------------


def run_training(args: argparse.Namespace) -> None:
    """Full training from scratch or resume."""
    cfg = config_from_args(args)
    trainer = XNLPTrainer(cfg)
    trainer.train()


def run_generation(args: argparse.Namespace) -> None:
    """Load a single-file checkpoint and generate text."""
    if not os.path.isfile(args.generate):
        print(f"ERROR: checkpoint not found: {args.generate}", file=sys.stderr)
        sys.exit(1)

    try:
        predictor = XNLPPredictor.load(args.generate)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    kwargs: dict = {}
    for k in ("max_new_tokens", "temperature", "top_k", "top_p", "repetition_penalty"):
        v = getattr(args, k)
        if v is not None:
            kwargs[k] = v

    prompts = [args.prompt] if args.prompt else [
        "Molo, ndiyabulela", "Umntu ngumntu", "Umthetho wamaXhosa",
        "Izibongo zethu", "Ndiyavuya",
    ]
    print("\n" + "=" * 64)
    print("  Text Generation")
    print("=" * 64)
    for p in prompts:
        try:
            text = predictor.generate(p, **kwargs)
            print(f"\n  Prompt:  {p}")
            print(f"  Output:  {text}")
        except Exception as e:
            print(f"\n  Prompt:  {p}")
            print(f"  ERROR:   {e}", file=sys.stderr)
    print("\n" + "=" * 64)


# -- Main --------------------------------------------------------------------


def main(argv: list | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.generate:
            run_generation(args)
        else:
            run_training(args)
    except KeyboardInterrupt:
        print("\n[interrupt] Training interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"\nFATAL: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
