# XNLP Core LLM - isiXhosa Language Model

## Overview

This project contains the **XNLP Core Large Language Model**, a custom LLaMA-style transformer specifically designed for isiXhosa language processing, along with a professional, independent training pipeline (`xnlp_trainer`).

### Key Design Principle: Single-File Checkpoints

Every model artifact is a **single self-contained `.pt` file** — model weights, model configuration, tokenizer state, optimizer state, scheduler state, and training metadata are all bundled together. There are no scattered tokenizer `vocab.json`, `merges.json`, or `config.json` files to manage. Simply copy one `best_model.pt` and it works anywhere.

## Current Foundation Gate

**Model training, tokenizer benchmarking, scaling, and generation optimization are frozen.** XNLP now establishes an evidence-first isiXhosa linguistic foundation before progressing to the neural-model layer.

The foundation is built reproducibly from reconciled authoritative Mqhayi and Masikhanyise records. It preserves historical and modern source forms, links every generated observation to record-level provenance, and distinguishes `OBSERVED`, `SUPPORTED`, `PROVISIONAL`, and `INSUFFICIENT_EVIDENCE` rather than treating corpus absence as a linguistic rule.

### Language integrity and preservation

Language integrity is a highest-priority requirement. Original corpus text is immutable: historical spellings, borrowed forms, code-mixed passages, and uncertain passages are retained rather than silently corrected or deleted. Only records already classified `TARGET_LANGUAGE` feed automated linguistic observations; `MIXED_LANGUAGE`, `FOREIGN_LANGUAGE`, and `UNCERTAIN` records remain traceable in a human-review queue. This protects isiXhosa evidence without imposing a false or externally defined standard of purity.

```bash
python xnlp_language/build_foundation.py
python xnlp_language/validation/validate_foundation.py
```

Generated artifacts and the release-gate report are under `xnlp_language/generated/`. The present gate decision is intentionally **FOUNDATION INCOMPLETE — CONTINUE LINGUISTIC ANALYSIS**: corpus text provides reliable orthographic and lexical observations, but reviewed corpus annotation is still required before asserting structural analyses of agreement, morphology, verbs, syntax, and related domains.

## Package Architecture

```
xnlp/
├── README.md                  # This file
├── core_llm/                  # Model architecture (inference only)
│   ├── __init__.py            # Public API: XNLPCoreLLM, XNLPConfig, XNLPTokenizer
│   ├── architecture.py        # LLaMA-style transformer (RoPE, GQA, SwiGLU, RMSNorm)
│   └── tokenizer.py           # BPE tokenizer with special tokens
├── xnlp_trainer/              # Professional training pipeline (independent)
│   ├── __init__.py            # Public API + version
│   ├── __main__.py            # Enables `python -m xnlp_trainer`
│   ├── config.py              # TrainingConfig, model presets, checkpoint validation
│   ├── data.py                # Corpus loading, dataset, tokenizer serialization
│   ├── trainer.py             # XNLPTrainer: core training engine
│   ├── evaluate.py            # Loss, perplexity, sample generation, eval suite
│   ├── inference.py           # XNLPPredictor: single-file model loading
│   └── run.py                 # CLI entry point with argparse
├── corpus/                    # isiXhosa source text (authoritative)
│   ├── mqhayi_complete.txt    # Mqhayi poems and literary works
│   ├── mqhayi_basic.txt        # (synthetic - excluded from training)
│   └── masikhanyise_complete.txt  # Masikhanyise textbook content
├── corpus_extractor.py        # Phase 2: Deep extraction from corpus sources
├── data_validator.py          # Phases 5-10: Validation & correction engine
├── data_extractor.py          # Phase 2: Linguistic data extraction
├── data/                      # Data pipeline outputs
│   ├── authoritative/         # Records from authoritative sources only
│   ├── candidate/             # Records needing validation/review
│   ├── validated/             # Records that passed validation
│   ├── rejected/              # Records that failed validation
│   ├── corrections/            # Correction logs
│   ├── manifests/             # Corpus manifests
│   ├── processed/             # Final clean training corpus
│   └── reports/               # Dataset quality & evaluation reports
├── pyproject.toml
├── start_v2_training.py        # V2 durable training launcher
├── launch_training.py          # V2 detached process launcher
├── resume_v2_training.py       # V2 resume from checkpoint
├── monitor_training.py         # V2 training monitor
├── evaluate_checkpoint.py      # V2 comprehensive evaluation
├── train_v2_tokenizer_fixed.py # V2 full training script (30 epochs)
├── train_v2_quick.py           # V2 quick pipeline-validation (300 sentences)
├── release_gate_v2_tokenizer_fixed.py  # V2 release gate
├── benchmark_v2_tokenizer_fixed.py     # V2 tokenizer benchmark
├── _train_v2_all.py            # V2 pipeline orchestration
├── xnlp_language/              # Linguistic foundation (isiXhosa grammar/evidence)
├── artifacts/                  # V2 model & tokenizer artifacts (gitignored)
│   └── xnlp_v2_tiny_tokenizer_fixed/
│       ├── config.json
│       ├── vocab.json
│       ├── merges.json
│       └── model/
│           ├── best_model.pt
│           └── last_model.pt
└── runs/                       # Training run logs (gitignored)
```

## Model Specifications

| Property | Value |
|----------|-------|
| **Architecture** | LLaMA-style Transformer |
| **Attention** | Grouped Query Attention (GQA) |
| **Position Encoding** | RoPE (Rotary Position Embeddings) |
| **Activation** | SwiGLU |
| **Normalization** | RMSNorm |
| **Special Tokens** | `[PAD]`, `[BOS]`, `[EOS]`, `[UNK]`, `[SEP]`, `[CLS]` |

## Model Presets

| Preset | Parameters | Hidden Size | Layers | Heads (KV) | Max Seq Len | Use Case |
|--------|-----------|-------------|--------|------------|-------------|----------|
| tiny | ~4.2M | 256 | 6 | 4 (2) | 512 | Testing, quick experiments |
| small | ~22M | 512 | 8 | 8 (4) | 1024 | Development, prototyping |
| medium | ~78M | 768 | 12 | 12 (6) | 2048 | Production, quality generation |
| large | ~189M | 1024 | 16 | 16 (8) | 2048 | High-quality applications |
| xlarge | ~374M | 1280 | 20 | 20 (10) | 2048 | Research, advanced tasks |

## Quick Start

### Training from Scratch

```bash
# Train the tiny preset from scratch (fastest on CPU)
python -m xnlp_trainer.run --preset tiny --epochs 30

# Train a small model
python -m xnlp_trainer.run --preset small --epochs 30

# Full options
python -m xnlp_trainer.run --help
```

**Programmatic API:**

```python
from xnlp_trainer.config import TrainingConfig
from xnlp_trainer.trainer import XNLPTrainer

cfg = TrainingConfig(preset="small", max_epochs=30, batch_size=8)
trainer = XNLPTrainer(cfg)
trainer.train()
```

### Resuming Training

```bash
# Resume from the last checkpoint
python -m xnlp_trainer.run --resume outputs/last_model.pt --epochs 50

# Resume with a larger model (overrides are applied)
python -m xnlp_trainer.run --resume outputs/last_model.pt --preset medium --epochs 100
```

When resuming, the tokenizer, optimizer, and scheduler state are all restored from the checkpoint. No external files are needed.

### Text Generation (Production Inference)

```bash
# Generate from a single checkpoint file
python -m xnlp_trainer.run --generate outputs/best_model.pt --prompt "Molo, ndiyabulela"
```

```python
from xnlp_trainer.inference import XNLPPredictor

# Load the single-file model — nothing else needed
predictor = XNLPPredictor.load("outputs/best_model.pt")

# Generate text
text = predictor.generate("Molo, ndiyabulela", max_new_tokens=50, temperature=0.8)

# Or use call syntax
text = predictor("Umntu ngumntu ngabantu", temperature=0.7, top_k=40, top_p=0.9)
```

### Checkpoint Inspection

```python
import torch
ckpt = torch.load("outputs/best_model.pt", map_location="cpu", weights_only=False)
print(ckpt["training_metadata"])
```

## Training Features

- **Single-file checkpoints** — everything (weights, config, tokenizer, optimizer, scheduler, metadata) in one `.pt` file
- **Atomic saves** — checkpoints are written to a temp file then atomically renamed, so crashes never corrupt existing files
- **Cosine LR with linear warm-up** — standard professional schedule via `LambdaLR`
- **Gradient clipping** — prevents exploding gradients
- **Train/val splitting** — configurable ratios with deterministic seeding
- **Early stopping** — halts when validation loss plateaus (configurable patience)
- **Best + last + periodic checkpoints** — all are fully resumable single-file checkpoints
- **Reproducibility metadata** — Python/PyTorch versions, seed, parameter count, training timestamps, RNG state
- **Automatic sample generation** — model produces text samples at end of training
- **Training history** — JSON log of loss, LR, and perplexity per epoch
- **Windows UTF-8 support** — output encoding is safely configured

## Checkpoint Format (v1)

Every checkpoint contains these top-level keys:

```text
checkpoint_format_version   # int (currently 1)
model_state_dict            # model weights
config                      # model architecture config
training_config             # full TrainingConfig as dict
tokenizer_state             # vocab, merges, special tokens, merge_ranks
optimizer_state_dict        # AdamW optimizer state
scheduler_state_dict        # LambdaLR scheduler state
training_metadata           # epoch, step, best_val_loss, param_count, etc.
val_loss                    # validation loss (when available)
```

### Checkpoint Contents Detail

**config** — model architecture:
- `vocab_size`, `hidden_size`, `intermediate_size`, `num_hidden_layers`
- `num_attention_heads`, `num_key_value_heads`, `max_position_embeddings`
- `pad_token_id`, `bos_token_id`, `eos_token_id`, `unk_token_id`
- `rope_theta`, `layer_norm_eps`, `dropout_prob`

**tokenizer_state** — full BPE tokenizer:
- `tokenizer_type` (`"xnlp_bpe"`), `tokenizer_version`
- `vocab_size`, `min_frequency`, `num_merges`, `training_data_size`
- `special_tokens` (pad, bos, eos, unk, mask, sep, cls)
- `token2id`, `id2token`, `merges`, `merge_ranks`

**training_metadata** — reproducibility & provenance:
- `epoch`, `global_step`, `best_val_loss`
- `param_count`, `vocab_size`, `seed`
- `python_version`, `torch_version`, `device`
- `preset`, `max_seq_len`, `batch_size`
- `learning_rate`, `weight_decay`, `warmup_steps`, `max_steps`
- `training_start_time`, `training_end_time`
- `rng_state` (CPU + CUDA RNG states)

## Release Artifact

A trained model can be distributed as a **single file**:

```text
best_model.pt
```

No tokenizer files, no config files, no vocab files, no merges files. The checkpoint is fully self-contained. Copy it anywhere and load it:

```python
from xnlp_trainer.inference import XNLPPredictor
predictor = XNLPPredictor.load("best_model.pt")
```

## CLI Reference

```bash
python -m xnlp_trainer.run [OPTIONS]

Options:
  --preset {tiny,small,medium,large,xlarge}  Model size (default: tiny)
  --epochs N                               Max training epochs (default: 50)
  --batch-size N                           Batch size (default: 8)
  --lr FLOAT                               Peak learning rate (default: 3e-4)
  --warmup-steps N                         Linear warm-up (default: 200)
  --max-seq-len N                          Max sequence length (default: 256)
  --vocab-size N                           Target BPE vocab size (default: 8000)
  --seed N                                 Random seed (default: 42)
  --grad-clip FLOAT                        Max grad norm (default: 1.0)
  --dropout FLOAT                          Dropout (default: 0.1)
  --weight-decay FLOAT                     Weight decay (default: 0.1)
  --save-every N                           Periodic checkpoint every N epochs (default: 5)
  --patience N                             Early stopping patience (default: 5)
  --no-early-stop                          Disable early stopping
  --resume PATH                            Resume from checkpoint
  --output-dir DIR                         Output directory (default: outputs)
  --corpus DIR                             Corpus directory (default: corpus)

  --generate CHECKPOINT                    Generate text only (skip training)
  --prompt TEXT                            Prompt for generation (default: "Molo, ndiyabulela")
  --max-new-tokens N                       Max new tokens (default: 50)
  --temperature FLOAT                      Sampling temperature (default: 0.8)
  --top-k N                                Top-k sampling (default: 40)
  --top-p FLOAT                            Top-p nucleus sampling (default: 0.9)
  --repetition-penalty FLOAT               Repetition penalty (default: 1.1)
```

## Training Data

- **Authoritative Sources**: S.E.K. Mqhayi's works and Masikhanyise textbook series
- **Pipeline**: `corpus_extractor.py` -> `data_validator.py` -> clean training corpus
- **Training Corpus**: 233 records (150 Mqhayi + 83 Masikhanyise), 0 synthetic/generated
- **Data Pipeline**:
  - Phase 2: Deep extraction from authoritative sources
  - Phase 5-10: Validation (CANDIDATE -> VALIDATED/REJECTED/UNCERTAIN)
  - Phase 10: Source-grounded correction engine (no hallucination)
  - Phase 13: No recursive model-generated training data

## Trained Model

| Property | Value |
|----------|-------|
| **Preset** | tiny |
| **Parameters** | 4,321,280 |
| **Vocab Size** | 739 (BPE) |
| **Best Validation Loss** | 4.7701 |
| **Best Epoch** | 14 |
| **Checkpoint Size** | 49.6 MB (single file) |
| **Training Time** | ~26 minutes (CPU) |
| **File** | `outputs/best_model.pt` |

### Evaluation Results

| Test | Score |
|------|-------|
| Spelling Validation | 100% |
| Vocabulary Coverage | 25% |
| Generation Quality | 100% |
| Novelty / Contamination | 100% |
| Offline Inference | 100% |
| **Overall** | **85%** |

The model generates novel isiXhosa language content from learned parameters - verified by the offline capability test where `best_model.pt` is loaded in an isolated directory with no network, no source corpus, and no retrieval system.

### Training Time

| Property | Value |
|----------|-------|
| **Preset** | tiny |
| **Parameters** | 5,180,672 |
| **Vocab Size** | 4096 (BPE, V2 whitespace-preserving) |
| **Training Corpus** | 12,051 sentences (V2 full corpus, 16,462 lines) |
| **Max Seq Len** | 64 (captures 100% of V2 corpus sentences; avg 18 tokens) |
| **Hardware** | 2 CPU cores, PyTorch CPU, no GPU |
| **Est. Time / Epoch** | ~120 minutes |

## V2 Pipeline (Full-Corpus Training)

The V2 pipeline trains on the full authoritative isiXhosa corpus with the
whitespace-preserving BPE tokenizer, dynamic padding for CPU efficiency, and
attention masking for proper padding handling.

### Scripts

| Script | Purpose |
|--------|---------|
| `start_v2_training.py` | Durable training launcher (creates run dir, PID file, log) |
| `launch_training.py` | Detached process launcher (survives shell expiry) |
| `resume_v2_training.py` | Resume from `last_model.pt` with full state |
| `monitor_training.py` | Check training progress from log files |
| `evaluate_checkpoint.py` | Comprehensive evaluation report (generation, morphology, contamination) |
| `benchmark_v2_tokenizer_fixed.py` | Tokenizer benchmark across vocab sizes |
| `release_gate_v2_tokenizer_fixed.py` | Release-gate verification |
| `train_v2_tokenizer_fixed.py` | Full training script (30 epochs, seq_len=256) |
| `train_v2_quick.py` | Quick pipeline-validation run (300 sentences, 3 epochs) |

### V2 Training on CPU

For CPU-only training, the following configuration is recommended:

```bash
# Full-corpus V2 training (detached, survives shell expiry)
python launch_training.py --epochs 20 --batch-size 8 --max-seq-len 64 --patience 5

# Monitor progress
python monitor_training.py

# Resume if interrupted
python resume_v2_training.py --epochs 30
```

**Key optimizations:**
- `max_seq_len=64`: The V2 corpus averages 18 tokens/sentence (max 62), so 64
  captures 100% of sentences without truncation while halving compute vs 128.
- Dynamic padding: batches are padded to the longest sequence in the batch
  (avg ~29 tokens), not to `max_seq_len` (64), giving ~2x speedup.
- Attention masking: padding tokens are properly excluded from attention.
- `CREATE_NEW_PROCESS_GROUP`: training runs as a detached process.

### V2 Artifacts

```
artifacts/xnlp_v2_tiny_tokenizer_fixed/
  config.json              # tokenizer config (vocab=4096)
  vocab.json               # 4096-token vocabulary
  merges.json              # BPE merge table
  training_config.json     # training configuration
  model/
    best_model.pt          # best checkpoint (62.5 MB)
    last_model.pt          # resumable checkpoint
    training_history.json  # per-epoch loss/LR history
runs/
  v2_full_<timestamp>/     # run metadata, logs
  v2_run_<timestamp>/      # detached training logs
  v2_resume_<timestamp>/   # resume run logs
```

### V2 Evaluation

```bash
# Run comprehensive evaluation on a checkpoint
python evaluate_checkpoint.py artifacts/xnlp_v2_tiny_tokenizer_fixed/model/best_model.pt
```

Produces a machine-readable JSON report in `data/reports/evaluation_*.json`
covering: tokenizer round-trip, generation quality, morphology/orthography,
and contamination detection.

## Technical Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- tqdm (optional, for progress bars)

## Troubleshooting

### "UnicodeEncodeError on Windows"

This is fixed automatically by the trainer's UTF-8 stdout configuration. If you still see encoding errors, set:

```bash
set PYTHONIOENCODING=utf-8
```

### "Checkpoint file not found" (exit code 1)

Verify the path is correct. The checkpoint must be a single `.pt` file produced by the trainer.

### "Unsupported checkpoint format version" (exit code 1)

The checkpoint was created with an incompatible version of the trainer. Re-train or convert the checkpoint.

### "Vocabulary size mismatch" (exit code 1)

The model config and tokenizer in the checkpoint disagree. The checkpoint is likely corrupt.

### "Checkpoint is missing required field(s)" (exit code 1)

The checkpoint is incomplete or corrupt. Retrain from scratch.

### Slow tokenizer training on CPU

The BPE tokenizer is implemented in pure Python. For large corpora, consider:
- Using `--vocab-size` with a smaller target
- Using `--preset tiny` for quick iteration
- Pre-training the tokenizer separately and passing it to `prepare_data()`

## Professional training safeguards

The training pipeline now treats evaluation and reproducibility as first-class
concerns:

- the corpus is split deterministically into train/validation/test partitions;
- the tokenizer is trained only from the training partition;
- the held-out test partition is evaluated only after model selection;
- tokenizer BPE application is occurrence-safe and deterministic;
- checkpoints carry a SHA-256 fingerprint of the corpus inputs;
- resuming against a changed corpus is rejected rather than silently mixing
  training states;
- regression tests cover tokenizer merge correctness and split reproducibility.

## License

This model is part of the XNLP project for isiXhosa natural language processing.

## Version Information

- **Model Version**: 1.0.0-core
- **Trainer Version**: 3.0.0
- **Checkpoint Format**: v1
- **Last Updated**: August 2026
