"""
XNLP Trainer - Core Training Engine
===================================
A professional, independent training pipeline for the XNLP Core LLM.

Design highlights
-----------------
* **Single-file checkpoints** - every ``.pt`` file contains the model
  weights, model config, tokenizer state, optimizer state, scheduler
  state, and training metadata all in one self-contained blob.
  No scattered files.
* **Atomic saves** - checkpoints are written to a temporary file first
  and then atomically renamed, so a crash never corrupts an existing
  checkpoint.
* **Resume / restart** - pass ``--resume checkpoint.pt`` to pick up
  exactly where training stopped, with optimizer + scheduler state.
* **Early stopping** - training halts when validation loss plateaus.
* **Cosine LR with linear warm-up** - standard professional schedule.
* **Best + last** - the best model (by val loss) and the final model
  are always saved as single self-contained files.

Checkpoint format version: 1
"""

from __future__ import annotations

import json
import math
import os
import platform
import time
import torch
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from core_llm.architecture import XNLPCoreLLM, XNLPConfig
from core_llm.tokenizer import XNLPTokenizer

from .config import (
    TrainingConfig,
    CHECKPOINT_FORMAT_VERSION,
    validate_checkpoint,
)
from .data import (
    prepare_data,
    tokenizer_to_state_dict,
    tokenizer_from_state_dict,
)
from .evaluate import evaluate, compute_perplexity, generate_samples


class XNLPTrainer:
    """End-to-end trainer that produces a single self-contained model file."""

    # -- Construction ---------------------------------------------------------

    def __init__(self, config: TrainingConfig):
        config.validate()
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)

        # Paths to the single-file artefacts
        self.checkpoint_path = os.path.join(config.output_dir, config.checkpoint_name)
        self.best_path = os.path.join(config.output_dir, "best_model.pt")
        self.last_path = os.path.join(config.output_dir, "last_model.pt")
        self.history_path = os.path.join(config.output_dir, config.history_name)

        # Determinism
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # State containers
        self.model: Optional[XNLPCoreLLM] = None
        self.tokenizer: Optional[XNLPTokenizer] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self._steps_per_epoch = 1
        self._max_steps = 1
        self._training_start_time: str = ""
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [], "test_loss": [], "lr": [], "epoch": [],
        }
        self.early_stop_counter = 0

    # -- LR schedule ---------------------------------------------------------

    def _make_lr_lambda(self):
        """Return a closure for cosine decay with linear warm-up."""
        warmup = self.config.warmup_steps
        max_steps = self._max_steps

        def lr_lambda(step: int) -> float:
            if max_steps <= warmup:
                return 1.0
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(max_steps - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return lr_lambda

    def _get_lr(self) -> float:
        """Current learning rate from the optimizer."""
        return self.optimizer.param_groups[0]["lr"]

    # -- Model & data --------------------------------------------------------

    def _build_model(self, vocab_size: int) -> None:
        """Create the LLM from the preset config."""
        kwargs = self.config.model_config_kwargs()
        kwargs["vocab_size"] = vocab_size
        model_cfg = XNLPConfig(**kwargs)
        self.model = XNLPCoreLLM(model_cfg)
        self.model.to(self.config.device)

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n[model] Built '{self.config.preset}' preset")
        print(f"  Parameters: {total:,}  (trainable: {trainable:,})")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Hidden:     {model_cfg.hidden_size}  Layers: {model_cfg.num_hidden_layers}")
        print(f"  Heads:      {model_cfg.num_attention_heads}  KV: {model_cfg.num_key_value_heads}")

    def _build_scheduler(self) -> None:
        """Create the LambdaLR scheduler with cosine-warmup schedule."""
        self._max_steps = self.config.max_epochs * self._steps_per_epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._make_lr_lambda(),
        )

    def _prepare(self) -> None:
        """Load data, train/load tokenizer, build model, optimizer & scheduler."""
        cfg = self.config
        print("=" * 64)
        print("  XNLP Professional Trainer")
        print("=" * 64)
        print(f"  Device:          {cfg.device}")
        print(f"  Preset:          {cfg.preset}")
        print(f"  Epochs:          {cfg.max_epochs}")
        print(f"  Batch size:      {cfg.batch_size}")
        print(f"  LR:              {cfg.learning_rate}")
        print(f"  Warmup steps:    {cfg.warmup_steps}")
        print(f"  Max seq len:     {cfg.max_seq_len}")
        print(f"  Early stopping:  {cfg.early_stopping}  (patience={cfg.patience})")
        print(f"  Output dir:      {cfg.output_dir}")
        print("=" * 64)

        # -- Data + tokenizer ------------------------------------------------
        ckpt = None
        if cfg.resume_from and os.path.isfile(cfg.resume_from):
            print(f"\n[load] Resuming from checkpoint: {cfg.resume_from}")
            ckpt = torch.load(cfg.resume_from, map_location="cpu", weights_only=False)
            # Validate before using
            validate_checkpoint(ckpt)
            tok_state = ckpt["tokenizer_state"]
            self.tokenizer = tokenizer_from_state_dict(tok_state)
            print(f"[load] Restored tokenizer (vocab={self.tokenizer.vocab_size_actual})")
            # Rebuild data loaders with the restored tokenizer (no retraining)
            self.train_loader, self.val_loader, self.test_loader, _ = prepare_data(
                cfg, tokenizer=self.tokenizer, verbose=True, include_test=True,
            )
            # Restore training start time from checkpoint
            meta = ckpt.get("training_metadata", {})
            self._training_start_time = meta.get("training_start_time", "")
        else:
            self.train_loader, self.val_loader, self.test_loader, self.tokenizer = prepare_data(
                cfg, tokenizer=None, verbose=True, include_test=True,
            )
            self._training_start_time = datetime.now(timezone.utc).isoformat()

        vocab_size = self.tokenizer.vocab_size_actual

        # -- Model -----------------------------------------------------------
        self._build_model(vocab_size)
        self._steps_per_epoch = len(self.train_loader) if self.train_loader else 1

        # -- Optimizer -------------------------------------------------------
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )

        # -- Scheduler -------------------------------------------------------
        self._build_scheduler()

        # -- Resume ----------------------------------------------------------
        if ckpt is not None:
            self.model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.global_step = ckpt.get("training_metadata", {}).get("global_step", 0)
            self.start_epoch = ckpt.get("training_metadata", {}).get("epoch", 0)
            self.best_val_loss = ckpt.get("training_metadata", {}).get("best_val_loss", float("inf"))
            print(f"[load] Resumed at epoch {self.start_epoch}, "
                  f"step {self.global_step}, best_val_loss={self.best_val_loss:.4f}")

    # -- Atomic checkpoint saving -------------------------------------------

    def _save_checkpoint_atomic(
        self,
        final_path: str,
        epoch: int,
        val_loss: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save a single self-contained checkpoint atomically.

        Writes to a temporary file first, then atomically renames it
        to the final path.  If the save fails, the existing file
        (if any) remains untouched.
        """
        model_cfg = self.model.config
        param_count = sum(p.numel() for p in self.model.parameters())

        # RNG state for reproducibility
        rng_state = {
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()

        metadata: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "param_count": param_count,
            "vocab_size": self.tokenizer.vocab_size_actual,
            "seed": self.config.seed,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "device": self.config.device,
            "preset": self.config.preset,
            "max_seq_len": self.config.max_seq_len,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "warmup_steps": self.config.warmup_steps,
            "max_steps": self._max_steps,
            "steps_per_epoch": self._steps_per_epoch,
            "training_start_time": self._training_start_time,
            "training_end_time": datetime.now(timezone.utc).isoformat(),
            "rng_state": rng_state,
        }

        ckpt: Dict[str, Any] = {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": self.model.state_dict(),
            "config": {
                "vocab_size": self.tokenizer.vocab_size_actual,
                "hidden_size": model_cfg.hidden_size,
                "intermediate_size": model_cfg.intermediate_size,
                "num_hidden_layers": model_cfg.num_hidden_layers,
                "num_attention_heads": model_cfg.num_attention_heads,
                "num_key_value_heads": model_cfg.num_key_value_heads,
                "max_position_embeddings": model_cfg.max_position_embeddings,
                "pad_token_id": model_cfg.pad_token_id,
                "bos_token_id": model_cfg.bos_token_id,
                "eos_token_id": model_cfg.eos_token_id,
                "unk_token_id": model_cfg.unk_token_id,
                "rope_theta": model_cfg.rope_theta,
                "layer_norm_eps": model_cfg.layer_norm_eps,
                "dropout_prob": model_cfg.dropout_prob,
            },
            "training_config": self.config.as_dict(),
            "tokenizer_state": tokenizer_to_state_dict(self.tokenizer),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_metadata": metadata,
        }
        if val_loss is not None:
            ckpt["val_loss"] = val_loss
        if extra:
            ckpt["training_metadata"].update(extra)

        # Atomic write: write to temp file, then rename
        tmp_path = final_path + ".tmp"
        try:
            torch.save(ckpt, tmp_path)
            os.replace(tmp_path, final_path)
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        print(f"  [save] {os.path.basename(final_path)}  "
              f"(epoch={epoch}, val_loss={val_loss})")

    # -- Training loop --------------------------------------------------------

    def train(self) -> None:
        """Full training run with early stopping and best-model tracking."""
        self._prepare()

        cfg = self.config

        # If history exists from a previous run, load it (for continuity)
        if os.path.exists(self.history_path) and cfg.resume_from:
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except (json.JSONDecodeError, KeyError):
                pass

        print(f"\n[train] Starting training from epoch {self.start_epoch} "
              f"({len(self.train_loader)} batches/epoch, "
              f"{self._max_steps} total steps)")
        print(f"  Training started: {self._training_start_time}")

        current_epoch = self.start_epoch
        try:
            for epoch in range(self.start_epoch, cfg.max_epochs):
                current_epoch = epoch
                self._train_epoch(epoch)

                # -- Validation --
                val_loss = evaluate(self.model, self.val_loader, cfg.device)
                perplexity = compute_perplexity(val_loss)
                self.history["val_loss"].append(val_loss)
                self.history["epoch"].append(epoch + 1)

                print(f"\n[epoch {epoch + 1}/{cfg.max_epochs}]  "
                      f"val_loss={val_loss:.4f}  ppl={perplexity:.1f}")

                # -- Best model --
                if val_loss < self.best_val_loss - cfg.min_delta:
                    self.best_val_loss = val_loss
                    self._save_checkpoint_atomic(self.best_path, epoch + 1, val_loss)
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    print(f"  [early-stop] no improvement for "
                          f"{self.early_stop_counter}/{cfg.patience} epochs")

                # -- Periodic checkpoint --
                if (epoch + 1) % cfg.save_every_n_epochs == 0:
                    cp_path = os.path.join(
                        cfg.output_dir, f"checkpoint_epoch{epoch + 1}.pt",
                    )
                    self._save_checkpoint_atomic(cp_path, epoch + 1, val_loss)

                # -- Always save last (resumable) --
                self._save_checkpoint_atomic(self.last_path, epoch + 1, val_loss)

                # -- Persist history --
                self._save_history()

                # -- Early stopping --
                if cfg.early_stopping and self.early_stop_counter >= cfg.patience:
                    print(f"\n[early-stop] Stopping - no improvement for "
                          f"{cfg.patience} epochs.")
                    current_epoch = epoch
                    break
                current_epoch = epoch

        except KeyboardInterrupt:
            print(f"\n[interrupt] Training stopped at epoch {current_epoch + 1}.")
        finally:
            # Ensure last + best are saved even on interrupt or error
            if self.model is not None:
                final_epoch = current_epoch + 1 if current_epoch >= self.start_epoch else self.start_epoch
                self._save_checkpoint_atomic(self.last_path, final_epoch)
                self._save_history()
                if not os.path.exists(self.best_path):
                    self._save_checkpoint_atomic(self.best_path, final_epoch)

        # Evaluate the final held-out test split exactly once after model
        # selection. The test set must not influence early stopping or tuning.
        if self.test_loader is not None and os.path.exists(self.best_path):
            best_ckpt = torch.load(self.best_path, map_location=cfg.device, weights_only=False)
            self.model.load_state_dict(best_ckpt["model_state_dict"])
            test_loss = evaluate(self.model, self.test_loader, cfg.device)
            test_ppl = compute_perplexity(test_loss)
            self.history["test_loss"].append(test_loss)
            print(f"  Held-out test loss : {test_loss:.4f}")
            print(f"  Held-out test ppl  : {test_ppl:.1f}")

        # Restore best model state for final sample generation
        best_exists = os.path.exists(self.best_path)

        print("\n" + "=" * 64)
        print("  Training Complete")
        print(f"  Best val loss : {self.best_val_loss:.4f}")
        print(f"  Best model    : {self.best_path}")
        print(f"  Last model    : {self.last_path}")
        print(f"  History       : {self.history_path}")
        print("=" * 64)

        # -- Sample generation from best model --
        if cfg.generate_samples and best_exists:
            print("\n[generate] Sample outputs from best model:")
            best_ckpt = torch.load(self.best_path, map_location=cfg.device, weights_only=False)
            self.model.load_state_dict(best_ckpt["model_state_dict"])
            samples = generate_samples(
                self.model, self.tokenizer, cfg.sample_prompts, cfg.device,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                repetition_penalty=cfg.repetition_penalty,
            )
            for prompt, text in samples:
                print(f"  Prompt:  {prompt}")
                print(f"  Output:  {text}")
                print()

    def _train_epoch(self, epoch: int) -> None:
        """Train one epoch and print periodic logs."""
        cfg = self.config
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        lr = self._get_lr()

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(cfg.device)
            labels = batch["labels"].to(cfg.device)

            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            self.global_step += 1
            lr = self._get_lr()

            if self.global_step % 50 == 0:
                avg = epoch_loss / max(n_batches, 1)
                print(f"  Ep {epoch + 1}/{cfg.max_epochs}  "
                      f"step {self.global_step}  "
                      f"loss {loss.item():.4f}  avg {avg:.4f}  "
                      f"lr {lr:.2e}")

        avg_epoch_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        self.history["train_loss"].append(avg_epoch_loss)
        self.history["lr"].append(lr)

        print(f"  Epoch {epoch + 1} done - "
              f"loss {avg_epoch_loss:.4f}  time {elapsed:.1f}s")

    def _save_history(self) -> None:
        """Write training history JSON alongside checkpoints."""
        path = os.path.join(self.config.output_dir, self.config.history_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
