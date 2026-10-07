"""
XNLP Trainer - Inference / Production Loading
==============================================
Loads a **single self-contained checkpoint** file and exposes a clean
``XNLPPredictor`` API for text generation.

Because every checkpoint file already embeds the model weights, model
config, tokenizer state, and training metadata, production code never
needs to chase down multiple files - just point at one ``.pt`` file.

Example::

    from xnlp_trainer.inference import XNLPPredictor

    predictor = XNLPPredictor.load("outputs/best_model.pt")
    text = predictor.generate("Molo, ndiyabulela", max_new_tokens=50)
    print(text)
"""

from __future__ import annotations

import os
import torch
from typing import Optional, Dict, Any

from core_llm.architecture import XNLPCoreLLM, XNLPConfig
from core_llm.tokenizer import XNLPTokenizer

from xnlp_language.grammar import validate_generated_text

from .config import validate_checkpoint
from .data import tokenizer_from_state_dict


class GeneratedTextRejected(ValueError):
    """Raised when generated text contains forms unsupported by the foundation."""

    def __init__(self, report: Dict[str, Any]):
        self.validation_report = report
        failed = report.get("checks", {}).get("observed_word_forms", {}).get("unknown_forms", [])
        forms = ", ".join(item["surface"] for item in failed[:8])
        super().__init__(
            "Generated text was blocked by the XNLP evidence gate."
            + (f" Unsupported forms: {forms}" if forms else "")
            + " Grammar and meaning are not certified by this gate."
        )


class XNLPPredictor:
    """
    Production-ready wrapper that bundles model + tokenizer into one
    object, loaded from a single checkpoint file.

    Attributes
    ----------
    model : XNLPCoreLLM
    tokenizer : XNLPTokenizer
    device : torch.device
    metadata : dict  (raw checkpoint metadata)
    """

    def __init__(self, model: XNLPCoreLLM, tokenizer: XNLPTokenizer,
                 device: str = "cpu", metadata: Optional[Dict[str, Any]] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.metadata = metadata or {}

    # -- Loading -------------------------------------------------------------

    @classmethod
    def load(cls, checkpoint_path: str, device: Optional[str] = None,
             **generate_defaults) -> "XNLPPredictor":
        """
        Load a model from a single self-contained ``.pt`` checkpoint.

        The checkpoint must contain model weights, model config,
        tokenizer state, and all metadata - no external files needed.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file (e.g. ``best_model.pt``).
        device : str, optional
            ``"cpu"``, ``"cuda"``, etc.  Auto-detected if not given.
        generate_defaults : dict
            Defaults stored on the predictor for ``generate()`` calls
            (e.g. ``temperature=0.7``).
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}. "
                f"Please verify the path and try again."
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load checkpoint
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint '{checkpoint_path}': {e}. "
                f"The file may be corrupt or incompatible."
            ) from e

        if not isinstance(ckpt, dict):
            raise ValueError(
                f"Checkpoint '{checkpoint_path}' is not a valid dictionary. "
                f"Expected a serialized model checkpoint."
            )

        # Strict validation of all required fields
        validate_checkpoint(ckpt)

        # -- Reconstruct model config from checkpoint --
        cfg = ckpt["config"]
        model_cfg = XNLPConfig(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            max_position_embeddings=cfg["max_position_embeddings"],
            pad_token_id=cfg.get("pad_token_id", 0),
            bos_token_id=cfg.get("bos_token_id", 1),
            eos_token_id=cfg.get("eos_token_id", 2),
            unk_token_id=cfg.get("unk_token_id", 3),
            rope_theta=cfg.get("rope_theta", 10000.0),
            layer_norm_eps=cfg.get("layer_norm_eps", 1e-6),
            dropout_prob=cfg.get("dropout_prob", 0.1),
            device=device,
        )
        model = XNLPCoreLLM(model_cfg)
        model.load_state_dict(ckpt["model_state_dict"])

        # -- Reconstruct tokenizer exclusively from checkpoint --
        tokenizer = tokenizer_from_state_dict(ckpt["tokenizer_state"])

        # -- Validate vocab size agreement --
        model_vocab = model_cfg.vocab_size
        tok_vocab = tokenizer.vocab_size_actual
        if model_vocab != tok_vocab:
            raise ValueError(
                f"Vocabulary size mismatch: model config expects {model_vocab} "
                f"but tokenizer has {tok_vocab} tokens. The checkpoint may be "
                f"corrupt or incomplete."
            )

        # -- Parameter count verification (non-fatal warning) --
        actual_params = sum(p.numel() for p in model.parameters())
        meta = ckpt.get("training_metadata", {})
        stored_params = meta.get("param_count")
        if stored_params is not None and stored_params != actual_params:
            import warnings
            warnings.warn(
                f"Parameter count mismatch: checkpoint says {stored_params:,} "
                f"but model has {actual_params:,}. Proceeding anyway.",
                stacklevel=2,
            )

        # -- Metadata --
        metadata = {
            "epoch": meta.get("epoch", 0),
            "global_step": meta.get("global_step", 0),
            "best_val_loss": meta.get("best_val_loss"),
            "param_count": actual_params,
            "training_start_time": meta.get("training_start_time"),
            "training_end_time": meta.get("training_end_time"),
            "python_version": meta.get("python_version"),
            "torch_version": meta.get("torch_version"),
            "device": meta.get("device"),
            "preset": meta.get("preset"),
        }

        predictor = cls(model, tokenizer, device=device, metadata=metadata)
        predictor._generate_defaults = generate_defaults
        print(f"[XNLPPredictor] Loaded {checkpoint_path}")
        print(f"  Format version : {ckpt.get('checkpoint_format_version', 'unknown')}")
        print(f"  Epoch:          {metadata['epoch']}")
        print(f"  Vocab:          {tokenizer.vocab_size_actual}")
        print(f"  Params:         {actual_params:,}")
        print(f"  Val loss:       {meta.get('val_loss', metadata.get('best_val_loss'))}")
        print(f"  Device:         {device}")
        return predictor

    # -- Generation API ------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        quality_gate: bool = True,
    ) -> str:
        """
        Generate text continuation for *prompt*.

        By default, reject a continuation containing word forms absent from
        the evidence-backed lexicon or letters outside the foundation
        orthography. This is a lexical check, not a grammar or meaning proof.
        """
        self.model.eval()
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            ids = [self.tokenizer.bos_token_id]
        inp = torch.tensor([ids], dtype=torch.long, device=self.device)

        out = self.model.generate(
            inp,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )
        decoded = self.tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        if quality_gate:
            continuation_ids = out[0][len(ids):].tolist()
            continuation = self.tokenizer.decode(
                continuation_ids, skip_special_tokens=True
            )
            report = validate_generated_text(continuation)
            if report["status"] != "LEXICALLY_SUPPORTED":
                raise GeneratedTextRejected(report)
        return decoded

    def __call__(self, prompt: str, **kwargs) -> str:
        """Shortcut: ``predictor(prompt, temperature=0.7)``."""
        defaults = getattr(self, "_generate_defaults", {})
        defaults.update(kwargs)
        return self.generate(prompt, **defaults)

    # -- Introspection -------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        """Return a summary dict with model and training info."""
        return {
            "vocab_size": self.tokenizer.vocab_size_actual,
            "num_merges": self.tokenizer.num_merges,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "metadata": self.metadata,
        }
