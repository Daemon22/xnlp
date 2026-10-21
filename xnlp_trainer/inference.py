"""Self-contained checkpoint inference and candidate generation."""
from __future__ import annotations

import os
from typing import Optional, Dict, Any

import torch

from core_llm.architecture import XNLPCoreLLM, XNLPConfig
from core_llm.tokenizer import XNLPTokenizer
from .config import validate_checkpoint
from .data import tokenizer_from_state_dict


class XNLPPredictor:
    """Production wrapper that loads model and tokenizer from one checkpoint."""

    def __init__(self, model: XNLPCoreLLM, tokenizer: XNLPTokenizer,
                 device: str = "cpu", metadata: Optional[Dict[str, Any]] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.metadata = metadata or {}

    @classmethod
    def load(cls, checkpoint_path: str, device: Optional[str] = None,
             **generate_defaults) -> "XNLPPredictor":
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load checkpoint '{checkpoint_path}': {exc}") from exc
        if not isinstance(ckpt, dict):
            raise ValueError("Checkpoint is not a serialized dictionary")
        validate_checkpoint(ckpt)
        cfg = ckpt["config"]
        model_cfg = XNLPConfig(
            vocab_size=cfg["vocab_size"], hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            max_position_embeddings=cfg["max_position_embeddings"],
            pad_token_id=cfg.get("pad_token_id", 0), bos_token_id=cfg.get("bos_token_id", 1),
            eos_token_id=cfg.get("eos_token_id", 2), unk_token_id=cfg.get("unk_token_id", 3),
            rope_theta=cfg.get("rope_theta", 10000.0), layer_norm_eps=cfg.get("layer_norm_eps", 1e-6),
            dropout_prob=cfg.get("dropout_prob", 0.1), device=device,
        )
        model = XNLPCoreLLM(model_cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        tokenizer = tokenizer_from_state_dict(ckpt["tokenizer_state"])
        if model_cfg.vocab_size != tokenizer.vocab_size_actual:
            raise ValueError("Vocabulary size mismatch between model and tokenizer")
        meta = ckpt.get("training_metadata", {})
        actual_params = sum(p.numel() for p in model.parameters())
        metadata = {
            "epoch": meta.get("epoch", 0), "global_step": meta.get("global_step", 0),
            "best_val_loss": meta.get("best_val_loss"), "param_count": actual_params,
            "training_start_time": meta.get("training_start_time"),
            "training_end_time": meta.get("training_end_time"),
            "python_version": meta.get("python_version"), "torch_version": meta.get("torch_version"),
            "device": meta.get("device"), "preset": meta.get("preset"),
        }
        predictor = cls(model, tokenizer, device=device, metadata=metadata)
        predictor._generate_defaults = generate_defaults
        return predictor

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.8,
                 top_k: int = 40, top_p: float = 0.9, repetition_penalty: float = 1.1,
                 do_sample: bool = True) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            ids = [self.tokenizer.bos_token_id]
        inp = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(inp, max_new_tokens=max_new_tokens, temperature=temperature,
                                  top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                                  do_sample=do_sample)
        return self.tokenizer.decode(out[0].tolist(), skip_special_tokens=True)

    def generate_candidates(self, prompt: str, *, count: int = 4, **kwargs: Any) -> list[str]:
        """Sample multiple continuations; quality is not defined by one draw."""
        if count < 1:
            raise ValueError("count must be >= 1")
        return [self.generate(prompt, **kwargs) for _ in range(count)]

    def understand(self, text: str) -> Dict[str, Any]:
        """Return validated structure rather than exposing malformed JSON text."""
        from xnlp_language import understand
        return understand(text)

    def analyze_word(self, text: str) -> Dict[str, Any]:
        from xnlp_language import analyze_word
        return analyze_word(text)

    def analyze_morphology(self, text: str) -> Dict[str, Any]:
        result = self.analyze_word(text)
        result["analysis_scope"] = "evidence_bound_morphology"
        return result

    def analyze_sentence(self, text: str) -> Dict[str, Any]:
        return self.understand(text)

    def explain_grammar(self, text: str) -> Dict[str, Any]:
        result = self.understand(text)
        result["explanation_status"] = "STRUCTURAL_EVIDENCE_ONLY"
        result["explanation_limit"] = "A corpus observation is not by itself a complete grammatical rule."
        return result

    def answer(self, context: str, question: str) -> Dict[str, Any]:
        """Return a grounded answer with support or an explicit abstention."""
        from xnlp_language import comprehend
        return comprehend(context, question)

    def summarize(self, text: str, max_sentences: int = 2) -> Dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        import re
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
        selected = sentences[:max(1, max_sentences)]
        return {"status": "EXTRACTIVE_SUMMARY", "summary": " ".join(selected), "support": selected}

    def check_grammar(self, text: str) -> Dict[str, Any]:
        result = self.understand(text)
        result["status"] = "REVIEW_REQUIRED"
        result["limitation"] = "No unsupported grammatical acceptability judgment is inferred."
        return result

    def generate_text(self, prompt: str, *, task: str = "CONTINUATION", **kwargs: Any) -> str:
        task_prompt = prompt if prompt.startswith("<TASK:") else f"<TASK:{task}>\n{prompt}\n<ANSWER>\n"
        return self.generate(task_prompt, **kwargs)

    def __call__(self, prompt: str, **kwargs) -> str:
        defaults = dict(getattr(self, "_generate_defaults", {}))
        defaults.update(kwargs)
        return self.generate(prompt, **defaults)

    def info(self) -> Dict[str, Any]:
        return {"vocab_size": self.tokenizer.vocab_size_actual,
                "num_merges": self.tokenizer.num_merges,
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "metadata": self.metadata}
