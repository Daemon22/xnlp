"""
XNLP Core Large Language Model (LLM)
=====================================
A LLaMA-style transformer architecture optimized for isiXhosa language processing.
"""

from .architecture import (
    XNLPCoreLLM,
    XNLPConfig,
    GroupedQueryAttention,
    XNLPDecoderLayer,
    RotaryEmbedding,
    RMSNorm,
    SwiGLUFFN,
    create_preset_model,
    XNLP_PRESETS,
)

from .tokenizer import XNLPTokenizer, SpecialTokens

__all__ = [
    'XNLPCoreLLM',
    'XNLPConfig',
    'XNLPTokenizer',
    'SpecialTokens',
    'create_preset_model',
    'XNLP_PRESETS',
    'GroupedQueryAttention',
    'XNLPDecoderLayer',
    'RotaryEmbedding',
    'RMSNorm',
    'SwiGLUFFN',
]
