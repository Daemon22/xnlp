"""XNLP isiXhosa deterministic grammar modules.

This package provides a deterministic word-analysis engine and
structured grammatical system builders. No neural models are used.
"""
from .word_analyzer import analyze_text, analyze_word, analyze_words, UNKNOWN_STATUS
from .output_quality import FoundationUnavailableError, validate_generated_text

__all__ = ["analyze_text", "analyze_word", "analyze_words", "UNKNOWN_STATUS", "FoundationUnavailableError", "validate_generated_text"]
