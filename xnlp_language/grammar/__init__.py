"""XNLP isiXhosa deterministic grammar modules.

This package provides a deterministic word-analysis engine and
structured grammatical system builders. No neural models are used.
"""
from .word_analyzer import analyze_word, analyze_words, UNKNOWN_STATUS

__all__ = ["analyze_word", "analyze_words", "UNKNOWN_STATUS"]
