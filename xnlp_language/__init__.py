"""Public, evidence-bound isiXhosa processing interface."""

from .grammar import UNKNOWN_STATUS, analyze_text, analyze_word, analyze_words

__version__ = "0.1.0"

__all__ = [
    "UNKNOWN_STATUS",
    "__version__",
    "analyze_text",
    "analyze_word",
    "analyze_words",
]