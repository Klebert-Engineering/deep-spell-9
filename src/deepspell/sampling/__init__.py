"""Synthetic training-phrase sampling (grammar + corruption)."""

from .corruption import corrupt
from .grammar import PhraseGrammar, render_phrase

__all__ = ["PhraseGrammar", "render_phrase", "corrupt"]
