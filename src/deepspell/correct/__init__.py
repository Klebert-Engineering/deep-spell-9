"""Spell-correction backends behind one interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Corrector(Protocol):
    def match(self, token: str, k: int = 3) -> list[tuple[str, float]]:
        """Return up to *k* ``(suggestion, distance)`` pairs, best first."""
        ...


from .embedding import EmbeddingCorrector  # noqa: E402
from .symspell import SymSpellCorrector  # noqa: E402

__all__ = ["Corrector", "SymSpellCorrector", "EmbeddingCorrector"]
