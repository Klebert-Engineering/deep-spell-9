"""SymSpell-based correction (default backend, successor of the DAWG baseline).

Pure-Python ``symspellpy`` with rapidfuzz Damerau-Levenshtein re-ranking.
The dictionary file is a plain ``term<space>count`` text file built from a
gazetteer; building from 9.5 M road names takes a few minutes once.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rapidfuzz.distance import DamerauLevenshtein
from symspellpy import SymSpell, Verbosity

from ..charset import CharVocab
from ..gazetteer import Gazetteer

logger = logging.getLogger(__name__)


class SymSpellCorrector:
    def __init__(self, dictionary_path: str | Path, max_edit_distance: int = 2):
        self.max_edit_distance = max_edit_distance
        self.symspell = SymSpell(max_dictionary_edit_distance=max_edit_distance)
        if not self.symspell.load_dictionary(str(dictionary_path), term_index=0, count_index=1):
            raise FileNotFoundError(f"cannot load symspell dictionary {dictionary_path}")

    def match(self, token: str, k: int = 3) -> list[tuple[str, float]]:
        query = token.replace(" ", "_")  # dictionary terms encode spaces as underscores
        suggestions = self.symspell.lookup(
            query, Verbosity.CLOSEST, max_edit_distance=self.max_edit_distance,
            include_unknown=False, transfer_casing=False,
        )
        ranked = sorted(
            (
                (s.term.replace("_", " "), float(DamerauLevenshtein.distance(query, s.term)))
                for s in suggestions
            ),
            key=lambda pair: pair[1],
        )
        return ranked[:k]


def build_dictionary(
    gaz: Gazetteer, out_path: str | Path, char_vocab: CharVocab | None = None
) -> Path:
    """Write a symspell frequency dictionary of normalized token names."""
    vocab = char_vocab or CharVocab()
    counts: dict[str, int] = {}
    for name, freq in gaz.iter_names():
        normalized = vocab.normalize(name).replace(" ", "_")  # symspell terms are single words
        if normalized:
            counts[normalized] = counts.get(normalized, 0) + max(int(freq), 1)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as dict_file:
        for term, count in sorted(counts.items()):
            dict_file.write(f"{term} {count}\n")
    logger.info("wrote %d terms to %s", len(counts), out)
    return out
