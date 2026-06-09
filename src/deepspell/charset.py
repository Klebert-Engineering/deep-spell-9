"""Character and class vocabularies.

The v1 system encoded characters as "2-hot" numpy vectors with special
characters (``^``, ``$``, ``_``) mixed into the charset. v2 uses integer ids
with dedicated special tokens (PAD/BOS/EOS/UNK) and embedding layers; the
charset itself contains only real characters.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

PAD, BOS, EOS, UNK = 0, 1, 2, 3
NUM_SPECIALS = 4

#: Matches the v1 lowercase charset minus its special characters.
DEFAULT_CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789-., /"

#: Default token classes, EOL last (the completer's "end of query" class).
DEFAULT_CLASSES = ("CITY", "STATE", "COUNTRY", "ROAD")

EOL_CLASS = "EOL"


def fold(text: str, lowercase: bool = True) -> str:
    """Normalize unicode text to the model alphabet.

    NFKD-decompose, strip combining marks (é -> e), optionally casefold.
    Characters that still fall outside the charset are kept verbatim; they
    map to UNK at encode time.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.casefold() if lowercase else stripped


@dataclass(frozen=True)
class CharVocab:
    charset: str = DEFAULT_CHARSET
    lowercase: bool = True
    _index: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        index = {ch: i + NUM_SPECIALS for i, ch in enumerate(self.charset)}
        if len(index) != len(self.charset):
            raise ValueError("charset contains duplicate characters")
        object.__setattr__(self, "_index", index)

    def __len__(self) -> int:
        return len(self.charset) + NUM_SPECIALS

    def normalize(self, text: str) -> str:
        return fold(text, self.lowercase)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode *normalized* text into ids; unknown characters become UNK."""
        ids = [self._index.get(ch, UNK) for ch in text]
        if add_bos:
            ids.insert(0, BOS)
        if add_eos:
            ids.append(EOS)
        return ids

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        chars = []
        for i in ids:
            if i in (PAD, BOS, EOS):
                continue
            chars.append("_" if i == UNK else self.charset[i - NUM_SPECIALS])
        return "".join(chars)


@dataclass(frozen=True)
class ClassVocab:
    """Token classes plus an implicit terminal EOL class (always last)."""

    names: tuple[str, ...] = DEFAULT_CLASSES

    def __post_init__(self) -> None:
        if EOL_CLASS in self.names:
            raise ValueError(f"{EOL_CLASS!r} is implicit and must not be listed")
        if len(set(self.names)) != len(self.names):
            raise ValueError("duplicate class names")

    def __len__(self) -> int:
        return len(self.names) + 1  # + EOL

    @property
    def eol_id(self) -> int:
        return len(self.names)

    def encode(self, name: str) -> int:
        if name == EOL_CLASS:
            return self.eol_id
        return self.names.index(name)

    def decode(self, class_id: int) -> str:
        if class_id == self.eol_id:
            return EOL_CLASS
        return self.names[class_id]

    def __contains__(self, name: str) -> bool:
        return name == EOL_CLASS or name in self.names
