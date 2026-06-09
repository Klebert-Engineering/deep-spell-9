"""Probabilistic phrase grammar, JSON-compatible with the v1 format.

A grammar describes how a training phrase is composed around a seed token:
the seed's transitive ancestors and one random descendant chain are made
available, then ``random-sequence`` rules decide (per class prior) which of
them join the phrase. The resulting token order is shuffled.

Example (``corpora/grammar-address-na.json``)::

    {
      "root-nonterminal": "us-address",
      "rules": [
        {"class": "city-road", "type": "random-sequence",
         "symbols": [{"class": "CITY", "prior": 1.0},
                     {"class": "ROAD", "prior": 0.5}]},
        {"class": "us-address", "type": "random-sequence",
         "symbols": [{"class": "COUNTRY", "prior": 0.1},
                     {"class": "STATE", "prior": 0.2},
                     {"class": "city-road", "prior": 0.8}]}
      ],
      "corruption": {"mean": 1.0, "stddev": 0.5}
    }
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from ..charset import CharVocab, ClassVocab
from ..gazetteer import Gazetteer, Token

RULE_TYPE_RANDOM_SEQUENCE = "random-sequence"


@dataclass
class _Rule:
    #: (symbol, prior); symbol is a nested _Rule or a terminal class name.
    symbols: list[tuple[_Rule | str, float]]

    def expand(self, seed_class: str, available: set[str], rng: random.Random, deck: list[str]) -> None:
        for symbol, prior in self.symbols:
            if isinstance(symbol, _Rule):
                if rng.random() <= prior or symbol.contains_terminal(seed_class):
                    symbol.expand(seed_class, available, rng, deck)
            elif symbol != seed_class and symbol in available and rng.random() <= prior:
                deck.append(symbol)

    def contains_terminal(self, class_name: str) -> bool:
        for symbol, _ in self.symbols:
            if symbol == class_name:
                return True
            if isinstance(symbol, _Rule) and symbol.contains_terminal(class_name):
                return True
        return False


class PhraseGrammar:
    def __init__(self, spec: dict):
        self.corruption_mean = float(spec["corruption"]["mean"])
        self.corruption_stddev = float(spec["corruption"]["stddev"])
        self._root_name = spec["root-nonterminal"]
        rules: dict[str, _Rule] = {}
        for rule_spec in spec["rules"]:
            if rule_spec["type"] != RULE_TYPE_RANDOM_SEQUENCE:
                raise ValueError(f"unsupported rule type {rule_spec['type']!r}")
            symbols: list[tuple[_Rule | str, float]] = []
            for symbol in rule_spec["symbols"]:
                name, prior = symbol["class"], float(symbol["prior"])
                symbols.append((rules.get(name, name), prior))
            rules[rule_spec["class"]] = _Rule(symbols)
        if self._root_name not in rules:
            raise ValueError(f"root nonterminal {self._root_name!r} has no rule")
        self._root = rules[self._root_name]

    @classmethod
    def from_file(cls, path: str | Path) -> PhraseGrammar:
        with open(path, encoding="utf-8") as grammar_file:
            return cls(json.load(grammar_file))

    def sample_phrase(self, gaz: Gazetteer, seed: Token, rng: random.Random) -> list[Token]:
        """Compose a shuffled phrase of related tokens around *seed*."""
        available: dict[str, Token] = {seed.class_name: seed}
        available.update(gaz.ancestors(seed))
        available.update(gaz.random_descendants(seed, rng))
        deck = [seed.class_name]
        self._root.expand(seed.class_name, set(available), rng, deck)
        rng.shuffle(deck)
        return [available[class_name] for class_name in deck]


def render_phrase(
    phrase: list[Token], char_vocab: CharVocab, class_vocab: ClassVocab
) -> tuple[str, list[int]]:
    """Render tokens to ``(text, class_id_per_char)``.

    Tokens are joined with a single space; the separator carries the class of
    the *following* token (v1 convention).
    """
    text_parts: list[str] = []
    class_ids: list[int] = []
    for i, token in enumerate(phrase):
        rendered = char_vocab.normalize(("" if i == 0 else " ") + token.name)
        text_parts.append(rendered)
        class_ids.extend([class_vocab.encode(token.class_name)] * len(rendered))
    return "".join(text_parts), class_ids
