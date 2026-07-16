"""Per-token class assignment from per-character class distributions.

Port of the v1 ``extract_best_class_sequence`` / ``tokenize_class_annotated_
characters``: a query is split into token units at separator characters; each
unit is assigned the class with the highest cumulative character log-prob.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable

#: Per character: list of (class_name, probability), any order.
CharClassProbs = list[list[tuple[str, float]]]

_EPS = 1e-9


def _default_separator(ch: str) -> bool:
    return ch in " -,"


def best_class_sequence(
    text: str,
    probs: CharClassProbs,
    separator: Callable[[str], bool] = _default_separator,
) -> list[str]:
    """Best class per character, constant within each token unit."""
    result: list[str] = []
    unit_length = 0
    logprob: dict[str, float] = defaultdict(float)

    def close_unit() -> None:
        nonlocal unit_length
        if unit_length and logprob:
            best = max(logprob, key=logprob.__getitem__)
            result.extend([best] * unit_length)
        unit_length = 0
        logprob.clear()

    for ch, char_probs in zip(text, probs):
        if separator(ch):
            close_unit()
        unit_length += 1
        for class_name, p in char_probs:
            logprob[class_name] += math.log(p + _EPS)
    close_unit()
    return result


def split_by_class(
    text: str,
    classes: list[str],
    separator: Callable[[str], bool] = _default_separator,
) -> dict[str, str]:
    """Concatenate the characters of each class into per-class token strings.

    ``classes`` must be per-character (e.g. the output of
    :func:`best_class_sequence`); separators joining two units of the same
    class are preserved ("los angeles" stays one CITY string).
    """
    result: dict[str, str] = defaultdict(str)
    for ch, class_name in zip(text, classes):
        result[class_name] += ch
    return {class_name: token.strip(" -,") for class_name, token in result.items() if token.strip(" -,")}
