"""Random typo generation, ported from the v1 grammar corruption model.

The number of edits applied to a string is drawn from a normal distribution
(``mean``, ``stddev``), floored, and capped at ``max_corruptions``. Strings
shorter than 3 characters are never corrupted further.
"""

from __future__ import annotations

import math
import random
import string


def max_corruptions(mean: float, stddev: float) -> int:
    return int(mean + stddev * 5.0)


def _delete(s: str, rng: random.Random) -> str:
    pos = rng.randrange(len(s))
    return s[:pos] + s[pos + 1 :]


def _substitute(s: str, rng: random.Random) -> str:
    pos = rng.randrange(len(s))
    return s[:pos] + rng.choice(string.ascii_lowercase) + s[pos + 1 :]


def _transpose(s: str, rng: random.Random) -> str:
    pos1 = rng.randrange(len(s) - 1)
    pos2 = rng.randrange(pos1 + 1, len(s))
    return s[:pos1] + s[pos2] + s[pos1 + 1 : pos2] + s[pos1] + s[pos2 + 1 :]


def _insert(s: str, rng: random.Random) -> str:
    pos = rng.randrange(len(s) + 1)
    return s[:pos] + rng.choice(string.ascii_lowercase) + s[pos:]


_OPS = (_delete, _substitute, _transpose, _insert)


def corrupt(text: str, rng: random.Random, mean: float = 1.0, stddev: float = 0.5) -> str:
    """Apply a random number of random single-character edits to *text*."""
    n = min(max_corruptions(mean, stddev), int(math.floor(rng.gauss(mean, stddev))))
    for _ in range(n):
        if len(text) < 3:
            break
        text = rng.choice(_OPS)(text, rng)
    return text
