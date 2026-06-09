"""Streaming batch sampling from a gazetteer.

No DataLoader workers: samples are drawn on demand from SQLite + grammar,
which keeps memory flat regardless of corpus size (the v1 loader held all
9.5 M tokens as Python objects). Each sampler owns a seeded RNG, so batches
are reproducible.
"""

from __future__ import annotations

import random

import torch

from ..charset import BOS, EOS, CharVocab, ClassVocab
from ..gazetteer import Gazetteer
from ..sampling.corruption import corrupt
from ..sampling.grammar import PhraseGrammar

IGNORE_INDEX = -100


class PhraseSampler:
    """Samples rendered (text, class-ids-per-char) phrases."""

    def __init__(
        self,
        gaz: Gazetteer,
        grammar: PhraseGrammar,
        char_vocab: CharVocab,
        class_vocab: ClassVocab,
        seed: int = 0,
        truncate_min: int | None = None,
        corrupt_tokens: bool = False,
    ):
        self.gaz = gaz
        self.grammar = grammar
        self.char_vocab = char_vocab
        self.class_vocab = class_vocab
        self.rng = random.Random(seed)
        self.truncate_min = truncate_min
        self.corrupt_tokens = corrupt_tokens

    def sample(self) -> tuple[str, list[int]]:
        phrase = self.grammar.sample_phrase(self.gaz, self.gaz.random_token(self.rng), self.rng)
        text_parts: list[str] = []
        class_ids: list[int] = []
        for i, token in enumerate(phrase):
            name = token.name
            if self.corrupt_tokens:
                name = corrupt(name, self.rng, self.grammar.corruption_mean, self.grammar.corruption_stddev)
            rendered = self.char_vocab.normalize(("" if i == 0 else " ") + name)
            text_parts.append(rendered)
            class_ids.extend([self.class_vocab.encode(token.class_name)] * len(rendered))
        text = "".join(text_parts)
        if self.truncate_min is not None and len(text) > self.truncate_min:
            cut = self.rng.randint(self.truncate_min, len(text))
            text, class_ids = text[:cut], class_ids[:cut]
        return text, class_ids


class TokenPairSampler:
    """Samples (corrupted, clean) normalized single-token pairs for the encoder."""

    def __init__(
        self,
        gaz: Gazetteer,
        char_vocab: CharVocab,
        seed: int = 0,
        corruption_mean: float = 1.0,
        corruption_stddev: float = 0.5,
    ):
        self.gaz = gaz
        self.char_vocab = char_vocab
        self.rng = random.Random(seed)
        self.corruption_mean = corruption_mean
        self.corruption_stddev = corruption_stddev

    def sample(self) -> tuple[str, str]:
        clean = self.char_vocab.normalize(self.gaz.random_token(self.rng).name)
        corrupted = corrupt(clean, self.rng, self.corruption_mean, self.corruption_stddev)
        return corrupted or clean, clean


def _pad(rows: list[list[int]], fill: int) -> torch.Tensor:
    width = max(len(r) for r in rows)
    out = torch.full((len(rows), width), fill, dtype=torch.long)
    for i, row in enumerate(rows):
        out[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    return out


def tagger_batch(
    sampler: PhraseSampler, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (char_ids [B,T], target classes [B,T] with IGNORE on padding, lengths)."""
    ids_rows, class_rows = [], []
    for _ in range(batch_size):
        text, class_ids = sampler.sample()
        while not text:
            text, class_ids = sampler.sample()
        ids_rows.append(sampler.char_vocab.encode(text, add_eos=True))
        class_rows.append(class_ids + [sampler.class_vocab.eol_id])
    lengths = torch.tensor([len(r) for r in ids_rows])
    return _pad(ids_rows, 0).to(device), _pad(class_rows, IGNORE_INDEX).to(device), lengths


def completer_batch(
    sampler: PhraseSampler, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Teacher-forcing batch: input (BOS+text), targets (text+EOS, classes+EOL)."""
    in_chars, in_classes, tgt_chars, tgt_classes = [], [], [], []
    for _ in range(batch_size):
        text, class_ids = sampler.sample()
        while not text:
            text, class_ids = sampler.sample()
        char_ids = sampler.char_vocab.encode(text)
        in_chars.append([BOS] + char_ids)
        in_classes.append([class_ids[0]] + class_ids)
        tgt_chars.append(char_ids + [EOS])
        tgt_classes.append(class_ids + [sampler.class_vocab.eol_id])
    lengths = torch.tensor([len(r) for r in in_chars])
    return (
        _pad(in_chars, 0).to(device),
        _pad(in_classes, 0).to(device),
        _pad(tgt_chars, IGNORE_INDEX).to(device),
        _pad(tgt_classes, IGNORE_INDEX).to(device),
        lengths,
    )


def encoder_batch(
    sampler: TokenPairSampler, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (corrupt_ids, corrupt_lengths, clean_ids, clean_lengths)."""
    corrupt_rows, clean_rows = [], []
    seen: set[str] = set()
    attempts = 0
    while len(clean_rows) < batch_size:
        corrupted, clean = sampler.sample()
        attempts += 1
        if clean in seen and attempts < batch_size * 50:
            continue  # in-batch duplicates break the contrastive labels
        seen.add(clean)
        corrupt_rows.append(sampler.char_vocab.encode(corrupted) or [0])
        clean_rows.append(sampler.char_vocab.encode(clean) or [0])
    corrupt_lengths = torch.tensor([len(r) for r in corrupt_rows])
    clean_lengths = torch.tensor([len(r) for r in clean_rows])
    return (
        _pad(corrupt_rows, 0).to(device),
        corrupt_lengths,
        _pad(clean_rows, 0).to(device),
        clean_lengths,
    )
