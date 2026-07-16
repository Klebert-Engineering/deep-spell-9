"""Quality metrics, successor of the v1 ``eval-*.py`` scripts."""

from __future__ import annotations

import random

from ..charset import CharVocab, ClassVocab
from ..correct import Corrector
from ..decode import beam_search, best_class_sequence
from ..models.completer import Completer
from ..models.tagger import Tagger
from ..sampling.corruption import corrupt
from .dataset import PhraseSampler


def evaluate_tagger(
    model: Tagger,
    sampler: PhraseSampler,
    char_vocab: CharVocab,
    class_vocab: ClassVocab,
    samples: int = 200,
) -> dict:
    """Character- and unit-level accuracy on freshly sampled (truncated) phrases."""
    char_correct = char_total = unit_correct = unit_total = 0
    for _ in range(samples):
        text, class_ids = sampler.sample()
        if not text:
            continue
        truth = [class_vocab.decode(i) for i in class_ids]
        probs = model.predict(text, char_vocab, class_vocab)
        argmax = [p[0][0] for p in probs]
        char_correct += sum(a == t for a, t in zip(argmax, truth))
        char_total += len(truth)
        units = best_class_sequence(text, probs)
        unit_correct += sum(u == t for u, t in zip(units, truth))
        unit_total += len(truth)
    return {
        "char_accuracy": round(char_correct / max(char_total, 1), 4),
        "unit_accuracy": round(unit_correct / max(unit_total, 1), 4),
        "samples": samples,
    }


def evaluate_completer(
    completer: Completer,
    sampler: PhraseSampler,
    char_vocab: CharVocab,
    class_vocab: ClassVocab,
    samples: int = 100,
    beam_width: int = 6,
    min_prefix: int = 3,
) -> dict:
    """Completion quality with ground-truth prefix classes.

    For each sampled phrase, cut at a random point and ask the completer to
    finish the *current token* (up to the next class switch). Reports
    exact-match@1/@k and mean saved keystrokes (common-prefix length of the
    top completion vs. truth).
    """
    exact_top1 = exact_topk = 0
    saved_chars = 0.0
    evaluated = 0
    for _ in range(samples):
        text, class_ids = sampler.sample()
        if len(text) <= min_prefix + 1:
            continue
        cut = sampler.rng.randint(min_prefix, len(text) - 1)
        prefix, prefix_classes = text[:cut], class_ids[:cut]
        current_class = class_ids[cut - 1]
        end = cut
        while end < len(text) and class_ids[end] == current_class:
            end += 1
        truth = text[cut:end]
        if not truth:
            continue
        completions = beam_search(
            completer, char_vocab, class_vocab, prefix, prefix_classes, beam_width=beam_width,
            max_len=max(len(truth) + 4, 8),
        )
        evaluated += 1
        if not completions:
            continue
        texts = [c.text for c in completions]
        if texts[0] == truth:
            exact_top1 += 1
        if truth in texts:
            exact_topk += 1
        common = 0
        for a, b in zip(texts[0], truth):
            if a != b:
                break
            common += 1
        saved_chars += common
    return {
        "exact_top1": round(exact_top1 / max(evaluated, 1), 4),
        f"exact_top{beam_width}": round(exact_topk / max(evaluated, 1), 4),
        "mean_saved_chars": round(saved_chars / max(evaluated, 1), 2),
        "samples": evaluated,
    }


def evaluate_corrector(
    corrector: Corrector,
    names: list[str],
    seed: int = 0,
    samples: int = 200,
    k: int = 3,
    corruption_mean: float = 1.0,
    corruption_stddev: float = 0.5,
) -> dict:
    """Recall@1/@k of recovering the clean name from a corrupted query."""
    rng = random.Random(seed)
    top1 = topk = 0
    for _ in range(samples):
        clean = rng.choice(names)
        query = corrupt(clean, rng, corruption_mean, corruption_stddev)
        suggestions = [s for s, _ in corrector.match(query, k=k)]
        if suggestions and suggestions[0] == clean:
            top1 += 1
        if clean in suggestions:
            topk += 1
    return {
        "recall_at_1": round(top1 / samples, 4),
        f"recall_at_{k}": round(topk / samples, 4),
        "samples": samples,
    }
