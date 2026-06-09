"""End-to-end CPU smoke training on the minimal corpus.

The corpus has 7 tokens, so the toy models must learn it almost perfectly
within a few hundred steps; thresholds are deliberately lenient to stay
robust across torch versions.
"""

import json
from pathlib import Path

from deepspell.decode import beam_search
from deepspell.models import load_model
from deepspell.train.dataset import PhraseSampler
from deepspell.train.evaluate import evaluate_completer, evaluate_tagger


def test_trained_tagger_quality(trained_tagger_path, gaz, grammar):
    tagger, card = load_model(trained_tagger_path)
    assert card.training["char_accuracy"] >= 0.85

    sampler = PhraseSampler(
        gaz, grammar, card.char_vocab(), card.class_vocab(), seed=123, truncate_min=3
    )
    metrics = evaluate_tagger(tagger, sampler, card.char_vocab(), card.class_vocab(), samples=50)
    assert metrics["unit_accuracy"] >= 0.8


def test_trained_completer_quality(trained_completer_path, gaz, grammar):
    completer, card = load_model(trained_completer_path)
    char_vocab, class_vocab = card.char_vocab(), card.class_vocab()

    # "los angeles cali" can only continue as "fornia" in this corpus
    prefix = "los angeles cali"
    classes = [class_vocab.encode("CITY")] * 12 + [class_vocab.encode("STATE")] * 4
    completions = beam_search(completer, char_vocab, class_vocab, prefix, classes, beam_width=6)
    assert completions
    assert "fornia" in [c.text for c in completions]

    sampler = PhraseSampler(gaz, grammar, char_vocab, class_vocab, seed=123)
    metrics = evaluate_completer(completer, sampler, char_vocab, class_vocab, samples=40)
    assert metrics["samples"] > 0
    assert metrics["exact_top6"] >= 0.3


def test_trained_encoder_metrics(trained_encoder_path):
    card = json.loads(Path(trained_encoder_path).read_text())
    assert card["training"]["val_retrieval_top1"] >= 0.6
