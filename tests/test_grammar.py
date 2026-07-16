import random

from deepspell.sampling.corruption import corrupt, max_corruptions
from deepspell.sampling.grammar import render_phrase
from deepspell.train.dataset import PhraseSampler


def test_phrase_always_contains_seed(gaz, grammar):
    rng = random.Random(0)
    for _ in range(50):
        seed = gaz.random_token(rng)
        phrase = grammar.sample_phrase(gaz, seed, rng)
        assert seed.id in [token.id for token in phrase]
        assert len({token.class_name for token in phrase}) == len(phrase)  # one token per class


def test_phrases_are_seeded_deterministic(gaz, grammar):
    def run(seed):
        rng = random.Random(seed)
        return [
            [t.id for t in grammar.sample_phrase(gaz, gaz.random_token(rng), rng)] for _ in range(20)
        ]

    assert run(42) == run(42)
    assert run(42) != run(43)


def test_render_phrase_classes_per_char(gaz, grammar, char_vocab, class_vocab):
    rng = random.Random(1)
    phrase = grammar.sample_phrase(gaz, gaz.random_token(rng), rng)
    text, class_ids = render_phrase(phrase, char_vocab, class_vocab)
    assert len(text) == len(class_ids)
    assert text == char_vocab.normalize(text)
    # the separator carries the class of the following token
    rendered_names = [char_vocab.normalize(t.name) for t in phrase]
    assert text == " ".join(rendered_names)


def test_corruption_seeded_and_bounded():
    rng_a, rng_b = random.Random(5), random.Random(5)
    results_a = [corrupt("california", rng_a) for _ in range(20)]
    results_b = [corrupt("california", rng_b) for _ in range(20)]
    assert results_a == results_b
    cap = max_corruptions(1.0, 0.5)
    for result in results_a:
        assert abs(len(result) - len("california")) <= cap


def test_corruption_leaves_short_strings():
    rng = random.Random(0)
    assert corrupt("ab", rng) == "ab"


def test_sampler_truncation(gaz, grammar, char_vocab, class_vocab):
    sampler = PhraseSampler(gaz, grammar, char_vocab, class_vocab, seed=2, truncate_min=3)
    for _ in range(30):
        text, class_ids = sampler.sample()
        assert len(text) == len(class_ids)
        assert len(text) >= 3


def test_sampler_corruption_mode(gaz, grammar, char_vocab, class_vocab):
    clean = PhraseSampler(gaz, grammar, char_vocab, class_vocab, seed=3)
    noisy = PhraseSampler(gaz, grammar, char_vocab, class_vocab, seed=3, corrupt_tokens=True)
    clean_texts = [clean.sample()[0] for _ in range(20)]
    noisy_texts = [noisy.sample()[0] for _ in range(20)]
    assert clean_texts != noisy_texts
