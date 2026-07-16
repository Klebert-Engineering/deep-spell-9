import torch

from deepspell.charset import CharVocab, ClassVocab
from deepspell.decode import beam_search, best_class_sequence, split_by_class
from deepspell.models import Completer


def _probs(per_char: list[str], confident: float = 0.9):
    classes = ("CITY", "ROAD")
    return [
        [(name, confident if name == best else (1 - confident) / 1) for name in classes]
        for best in per_char
    ]


def test_best_class_sequence_is_unit_constant():
    text = "los angeles"
    # one noisy char inside the second unit must not flip the whole unit
    per_char = ["CITY"] * 4 + ["CITY", "ROAD", "CITY", "CITY", "CITY", "CITY", "CITY"]
    result = best_class_sequence(text, _probs(per_char))
    assert result == ["CITY"] * len(text)


def test_best_class_sequence_splits_units():
    text = "main st la"
    per_char = ["ROAD"] * 7 + ["CITY"] * 3
    result = best_class_sequence(text, _probs(per_char))
    assert result[:7] == ["ROAD"] * 7
    assert result[7:] == ["CITY"] * 3


def test_split_by_class_merges_units():
    text = "los angeles main st"
    classes = ["CITY"] * 11 + ["ROAD"] * 8
    tokens = split_by_class(text, classes)
    assert tokens == {"CITY": "los angeles", "ROAD": "main st"}


def test_beam_search_contract():
    torch.manual_seed(0)
    char_vocab = CharVocab()
    class_vocab = ClassVocab(names=("CITY", "ROAD"))
    completer = Completer(len(char_vocab), len(class_vocab), emb_dim=16, hidden=32, layers=1)
    completer.eval()
    text = "los ange"
    completions = beam_search(
        completer, char_vocab, class_vocab, text, [0] * len(text), beam_width=4, max_len=8
    )
    assert 0 < len(completions) <= 4
    logprobs = [c.logprob for c in completions]
    assert logprobs == sorted(logprobs, reverse=True)
    texts = [c.text for c in completions]
    assert len(set(texts)) == len(texts)  # deduplicated
    for completion in completions:
        assert len(completion.text) <= 8
        assert completion.class_name in ("CITY", "ROAD", "EOL")


def test_beam_search_empty_prefix():
    char_vocab = CharVocab()
    class_vocab = ClassVocab(names=("CITY",))
    completer = Completer(len(char_vocab), len(class_vocab), emb_dim=8, hidden=16, layers=1)
    assert beam_search(completer, char_vocab, class_vocab, "", []) == []
