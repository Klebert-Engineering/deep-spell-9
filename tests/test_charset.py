import pytest

from deepspell.charset import BOS, EOS, UNK, CharVocab, ClassVocab, fold


def test_fold_strips_diacritics_and_casefolds():
    assert fold("Abalá") == "abala"
    assert fold("Müllerstraße") == "mullerstrasse"
    assert fold("Los Angeles", lowercase=False) == "Los Angeles"


def test_encode_decode_roundtrip():
    vocab = CharVocab()
    text = "los angeles, ca 90210/5"
    assert vocab.decode(vocab.encode(text)) == text


def test_encode_unknown_and_specials():
    vocab = CharVocab()
    ids = vocab.encode("a#b", add_bos=True, add_eos=True)
    assert ids[0] == BOS
    assert ids[-1] == EOS
    assert ids[2] == UNK
    assert vocab.decode(ids) == "a_b"


def test_charset_duplicate_rejected():
    with pytest.raises(ValueError):
        CharVocab(charset="aab")


def test_class_vocab():
    vocab = ClassVocab(names=("CITY", "ROAD"))
    assert len(vocab) == 3
    assert vocab.eol_id == 2
    assert vocab.encode("ROAD") == 1
    assert vocab.encode("EOL") == vocab.eol_id
    assert vocab.decode(2) == "EOL"
    assert "EOL" in vocab and "CITY" in vocab and "STATE" not in vocab


def test_class_vocab_rejects_explicit_eol():
    with pytest.raises(ValueError):
        ClassVocab(names=("CITY", "EOL"))
