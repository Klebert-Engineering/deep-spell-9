from deepspell.correct import EmbeddingCorrector, SymSpellCorrector
from deepspell.correct.symspell import build_dictionary
from deepspell.models import load_model


def test_symspell_corrector(gaz, tmp_path):
    dictionary = build_dictionary(gaz, tmp_path / "symspell.txt")
    corrector = SymSpellCorrector(dictionary)

    suggestions = corrector.match("los angles")
    assert suggestions
    assert suggestions[0][0] == "los angeles"

    assert corrector.match("califonria")[0][0] == "california"
    assert corrector.match("main s")[0][0] == "main st"


def test_embedding_corrector_roundtrip(gaz, trained_encoder_path, tmp_path, char_vocab):
    encoder, card = load_model(trained_encoder_path)
    names = [name for name, _ in gaz.iter_names()]
    space = EmbeddingCorrector.build(encoder, card.char_vocab(), names)
    assert len(space.tokens) == len(space.vectors)

    base = tmp_path / "space"
    space.save(base)
    loaded = EmbeddingCorrector.load(encoder, card.char_vocab(), base)
    assert loaded.tokens == space.tokens

    suggestions = loaded.match("los angeels", k=3)
    assert len(suggestions) == 3
    assert suggestions[0][0] == "los angeles"
    # distances ascend
    distances = [d for _, d in suggestions]
    assert distances == sorted(distances)
