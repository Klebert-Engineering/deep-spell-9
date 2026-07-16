import torch

from deepspell.charset import CharVocab, ClassVocab
from deepspell.models import Completer, ModelCard, TokenEncoder, load_model, save_model
from deepspell.models.tagger import Tagger


def test_tagger_shapes_and_predict():
    char_vocab = CharVocab()
    class_vocab = ClassVocab(names=("CITY", "ROAD"))
    tagger = Tagger(len(char_vocab), len(class_vocab), emb_dim=16, hidden=16, layers=1)
    ids = torch.tensor([[5, 6, 7, 0], [5, 6, 0, 0]])
    logits = tagger(ids, torch.tensor([3, 2]))
    assert logits.shape == (2, 4, 3)

    probs = tagger.predict("los angeles", char_vocab, class_vocab)
    assert len(probs) == len("los angeles")
    for char_probs in probs:
        assert abs(sum(p for _, p in char_probs) - 1.0) < 1e-4
        assert [p for _, p in char_probs] == sorted((p for _, p in char_probs), reverse=True)
    assert tagger.predict("", char_vocab, class_vocab) == []


def test_completer_shapes_and_state():
    completer = Completer(50, 4, emb_dim=16, hidden=24, layers=2)
    chars = torch.randint(1, 50, (3, 7))
    classes = torch.randint(0, 4, (3, 7))
    char_logits, class_logits, state = completer(chars, classes, torch.tensor([7, 5, 3]))
    assert char_logits.shape == (3, 7, 50)
    assert class_logits.shape == (3, 7, 4)
    # stepwise continuation with the returned state
    char_logits2, _, _ = completer(chars[:, :1], classes[:, :1], state=state)
    assert char_logits2.shape == (3, 1, 50)


def test_encoder_normalized_output():
    char_vocab = CharVocab()
    encoder = TokenEncoder(len(char_vocab), emb_dim=16, hidden=16, out_dim=8)
    vectors = encoder.encode_strings(["los angeles", "main st"], char_vocab)
    assert vectors.shape == (2, 8)
    norms = (vectors**2).sum(axis=1) ** 0.5
    assert abs(norms - 1.0).max() < 1e-5


def test_model_card_roundtrip(tmp_path):
    char_vocab = CharVocab()
    card = ModelCard(
        kind="tagger",
        charset=char_vocab.charset,
        classes=["CITY", "ROAD"],
        hparams={"emb_dim": 16, "hidden": 16, "layers": 1},
    )
    model = card.build()
    base = tmp_path / "tagger-roundtrip"
    json_path = save_model(model, card, base)
    assert json_path.exists() and base.with_suffix(".pt").exists()

    loaded, loaded_card = load_model(json_path)
    assert loaded_card == card
    text_ids = torch.tensor([[5, 6, 7]])
    lengths = torch.tensor([3])
    model.eval()
    assert torch.allclose(model(text_ids, lengths), loaded(text_ids, lengths))
