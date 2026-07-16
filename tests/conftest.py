import pytest
from tests.paths import GRAMMAR_JSON, MINIMAL_TSV

from deepspell.charset import CharVocab, ClassVocab
from deepspell.gazetteer import Gazetteer
from deepspell.gazetteer.legacy_tsv import import_legacy_tsv
from deepspell.sampling.grammar import PhraseGrammar
from deepspell.train import TrainSettings, train_completer, train_encoder, train_tagger


@pytest.fixture(scope="session")
def gazetteer_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("gaz") / "minimal.sqlite"
    with Gazetteer(path) as gaz:
        import_legacy_tsv(gaz, MINIMAL_TSV)
    return path


@pytest.fixture(scope="session")
def gaz(gazetteer_path):
    gazetteer = Gazetteer(gazetteer_path, readonly=True)
    yield gazetteer
    gazetteer.close()


@pytest.fixture(scope="session")
def grammar():
    return PhraseGrammar.from_file(GRAMMAR_JSON)


@pytest.fixture(scope="session")
def char_vocab():
    return CharVocab()


@pytest.fixture(scope="session")
def class_vocab(gaz):
    return ClassVocab(names=tuple(gaz.classes()))


@pytest.fixture(scope="session")
def trained_tagger_path(gaz, grammar, tmp_path_factory):
    out = tmp_path_factory.mktemp("models") / "tagger-test"
    settings = TrainSettings(
        steps=250, batch_size=16, val_every=250, val_batches=4, log_every=1000, device="cpu",
        hparams={"hidden": 32, "emb_dim": 16, "layers": 1},
    )
    return train_tagger(gaz, grammar, str(out), settings)


@pytest.fixture(scope="session")
def trained_completer_path(gaz, grammar, tmp_path_factory):
    out = tmp_path_factory.mktemp("models") / "completer-test"
    settings = TrainSettings(
        steps=500, batch_size=16, val_every=500, val_batches=4, log_every=1000, device="cpu",
        hparams={"hidden": 64, "emb_dim": 32, "layers": 1},
    )
    return train_completer(gaz, grammar, str(out), settings)


@pytest.fixture(scope="session")
def trained_encoder_path(gaz, tmp_path_factory):
    out = tmp_path_factory.mktemp("models") / "encoder-test"
    settings = TrainSettings(
        steps=300, batch_size=6, val_every=300, val_batches=4, log_every=1000, device="cpu",
        hparams={"hidden": 32, "emb_dim": 16, "out_dim": 16},
    )
    return train_encoder(gaz, str(out), settings)
