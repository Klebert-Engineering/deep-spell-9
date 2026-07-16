import pytest
from fastapi.testclient import TestClient

from deepspell.correct import SymSpellCorrector
from deepspell.correct.symspell import build_dictionary
from deepspell.lookup import FtsLookup, build_lookup_index
from deepspell.models import load_model
from deepspell.service import Pipeline, create_app


@pytest.fixture(scope="module")
def pipeline(gaz, trained_tagger_path, trained_completer_path, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("service")
    tagger, tagger_card = load_model(trained_tagger_path)
    completer, completer_card = load_model(trained_completer_path)
    corrector = SymSpellCorrector(build_dictionary(gaz, tmp / "symspell.txt"))
    lookup = FtsLookup(build_lookup_index(gaz, tmp / "lookup.sqlite"))
    return Pipeline(
        tagger, tagger_card, completer, completer_card, corrector=corrector, lookup=lookup
    )


@pytest.fixture(scope="module")
def client(pipeline):
    with TestClient(create_app(pipeline=pipeline)) as test_client:
        yield test_client


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["with_corrector"] is True
    assert body["with_lookup"] is True


def test_complete_endpoint(client):
    response = client.get("/api/complete", params={"q": "Los Angeles Cali"})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "los angeles cali"
    assert len(body["classes"]) == len(body["query"])
    for char_probs in body["classes"]:
        assert char_probs[0]["p"] >= char_probs[-1]["p"]
    assert body["completions"], "expected at least one completion"
    assert {"classification", "completion", "correction"} <= set(body["timings_ms"])
    assert "CITY" in body["tokens"]


def test_complete_applies_correction(client):
    body = client.get("/api/complete", params={"q": "los angles"}).json()
    corrections = body["corrections"]
    assert corrections
    suggestions = [s["text"] for c in corrections.values() for s in c["suggestions"]]
    assert "los angeles" in suggestions


def test_complete_empty_query(client):
    body = client.get("/api/complete", params={"q": ""}).json()
    assert body["query"] == ""
    assert body["classes"] == []
    assert body["completions"] == []


def test_lookup_endpoint(client):
    response = client.get("/api/lookup", params={"CITY": "los angeles", "n": 5})
    assert response.status_code == 200
    rows = response.json()
    assert rows and rows[0]["CITY"] == "los angeles"


def test_lookup_rejects_unknown_class(client):
    response = client.get("/api/lookup", params={"EVIL": "x"})
    assert response.status_code == 400


def test_index_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Deep Spell" in response.text


def test_incompatible_cards_rejected(trained_tagger_path, trained_completer_path):
    tagger, tagger_card = load_model(trained_tagger_path)
    completer, completer_card = load_model(trained_completer_path)
    broken = completer_card.model_copy(update={"classes": ["OTHER"]})
    with pytest.raises(ValueError):
        Pipeline(tagger, tagger_card, completer, broken)
