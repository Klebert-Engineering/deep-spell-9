import random

from deepspell.gazetteer import Gazetteer
from deepspell.gazetteer.geonames import import_geonames


def test_import_counts(gaz):
    assert gaz.count("ROAD") == 4
    assert gaz.count("CITY") == 1
    assert gaz.count("STATE") == 1
    assert gaz.count("COUNTRY") == 1
    assert gaz.count() == 7


def test_classes_ordered_by_count(gaz):
    assert gaz.classes()[0] == "ROAD"
    assert set(gaz.classes()) == {"ROAD", "CITY", "STATE", "COUNTRY"}


def test_dense_ids(gaz):
    ids = [token.id for token in gaz.iter_tokens()]
    assert ids == list(range(1, 8))


def test_hierarchy_and_abbrev(gaz):
    roads = list(gaz.iter_tokens("ROAD"))
    ancestors = gaz.ancestors(roads[0])
    assert ancestors["CITY"].name == "Los Angeles"
    assert ancestors["STATE"].name == "California"
    assert ancestors["STATE"].abbrev == "CA"
    assert ancestors["COUNTRY"].name == "United States"
    assert ancestors["COUNTRY"].abbrev == "USA"


def test_random_descendants_reach_roads(gaz):
    country = next(gaz.iter_tokens("COUNTRY"))
    rng = random.Random(7)
    descendants = gaz.random_descendants(country, rng)
    assert descendants["STATE"].name == "California"
    assert descendants["CITY"].name == "Los Angeles"
    assert descendants["ROAD"].class_name == "ROAD"


def test_random_token_seeded(gaz):
    rng = random.Random(3)
    sampled = {gaz.random_token(rng).id for _ in range(50)}
    assert sampled.issubset(set(range(1, 8)))
    assert len(sampled) > 3  # actually mixes
    city = gaz.random_token(random.Random(1), class_name="CITY")
    assert city.class_name == "CITY"


def test_iter_names_accumulates_freq(gaz):
    names = dict(gaz.iter_names("ROAD"))
    assert "Main St" in names


def test_geonames_import(tmp_path):
    directory = tmp_path / "geonames"
    directory.mkdir()
    (directory / "countryInfo.txt").write_text(
        "# comment line\n"
        "US\tUSA\t840\tUS\tUnited States\tWashington\t9629091\t310232863\tNA\n",
        encoding="utf-8",
    )
    (directory / "admin1CodesASCII.txt").write_text(
        "US.CA\tCalifornia\tCalifornia\t5332921\n", encoding="utf-8"
    )
    cols = ["5368361", "Los Angeles", "Los Angeles", "", "34.05", "-118.24", "P", "PPLA2", "US", "", "CA"]
    (directory / "cities500.txt").write_text("\t".join(cols) + "\n", encoding="utf-8")

    with Gazetteer(tmp_path / "geo.sqlite") as gaz:
        assert import_geonames(gaz, directory) == 3
        city = next(gaz.iter_tokens("CITY"))
        assert city.name == "Los Angeles"
        state = gaz.ancestors(city)["STATE"]
        assert state.name == "California"
        assert state.abbrev == "CA"
        assert gaz.ancestors(city)["COUNTRY"].abbrev == "US"
