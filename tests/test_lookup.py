import sqlite3

import pytest

from deepspell.lookup import FtsLookup, build_lookup_index


@pytest.fixture(scope="module")
def lookup_db(gaz, tmp_path_factory):
    path = tmp_path_factory.mktemp("lookup") / "lookup.sqlite"
    build_lookup_index(gaz, path)
    return path


def test_native_lookup_by_city(lookup_db):
    lookup = FtsLookup(lookup_db)
    rows = lookup.query({"CITY": "los angeles"}, limit=10)
    # grouped by the most specific criterion (CITY) -> one row for the city
    assert len(rows) == 1
    assert rows[0]["CITY"] == "los angeles"
    assert rows[0]["group_size"] >= 4  # the four roads collapse into the group


def test_native_lookup_multi_criteria(lookup_db):
    lookup = FtsLookup(lookup_db)
    rows = lookup.query({"CITY": "los angeles", "ROAD": "universal"}, limit=10)
    assert rows
    assert all("universal" in row["ROAD"] for row in rows)


def test_lookup_rejects_unknown_class(lookup_db):
    lookup = FtsLookup(lookup_db)
    with pytest.raises(ValueError):
        lookup.query({"DROPTABLE": "x"})


def test_lookup_is_injection_safe(lookup_db):
    lookup = FtsLookup(lookup_db)
    # hostile values must neither raise sqlite errors nor leak rows
    assert lookup.query({"CITY": 'x" OR docid MATCH "*'}) == []
    assert lookup.query({"CITY": "los angeles; DROP TABLE lookup"}) == []
    assert lookup.query({"CITY": 'los "angeles"'}) != []  # quotes are escaped, not fatal


def test_lookup_empty_criteria(lookup_db):
    assert FtsLookup(lookup_db).query({}) == []
    assert FtsLookup(lookup_db).query({"CITY": "  "}) == []


def test_nds_adapter(tmp_path):
    path = tmp_path / "nds.nds"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE VIRTUAL TABLE nameFtsTable USING fts5(
            namedObjectId UNINDEXED, mortonCode UNINDEXED,
            criterionA, criterionB, criterionC, criterionD,
            tokenize = "unicode61 remove_diacritics 0"
        );
        INSERT INTO nameFtsTable VALUES
            (1, 99, 'W 29th Pl', 'Los Angeles', 'CA☱California', 'USA☱United States'),
            (2, 98, 'Main St', 'Los Angeles', 'CA☱California', 'USA☱United States');
        """
    )
    conn.commit()
    conn.close()

    lookup = FtsLookup.nds({"path": str(path)})
    rows = lookup.query({"STATE": "california", "ROAD": "main"}, limit=5)
    assert len(rows) == 1
    assert rows[0]["docid"] == 2
    assert rows[0]["morton"] == 98
    assert rows[0]["ROAD"] == "Main St"
    assert rows[0]["STATE"] == "CA / California"  # alt-name separator resolved
