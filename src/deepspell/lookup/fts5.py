"""SQLite FTS5 lookup — native v2 indexes and licensed NDS databases.

Replaces the v1 ``DSFtsDatabaseConnection``, whose statement builder
interpolated raw request input into SQL (injectable). Here, class names are
validated against the configured column mapping, values are escaped as FTS5
phrase strings, and the MATCH expression plus limit are bound as parameters.

Verified: NDS ``RoadFTS5_*`` files use a plain ``unicode61`` tokenizer and
work with stock Python sqlite3 — no proprietary build needed. The ``☱``
character separates alternate names inside NDS columns.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path

from ..charset import CharVocab
from ..gazetteer import Gazetteer

NDS_DEFAULT_COLUMNS = {
    "ROAD": "criterionA",
    "CITY": "criterionB",
    "STATE": "criterionC",
    "COUNTRY": "criterionD",
}
NATIVE_COLUMNS = {"ROAD": "road", "CITY": "city", "STATE": "state", "COUNTRY": "country"}
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
NDS_ALT_SEPARATOR = "☱"  # ☱


def _check_identifier(name: str) -> str:
    if not _IDENTIFIER.match(name):
        raise ValueError(f"invalid SQL identifier: {name!r}")
    return name


class FtsLookup:
    """Class-scoped FTS5 MATCH queries over a lookup table."""

    def __init__(
        self,
        path: str | Path,
        table: str = "lookup",
        columns: dict[str, str] | None = None,
        docid_column: str = "doc_id",
        morton_column: str | None = None,
        specificity: tuple[str, ...] = ("COUNTRY", "STATE", "CITY", "ROAD"),
        alt_separator: str | None = None,
    ):
        # served from FastAPI's threadpool: share the read-only connection behind a lock
        self.conn = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True, check_same_thread=False)
        self._lock = threading.Lock()
        self.table = _check_identifier(table)
        self.columns = {k: _check_identifier(v) for k, v in (columns or NATIVE_COLUMNS).items()}
        self.docid_column = _check_identifier(docid_column)
        self.morton_column = _check_identifier(morton_column) if morton_column else None
        self.specificity = specificity
        self.alt_separator = alt_separator

    @classmethod
    def nds(cls, config: dict) -> FtsLookup:
        """Open a licensed NDS database from a v1 ``fts_db`` config block."""
        return cls(
            path=config["path"],
            table=config.get("fts_table_name", "nameFtsTable"),
            columns=config.get("column_names", NDS_DEFAULT_COLUMNS),
            docid_column=config.get("docid_column_name", "namedObjectId"),
            morton_column=config.get("morton_column_name", "mortonCode"),
            specificity=tuple(config.get("class_specificity", ("COUNTRY", "STATE", "CITY", "ROAD"))),
            alt_separator=NDS_ALT_SEPARATOR,
        )

    def query(self, criteria: dict[str, str], limit: int = 10) -> list[dict]:
        """Look up entries matching ``{class_name: token}`` criteria.

        Unknown class names raise ``ValueError``; token values are FTS5
        phrase-escaped and bound, never interpolated.
        """
        criteria = {k: v.strip() for k, v in criteria.items() if v and v.strip()}
        if not criteria:
            return []
        unknown = set(criteria) - set(self.columns)
        if unknown:
            raise ValueError(f"unknown lookup classes: {sorted(unknown)}")

        most_specific = self.columns[
            max(criteria, key=lambda class_name: self.specificity.index(class_name))
        ]
        match_expression = " ".join(
            '{}: "{}"'.format(self.columns[class_name], value.replace('"', '""'))
            for class_name, value in criteria.items()
        )
        aliases = ", ".join(
            f"{column} AS {_check_identifier(class_name)}" for class_name, column in self.columns.items()
        )
        length_sum = " + ".join(f"length({self.columns[c]})" for c in criteria)
        morton = f"{self.morton_column} AS morton," if self.morton_column else ""
        statement = f"""
            SELECT {self.docid_column} AS docid, {morton}
                   COUNT({most_specific}) AS group_size, {aliases}
            FROM {self.table}
            WHERE {self.table} MATCH ?
            GROUP BY {most_specific}
            ORDER BY {length_sum}, group_size
            LIMIT ?
        """
        with self._lock:
            cursor = self.conn.execute(statement, (match_expression, int(limit)))
            names = [d[0] for d in cursor.description]
            rows = [dict(zip(names, row)) for row in cursor.fetchall()]
        if self.alt_separator:
            for row in rows:
                for key, value in row.items():
                    if isinstance(value, str) and self.alt_separator in value:
                        row[key] = " / ".join(value.split(self.alt_separator))
        return rows

    def close(self) -> None:
        self.conn.close()


def build_lookup_index(
    gaz: Gazetteer, out_path: str | Path, char_vocab: CharVocab | None = None, batch_size: int = 5000
) -> Path:
    """Build a native FTS5 lookup index from a gazetteer.

    One row per most-specific token (typically ROAD) carrying its full
    ancestor hierarchy; tokens without children also get a row so cities can
    be found without a road.
    """
    vocab = char_vocab or CharVocab()
    out = Path(out_path)
    conn = sqlite3.connect(out)
    conn.executescript(
        """
        DROP TABLE IF EXISTS lookup;
        CREATE VIRTUAL TABLE lookup USING fts5(
            doc_id UNINDEXED, road, city, state, country,
            tokenize = "unicode61 remove_diacritics 2"
        );
        """
    )
    class_to_column = NATIVE_COLUMNS
    batch: list[tuple] = []
    for token in gaz.iter_tokens():
        values = {column: "" for column in class_to_column.values()}
        column = class_to_column.get(token.class_name)
        if column is None:
            continue
        values[column] = vocab.normalize(token.name)
        for ancestor in gaz.ancestors(token).values():
            ancestor_column = class_to_column.get(ancestor.class_name)
            if ancestor_column:
                values[ancestor_column] = vocab.normalize(ancestor.name)
        batch.append((token.id, values["road"], values["city"], values["state"], values["country"]))
        if len(batch) >= batch_size:
            conn.executemany("INSERT INTO lookup VALUES (?, ?, ?, ?, ?)", batch)
            batch.clear()
    if batch:
        conn.executemany("INSERT INTO lookup VALUES (?, ?, ?, ?, ?)", batch)
    conn.commit()
    conn.close()
    return out
