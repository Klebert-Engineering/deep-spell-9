"""SQLite gazetteer — the single canonical data artifact of DS9 v2.

One database holds the hierarchical token graph (road -> city -> state ->
country) that powers training-phrase sampling, corrector vocabularies and
the FTS5 lookup index builder. Importers fill it from the legacy TSV corpus,
GeoNames dumps, or (later) OSM extracts.
"""

from __future__ import annotations

import random
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS token (
    id        INTEGER PRIMARY KEY,
    class     TEXT NOT NULL,
    name      TEXT NOT NULL,
    abbrev    TEXT,
    parent_id INTEGER REFERENCES token(id),
    source    TEXT NOT NULL DEFAULT '',
    freq      REAL NOT NULL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_token_parent ON token(parent_id);
CREATE INDEX IF NOT EXISTS idx_token_class ON token(class);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


@dataclass(frozen=True)
class Token:
    id: int
    class_name: str
    name: str
    parent_id: int | None
    abbrev: str | None = None

    @staticmethod
    def from_row(row: sqlite3.Row | tuple) -> Token:
        return Token(id=row[0], class_name=row[1], name=row[2], parent_id=row[3], abbrev=row[4])


_TOKEN_COLS = "id, class, name, parent_id, abbrev"


class Gazetteer:
    """Read/write access to a gazetteer database."""

    def __init__(self, path: str | Path, readonly: bool = False):
        self.path = str(path)
        if readonly:
            self.conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        else:
            self.conn = sqlite3.connect(self.path)
            self.conn.executescript(SCHEMA)
        self.conn.execute("PRAGMA foreign_keys = OFF")  # parents may be inserted after children
        self._id_range_cache: dict[str | None, tuple[int, int]] = {}

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Gazetteer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------ writing

    def add_tokens(
        self, rows: Iterable[tuple[int, str, str, str | None, int | None, str]], batch_size: int = 10000
    ) -> int:
        """Insert ``(id, class, name, abbrev, parent_id, source)`` rows in batches."""
        cursor = self.conn.cursor()
        total = 0
        batch: list[tuple] = []
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_size:
                cursor.executemany(
                    "INSERT OR REPLACE INTO token(id, class, name, abbrev, parent_id, source) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    batch,
                )
                total += len(batch)
                batch.clear()
        if batch:
            cursor.executemany(
                "INSERT OR REPLACE INTO token(id, class, name, abbrev, parent_id, source) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                batch,
            )
            total += len(batch)
        self.conn.commit()
        self._id_range_cache.clear()
        return total

    def prune_dangling_parents(self) -> int:
        """NULL out parent references that point to non-existent tokens."""
        cursor = self.conn.execute(
            "UPDATE token SET parent_id = NULL WHERE parent_id IS NOT NULL "
            "AND NOT EXISTS (SELECT 1 FROM token p WHERE p.id = token.parent_id)"
        )
        self.conn.commit()
        return cursor.rowcount

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))
        self.conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------ reading

    def classes(self) -> list[str]:
        """Distinct token classes, most frequent first (deterministic tiebreak)."""
        rows = self.conn.execute(
            "SELECT class, COUNT(*) AS n FROM token GROUP BY class ORDER BY n DESC, class ASC"
        ).fetchall()
        return [r[0] for r in rows]

    def count(self, class_name: str | None = None) -> int:
        if class_name is None:
            return self.conn.execute("SELECT COUNT(*) FROM token").fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM token WHERE class = ?", (class_name,)).fetchone()[0]

    def get(self, token_id: int) -> Token | None:
        row = self.conn.execute(f"SELECT {_TOKEN_COLS} FROM token WHERE id = ?", (token_id,)).fetchone()
        return Token.from_row(row) if row else None

    def parent(self, token: Token) -> Token | None:
        if token.parent_id is None:
            return None
        return self.get(token.parent_id)

    def ancestors(self, token: Token) -> dict[str, Token]:
        """All transitive parents keyed by class name (cycle-safe)."""
        result: dict[str, Token] = {}
        seen = {token.id}
        current = self.parent(token)
        while current is not None and current.id not in seen:
            result[current.class_name] = current
            seen.add(current.id)
            current = self.parent(current)
        return result

    def random_child(self, token: Token, rng: random.Random) -> Token | None:
        n = self.conn.execute(
            "SELECT COUNT(*) FROM token WHERE parent_id = ?", (token.id,)
        ).fetchone()[0]
        if n == 0:
            return None
        row = self.conn.execute(
            f"SELECT {_TOKEN_COLS} FROM token WHERE parent_id = ? LIMIT 1 OFFSET ?",
            (token.id, rng.randrange(n)),
        ).fetchone()
        return Token.from_row(row)

    def random_descendants(self, token: Token, rng: random.Random) -> dict[str, Token]:
        """One random child per level below the token, keyed by class name."""
        result: dict[str, Token] = {}
        seen = {token.id}
        current = self.random_child(token, rng)
        while current is not None and current.id not in seen:
            result[current.class_name] = current
            seen.add(current.id)
            current = self.random_child(current, rng)
        return result

    def _id_range(self, class_name: str | None) -> tuple[int | None, int | None]:
        if class_name not in self._id_range_cache:
            where = "" if class_name is None else "WHERE class = ?"
            args: tuple = () if class_name is None else (class_name,)
            # separate statements: SQLite only applies the min/max index
            # optimization to single-aggregate queries (a combined
            # MIN(id), MAX(id) falls back to a full table scan)
            lo = self.conn.execute(f"SELECT MIN(id) FROM token {where}", args).fetchone()[0]
            hi = self.conn.execute(f"SELECT MAX(id) FROM token {where}", args).fetchone()[0]
            self._id_range_cache[class_name] = (lo, hi)
        return self._id_range_cache[class_name]

    def random_token(self, rng: random.Random, class_name: str | None = None) -> Token:
        """Uniform-ish random token via id probing (importer ids are dense per class)."""
        args: tuple = () if class_name is None else (class_name,)
        lo, hi = self._id_range(class_name)
        if lo is None:
            raise LookupError("gazetteer is empty")
        for _ in range(64):
            candidate = self.conn.execute(
                f"SELECT {_TOKEN_COLS} FROM token WHERE id >= ? "
                + ("" if class_name is None else "AND class = ? ")
                + "ORDER BY id LIMIT 1",
                (rng.randint(lo, hi), *args),
            ).fetchone()
            if candidate:
                return Token.from_row(candidate)
        raise LookupError("could not sample a random token")

    def iter_tokens(self, class_name: str | None = None) -> Iterator[Token]:
        where = "" if class_name is None else "WHERE class = ?"
        args: tuple = () if class_name is None else (class_name,)
        for row in self.conn.execute(f"SELECT {_TOKEN_COLS} FROM token {where} ORDER BY id", args):
            yield Token.from_row(row)

    def iter_names(self, class_name: str | None = None) -> Iterator[tuple[str, float]]:
        """Distinct token names with accumulated frequency (corrector vocabularies)."""
        where = "" if class_name is None else "WHERE class = ?"
        args: tuple = () if class_name is None else (class_name,)
        query = f"SELECT name, SUM(freq) FROM token {where} GROUP BY name"
        for name, freq in self.conn.execute(query, args):
            yield name, float(freq)
