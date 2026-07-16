"""Bulk loading with id renumbering.

Importers stream rows with *temporary* ids (any scheme that is unique within
the import, e.g. ``class_index * stride + source_local_id``). The staged
loader then renumbers everything into dense ids ``1..N``, ordered by
``(class, temp_id)``, and rewrites parent references through the same
mapping. Dense, per-class-contiguous ids are what makes
:meth:`Gazetteer.random_token` O(log n) and unbiased.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from . import Gazetteer

logger = logging.getLogger(__name__)

#: Row shape importers must produce: (temp_id, class, name, abbrev, parent_temp_id, source)
StagedRow = tuple[int, str, str, str | None, int | None, str]


def load_staged(gaz: Gazetteer, rows: Iterable[StagedRow], batch_size: int = 20000) -> int:
    """Load rows through a staging table and renumber to dense ids. Returns row count."""
    conn = gaz.conn
    conn.execute("DROP TABLE IF EXISTS token_staging")
    conn.execute(
        "CREATE TABLE token_staging ("
        "id INTEGER PRIMARY KEY, class TEXT, name TEXT, abbrev TEXT, parent_id INTEGER, source TEXT)"
    )
    cursor = conn.cursor()
    batch: list[StagedRow] = []
    total = 0

    def flush() -> None:
        nonlocal total
        if batch:
            cursor.executemany(
                "INSERT OR REPLACE INTO token_staging VALUES (?, ?, ?, ?, ?, ?)", batch
            )
            total += len(batch)
            batch.clear()

    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            flush()
            if total % 1_000_000 < batch_size:
                logger.info("staged %d rows ...", total)
    flush()
    conn.commit()

    logger.info("renumbering %d staged rows ...", total)
    base = conn.execute("SELECT COALESCE(MAX(id), 0) FROM token").fetchone()[0]
    conn.executescript(
        f"""
        CREATE TEMP TABLE idmap AS
            SELECT id AS old_id,
                   {base} + ROW_NUMBER() OVER (ORDER BY class, id) AS new_id
            FROM token_staging;
        CREATE UNIQUE INDEX idmap_old ON idmap(old_id);
        INSERT INTO token(id, class, name, abbrev, parent_id, source)
            SELECT m.new_id, s.class, s.name, s.abbrev, pm.new_id, s.source
            FROM token_staging s
            JOIN idmap m ON m.old_id = s.id
            LEFT JOIN idmap pm ON pm.old_id = s.parent_id;
        DROP TABLE token_staging;
        DROP TABLE idmap;
        """
    )
    conn.commit()
    gaz._id_range_cache.clear()
    inserted = conn.execute("SELECT COUNT(*) FROM token").fetchone()[0]
    logger.info("gazetteer now holds %d tokens", inserted)
    return total
