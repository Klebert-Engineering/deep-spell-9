"""Importer for the v1 training corpus TSV format.

Line format (tab-separated)::

    CLASS  LOCAL_ID  NAME  *  PARENT_CLASS  PARENT_LOCAL_ID  [ABBREV]

``*`` in the parent class column means "no parent". Column 3 is unused in
the v1 corpus files. Local ids are only unique per class; the staged loader
renumbers them into dense global ids.
"""

from __future__ import annotations

import codecs
import logging
from collections.abc import Iterator
from pathlib import Path

from . import Gazetteer
from .staging import StagedRow, load_staged

logger = logging.getLogger(__name__)

WILDCARD = "*"
_STRIDE = 100_000_000  # legacy local ids stay well below this


def _rows(tsv_path: Path, source: str) -> Iterator[StagedRow]:
    class_index: dict[str, int] = {}

    def cls_idx(name: str) -> int:
        return class_index.setdefault(name, len(class_index))

    with codecs.open(str(tsv_path), encoding="utf-8") as tsv_file:
        for line_no, line in enumerate(tsv_file, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            class_name = parts[0].strip()
            name = parts[2].strip()
            if not class_name or not name:
                continue
            try:
                temp_id = cls_idx(class_name) * _STRIDE + int(parts[1])
                parent_temp_id = None
                if parts[4].strip() != WILDCARD:
                    parent_temp_id = cls_idx(parts[4].strip()) * _STRIDE + int(parts[5])
            except ValueError:
                logger.warning("skipping malformed line %d: %r", line_no, line[:80])
                continue
            abbrev = parts[6].strip() if len(parts) > 6 and parts[6].strip() else None
            yield temp_id, class_name, name, abbrev, parent_temp_id, source


def import_legacy_tsv(gaz: Gazetteer, tsv_path: str | Path, source: str = "legacy_tsv") -> int:
    """Import a v1 TSV corpus file. Returns the number of imported rows."""
    count = load_staged(gaz, _rows(Path(tsv_path), source))
    gaz.set_meta(f"import:{source}", str(tsv_path))
    return count
