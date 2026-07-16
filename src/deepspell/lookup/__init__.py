"""FTS5 lookup of real database entries for classified query tokens."""

from .fts5 import FtsLookup, build_lookup_index

__all__ = ["FtsLookup", "build_lookup_index"]
