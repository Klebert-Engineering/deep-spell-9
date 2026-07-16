"""Service configuration (successor of ``service.json``).

Loadable from a JSON file and overridable via ``DS9_``-prefixed environment
variables (e.g. ``DS9_PORT=9000``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DS9_", extra="ignore")

    tagger: Path
    completer: Path

    corrector: Literal["none", "symspell", "embedding"] = "none"
    symspell_dictionary: Path | None = None
    encoder: Path | None = None  # embedding backend: model card json
    embedding_space: Path | None = None  # embedding backend: .npz/.tokens base path

    lookup: Path | None = None  # FTS5 database (native or NDS)
    lookup_kind: Literal["native", "nds"] = "native"
    nds: dict | None = None  # v1 ``fts_db`` block (table/column mapping overrides)

    beam_width: int = 6
    max_completion_length: int = 16
    corrections: int = 3
    device: str = "cpu"

    @classmethod
    def from_json(cls, path: str | Path) -> ServiceConfig:
        with open(path, encoding="utf-8") as config_file:
            return cls(**json.load(config_file))
