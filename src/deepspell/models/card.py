"""Model cards: JSON descriptors persisted next to the weight files.

Every trained artifact is a pair ``<base>.json`` (card) + ``<base>.pt``
(state dict). The card pins the charset, classes and hyperparameters, so a
model can always be reconstructed from the JSON path alone — the same role
the v1 model JSONs played for the TF checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field

from ..charset import CharVocab, ClassVocab
from .completer import Completer
from .encoder import TokenEncoder
from .tagger import Tagger

ModelKind = Literal["tagger", "completer", "encoder"]


class ModelCard(BaseModel):
    kind: ModelKind
    charset: str
    lowercase: bool = True
    classes: list[str] = Field(default_factory=list)  # without the implicit EOL class
    hparams: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)
    format_version: int = 2

    def char_vocab(self) -> CharVocab:
        return CharVocab(charset=self.charset, lowercase=self.lowercase)

    def class_vocab(self) -> ClassVocab:
        return ClassVocab(names=tuple(self.classes))

    def build(self) -> torch.nn.Module:
        vocab_size = len(self.char_vocab())
        if self.kind == "tagger":
            return Tagger(vocab_size, len(self.class_vocab()), **self.hparams)
        if self.kind == "completer":
            return Completer(vocab_size, len(self.class_vocab()), **self.hparams)
        if self.kind == "encoder":
            return TokenEncoder(vocab_size, **self.hparams)
        raise ValueError(f"unknown model kind {self.kind!r}")


def save_model(model: torch.nn.Module, card: ModelCard, base_path: str | Path) -> Path:
    """Write ``<base>.json`` + ``<base>.pt``; returns the JSON path."""
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    json_path = base.with_suffix(".json")
    json_path.write_text(card.model_dump_json(indent=2), encoding="utf-8")
    torch.save(model.state_dict(), base.with_suffix(".pt"))
    return json_path


def load_model(json_path: str | Path, device: str = "cpu") -> tuple[torch.nn.Module, ModelCard]:
    """Load a model from its card path; returns ``(model.eval(), card)``."""
    json_path = Path(json_path)
    card = ModelCard.model_validate_json(json_path.read_text(encoding="utf-8"))
    model = card.build()
    state = torch.load(json_path.with_suffix(".pt"), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, card
