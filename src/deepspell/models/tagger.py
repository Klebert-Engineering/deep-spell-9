"""Character-class tagger: BiLSTM over characters -> class logits per char.

Successor of the v1 ``DSLstmDiscriminator`` (which used a separate backward
RNN feeding a forward RNN; a standard bidirectional LSTM subsumes that).
"""

from __future__ import annotations

import torch
from torch import nn

from ..charset import CharVocab, ClassVocab
from ..decode.classes import CharClassProbs


class Tagger(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        emb_dim: int = 64,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(2 * hidden, num_classes)

    def forward(self, char_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """``char_ids`` [B, T], ``lengths`` [B] -> class logits [B, T, C]."""
        embedded = self.embedding(char_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.lstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True, total_length=char_ids.shape[1]
        )
        return self.head(unpacked)

    @torch.inference_mode()
    def predict(self, text: str, char_vocab: CharVocab, class_vocab: ClassVocab) -> CharClassProbs:
        """Per-character class distributions for *normalized* text.

        Returns one entry per input character, each a probability-descending
        list of ``(class_name, probability)``.
        """
        if not text:
            return []
        device = next(self.parameters()).device
        ids = torch.tensor([char_vocab.encode(text, add_eos=True)], device=device)
        lengths = torch.tensor([ids.shape[1]])
        probs = torch.softmax(self(ids, lengths)[0], dim=-1)[: len(text)].cpu()
        result: CharClassProbs = []
        for char_probs in probs:
            ranked = sorted(
                ((class_vocab.decode(i), float(p)) for i, p in enumerate(char_probs)),
                key=lambda entry: entry[1],
                reverse=True,
            )
            result.append(ranked)
        return result
