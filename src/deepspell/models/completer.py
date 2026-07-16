"""Class-conditioned character language model (successor of ``DSLstmExtrapolator``).

Input at each step is the sum of a character embedding and a class embedding;
the model predicts the next character and the next class. Completions are
decoded with :func:`deepspell.decode.beam.beam_search` in plain Python.
"""

from __future__ import annotations

import torch
from torch import nn

State = tuple[torch.Tensor, torch.Tensor]


class Completer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        emb_dim: int = 64,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.char_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.class_embedding = nn.Embedding(num_classes, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.char_head = nn.Linear(hidden, vocab_size)
        self.class_head = nn.Linear(hidden, num_classes)

    def forward(
        self,
        char_ids: torch.Tensor,
        class_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        state: State | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, State]:
        """``char_ids``/``class_ids`` [B, T] -> char logits, class logits, lstm state.

        ``lengths`` enables packing for padded training batches; leave None
        for stepwise decoding (T == 1) or unpadded batches.
        """
        embedded = self.char_embedding(char_ids) + self.class_embedding(class_ids)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs, new_state = self.lstm(packed, state)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True, total_length=char_ids.shape[1]
            )
        else:
            outputs, new_state = self.lstm(embedded, state)
        return self.char_head(outputs), self.class_head(outputs), new_state
